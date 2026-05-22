from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import logging
import warnings

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"urllib3 v2 only supports OpenSSL")

# Ensure src/ is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Force project-local keys to win over any stale shell-exported credentials.
load_dotenv(PROJECT_ROOT / ".env", override=True)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tcp_agent.utils.llm_utils import openai_compat_base_from_env

from tcp_agent.agent.tcp_agent import run_agent
from tcp_agent.agent.ranker import build_ranked_df, normalize_ranked_items
from tcp_agent.evaluation import apfd, apfdc, failure_recall_at_k
from tcp_agent.config import AgentMode
from tcp_agent.data_cache import load_dataset
from tcp_agent.tools.feature_extractor import candidate_risk_score
import pandas as pd

# default settings — can be overridden via CLI
_mode = AgentMode.PILOT
_batch_size = 40
_filter_model = "gpt-5-nano"
_ranking_model = "gemini-3-flash-preview"
_filter_gap = 0.0
_ranking_workers = 1
_ranking_batch_size = 8
_merge_agent = True
_candidate_cap: int | None = None

def _print_diagnosis(
    ranked_df, target_build, build_idx, total_builds, *, failed_builds_only: bool
):
    """Print a per-build failure breakdown: where each failing test was ranked."""
    n = len(ranked_df)
    failures = ranked_df[ranked_df["Verdict"] != 0]
    m = len(failures)
    build_apfd  = apfd(ranked_df)
    build_apfdc = apfdc(ranked_df)
    build_recall_at_10 = failure_recall_at_k(ranked_df, k=10)

    print(f"\n  [Build {target_build}  ({build_idx+1}/{total_builds})  —  {m} failure(s) / {n} tests]")
    if m == 0:
        if failed_builds_only:
            print("    (no failures in target build — unexpected for --failed-builds-only mode)")
        else:
            print("    (no failures in this build)")
    else:
        for idx, row in failures.iterrows():
            rank = idx + 1
            pct  = rank / n * 100
            flag = "✓" if rank <= 10 else ("✗" if rank > n // 2 else "")
            print(f"    rank {rank:>4} / {n}  ({pct:5.1f}%)  test {int(row['Test'])}  {flag}")
    print(
        f"    APFD={build_apfd:.4f}  APFDc={build_apfdc:.4f}  "
        f"Recall@10={build_recall_at_10:.4f}"
    )


def _extract_merge_status(ranked: list[dict]) -> str:
    for item in ranked:
        if isinstance(item, dict) and item.get("_merge_status"):
            return str(item["_merge_status"])
    return "unknown"


def _extract_merge_missing_count(ranked: list[dict]) -> int:
    for item in ranked:
        if not isinstance(item, dict):
            continue
        raw = item.get("_merge_missing_count")
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0
    return 0


_FORBIDDEN_AGENT_COLS = {"Verdict", "Duration"}


def _build_agent_feature_frame(target):
    """Target-build feature rows that are safe to show to the LLM.

    This matches the paper's prediction setup: rank tests in the target build
    from that build's feature vector, while hiding outcome/cost columns used
    later for APFD/APFDc evaluation.
    """
    drop_cols = [c for c in _FORBIDDEN_AGENT_COLS if c in target.columns]
    return target.drop(columns=drop_cols).copy()


def _feature_profile_score_map(features) -> tuple[set[int], dict[int, float], dict[int, float]]:
    latest = (
        features.sort_values("Build", ascending=False)
        .groupby("Test")
        .first()
        .reset_index()
    )
    score_map: dict[int, float] = {}
    exec_map: dict[int, float] = {}
    for rec in latest.to_dict("records"):
        tid = int(rec["Test"])
        score_map[tid] = candidate_risk_score({"test": tid, **rec})
        try:
            exec_map[tid] = float(rec.get("REC_RecentAvgExeTime", rec.get("REC_TotalAvgExeTime", rec.get("Duration", 0.0))) or 0.0)
        except (TypeError, ValueError):
            exec_map[tid] = 0.0
    return set(score_map), score_map, exec_map


def _apply_candidate_cap(agent_features, target, cap: int | None):
    """Return capped target feature rows and cap metadata for one target build."""
    if cap is None or cap <= 0:
        all_ids, score_map, exec_map = _feature_profile_score_map(agent_features)
        return agent_features, {
            "enabled": False,
            "cap": "",
            "selected_ids": sorted(all_ids),
            "score_map": score_map,
            "exec_map": exec_map,
            "tail_ids": [],
            "recall": "",
            "selected_count": len(all_ids),
            "tail_count": 0,
        }

    all_ids, score_map, exec_map = _feature_profile_score_map(agent_features)
    selected = sorted(
        all_ids,
        key=lambda tid: (score_map.get(tid, 0.0), -exec_map.get(tid, 0.0), tid),
        reverse=True,
    )[:cap]
    selected_set = set(selected)
    capped_features = agent_features[agent_features["Test"].isin(selected_set)]

    target_ids = {int(tid) for tid in target["Test"].unique().tolist()}
    target_fail_ids = {int(tid) for tid in target.loc[target["Verdict"] != 0, "Test"].unique().tolist()}
    included_failures = target_fail_ids & selected_set
    recall = (len(included_failures) / len(target_fail_ids)) if target_fail_ids else 1.0
    tail_ids = sorted(
        target_ids - selected_set,
        key=lambda tid: (-score_map.get(tid, 0.0), exec_map.get(tid, 0.0), tid),
    )
    return capped_features, {
        "enabled": True,
        "cap": cap,
        "selected_ids": selected,
        "score_map": score_map,
        "exec_map": exec_map,
        "tail_ids": tail_ids,
        "recall": recall,
        "selected_count": len(selected),
        "tail_count": len(tail_ids),
    }


def _filter_history_to_failed_builds(history):
    """Keep only historical builds that had at least one failing test."""
    failed_history_builds = history.loc[history["Verdict"] != 0, "Build"].unique()
    return history[history["Build"].isin(failed_history_builds)]


def _append_candidate_tail(ranked: list[dict], cap_meta: dict) -> list[dict]:
    if not cap_meta.get("enabled"):
        return ranked

    ranked_ids = {
        int(item["test"])
        for item in ranked
        if isinstance(item, dict) and str(item.get("test", "")).isdigit()
    }
    score_map = cap_meta.get("score_map", {})
    exec_map = cap_meta.get("exec_map", {})
    next_priority = max((int(item.get("priority", 0)) for item in ranked), default=0) + 1
    tail = []
    for tid in cap_meta.get("tail_ids", []):
        if tid in ranked_ids:
            continue
        tail.append({
            "test": str(tid),
            "priority": next_priority + len(tail),
            "confidence": 0.05,
            "reason": "Outside candidate cap; deterministic tail.",
            "_merge_status": _extract_merge_status(ranked),
            "_merge_missing_count": _extract_merge_missing_count(ranked),
            "_candidate_score": score_map.get(tid, 0.0),
            "_candidate_exec_time": exec_map.get(tid, 0.0),
        })
    return ranked + tail


def evaluate(
    csv_path,
    verbose=False,
    eval_window=5,
    gap=65.0,
    no_validation=False,
    failed_builds_only=False,
    history_failed_builds_only=False,
    diagnose=False,
    build_results_csv: Path | None = None,
    build_results_lock: threading.Lock | None = None,
    llm_trace_csv: Path | None = None,
    llm_trace_lock: threading.Lock | None = None,
):
    """
    Rolling-window evaluation over the last `eval_window` builds.

    When failed_builds_only=True, only builds that contain at least one test
    failure are used as targets — zero-failure builds carry no ranking signal
    and would inflate APFD/APFDc to 1.0.

    For each target build B:
      - history = all builds BEFORE B  (used to ensure prior data exists)
      - agent input = build B feature rows without Verdict/Duration
      - target = build B with Verdict/Duration for scoring after ranking

    When history_failed_builds_only=True, the prior-data availability check is
    restricted to historical builds that had at least one failing test. The LLM
    still receives only sanitized target-build feature rows.
    """
    df = load_dataset(csv_path)
    min_build = df["Build"].min()

    if failed_builds_only:
        candidate_builds = sorted(df[df["Verdict"] != 0]["Build"].unique())
    else:
        candidate_builds = sorted(df["Build"].unique())

    # drop the very first build — it has no history
    eligible = [b for b in candidate_builds if b > min_build]

    if not eligible:
        return 0.0, 0.0, 0.0, 0, {}

    if len(eligible) < eval_window:
        eval_window = len(eligible)

    target_builds = eligible[-eval_window:]

    all_apfd, all_apfdc, all_recall_at_10 = [], [], []
    merge_status_counts: dict[str, int] = {}
    merge_missing_total = 0
    cap_recalls: list[float] = []

    for i, target_build in enumerate(target_builds):
        build_start = time.time()
        tmp_path = None
        history = df[df["Build"] < target_build]
        target  = df[df["Build"] == target_build]
        if history_failed_builds_only:
            history = _filter_history_to_failed_builds(history)

        if history.empty:
            continue

        try:
            agent_features = _build_agent_feature_frame(target)
            agent_input, cap_meta = _apply_candidate_cap(agent_features, target, _candidate_cap)
            if cap_meta.get("enabled") and cap_meta.get("recall") != "":
                cap_recalls.append(float(cap_meta["recall"]))

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, prefix="tcp_target_features_"
            ) as tmp:
                agent_input.to_csv(tmp, index=False)
                tmp_path = tmp.name

            ranked = run_agent(
                tmp_path,
                mode=_mode,
                batch_size=_batch_size,
                filter_model=_filter_model,
                ranking_model=_ranking_model,
                no_validation=no_validation,
                filter_gap=_filter_gap,
                ranking_workers=_ranking_workers,
                ranking_batch_size=_ranking_batch_size,
                merge_agent=_merge_agent,
            )
            ranked = _append_candidate_tail(ranked, cap_meta)
            merge_status = _extract_merge_status(ranked)
            merge_missing_count = _extract_merge_missing_count(ranked)
            merge_status_counts[merge_status] = merge_status_counts.get(merge_status, 0) + 1
            merge_missing_total += merge_missing_count

            if verbose:
                print(f"\n  [Build {target_build} — {i+1}/{len(target_builds)}]")
                for item in ranked:
                    print(f"    #{item['priority']} test {item['test']} — {item['reason']}")

            ranked_df = build_ranked_df(ranked, target)
            build_apfd = apfd(ranked_df)
            build_apfdc = apfdc(ranked_df)
            build_recall_at_10 = failure_recall_at_k(ranked_df, k=10)
            build_wall_seconds = time.time() - build_start
            all_apfd.append(build_apfd)
            all_apfdc.append(build_apfdc)
            all_recall_at_10.append(build_recall_at_10)

            if llm_trace_csv:
                _append_llm_trace(
                    llm_trace_csv,
                    llm_trace_lock or build_results_lock or threading.Lock(),
                    dataset=Path(csv_path).name,
                    target_build=target_build,
                    agent_input=agent_input,
                    ranked=ranked,
                    target=target,
                    build_apfd=build_apfd,
                    build_apfdc=build_apfdc,
                    build_recall_at_10=build_recall_at_10,
                    build_wall_seconds=build_wall_seconds,
                    cap_meta=cap_meta,
                )

            if build_results_csv:
                _append_build_result(
                    build_results_csv,
                    build_results_lock or threading.Lock(),
                    {
                        "dataset": Path(csv_path).name,
                        "target_build": target_build,
                        "build_index": i + 1,
                        "total_builds": len(target_builds),
                        "apfd": f"{build_apfd:.6f}",
                        "apfdc": f"{build_apfdc:.6f}",
                        "recall_at_10": f"{build_recall_at_10:.6f}",
                        "num_tests": len(ranked_df),
                        "num_failures": int((ranked_df["Verdict"] != 0).sum()),
                        "filter_model": _filter_model,
                        "ranking_model": _ranking_model,
                        "eval_window": eval_window,
                        "failed_builds_only": "1" if failed_builds_only else "0",
                        "history_failed_builds_only": "1" if history_failed_builds_only else "0",
                        "batch_size": _batch_size,
                        "ranking_batch_size": _ranking_batch_size,
                        "ranking_workers": _ranking_workers,
                        "merge_agent": "1" if _merge_agent else "0",
                        "merge_status": merge_status,
                        "merge_missing_count": merge_missing_count,
                        "candidate_cap": cap_meta.get("cap", ""),
                        "candidate_selected_count": cap_meta.get("selected_count", ""),
                        "candidate_tail_count": cap_meta.get("tail_count", ""),
                        "candidate_cap_recall": (
                            f"{cap_meta['recall']:.6f}"
                            if isinstance(cap_meta.get("recall"), float)
                            else cap_meta.get("recall", "")
                        ),
                        "wall_seconds": f"{build_wall_seconds:.1f}",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "status": "ok",
                        "error": "",
                    },
                )

            if diagnose:
                _print_diagnosis(
                    ranked_df, target_build, i, len(target_builds),
                    failed_builds_only=failed_builds_only,
                )
        except Exception as e:
            if build_results_csv:
                _append_build_result(
                    build_results_csv,
                    build_results_lock or threading.Lock(),
                    {
                        "dataset": Path(csv_path).name,
                        "target_build": target_build,
                        "build_index": i + 1,
                        "total_builds": len(target_builds),
                        "apfd": "",
                        "apfdc": "",
                        "recall_at_10": "",
                        "num_tests": len(target),
                        "num_failures": int((target["Verdict"] != 0).sum()),
                        "filter_model": _filter_model,
                        "ranking_model": _ranking_model,
                        "eval_window": eval_window,
                        "failed_builds_only": "1" if failed_builds_only else "0",
                        "history_failed_builds_only": "1" if history_failed_builds_only else "0",
                        "batch_size": _batch_size,
                        "ranking_batch_size": _ranking_batch_size,
                        "ranking_workers": _ranking_workers,
                        "merge_agent": "1" if _merge_agent else "0",
                        "merge_status": "",
                        "merge_missing_count": "",
                        "candidate_cap": _candidate_cap or "",
                        "candidate_selected_count": "",
                        "candidate_tail_count": "",
                        "candidate_cap_recall": "",
                        "wall_seconds": f"{time.time() - build_start:.1f}",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "status": "failed",
                        "error": str(e)[:500],
                    },
                )
            raise
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass

        # rate-limit gap between agent calls (skip after the last one)
        if i < len(target_builds) - 1 and gap > 0:
            time.sleep(gap)

    if not all_apfd:
        return 0.0, 0.0, 0.0, 0, {}

    n = len(all_apfd)
    meta = {
        "merge_status_counts": ";".join(
            f"{k}:{v}" for k, v in sorted(merge_status_counts.items())
        ),
        "merge_repaired_builds": merge_status_counts.get("repaired", 0),
        "merge_missing_total": merge_missing_total,
        "candidate_cap_recall_avg": (
            sum(cap_recalls) / len(cap_recalls) if cap_recalls else ""
        ),
        "candidate_cap_recall_min": min(cap_recalls) if cap_recalls else "",
    }
    return (
        sum(all_apfd)          / n,
        sum(all_apfdc)         / n,
        sum(all_recall_at_10)  / n,
        n,
        meta,
    )


def main():
    global _mode, _batch_size, _filter_model, _ranking_model, _filter_gap, _ranking_workers, _ranking_batch_size, _merge_agent, _candidate_cap

    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", type=Path)
    g.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--mode",
        choices=["pilot", "production"],
        default="pilot",
        help="Agent mode: 'pilot' reads from CSV (default), 'production' extracts from real sources",
    )
    parser.add_argument(
        "--batch-size", type=int, default=40,
        help="Number of tests per Filter Agent batch (default: 40, sized for "
             "the full 150-feature Yaraghi 2022 set). Auto-splits on output-"
             "token-limit errors; lower if you see frequent splits.",
    )
    parser.add_argument(
        "--filter-model", type=str, default="gpt-5-nano",
        help="LLM model for the Filter Agent (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--ranking-model", type=str, default="gemini-3-flash-preview",
        help="LLM model for the Ranking Agent (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--eval-window", type=int, default=5,
        help="Number of most-recent builds to evaluate against (default: 5). "
             "The agent is called once per target build with sanitized "
             "target-build feature rows.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--gap", type=float, default=0.0,
        help="Seconds to wait between agent calls to avoid rate limits (0 to disable). "
             "Defaults to 0 because the default split-provider combo (OpenAI filter + "
             "Gemini ranking) has high RPM headroom; raise if you switch to lower-tier providers.",
    )
    parser.add_argument(
        "--filter-gap", type=float, default=0.0,
        help="Seconds to sleep between Filter Agent LLM batches (default: 0). "
             "Increase on rate-limited providers to space out filter calls within a build.",
    )
    parser.add_argument(
        "--ranking-workers", type=int, default=1,
        help="Number of concurrent Ranking Agent batches (default: 1, sequential). "
             "Raise to 4+ on higher-tier API accounts. Multiplies concurrent LLM "
             "calls — the main source of 429s on rate-limited providers.",
    )
    parser.add_argument(
        "--ranking-batch-size", type=int, default=8,
        help="Number of high-risk tests per Ranking Agent batch (default: 8). "
             "Try 12 or 15 on stronger/local models if JSON remains valid.",
    )
    parser.add_argument(
        "--no-merge-agent",
        dest="merge_agent",
        action="store_false",
        default=True,
        help="Disable the third Merge Agent that globally reorders ranking batches.",
    )
    parser.add_argument(
        "--candidate-cap",
        type=int,
        default=0,
        help="Deterministically preselect the top K risky candidate tests per build before LLM agents. 0 disables.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of datasets to evaluate in parallel when using --data-dir (default: 1). "
             "Use 3-4 for OpenAI/Anthropic tier-1 limits, higher on bigger tiers.",
    )
    parser.add_argument(
        "--results-csv", type=Path, default=Path("results/evaluation_summary.csv"),
        help="Path to the persistent results CSV. Existing rows are read on startup so already-"
             "evaluated datasets are skipped (automatic resume).",
    )
    parser.add_argument(
        "--build-results-csv",
        type=Path,
        default=None,
        help="Path to per-build detailed results CSV. Defaults to <results-csv stem>_builds.csv.",
    )
    parser.add_argument(
        "--llm-trace-csv",
        type=Path,
        default=None,
        help="Optional per-test trace CSV showing sanitized LLM input features, final LLM ranking output, true evaluation labels, and build wall time.",
    )
    parser.add_argument(
        "--no-validation", action="store_true",
        help="Bypass the validation layer (use LLM output even if invalid)",
    )
    parser.add_argument(
        "--failed-builds-only",
        dest="failed_builds_only",
        action="store_true",
        default=True,
        help="Only evaluate on builds that contain at least one test failure (default: on). "
             "Zero-failure builds carry no signal and inflate APFD/APFDc to 1.0.",
    )
    parser.add_argument(
        "--all-builds",
        dest="failed_builds_only",
        action="store_false",
        help="Evaluate on every eligible build, including those with no failing tests.",
    )
    parser.add_argument(
        "--history-failed-builds-only",
        action="store_true",
        help="Restrict the prior-data availability check to historical builds "
             "that had at least one failing test. Target selection is unchanged; "
             "the LLM still receives sanitized target-build feature rows.",
    )
    parser.add_argument(
        "--diagnose-failures",
        action="store_true",
        help="After each build, print a breakdown of where each failing test was ranked.",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        metavar="URL",
        help="OpenAI-compatible API base (e.g. Ollama: http://127.0.0.1:11434/v1). Sets OPENAI_BASE_URL for Qwen/OpenAI-compat backends.",
    )
    args = parser.parse_args()
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url.strip()
    if args.build_results_csv is None:
        args.build_results_csv = args.results_csv.with_name(
            f"{args.results_csv.stem}_builds{args.results_csv.suffix}"
        )
    if args.build_results_csv.resolve() == args.results_csv.resolve():
        sys.exit("--build-results-csv must be different from --results-csv")
    _mode = AgentMode.PILOT if args.mode == "pilot" else AgentMode.PRODUCTION
    _batch_size = args.batch_size
    _filter_model = args.filter_model
    _ranking_model = args.ranking_model
    _filter_gap = args.filter_gap
    _ranking_workers = args.ranking_workers
    _ranking_batch_size = args.ranking_batch_size
    _merge_agent = args.merge_agent
    _candidate_cap = args.candidate_cap if args.candidate_cap and args.candidate_cap > 0 else None

    needed_keys = set()
    for model_name in [_filter_model, _ranking_model]:
        name = model_name.lower()
        if "gemini" in name:
            needed_keys.add("GOOGLE_API_KEY")
        elif name.startswith("claude"):
            needed_keys.add("ANTHROPIC_API_KEY")
        elif "mistral" in name:
            needed_keys.add("MISTRAL_API_KEY")
        else:
            needed_keys.add("OPENAI_API_KEY")

    missing_keys: list[str] = []
    for k in sorted(needed_keys):
        if os.environ.get(k, "").strip():
            continue
        if k == "OPENAI_API_KEY" and openai_compat_base_from_env():
            continue
        missing_keys.append(k)
    if missing_keys:
        sys.exit(f"Missing required API key(s): {', '.join(missing_keys)}")

    if args.data:
        start = time.time()
        build_lock = threading.Lock()
        try:
            a, ac, recall_at_10, n_b, meta = evaluate(
                args.data,
                verbose=not args.quiet,
                eval_window=args.eval_window,
                gap=args.gap,
                no_validation=args.no_validation,
                failed_builds_only=args.failed_builds_only,
                history_failed_builds_only=args.history_failed_builds_only,
                diagnose=args.diagnose_failures,
                build_results_csv=args.build_results_csv,
                build_results_lock=build_lock,
                llm_trace_csv=args.llm_trace_csv,
                llm_trace_lock=build_lock,
            )
        except Exception as e:
            elapsed = time.time() - start
            _append_result(args.results_csv, threading.Lock(), {
                "dataset": args.data.name,
                "apfd": "", "apfdc": "", "recall_at_10": "",
                "filter_model": _filter_model,
                "ranking_model": _ranking_model,
                "eval_window": args.eval_window,
                "failed_builds_only": "1" if args.failed_builds_only else "0",
                "history_failed_builds_only": "1" if args.history_failed_builds_only else "0",
                "batch_size": _batch_size,
                "ranking_batch_size": _ranking_batch_size,
                "ranking_workers": _ranking_workers,
                "merge_agent": "1" if _merge_agent else "0",
                "merge_status_counts": "",
                "merge_repaired_builds": "",
                "merge_missing_total": "",
                "candidate_cap": _candidate_cap or "",
                "candidate_cap_recall_avg": "",
                "candidate_cap_recall_min": "",
                "build_results_csv": str(args.build_results_csv),
                "gap": args.gap,
                "filter_gap": args.filter_gap,
                "openai_base_url": os.environ.get("OPENAI_BASE_URL", ""),
                "wall_seconds": f"{elapsed:.1f}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "status": "failed",
                "error": str(e)[:500],
            })
            print(f"Failed result appended to {args.results_csv}")
            print(f"Per-build results appended to {args.build_results_csv}")
            if args.llm_trace_csv:
                print(f"LLM trace rows appended to {args.llm_trace_csv}")
            raise
        else:
            elapsed = time.time() - start
            _append_result(args.results_csv, threading.Lock(), {
                "dataset": args.data.name,
                "apfd": f"{a:.6f}",
                "apfdc": f"{ac:.6f}",
                "recall_at_10": f"{recall_at_10:.6f}",
                "filter_model": _filter_model,
                "ranking_model": _ranking_model,
                "eval_window": args.eval_window,
                "failed_builds_only": "1" if args.failed_builds_only else "0",
                "history_failed_builds_only": "1" if args.history_failed_builds_only else "0",
                "batch_size": _batch_size,
                "ranking_batch_size": _ranking_batch_size,
                "ranking_workers": _ranking_workers,
                "merge_agent": "1" if _merge_agent else "0",
                "merge_status_counts": meta.get("merge_status_counts", ""),
                "merge_repaired_builds": meta.get("merge_repaired_builds", ""),
                "merge_missing_total": meta.get("merge_missing_total", ""),
                "candidate_cap": _candidate_cap or "",
                "candidate_cap_recall_avg": (
                    f"{meta['candidate_cap_recall_avg']:.6f}"
                    if isinstance(meta.get("candidate_cap_recall_avg"), float)
                    else meta.get("candidate_cap_recall_avg", "")
                ),
                "candidate_cap_recall_min": (
                    f"{meta['candidate_cap_recall_min']:.6f}"
                    if isinstance(meta.get("candidate_cap_recall_min"), float)
                    else meta.get("candidate_cap_recall_min", "")
                ),
                "build_results_csv": str(args.build_results_csv),
                "gap": args.gap,
                "filter_gap": args.filter_gap,
                "openai_base_url": os.environ.get("OPENAI_BASE_URL", ""),
                "wall_seconds": f"{elapsed:.1f}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "status": "ok",
                "error": "",
            })
        print(
            f"APFD={a:.4f}  APFDc={ac:.4f}  Recall@10={recall_at_10:.4f}  "
            f"(avg over {n_b} build(s)"
            f"{'; failed-builds only' if args.failed_builds_only else ''}"
            f"{'; failed-history only' if args.history_failed_builds_only else ''})"
        )
        print(f"Result appended to {args.results_csv}")
        print(f"Per-build results appended to {args.build_results_csv}")
        if args.llm_trace_csv:
            print(f"LLM trace rows appended to {args.llm_trace_csv}")
        return

    _run_data_dir(args)


def _normalize_failed_builds_flag(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "y")


def _normalize_bool_key(raw) -> str:
    if isinstance(raw, bool):
        return "1" if raw else "0"
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "y"):
        return "1"
    if text in ("0", "false", "no", "n"):
        return "0"
    return str(raw)


def _completed_key_from_values(
    dataset: str,
    failed_builds_only: bool,
    history_failed_builds_only: bool,
    filter_model: str,
    ranking_model: str,
    eval_window: int | str,
    batch_size: int | str,
    ranking_batch_size: int | str,
    ranking_workers: int | str,
    merge_agent: bool | str,
    candidate_cap: int | str,
    gap: float | str,
    filter_gap: float | str,
    openai_base_url: str,
) -> tuple:
    return (
        dataset,
        failed_builds_only,
        history_failed_builds_only,
        str(filter_model),
        str(ranking_model),
        str(eval_window),
        str(batch_size),
        str(ranking_batch_size),
        str(ranking_workers),
        _normalize_bool_key(merge_agent),
        str(candidate_cap),
        str(gap),
        str(filter_gap),
        str(openai_base_url),
    )


def _load_completed(results_csv: Path) -> set[tuple]:
    """Rows with status ok for a specific experiment configuration.

    Older result files omit some config fields; those rows will only match if
    the current key also has blank values for those fields, so new experiments
    do not get skipped accidentally.
    """
    if not results_csv.exists():
        return set()
    done: set[tuple] = set()
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok" or not row.get("dataset"):
                continue
            fbo = _normalize_failed_builds_flag(row.get("failed_builds_only", ""))
            hfbo = _normalize_failed_builds_flag(row.get("history_failed_builds_only", ""))
            done.add(_completed_key_from_values(
                row["dataset"],
                fbo,
                hfbo,
                row.get("filter_model", ""),
                row.get("ranking_model", ""),
                row.get("eval_window", ""),
                row.get("batch_size", ""),
                row.get("ranking_batch_size", ""),
                row.get("ranking_workers", ""),
                row.get("merge_agent", ""),
                row.get("candidate_cap", ""),
                row.get("gap", ""),
                row.get("filter_gap", ""),
                row.get("openai_base_url", ""),
            ))
    return done


def _append_csv_row(csv_path: Path, lock: threading.Lock, fieldnames: list[str], row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        if csv_path.exists() and csv_path.stat().st_size > 0:
            with open(csv_path, newline="") as existing:
                reader = csv.DictReader(existing)
                existing_header = reader.fieldnames or []
                if existing_header != fieldnames:
                    old_rows = list(reader)
                    with open(csv_path, "w", newline="") as migrated:
                        writer = csv.DictWriter(migrated, fieldnames=fieldnames)
                        writer.writeheader()
                        for old_row in old_rows:
                            writer.writerow({
                                k: old_row.get(k, "") for k in fieldnames
                            })
                        migrated.flush()
                        os.fsync(migrated.fileno())

        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            f.flush()
            os.fsync(f.fileno())


def _append_csv_rows(csv_path: Path, lock: threading.Lock, fieldnames: list[str], rows: list[dict]):
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        if csv_path.exists() and csv_path.stat().st_size > 0:
            with open(csv_path, newline="") as existing:
                reader = csv.DictReader(existing)
                existing_header = reader.fieldnames or []
                if existing_header != fieldnames:
                    old_rows = list(reader)
                    with open(csv_path, "w", newline="") as migrated:
                        writer = csv.DictWriter(migrated, fieldnames=fieldnames)
                        writer.writeheader()
                        for old_row in old_rows:
                            writer.writerow({
                                k: old_row.get(k, "") for k in fieldnames
                            })
                        migrated.flush()
                        os.fsync(migrated.fileno())

        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
            f.flush()
            os.fsync(f.fileno())


def _json_safe(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _feature_json(row: dict, feature_cols: list[str]) -> str:
    return json.dumps(
        {col: _json_safe(row.get(col)) for col in feature_cols},
        sort_keys=True,
        separators=(",", ":"),
    )


def _append_llm_trace(
    trace_csv: Path,
    lock: threading.Lock,
    *,
    dataset: str,
    target_build,
    agent_input,
    ranked: list[dict],
    target,
    build_apfd: float,
    build_apfdc: float,
    build_recall_at_10: float,
    build_wall_seconds: float,
    cap_meta: dict,
):
    """Append per-test evidence of what the LLM saw and produced for one build."""
    ranked_by_test = {
        int(item["test"]): item
        for item in normalize_ranked_items(ranked)
    }
    target_by_test = {
        int(row["Test"]): row
        for row in target.to_dict("records")
    }
    feature_cols = [
        col for col in agent_input.columns
        if col not in ("Build", "Test", "Verdict", "Duration")
    ]
    input_has_verdict = "Verdict" in agent_input.columns
    input_has_duration = "Duration" in agent_input.columns

    rows = []
    for rec in agent_input.to_dict("records"):
        tid = int(rec["Test"])
        ranked_item = ranked_by_test.get(tid, {})
        target_row = target_by_test.get(tid, {})
        rows.append({
            "dataset": dataset,
            "target_build": target_build,
            "test": tid,
            "filter_model": _filter_model,
            "ranking_model": _ranking_model,
            "batch_size": _batch_size,
            "ranking_batch_size": _ranking_batch_size,
            "ranking_workers": _ranking_workers,
            "merge_agent": "1" if _merge_agent else "0",
            "candidate_cap": cap_meta.get("cap", ""),
            "candidate_selected_count": cap_meta.get("selected_count", ""),
            "input_has_verdict": "1" if input_has_verdict else "0",
            "input_has_duration": "1" if input_has_duration else "0",
            "input_feature_count": len(feature_cols),
            "llm_input_features_json": _feature_json(rec, feature_cols),
            "llm_priority": ranked_item.get("priority", ""),
            "llm_confidence": ranked_item.get("confidence", ""),
            "llm_reason": ranked_item.get("reason", ""),
            "actual_verdict": target_row.get("Verdict", ""),
            "actual_duration": target_row.get("Duration", ""),
            "build_apfd": f"{build_apfd:.6f}",
            "build_apfdc": f"{build_apfdc:.6f}",
            "build_recall_at_10": f"{build_recall_at_10:.6f}",
            "build_wall_seconds": f"{build_wall_seconds:.1f}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    fieldnames = [
        "dataset", "target_build", "test",
        "filter_model", "ranking_model",
        "batch_size", "ranking_batch_size", "ranking_workers", "merge_agent",
        "candidate_cap", "candidate_selected_count",
        "input_has_verdict", "input_has_duration", "input_feature_count",
        "llm_input_features_json",
        "llm_priority", "llm_confidence", "llm_reason",
        "actual_verdict", "actual_duration",
        "build_apfd", "build_apfdc", "build_recall_at_10",
        "build_wall_seconds", "timestamp",
    ]
    _append_csv_rows(trace_csv, lock, fieldnames, rows)


def _append_result(results_csv: Path, lock: threading.Lock, row: dict):
    """Atomically append one result row + fsync so a crash can't lose it.

    The lock serializes writes across worker threads (CSV append from multiple
    threads is a race). fsync forces the OS to commit bytes to disk hardware,
    so even a kernel panic mid-write leaves the file consistent.
    """
    fieldnames = [
        "dataset", "apfd", "apfdc", "recall_at_10",
        "filter_model", "ranking_model", "eval_window", "failed_builds_only",
        "history_failed_builds_only",
        "batch_size", "ranking_batch_size", "ranking_workers", "merge_agent",
        "merge_status_counts", "merge_repaired_builds", "merge_missing_total",
        "candidate_cap", "candidate_cap_recall_avg", "candidate_cap_recall_min",
        "build_results_csv", "gap", "filter_gap", "openai_base_url",
        "wall_seconds", "timestamp", "status", "error",
    ]
    _append_csv_row(results_csv, lock, fieldnames, row)


def _append_build_result(results_csv: Path, lock: threading.Lock, row: dict):
    fieldnames = [
        "dataset", "target_build", "build_index", "total_builds",
        "apfd", "apfdc", "recall_at_10", "num_tests", "num_failures",
        "filter_model", "ranking_model", "eval_window", "failed_builds_only",
        "history_failed_builds_only",
        "batch_size", "ranking_batch_size", "ranking_workers", "merge_agent",
        "merge_status", "merge_missing_count",
        "candidate_cap", "candidate_selected_count", "candidate_tail_count",
        "candidate_cap_recall",
        "wall_seconds", "timestamp", "status", "error",
    ]
    _append_csv_row(results_csv, lock, fieldnames, row)


def _evaluate_one(f: Path, args, results_csv: Path, lock: threading.Lock) -> tuple[Path, str]:
    """Run evaluation on a single dataset and durably append the result.
    Returns (path, status_string) for the progress log."""
    start = time.time()
    try:
        a, ac, recall_at_10, n_b, meta = evaluate(
            f,
            verbose=False,  # parallel runs — verbose output would interleave
            eval_window=args.eval_window,
            gap=args.gap,
            no_validation=args.no_validation,
            failed_builds_only=args.failed_builds_only,
            history_failed_builds_only=args.history_failed_builds_only,
            diagnose=args.diagnose_failures,
            build_results_csv=args.build_results_csv,
            build_results_lock=lock,
            llm_trace_csv=args.llm_trace_csv,
            llm_trace_lock=lock,
        )
        elapsed = time.time() - start
        _append_result(results_csv, lock, {
            "dataset": f.name,
            "apfd": f"{a:.6f}",
            "apfdc": f"{ac:.6f}",
            "recall_at_10": f"{recall_at_10:.6f}",
            "filter_model": _filter_model,
            "ranking_model": _ranking_model,
            "eval_window": args.eval_window,
            "failed_builds_only": "1" if args.failed_builds_only else "0",
            "history_failed_builds_only": "1" if args.history_failed_builds_only else "0",
            "batch_size": _batch_size,
            "ranking_batch_size": _ranking_batch_size,
            "ranking_workers": _ranking_workers,
            "merge_agent": "1" if _merge_agent else "0",
            "merge_status_counts": meta.get("merge_status_counts", ""),
            "merge_repaired_builds": meta.get("merge_repaired_builds", ""),
            "merge_missing_total": meta.get("merge_missing_total", ""),
            "candidate_cap": _candidate_cap or "",
            "candidate_cap_recall_avg": (
                f"{meta['candidate_cap_recall_avg']:.6f}"
                if isinstance(meta.get("candidate_cap_recall_avg"), float)
                else meta.get("candidate_cap_recall_avg", "")
            ),
            "candidate_cap_recall_min": (
                f"{meta['candidate_cap_recall_min']:.6f}"
                if isinstance(meta.get("candidate_cap_recall_min"), float)
                else meta.get("candidate_cap_recall_min", "")
            ),
            "build_results_csv": str(args.build_results_csv),
            "gap": args.gap,
            "filter_gap": args.filter_gap,
            "openai_base_url": os.environ.get("OPENAI_BASE_URL", ""),
            "wall_seconds": f"{elapsed:.1f}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "ok",
            "error": "",
        })
        fbo_note = "fbo" if args.failed_builds_only else "all"
        history_note = "+hfbo" if args.history_failed_builds_only else ""
        return f, (
            f"OK\tAPFD={a:.4f}\tAPFDc={ac:.4f}\t"
            f"Recall@10={recall_at_10:.4f}\t{n_b}b\t{fbo_note}{history_note}\t({elapsed:.0f}s)"
        )
    except Exception as e:
        elapsed = time.time() - start
        _append_result(results_csv, lock, {
            "dataset": f.name,
            "apfd": "", "apfdc": "", "recall_at_10": "",
            "filter_model": _filter_model,
            "ranking_model": _ranking_model,
            "eval_window": args.eval_window,
            "failed_builds_only": "1" if args.failed_builds_only else "0",
            "history_failed_builds_only": "1" if args.history_failed_builds_only else "0",
            "batch_size": _batch_size,
            "ranking_batch_size": _ranking_batch_size,
            "ranking_workers": _ranking_workers,
            "merge_agent": "1" if _merge_agent else "0",
            "merge_status_counts": "",
            "merge_repaired_builds": "",
            "merge_missing_total": "",
            "candidate_cap": _candidate_cap or "",
            "candidate_cap_recall_avg": "",
            "candidate_cap_recall_min": "",
            "build_results_csv": str(args.build_results_csv),
            "gap": args.gap,
            "filter_gap": args.filter_gap,
            "openai_base_url": os.environ.get("OPENAI_BASE_URL", ""),
            "wall_seconds": f"{elapsed:.1f}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "failed",
            "error": str(e)[:500],
        })
        return f, f"FAILED\t{type(e).__name__}: {e}"


def _run_data_dir(args):
    files = sorted(args.data_dir.glob("*.csv"))
    results_csv = args.results_csv
    completed = _load_completed(results_csv)

    pending = [
        f for f in files
        if _completed_key_from_values(
            f.name,
            args.failed_builds_only,
            args.history_failed_builds_only,
            _filter_model,
            _ranking_model,
            args.eval_window,
            _batch_size,
            _ranking_batch_size,
            _ranking_workers,
            _merge_agent,
            _candidate_cap or "",
            args.gap,
            args.filter_gap,
            os.environ.get("OPENAI_BASE_URL", ""),
        ) not in completed
    ]
    skipped = len(files) - len(pending)
    mode = "failed-builds only" if args.failed_builds_only else "all builds"
    history_mode = ", failed-history only" if args.history_failed_builds_only else ""
    if skipped:
        print(
            f"[resume] {skipped}/{len(files)} datasets ({mode}{history_mode}) already in {results_csv} — skipping",
            flush=True,
        )
    if not pending:
        print("[resume] nothing to do — all datasets evaluated", flush=True)
        return

    print(
        f"[start] {len(pending)} datasets, workers={args.workers}, "
        f"filter={_filter_model}, ranking={_ranking_model}, eval={mode}{history_mode}",
        flush=True,
    )

    write_lock = threading.Lock()
    done_count = 0
    total = len(pending)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_evaluate_one, f, args, results_csv, write_lock): f for f in pending}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                _, status_line = fut.result()
            except Exception as e:
                status_line = f"FAILED\t{type(e).__name__}: {e}"
            done_count += 1
            print(f"[{done_count}/{total}] {f.name}\t{status_line}", flush=True)


if __name__ == "__main__":
    main()
