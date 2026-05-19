from __future__ import annotations

import argparse
import csv
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
from tcp_agent.evaluation import apfd, apfdc, precision_at_k, failure_recall_at_k
from tcp_agent.config import AgentMode
from tcp_agent.data_cache import load_dataset
import pandas as pd

# default settings — can be overridden via CLI
_mode = AgentMode.PILOT
_batch_size = 40
_filter_model = "gpt-5-nano"
_ranking_model = "gemini-3-flash-preview"
_filter_gap = 0.0
_ranking_workers = 1

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


def evaluate(
    csv_path,
    verbose=False,
    eval_window=5,
    gap=65.0,
    no_validation=False,
    failed_builds_only=False,
    diagnose=False,
):
    """
    Rolling-window evaluation over the last `eval_window` builds.

    When failed_builds_only=True, only builds that contain at least one test
    failure are used as targets — zero-failure builds carry no ranking signal
    and would inflate APFD/APFDc to 1.0.

    For each target build B:
      - history = all builds BEFORE B  (what the agent can see)
      - target  = build B              (what we score against)
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
        return 0.0, 0.0, 0.0, 0

    if len(eligible) < eval_window:
        eval_window = len(eligible)

    target_builds = eligible[-eval_window:]

    all_apfd, all_apfdc, all_recall_at_10 = [], [], []

    for i, target_build in enumerate(target_builds):
        history = df[df["Build"] < target_build]
        target  = df[df["Build"] == target_build]

        if history.empty:
            continue

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, prefix="tcp_history_"
        ) as tmp:
            history.to_csv(tmp, index=False)
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
        )

        if verbose:
            print(f"\n  [Build {target_build} — {i+1}/{len(target_builds)}]")
            for item in ranked:
                print(f"    #{item['priority']} test {item['test']} — {item['reason']}")

        ranked_df = build_ranked_df(ranked, target)
        all_apfd.append(apfd(ranked_df))
        all_apfdc.append(apfdc(ranked_df))
        all_recall_at_10.append(failure_recall_at_k(ranked_df, k=10))

        if diagnose:
            _print_diagnosis(
                ranked_df, target_build, i, len(target_builds),
                failed_builds_only=failed_builds_only,
            )

        # rate-limit gap between agent calls (skip after the last one)
        if i < len(target_builds) - 1 and gap > 0:
            time.sleep(gap)

    if not all_apfd:
        return 0.0, 0.0, 0.0, 0

    n = len(all_apfd)
    return (
        sum(all_apfd)          / n,
        sum(all_apfdc)         / n,
        sum(all_recall_at_10)  / n,
        n,
    )


def main():
    global _mode, _batch_size, _filter_model, _ranking_model, _filter_gap, _ranking_workers

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
             "the full 148-feature Yaraghi 2022 set). Auto-splits on output-"
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
             "The agent is called once per build with all prior builds as history.",
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
    _mode = AgentMode.PILOT if args.mode == "pilot" else AgentMode.PRODUCTION
    _batch_size = args.batch_size
    _filter_model = args.filter_model
    _ranking_model = args.ranking_model
    _filter_gap = args.filter_gap
    _ranking_workers = args.ranking_workers

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
        a, ac, recall_at_10, n_b = evaluate(
            args.data,
            verbose=not args.quiet,
            eval_window=args.eval_window,
            gap=args.gap,
            no_validation=args.no_validation,
            failed_builds_only=args.failed_builds_only,
            diagnose=args.diagnose_failures,
        )
        print(
            f"APFD={a:.4f}  APFDc={ac:.4f}  Recall@10={recall_at_10:.4f}  "
            f"(avg over {n_b} build(s){'; failed-builds only' if args.failed_builds_only else ''})"
        )
        return

    _run_data_dir(args)


def _normalize_failed_builds_flag(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "y")


def _load_completed(results_csv: Path) -> set[tuple[str, bool]]:
    """Rows with status ok: (dataset filename, failed_builds_only).

    Older result files omit failed_builds_only — treated as False so resume
    behavior matches historical runs."""
    if not results_csv.exists():
        return set()
    done: set[tuple[str, bool]] = set()
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok" or not row.get("dataset"):
                continue
            key = row["dataset"]
            fbo = _normalize_failed_builds_flag(row.get("failed_builds_only", ""))
            done.add((key, fbo))
    return done


def _append_result(results_csv: Path, lock: threading.Lock, row: dict):
    """Atomically append one result row + fsync so a crash can't lose it.

    The lock serializes writes across worker threads (CSV append from multiple
    threads is a race). fsync forces the OS to commit bytes to disk hardware,
    so even a kernel panic mid-write leaves the file consistent.
    """
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "apfd", "apfdc", "recall_at_10",
        "filter_model", "ranking_model", "eval_window", "failed_builds_only",
        "wall_seconds", "timestamp", "status", "error",
    ]
    with lock:
        write_header = not results_csv.exists() or results_csv.stat().st_size == 0
        with open(results_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            f.flush()
            os.fsync(f.fileno())


def _evaluate_one(f: Path, args, results_csv: Path, lock: threading.Lock) -> tuple[Path, str]:
    """Run evaluation on a single dataset and durably append the result.
    Returns (path, status_string) for the progress log."""
    start = time.time()
    try:
        a, ac, recall_at_10, n_b = evaluate(
            f,
            verbose=False,  # parallel runs — verbose output would interleave
            eval_window=args.eval_window,
            gap=args.gap,
            no_validation=args.no_validation,
            failed_builds_only=args.failed_builds_only,
            diagnose=args.diagnose_failures,
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
            "wall_seconds": f"{elapsed:.1f}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "ok",
            "error": "",
        })
        fbo_note = "fbo" if args.failed_builds_only else "all"
        return f, (
            f"OK\tAPFD={a:.4f}\tAPFDc={ac:.4f}\t"
            f"Recall@10={recall_at_10:.4f}\t{n_b}b\t{fbo_note}\t({elapsed:.0f}s)"
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

    pending = [f for f in files if (f.name, args.failed_builds_only) not in completed]
    skipped = len(files) - len(pending)
    mode = "failed-builds only" if args.failed_builds_only else "all builds"
    if skipped:
        print(
            f"[resume] {skipped}/{len(files)} datasets ({mode}) already in {results_csv} — skipping",
            flush=True,
        )
    if not pending:
        print("[resume] nothing to do — all datasets evaluated", flush=True)
        return

    print(
        f"[start] {len(pending)} datasets, workers={args.workers}, "
        f"filter={_filter_model}, ranking={_ranking_model}, eval={mode}",
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
