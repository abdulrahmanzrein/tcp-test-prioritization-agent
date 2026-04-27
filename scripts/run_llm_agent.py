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
load_dotenv(PROJECT_ROOT / ".env")
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tcp_agent.agent.tcp_agent import run_agent
from tcp_agent.agent.ranker import build_ranked_df, normalize_ranked_items
from tcp_agent.evaluation import apfd, apfdc, precision_at_k
from tcp_agent.config import AgentMode
from tcp_agent.data_cache import load_dataset
import pandas as pd

# default settings — can be overridden via CLI
_mode = AgentMode.PILOT
_strategy = "multi"
_batch_size = 200
_filter_model = "gpt-4o-mini"
_ranking_model = "gpt-4o"

def evaluate(csv_path, verbose=False, eval_window=5, gap=65.0):
    """
    Rolling-window evaluation over the last `eval_window` builds.

    For each target build B in builds[-eval_window:]:
      - history = all builds BEFORE B  (what the agent can see)
      - target  = build B              (what we evaluate against)
      The agent ranks tests on the history, then we score its ordering
      against B's real verdicts.

    Scores are averaged across all target builds, giving a much more
    robust estimate than a single-build evaluation.
    """
    df = load_dataset(csv_path)
    builds = sorted(df["Build"].unique())

    # need at least eval_window + 1 builds (one for history, rest for targets)
    if len(builds) < eval_window + 1:
        eval_window = max(1, len(builds) - 1)

    target_builds = builds[-eval_window:]   # last N builds to evaluate on

    all_apfd, all_apfdc, all_p10 = [], [], []

    for i, target_build in enumerate(target_builds):
        # agent sees only builds BEFORE this target build
        history = df[df["Build"] < target_build]
        target  = df[df["Build"] == target_build]

        if history.empty:
            continue  # skip if no history available

        # write history to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, prefix="tcp_history_"
        ) as tmp:
            history.to_csv(tmp, index=False)
            tmp_path = tmp.name

        ranked = run_agent(
            tmp_path,
            mode=_mode,
            strategy=_strategy,
            batch_size=_batch_size,
            filter_model=_filter_model,
            ranking_model=_ranking_model,
        )

        if verbose:
            print(f"\n  [Build {target_build} — {i+1}/{len(target_builds)}]")
            for item in ranked:
                print(f"    #{item['priority']} test {item['test']} — {item['reason']}")

        ranked_df = build_ranked_df(ranked, target)
        all_apfd.append(apfd(ranked_df))
        all_apfdc.append(apfdc(ranked_df))
        all_p10.append(precision_at_k(ranked_df, k=10))

        # rate-limit gap between agent calls (skip after the last one)
        if i < len(target_builds) - 1 and gap > 0:
            time.sleep(gap)

    if not all_apfd:
        return 0.0, 0.0, 0.0

    return (
        sum(all_apfd)  / len(all_apfd),
        sum(all_apfdc) / len(all_apfdc),
        sum(all_p10)   / len(all_p10),
    )


def main():
    global _mode, _strategy, _batch_size, _filter_model, _ranking_model

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
        "--strategy",
        choices=["single", "multi"],
        default="multi",
        help="Agent strategy: 'single' = legacy one-agent loop, 'multi' = two-agent pipeline (default)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=200,
        help="Number of tests per Filter Agent batch (multi strategy only, default: 200)",
    )
    parser.add_argument(
        "--filter-model", type=str, default="gpt-4o-mini",
        help="LLM model for the Filter Agent (multi strategy only, default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--ranking-model", type=str, default="gpt-4o",
        help="LLM model for the Ranking Agent (multi strategy only, default: gpt-4o)",
    )
    parser.add_argument(
        "--eval-window", type=int, default=5,
        help="Number of most-recent builds to evaluate against (default: 5). "
             "The agent is called once per build with all prior builds as history.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--gap", type=float, default=65.0,
        help="Seconds to wait between agent calls to avoid rate limits (0 to disable)",
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
    args = parser.parse_args()
    _mode = AgentMode.PILOT if args.mode == "pilot" else AgentMode.PRODUCTION
    _strategy = args.strategy
    _batch_size = args.batch_size
    _filter_model = args.filter_model
    _ranking_model = args.ranking_model

    needed_keys = set()
    for model_name in (_filter_model, _ranking_model):
        name = model_name.lower()
        if "gemini" in name:
            needed_keys.add("GOOGLE_API_KEY")
        elif name.startswith("claude"):
            needed_keys.add("ANTHROPIC_API_KEY")
        else:
            needed_keys.add("OPENAI_API_KEY")

    missing = [k for k in needed_keys if not os.environ.get(k, "").strip()]
    if missing:
        sys.exit(f"Missing required API key(s): {', '.join(missing)}")

    if args.data:
        a, ac, p10 = evaluate(
            args.data,
            verbose=not args.quiet,
            eval_window=args.eval_window,
            gap=args.gap,
        )
        print(f"APFD={a:.4f}  APFDc={ac:.4f}  P@10={p10:.4f}  (avg over {args.eval_window} builds)")
        return

    _run_data_dir(args)


def _load_completed(results_csv: Path) -> set[str]:
    """Return the set of dataset filenames already present in the results CSV.
    Used to skip datasets we've already evaluated (automatic resume on rerun)."""
    if not results_csv.exists():
        return set()
    done = set()
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("dataset"):
                done.add(row["dataset"])
    return done


def _append_result(results_csv: Path, lock: threading.Lock, row: dict):
    """Atomically append one result row + fsync so a crash can't lose it.

    The lock serializes writes across worker threads (CSV append from multiple
    threads is a race). fsync forces the OS to commit bytes to disk hardware,
    so even a kernel panic mid-write leaves the file consistent.
    """
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "apfd", "apfdc", "p_at_10",
        "filter_model", "ranking_model", "eval_window",
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
        a, ac, p10 = evaluate(
            f,
            verbose=False,  # parallel runs — verbose output would interleave
            eval_window=args.eval_window,
            gap=args.gap,
        )
        elapsed = time.time() - start
        _append_result(results_csv, lock, {
            "dataset": f.name,
            "apfd": f"{a:.6f}",
            "apfdc": f"{ac:.6f}",
            "p_at_10": f"{p10:.6f}",
            "filter_model": _filter_model,
            "ranking_model": _ranking_model,
            "eval_window": args.eval_window,
            "wall_seconds": f"{elapsed:.1f}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "ok",
            "error": "",
        })
        return f, f"OK\tAPFD={a:.4f}\tAPFDc={ac:.4f}\tP@10={p10:.4f}\t({elapsed:.0f}s)"
    except Exception as e:
        elapsed = time.time() - start
        _append_result(results_csv, lock, {
            "dataset": f.name,
            "apfd": "", "apfdc": "", "p_at_10": "",
            "filter_model": _filter_model,
            "ranking_model": _ranking_model,
            "eval_window": args.eval_window,
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

    pending = [f for f in files if f.name not in completed]
    skipped = len(files) - len(pending)
    if skipped:
        print(f"[resume] {skipped}/{len(files)} datasets already in {results_csv} — skipping", flush=True)
    if not pending:
        print("[resume] nothing to do — all datasets evaluated", flush=True)
        return

    print(
        f"[start] {len(pending)} datasets, workers={args.workers}, "
        f"filter={_filter_model}, ranking={_ranking_model}",
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
