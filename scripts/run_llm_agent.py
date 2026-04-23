import argparse
import os
import sys
import time
import tempfile
from pathlib import Path

# Ensure src/ is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tcp_agent.agent.tcp_agent import run_agent
from tcp_agent.agent.ranker import build_ranked_df, normalize_ranked_items
from tcp_agent.evaluation import apfd, apfdc, precision_at_k
from tcp_agent.config import AgentMode
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
    df = pd.read_csv(csv_path)
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
    args = parser.parse_args()
    _mode = AgentMode.PILOT if args.mode == "pilot" else AgentMode.PRODUCTION
    _strategy = args.strategy
    _batch_size = args.batch_size
    _filter_model = args.filter_model
    _ranking_model = args.ranking_model

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        sys.exit("OPENAI_API_KEY not set.")

    if args.data:
        a, ac, p10 = evaluate(
            args.data,
            verbose=not args.quiet,
            eval_window=args.eval_window,
            gap=args.gap,
        )
        print(f"APFD={a:.4f}  APFDc={ac:.4f}  P@10={p10:.4f}  (avg over {args.eval_window} builds)")
        return

    files = sorted(args.data_dir.glob("*.csv"))
    for i, f in enumerate(files, 1):
        try:
            a, ac, p10 = evaluate(
                f,
                verbose=not args.quiet,
                eval_window=args.eval_window,
                gap=args.gap,
            )
            print(
                f"[{i}/{len(files)}] {f.name}\t"
                f"APFD={a:.4f}\tAPFDc={ac:.4f}\tP@10={p10:.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"[{i}/{len(files)}] {f.name}\tFAILED\t{e}", flush=True)


if __name__ == "__main__":
    main()
