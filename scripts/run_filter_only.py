"""
Diagnostic script: Run only the Filter Agent and inspect its output.

Usage:
    python scripts/run_filter_only.py --data dataset.csv
    python scripts/run_filter_only.py --data dataset.csv --batch-size 100 --filter-model gpt-4o-mini
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure src/ is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tcp_agent.agent.filter_agent import run_filter_agent
from tcp_agent.config import AgentMode, set_mode


def main():
    parser = argparse.ArgumentParser(
        description="Run the Filter Agent and display classification results."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset CSV")
    parser.add_argument(
        "--batch-size", type=int, default=200,
        help="Number of tests per batch (default: 200)",
    )
    parser.add_argument(
        "--filter-model", type=str, default="gpt-4o-mini",
        help="LLM model for the Filter Agent (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        sys.exit("OPENAI_API_KEY not set.")

    set_mode(AgentMode.PILOT, dataset_path=str(args.data))

    print(f"Running Filter Agent on {args.data}")
    print(f"  batch_size={args.batch_size}, model={args.filter_model}")
    print()

    result = run_filter_agent(
        str(args.data),
        batch_size=args.batch_size,
        filter_model=args.filter_model,
    )

    # ── Display results ──────────────────────────────────────────────
    print("=" * 70)
    print(result.summary())
    print("=" * 70)

    print(f"\nMetadata: {result.metadata}")

    if result.high_risk_tests:
        print(f"\n{'─' * 70}")
        print(f"HIGH-RISK TESTS  (T1-T5) — {len(result.high_risk_tests)} tests")
        print(f"{'─' * 70}")
        for t in sorted(result.high_risk_tests, key=lambda x: x["tier"]):
            tier = t["tier"]
            tid = t["test_id"]
            signals = ", ".join(t["key_signals"])
            print(f"  T{tier}  test {tid:>6}  →  {signals}")

    if result.low_signal_tests:
        print(f"\n{'─' * 70}")
        print(f"LOW-SIGNAL TESTS (T6) — {len(result.low_signal_tests)} tests")
        print(f"{'─' * 70}")
        sorted_t6 = sorted(result.low_signal_tests, key=lambda x: x.get("avg_exec_time", 0))
        for t in sorted_t6:
            tid = t["test_id"]
            et = t.get("avg_exec_time", 0.0)
            print(f"  T6  test {tid:>6}  →  avg_exec_time={et:.2f}ms")

    # ── Tier distribution ────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("TIER DISTRIBUTION")
    print(f"{'─' * 70}")
    tier_counts = {}
    for t in result.high_risk_tests + result.low_signal_tests:
        tier = t["tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    for tier in sorted(tier_counts):
        count = tier_counts[tier]
        pct = 100 * count / (len(result.high_risk_tests) + len(result.low_signal_tests))
        bar = "█" * int(pct / 2)
        print(f"  T{tier}: {count:>4} ({pct:5.1f}%)  {bar}")


if __name__ == "__main__":
    main()
