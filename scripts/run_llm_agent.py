import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tcp_agent.agent.tcp_agent import run_agent
from tcp_agent.agent.ranker import build_ranked_df
from tcp_agent.evaluation import apfd, apfdc, precision_at_k
from tcp_agent.config import AgentMode
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to dataset CSV")
    parser.add_argument(
        "--mode",
        choices=["pilot", "production"],
        default="pilot",
        help="Agent mode: 'pilot' reads from CSV (default), 'production' extracts from real sources",
    )
    args = parser.parse_args()

    mode = AgentMode.PILOT if args.mode == "pilot" else AgentMode.PRODUCTION

    # run the LLM agent — gathers context and asks Claude to rank tests
    df = pd.read_csv(args.data)
    builds = sorted(df["Build"].unique())
    target_build = builds[-1]
    history = df[df["Build"] != target_build]
    target = df[df["Build"] == target_build]
    history.to_csv("/tmp/history.csv", index=False)

    ranked = run_agent("/tmp/history.csv", mode=mode)

    # print the agent's reasoning
    print("\n--- Agent Ranking ---")
    for item in ranked:
        print(f"#{item['priority']} Test {item['test']} — {item['reason']}")
    print()

    # merge Claude's ranking with real Verdict and Duration
    ranked_df = build_ranked_df(ranked, target)

    # evaluate how good Claude's ranking is
    print(f"APFD:          {apfd(ranked_df):.4f}")
    print(f"APFDc:         {apfdc(ranked_df):.4f}")
    print(f"Precision@10:  {precision_at_k(ranked_df, k=10):.4f}")

if __name__ == "__main__":
    main()
