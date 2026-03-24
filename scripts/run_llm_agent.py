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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to dataset CSV")
    args = parser.parse_args()

    # run the LLM agent — gathers context and asks Claude to rank tests
    ranked = run_agent(args.data)

    # print the agent's reasoning
    print("\n--- Agent Ranking ---")
    for item in ranked:
        print(f"#{item['priority']} Test {item['test']} — {item['reason']}")
    print()

    # merge Claude's ranking with real Verdict and Duration
    ranked_df = build_ranked_df(ranked, args.data)

    # evaluate how good Claude's ranking is
    print(f"APFD:          {apfd(ranked_df):.4f}")
    print(f"APFDc:         {apfdc(ranked_df):.4f}")
    print(f"Precision@10:  {precision_at_k(ranked_df, k=10):.4f}")

if __name__ == "__main__":
    main()
