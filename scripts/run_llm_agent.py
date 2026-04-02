import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r"urllib3 v2 only supports OpenSSL")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tcp_agent.agent.tcp_agent import run_agent
from tcp_agent.agent.ranker import build_ranked_df, normalize_ranked_items
from tcp_agent.evaluation import apfd, apfdc, precision_at_k
import pandas as pd


def evaluate(csv_path, verbose=False):
    df = pd.read_csv(csv_path)
    builds = sorted(df["Build"].unique())
    target_build = builds[-1]
    history = df[df["Build"] != target_build]
    target = df[df["Build"] == target_build]
    history.to_csv("/tmp/history.csv", index=False)

    ranked = normalize_ranked_items(run_agent("/tmp/history.csv"))

    if verbose:
        for item in ranked:
            print(f"  #{item['priority']} test {item['test']} — {item['reason']}")
        print()

    ranked_df = build_ranked_df(ranked, target)
    return apfd(ranked_df), apfdc(ranked_df), precision_at_k(ranked_df, k=10)


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", type=Path)
    g.add_argument("--data-dir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gap", type=float, default=65.0,
                        help="seconds between batch runs to avoid rate limits (0 to disable)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        sys.exit("ANTHROPIC_API_KEY not set.")

    if args.data:
        a, ac, p10 = evaluate(args.data, verbose=True)
        print(f"APFD={a:.4f}  APFDc={ac:.4f}  P@10={p10:.4f}")
        return

    files = sorted(args.data_dir.glob("*.csv"))
    for i, f in enumerate(files, 1):
        try:
            a, ac, p10 = evaluate(f, verbose=not args.quiet)
            print(f"[{i}/{len(files)}] {f.name}\tAPFD={a:.4f}\tAPFDc={ac:.4f}\tP@10={p10:.4f}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(files)}] {f.name}\tFAILED\t{e}", flush=True)
        if i < len(files) and args.gap > 0:
            time.sleep(args.gap)


if __name__ == "__main__":
    main()
