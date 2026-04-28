import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tcp_agent.data_loader import load_data, get_features_and_labels, get_metadata
from tcp_agent.features import clean_features
from tcp_agent.model import train_model
from tcp_agent.ranking import rank_tests
from tcp_agent.evaluation import apfd, apfdc, precision_at_k

def main():
    # accept the dataset path from the command line so we dont hardcode it
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to dataset CSV")
    args = parser.parse_args()

    # load the raw CSV into a dataframe
    df = load_data(args.data)

    # split into features (X), labels (y), and metadata (test names, durations, verdicts)
    X, y = get_features_and_labels(df)
    X = clean_features(X)
    metadata = get_metadata(df)

    # train the model — splits 80/20 internally, returns the 20% holdout + matching metadata
    model, X_test, y_test, metadata_test = train_model(X, y, metadata)

    # rank the holdout tests by predicted failure probability, highest first
    ranked_df = rank_tests(model, X_test, metadata_test)

    # evaluate how good the ranking is
    print(f"APFD:          {apfd(ranked_df):.4f}")
    print(f"APFDc:         {apfdc(ranked_df):.4f}")
    print(f"Precision@10:  {precision_at_k(ranked_df, k=10):.4f}")

if __name__ == "__main__":
    main()
