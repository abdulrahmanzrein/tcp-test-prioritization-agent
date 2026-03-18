import pandas as pd


def build_ranked_df(ranked, dataset_path):
    """
    Takes Claude's ranked output and merges it with real Verdict data
    so evaluation.py can score it.
    """
    df_ranked = pd.DataFrame(ranked)

    df = pd.read_csv(dataset_path)

    # average Verdict and Duration per test across all build
    summary = df.groupby("Test")[["Verdict", "Duration"]].mean().reset_index()
   
    
    # attach real Verdict and Duration to Claude's ranked list
    merged = pd.merge(df_ranked, summary, on="Test")

    result = merged.sort_values("priority", ascending=True)
    return result

    # final output looks like:
    # test   priority  confidence  reason                        Verdict  Duration
    # 2757   1         0.91        high failure rate             0.8      3.2
    # 3102   2         0.74        failed in recent builds       0.5      8.1
    # 4891   3         0.61        high coverage score           0.2      2.4
