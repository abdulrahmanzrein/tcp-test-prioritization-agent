import pandas as pd


def build_ranked_df(ranked, target_df):
    """
    Takes Claude's ranked output and merges it with the target build's
    actual Verdict and Duration so evaluation.py can score it.
    """
    df_ranked = pd.DataFrame(ranked)
    df_ranked = df_ranked.rename(columns={"test": "Test"})  # Claude outputs lowercase "test", dataset uses "Test"
    df_ranked["Test"] = pd.to_numeric(df_ranked["Test"], errors="coerce")  # Claude returns strings, dataset uses ints

    # grab actual Verdict and Duration from the target build — no averaging
    target = target_df[["Test", "Verdict", "Duration"]].copy()

    # attach real Verdict and Duration to Claude's ranked list
    merged = pd.merge(df_ranked, target, on="Test")

    result = merged.sort_values("priority", ascending=True)
    return result

    # final output looks like:
    # test   priority  confidence  reason                        Verdict  Duration
    # 2757   1         0.91        high failure rate             1        3.2
    # 3102   2         0.74        failed in recent builds       0        8.1
    # 4891   3         0.61        high coverage score           0        2.4
