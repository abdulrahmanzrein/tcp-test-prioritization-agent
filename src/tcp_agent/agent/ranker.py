import pandas as pd


def normalize_ranked_items(ranked: list) -> list:
    """
    Sort by priority, drop rows with invalid test/priority, dedupe by test (keep best priority).
    Stops bogus JSON rows (e.g. 'duplicate removed') from skewing merge/eval.
    """
    cleaned = []
    for item in ranked:
        if not isinstance(item, dict):
            continue
        t, p = item.get("test"), item.get("priority")
        try:
            tid = int(t) if t is not None and str(t).strip() != "" else None
            pid = int(p) if p is not None and str(p).strip() != "" else None
        except (TypeError, ValueError):
            continue
        if tid is None or pid is None:
            continue
        row = dict(item)
        row["test"] = tid
        row["priority"] = pid
        cleaned.append(row)
    cleaned.sort(key=lambda x: (x["priority"], x["test"]))
    seen = set()
    out = []
    for row in cleaned:
        tid = row["test"]
        if tid in seen:
            continue
        seen.add(tid)
        out.append(row)
    return out


def build_ranked_df(ranked, target_df):
    """
    Takes Claude's ranked output and merges it with the target build's
    actual Verdict and Duration so evaluation.py can score it.
    """
    ranked = normalize_ranked_items(ranked)
    df_ranked = pd.DataFrame(ranked)
    df_ranked = df_ranked.rename(columns={"test": "Test"})  # Claude outputs lowercase "test", dataset uses "Test"
    df_ranked["Test"] = pd.to_numeric(df_ranked["Test"], errors="coerce")  # Claude returns strings, dataset uses ints

    # grab actual Verdict and Duration from the target build
    target = target_df[["Test", "Verdict", "Duration"]].copy()

    # attach real Verdict and Duration to Claude's ranked list
    # use OUTER merge so tests the LLM missed still appear (ranked last)
    merged = pd.merge(df_ranked, target, on="Test", how="outer")

    # tests the LLM missed get worst priority (ranked last) — this prevents
    # inflated APFDc from silently dropping missed failures
    max_priority = merged["priority"].max()
    if pd.isna(max_priority):
        max_priority = 0
    merged["priority"] = merged["priority"].fillna(max_priority + 1)
    merged = merged.dropna(subset=["Verdict"])

    # final output looks like:
    # Test   priority  confidence  reason                        Verdict  Duration
    # 2757   1         0.91        high failure rate             1        3.2
    # 3102   2         0.74        failed in recent builds       0        8.1
    result = merged.sort_values("priority", ascending=True).reset_index(drop=True)
    return result
