from __future__ import annotations

"""
Pure-Python feature extraction for the Filter Agent.

These functions extract the same data as the LangChain tools but return it
directly without going through the tool-calling loop.  This avoids wasting
an LLM round-trip on data that can be fetched deterministically.

Used by the Filter Agent to pre-load features before sending them to the
LLM in batches.
"""

import pandas as pd
from typing import Optional

from tcp_agent.data_cache import load_dataset


# ── Feature column selection (Yaraghi et al. 2022 — full 148-feature set) ─

# Columns that are NOT features: identifiers, labels, and execution outcomes.
_NON_FEATURE_COLS = {"Build", "Test", "Verdict", "Duration"}

# Illegal features (data leakage). DET_COV_*Faults count faults DETECTED by the
# test in the build being evaluated — only knowable AFTER the test has run, so
# using them at prediction time leaks the label. Yaraghi 2022 § 3.2.5 defines
# them; their RF model uses them only because PDF (Previously Detected Faults)
# is recomputed from prior builds, not the target build. In our pipeline we
# evaluate on the target build's verdicts, so we must exclude these two.
_ILLEGAL_FEATURE_COLS = {"DET_COV_C_Faults", "DET_COV_IMP_Faults"}


def _legal_feature_cols(df_columns) -> list[str]:
    """All legal feature columns in the dataset (148 in TCP-CI, fewer if a
    subject's CSV is missing some). Excludes identifiers, labels, and the
    two leakage features."""
    excluded = _NON_FEATURE_COLS | _ILLEGAL_FEATURE_COLS
    return [c for c in df_columns if c not in excluded]


def extract_risk_profiles(
    dataset_path: str,
    sparse: bool = True,
    test_ids: Optional[list[int]] = None,
) -> list[dict]:
    """Extract the full Yaraghi 2022 feature set per test (148 features when the
    CSV has all of them, fewer if a subject's dataset is missing some columns).

    The two illegal columns (DET_COV_C_Faults, DET_COV_IMP_Faults) are excluded
    to prevent label leakage — see _legal_feature_cols above.

    Parameters
    ----------
    dataset_path : str
        Path to the CSV dataset.
    sparse : bool
        If True, omit keys whose value is -1 (the TCP-CI "no data" sentinel).
        Real-zero values are KEPT because they carry meaningful signal — e.g.,
        REC_LastVerdict=0 ("passed last build"), REC_LastFailureAge=0 ("failed
        in current build"), REC_TotalFailRate=0 ("never failed"). Conflating
        these with "no data" causes the Filter Agent to misclassify tests.
    test_ids : list[int] | None
        If provided, only extract profiles for these test IDs.

    Returns
    -------
    list[dict]
        One dict per test, keyed by feature name. Includes a "test" key
        containing the test ID.
    """
    df = load_dataset(dataset_path)

    # latest build snapshot per test
    latest = (
        df.sort_values("Build", ascending=False)
        .groupby("Test")
        .first()
        .reset_index()
    )

    if test_ids is not None:
        latest = latest[latest["Test"].isin(test_ids)]

    feature_cols = _legal_feature_cols(latest.columns)
    keep = ["Test"] + feature_cols
    result = latest[keep].rename(columns={"Test": "test"})
    if "REC_RecentFailRate" in result.columns:
        result = result.sort_values("REC_RecentFailRate", ascending=False)

    records = result.to_dict("records")

    if sparse:
        # Only drop -1 (TCP-CI "no data" sentinel). Keep 0s — they're real signal.
        records = [
            {k: v for k, v in rec.items() if k == "test" or v != -1}
            for rec in records
        ]

    return records


def extract_failure_rates(dataset_path: str) -> dict:
    """Return {test_id: failure_rate} mapping for every test."""
    df = load_dataset(dataset_path)
    rates = (
        df.assign(_fail=df["Verdict"].ne(0))
        .groupby("Test")["_fail"]
        .mean()
    )
    return rates.to_dict()


def extract_exec_times(dataset_path: str) -> dict:
    """Return {test_id: avg_exec_time} mapping for every test."""
    df = load_dataset(dataset_path)
    times = df.groupby("Test")["Duration"].mean()
    return times.to_dict()


def extract_all_test_ids(dataset_path: str) -> set:
    """Return the set of all unique test IDs in the dataset."""
    df = load_dataset(dataset_path)
    return set(df["Test"].unique().tolist())


def extract_latest_features_for_fallback(dataset_path: str) -> pd.DataFrame:
    """Return a DataFrame with the latest build snapshot for all tests,
    including the key columns needed by the deterministic fallback ranker.

    Columns guaranteed present (if they exist in the dataset):
        Test, REC_TotalFailRate, REC_RecentFailRate, DET_COV_C_Faults,
        DET_COV_IMP_Faults, REC_RecentAvgExeTime
    """
    df = load_dataset(dataset_path)
    latest = (
        df.sort_values("Build", ascending=False)
        .groupby("Test")
        .first()
        .reset_index()
    )
    return latest
