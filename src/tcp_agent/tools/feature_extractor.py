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


# ── Risk profile features (REC_ + DET_COV_ + COV_ + TES_CHN_) ────────

_RISK_PROFILE_COLS = [
    "Test",
    # ── REC: execution history (paper Table 12 — top features) ───────
    "REC_RecentFailRate", "REC_TotalFailRate",
    "REC_LastVerdict", "REC_LastFailureAge",
    "REC_RecentTransitionRate", "REC_TotalTransitionRate",
    "REC_Age",                       # #1 most predictive feature paper-wide
    "REC_RecentAssertRate", "REC_TotalAssertRate",
    "REC_RecentExcRate", "REC_TotalExcRate",
    "REC_LastExeTime",
    "REC_RecentAvgExeTime", "REC_RecentMaxExeTime",
    "REC_TotalAvgExeTime", "REC_TotalMaxExeTime",
    "REC_LastTransitionAge",
    "REC_MaxTestFileFailRate", "REC_MaxTestFileTransitionRate",
    # ── DET_COV: fault history of covered files ──────────────────────
    "DET_COV_C_Faults", "DET_COV_IMP_Faults",
    # ── COV: change/impact coverage scores ───────────────────────────
    "COV_ChnScoreSum", "COV_ImpScoreSum",
    "COV_ChnCount", "COV_ImpCount",
    # ── TES_CHN: test file churn (this build) ────────────────────────
    "TES_CHN_LinesAdded", "TES_CHN_LinesDeleted",
    "TES_CHN_AddedChangeScattering", "TES_CHN_DeletedChangeScattering",
    "TES_CHN_DMMSize", "TES_CHN_DMMComplexity", "TES_CHN_DMMInterfacing",
    # ── TES_PRO: test ownership / process metrics ─────────────────────
    # Per Yaraghi 2022 Table 12, OwnersExperience (#2), AllCommitersExperience
    # (#3), OwnersContribution, and CommitCount are all among the top-7 most
    # predictive features. On some subjects (e.g. thinkaurelius/titan) the
    # TES_M model (TES_PRO + TES_COM + TES_CHN) is the single best feature
    # group, beating REC_M. The Filter MUST see them.
    "TES_PRO_OwnersExperience", "TES_PRO_AllCommitersExperience",
    "TES_PRO_OwnersContribution", "TES_PRO_CommitCount",
    "TES_PRO_DistinctDevCount", "TES_PRO_MinorContributorCount",
    # ── TES_COM: test source-code complexity / size ──────────────────
    "TES_COM_SumCyclomatic", "TES_COM_CountLineCode",
    "TES_COM_CountStmtDecl", "TES_COM_CountStmtExe",
    "TES_COM_RatioCommentToCode",
]


def extract_risk_profiles(
    dataset_path: str,
    sparse: bool = True,
    test_ids: Optional[list[int]] = None,
) -> list[dict]:
    """Extract REC_, DET_COV_, COV_, TES_CHN_ features for all (or selected) tests.

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
        One dict per test, keyed by feature name.
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

    keep = [c for c in _RISK_PROFILE_COLS if c in latest.columns]
    result = latest[keep].rename(columns={"Test": "test"})
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
