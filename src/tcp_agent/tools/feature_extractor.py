from __future__ import annotations

"""
Pure-Python feature extraction for the Filter Agent.

These functions extract the same data as the LangChain tools but return it
directly without going through the tool-calling loop.  This avoids wasting
an LLM round-trip on data that can be fetched deterministically.

Used by the Filter Agent to pre-load features before sending them to the
LLM in batches.
"""

from typing import Optional

from tcp_agent.data_cache import load_dataset


# ── Feature column selection (Yaraghi et al. 2022 — full 150-feature set) ─

# Columns that are NOT features: identifiers, labels, and execution outcomes.
_NON_FEATURE_COLS = {"Build", "Test", "Verdict", "Duration"}


def _legal_feature_cols(df_columns) -> list[str]:
    """All legal feature columns in the dataset (150 in TCP-CI, fewer if a
    subject's CSV is missing some). Excludes identifiers, labels, and execution
    outcomes. DET_COV_*_Faults are retained as historical previously-detected-
    fault features from the Yaraghi feature model."""
    excluded = _NON_FEATURE_COLS
    return [c for c in df_columns if c not in excluded]


def extract_risk_profiles(
    dataset_path: str,
    sparse: bool = True,
    test_ids: Optional[list[int]] = None,
) -> list[dict]:
    """Extract the full Yaraghi 2022 feature set per test (150 features when the
    CSV has all of them, fewer if a subject's dataset is missing some columns).

    DET_COV_C_Faults and DET_COV_IMP_Faults are included as historical
    previously-detected-fault features defined in the TCP-CI feature model.

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
    if "Verdict" not in df.columns:
        return {}
    rates = (
        df.assign(_fail=df["Verdict"].ne(0))
        .groupby("Test")["_fail"]
        .mean()
    )
    return rates.to_dict()


def extract_exec_times(dataset_path: str) -> dict:
    """Return {test_id: avg_exec_time} mapping for every test.

    If target Duration is hidden from the LLM input, fall back to historical REC
    execution-time features already present in the target feature rows.
    """
    df = load_dataset(dataset_path)
    if "Duration" in df.columns:
        times = df.groupby("Test")["Duration"].mean()
        return times.to_dict()

    latest = (
        df.sort_values("Build", ascending=False)
        .groupby("Test")
        .first()
        .reset_index()
    )
    for col in ("REC_RecentAvgExeTime", "REC_TotalAvgExeTime", "REC_LastExeTime"):
        if col in latest.columns:
            return latest.set_index("Test")[col].fillna(0.0).to_dict()
    return {int(tid): 0.0 for tid in df["Test"].unique().tolist()}


def extract_all_test_ids(dataset_path: str) -> set:
    """Return the set of all unique test IDs in the dataset."""
    df = load_dataset(dataset_path)
    return set(df["Test"].unique().tolist())


def candidate_risk_score(profile: dict) -> float:
    """Cheap deterministic TCP-CI risk score for candidate preselection.

    This is only a retrieval score. The LLM agents still classify/rank the
    selected candidates. We deliberately use prediction-time legal features and
    avoid Verdict/Duration from the target build.
    """
    def val(name: str, default: float = 0.0) -> float:
        raw = profile.get(name, default)
        try:
            if raw == -1:
                return default
            return float(raw)
        except (TypeError, ValueError):
            return default

    score = 0.0

    # History dominates in TCP-CI/Yaraghi and is the safest cheap signal.
    score += 1000.0 * val("REC_RecentFailRate")
    score += 700.0 * val("REC_TotalFailRate")
    if val("REC_LastVerdict") != 0:
        score += 450.0

    last_fail_age = val("REC_LastFailureAge", default=9999.0)
    if last_fail_age <= 2:
        score += 260.0
    elif last_fail_age <= 10:
        score += 120.0

    score += 90.0 * val("REC_TotalAssertRate")
    score += 70.0 * val("REC_TotalExcRate")
    score += 60.0 * val("REC_RecentTransitionRate")
    score += 40.0 * val("REC_TotalTransitionRate")

    # Coverage/change signals retrieve tests relevant to the current build.
    score += 160.0 * val("COV_ChnScoreSum")
    score += 100.0 * val("COV_ImpScoreSum")
    score += 10.0 * val("COV_ChnCount")
    score += 5.0 * val("COV_ImpCount")

    # Young, complex, or low-owner-experience tests are useful backstops.
    age = val("REC_Age", default=9999.0)
    if age <= 5:
        score += 80.0
    elif age <= 20:
        score += 25.0

    owner_exp = val("TES_PRO_OwnersExperience", default=1.0)
    if owner_exp <= 0.5:
        score += 35.0
    score += min(val("TES_PRO_CommitCount"), 20.0) * 3.0
    score += min(val("TES_COM_SumCyclomatic"), 250.0) * 0.12
    score += min(val("TES_COM_CountLineCode"), 2000.0) * 0.01

    # Long-running tests often cover more behavior, but keep this capped so cost
    # never overwhelms failure evidence.
    score += min(val("REC_RecentAvgExeTime"), 60000.0) / 2000.0
    score += min(val("REC_TotalAvgExeTime"), 60000.0) / 3000.0

    return score


def select_candidate_test_ids(dataset_path: str, cap: int) -> tuple[list[int], dict[int, float]]:
    """Return top-K test IDs and their deterministic retrieval scores."""
    profiles = extract_risk_profiles(dataset_path, sparse=False)
    scored = []
    for profile in profiles:
        tid = int(profile["test"])
        score = candidate_risk_score(profile)
        exec_time = float(profile.get("REC_RecentAvgExeTime", profile.get("REC_TotalAvgExeTime", 0.0)) or 0.0)
        scored.append((score, -exec_time, tid))
    scored.sort(reverse=True)
    selected = [tid for _, _, tid in scored[:cap]]
    score_map = {tid: score for score, _, tid in scored}
    return selected, score_map
