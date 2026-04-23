"""
get_test_complexity — Expose TES_COM_ (31) + TES_PRO_ (6) = 37 features
about test file complexity and ownership.

Pilot mode:      Reads from the CSV dataset.
Production mode: Extracts complexity via SciTools Understand and process
                 metrics via pydriller/git history (matches TCP-CI's
                 DatasetFactory.compute_tes_features).
"""

import pandas as pd
from langchain_core.tools import tool
from tcp_agent.config import get_mode, get_config, AgentMode

# ── Feature column definitions ───────────────────────────────────────
# Mirrors TCP-CI Feature class exactly

_COMPLEXITY_METRICS = [
    "CountDeclFunction",
    "CountLine",
    "CountLineBlank",
    "CountLineCode",
    "CountLineCodeDecl",
    "CountLineCodeExe",
    "CountLineComment",
    "CountStmt",
    "CountStmtDecl",
    "CountStmtExe",
    "RatioCommentToCode",
    "MaxCyclomatic",
    "MaxCyclomaticModified",
    "MaxCyclomaticStrict",
    "MaxEssential",
    "MaxNesting",
    "SumCyclomatic",
    "SumCyclomaticModified",
    "SumCyclomaticStrict",
    "SumEssential",
    "CountDeclClass",
    "CountDeclClassMethod",
    "CountDeclClassVariable",
    "CountDeclExecutableUnit",
    "CountDeclInstanceMethod",
    "CountDeclInstanceVariable",
    "CountDeclMethod",
    "CountDeclMethodDefault",
    "CountDeclMethodPrivate",
    "CountDeclMethodProtected",
    "CountDeclMethodPublic",
]

_PROCESS_METRICS = [
    "CommitCount",
    "DistinctDevCount",
    "OwnersContribution",
    "MinorContributorCount",
    "OwnersExperience",
    "AllCommitersExperience",
]

# Full column names as they appear in the dataset (31 + 6 = 37)
TES_COM_COLS = [f"TES_COM_{m}" for m in _COMPLEXITY_METRICS]
TES_PRO_COLS = [f"TES_PRO_{m}" for m in _PROCESS_METRICS]
ALL_COLS = TES_COM_COLS + TES_PRO_COLS  # 37 total


def _pilot_get_test_complexity(dataset_path: str, test_ids=None) -> list[dict]:
    """Pilot mode: read TES_COM_ + TES_PRO_ features from the CSV."""
    df = pd.read_csv(dataset_path)

    # get the latest build for each test (same pattern as get_test_risk_profile)
    latest = df.sort_values("Build", ascending=False).groupby("Test").first()
    latest = latest.reset_index().copy()

    if test_ids is not None:
        latest = latest[latest["Test"].isin(test_ids)]

    keep_cols = ["Test"] + [c for c in ALL_COLS if c in latest.columns]
    result = latest[keep_cols].rename(columns={"Test": "test"})

    # sort by cyclomatic complexity descending — most complex tests first
    if "TES_COM_SumCyclomatic" in result.columns:
        result = result.sort_values("TES_COM_SumCyclomatic", ascending=False)

    return result.to_dict("records")


def _production_get_test_complexity() -> list[dict]:
    """
    Production mode: extract TES_COM_ + TES_PRO_ features from real sources.

    This mirrors TCP-CI's DatasetFactory.compute_tes_features which:
    1. Runs SciTools Understand on each test file to get static complexity
       metrics (CountDeclFunction, MaxCyclomatic, SumCyclomatic, etc.)
    2. Mines git history via pydriller to compute process metrics
       (CommitCount, DistinctDevCount, OwnersContribution, etc.)

    See: TCP-CI-main/src/python/dataset_factory.py::compute_tes_features()
         TCP-CI-main/src/python/dataset_factory.py::compute_all_metrics()
         TCP-CI-main/src/python/dataset_factory.py::compute_static_metrics()
         TCP-CI-main/src/python/dataset_factory.py::compute_process_metrics()
    """
    raise NotImplementedError(
        "Production mode for get_test_complexity is not yet implemented.\n"
        "To implement, wire up:\n"
        "  1. SciTools Understand (UnderstandFileAnalyzer) to compute TES_COM_ metrics\n"
        "  2. pydriller + git history to compute TES_PRO_ metrics\n"
        "See TCP-CI-main/src/python/dataset_factory.py::compute_tes_features()"
    )


@tool
def get_test_complexity(dataset_path: str, test_ids=None) -> list[dict]:
    """Get the complexity and ownership profile for every test (or a subset).

    Returns 37 features per test:
    - TES_COM_ (31 features): Static complexity metrics of the test file itself —
      cyclomatic complexity (MaxCyclomatic, SumCyclomatic), size (CountLineCode,
      CountStmtExe), nesting depth (MaxNesting), class/method counts, etc.
      More complex tests tend to be more failure-prone and cover more behavior.
    - TES_PRO_ (6 features): Development process metrics of the test file —
      CommitCount, DistinctDevCount, OwnersContribution, MinorContributorCount,
      OwnersExperience, AllCommitersExperience.
      Tests with many contributors or high churn may be less stable.

    Use this to identify tests that are structurally complex or have
    volatile ownership — both are risk indicators for failure.
    Optional: pass test_ids (list of ints) to get profiles for specific tests only.
    """
    mode = get_mode()
    if mode == AgentMode.PILOT:
        return _pilot_get_test_complexity(dataset_path, test_ids=test_ids)
    else:
        return _production_get_test_complexity()
