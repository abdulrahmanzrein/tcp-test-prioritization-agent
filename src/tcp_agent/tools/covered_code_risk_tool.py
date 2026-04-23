"""
get_covered_code_risk — Expose COD_COV_ features (81 total) about the
complexity, churn, and process metrics of the production code each test covers.

Pilot mode:      Reads from the CSV dataset.
Production mode: Extracts via SciTools Understand + pydriller (matches TCP-CI's
                 DatasetFactory.compute_cod_cov_features).
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

_CHANGE_METRICS = [
    "LinesAdded",
    "LinesDeleted",
    "AddedChangeScattering",
    "DeletedChangeScattering",
    "DMMSize",
    "DMMComplexity",
    "DMMInterfacing",
]

# Full column names as they appear in the dataset
# Changed coverage: COM (31) + PRO (6) + CHN (7) = 44
COD_COV_COM_C_COLS = [f"COD_COV_COM_C_{m}" for m in _COMPLEXITY_METRICS]      # 31
COD_COV_PRO_C_COLS = [f"COD_COV_PRO_C_{m}" for m in _PROCESS_METRICS]        # 6
COD_COV_CHN_C_COLS = [f"COD_COV_CHN_C_{m}" for m in _CHANGE_METRICS]         # 7

# Impacted coverage: COM (31) + PRO (6) = 37
COD_COV_COM_IMP_COLS = [f"COD_COV_COM_IMP_{m}" for m in _COMPLEXITY_METRICS]  # 31
COD_COV_PRO_IMP_COLS = [f"COD_COV_PRO_IMP_{m}" for m in _PROCESS_METRICS]    # 6

# Grand total: 31 + 6 + 7 + 31 + 6 = 81
ALL_COLS = (
    COD_COV_COM_C_COLS
    + COD_COV_PRO_C_COLS
    + COD_COV_CHN_C_COLS
    + COD_COV_COM_IMP_COLS
    + COD_COV_PRO_IMP_COLS
)


def _pilot_get_covered_code_risk(dataset_path: str, test_ids=None) -> list[dict]:
    """Pilot mode: read COD_COV_ features from the CSV."""
    df = pd.read_csv(dataset_path)

    # get the latest build for each test
    latest = df.sort_values("Build", ascending=False).groupby("Test").first()
    latest = latest.reset_index().copy()

    if test_ids is not None:
        latest = latest[latest["Test"].isin(test_ids)]

    keep_cols = ["Test"] + [c for c in ALL_COLS if c in latest.columns]
    result = latest[keep_cols].rename(columns={"Test": "test"})

    # sort by total complexity of changed covered code — riskiest tests first
    if "COD_COV_COM_C_SumCyclomatic" in result.columns:
        result = result.sort_values("COD_COV_COM_C_SumCyclomatic", ascending=False)

    return result.to_dict("records")


def _production_get_covered_code_risk() -> list[dict]:
    """
    Production mode: extract COD_COV_ features from real sources.

    This mirrors TCP-CI's DatasetFactory.compute_cod_cov_features which:
    1. Uses test coverage (compute_test_coverage) to find which production
       code files each test covers — split into "changed" (C) and
       "impacted" (IMP) sets.
    2. For each covered production file, computes:
       - Static complexity (via SciTools Understand): COD_COV_COM_C_*, COD_COV_COM_IMP_*
       - Process metrics (via git history):           COD_COV_PRO_C_*, COD_COV_PRO_IMP_*
       - Change metrics (via pydriller commits):      COD_COV_CHN_C_*
    3. Aggregates per-file metrics into per-test metrics using weighted sums
       (aggregate_cov_metrics), weighted by the coverage scores.

    See: TCP-CI-main/src/python/dataset_factory.py::compute_cod_cov_features()
         TCP-CI-main/src/python/dataset_factory.py::compute_test_coverage()
         TCP-CI-main/src/python/dataset_factory.py::aggregate_cov_metrics()
         TCP-CI-main/src/python/dataset_factory.py::compute_all_metrics()
    """
    raise NotImplementedError(
        "Production mode for get_covered_code_risk is not yet implemented.\n"
        "To implement, wire up:\n"
        "  1. Coverage analysis to find changed/impacted production files per test\n"
        "  2. SciTools Understand to compute complexity of those files\n"
        "  3. pydriller + git history for process and change metrics\n"
        "  4. Coverage-weighted aggregation (aggregate_cov_metrics)\n"
        "See TCP-CI-main/src/python/dataset_factory.py::compute_cod_cov_features()"
    )


@tool
def get_covered_code_risk(dataset_path: str, test_ids=None) -> list[dict]:
    """Get the risk profile of production code each test covers (81 features).

    Returns complexity, churn, and process metrics for the production code
    that each test covers, split by:

    **Changed code (C) — 44 features:**
    - COD_COV_COM_C_* (31): Complexity of directly changed production code
      (cyclomatic complexity, line counts, nesting depth, class/method counts).
    - COD_COV_PRO_C_* (6): Process metrics of changed code (commit count,
      distinct developers, ownership concentration).
    - COD_COV_CHN_C_* (7): Churn metrics of changed code (lines added/deleted,
      change scattering, DMM size/complexity/interfacing).

    **Impacted code (IMP) — 37 features:**
    - COD_COV_COM_IMP_* (31): Complexity of downstream-impacted production code.
    - COD_COV_PRO_IMP_* (6): Process metrics of impacted code.

    Tests covering complex, high-churn, multi-author production code are
    higher risk and should be prioritized. High COD_COV_COM_C_SumCyclomatic
    or COD_COV_CHN_C_LinesAdded values signal risky code under test.
    Optional: pass test_ids (list of ints) to get profiles for specific tests only.
    """
    mode = get_mode()
    if mode == AgentMode.PILOT:
        return _pilot_get_covered_code_risk(dataset_path, test_ids=test_ids)
    else:
        return _production_get_covered_code_risk()
