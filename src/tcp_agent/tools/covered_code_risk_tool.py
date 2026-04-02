"""
get_covered_code_risk — Expose COD_COV_ features (81 total) about the
complexity, churn, and process metrics of the production code each test covers.

Reads from the CSV dataset (pilot mode).
"""

import pandas as pd
from langchain_core.tools import tool

# Shared metric names (same structure used for test files and covered code)
_COMPLEXITY = [
    "CountDeclFunction", "CountLine", "CountLineBlank", "CountLineCode",
    "CountLineCodeDecl", "CountLineCodeExe", "CountLineComment",
    "CountStmt", "CountStmtDecl", "CountStmtExe", "RatioCommentToCode",
    "MaxCyclomatic", "MaxCyclomaticModified", "MaxCyclomaticStrict",
    "MaxEssential", "MaxNesting",
    "SumCyclomatic", "SumCyclomaticModified", "SumCyclomaticStrict", "SumEssential",
    "CountDeclClass", "CountDeclClassMethod", "CountDeclClassVariable",
    "CountDeclExecutableUnit", "CountDeclInstanceMethod", "CountDeclInstanceVariable",
    "CountDeclMethod", "CountDeclMethodDefault", "CountDeclMethodPrivate",
    "CountDeclMethodProtected", "CountDeclMethodPublic",
]

_PROCESS = [
    "CommitCount", "DistinctDevCount", "OwnersContribution",
    "MinorContributorCount", "OwnersExperience", "AllCommitersExperience",
]

_CHANGE = [
    "LinesAdded", "LinesDeleted", "AddedChangeScattering",
    "DeletedChangeScattering", "DMMSize", "DMMComplexity", "DMMInterfacing",
]

# Changed coverage: COM (31) + PRO (6) + CHN (7) = 44
# Impacted coverage: COM (31) + PRO (6) = 37
# Grand total: 81
ALL_COLS = (
    [f"COD_COV_COM_C_{m}" for m in _COMPLEXITY]       # 31
    + [f"COD_COV_PRO_C_{m}" for m in _PROCESS]        # 6
    + [f"COD_COV_CHN_C_{m}" for m in _CHANGE]         # 7
    + [f"COD_COV_COM_IMP_{m}" for m in _COMPLEXITY]   # 31
    + [f"COD_COV_PRO_IMP_{m}" for m in _PROCESS]      # 6
)


@tool
def get_covered_code_risk(dataset_path: str) -> list[dict]:
    """Get the risk profile of production code each test covers (81 features).

    Returns per test:
    - COD_COV_COM_C_* (31): Complexity of directly changed production code.
    - COD_COV_PRO_C_* (6): Process metrics of changed code.
    - COD_COV_CHN_C_* (7): Churn metrics of changed code (lines added/deleted, DMM).
    - COD_COV_COM_IMP_* (31): Complexity of downstream-impacted production code.
    - COD_COV_PRO_IMP_* (6): Process metrics of impacted code.

    Tests covering complex, high-churn production code are higher risk.
    """
    df = pd.read_csv(dataset_path)

    latest = df.sort_values("Build", ascending=False).groupby("Test").first()
    latest = latest.reset_index().copy()

    keep_cols = ["Test"] + [c for c in ALL_COLS if c in latest.columns]
    result = latest[keep_cols].rename(columns={"Test": "test"})

    if "COD_COV_COM_C_SumCyclomatic" in result.columns:
        result = result.sort_values("COD_COV_COM_C_SumCyclomatic", ascending=False)

    return result.to_dict("records")
