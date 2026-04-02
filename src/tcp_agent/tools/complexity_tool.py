"""
get_test_complexity — Expose TES_COM_ (31) + TES_PRO_ (6) = 37 features
about test file complexity and ownership.

Reads from the CSV dataset (pilot mode).
"""

import pandas as pd
from langchain_core.tools import tool

# Column names as they appear in the dataset
TES_COM_COLS = [
    "TES_COM_CountDeclFunction", "TES_COM_CountLine", "TES_COM_CountLineBlank",
    "TES_COM_CountLineCode", "TES_COM_CountLineCodeDecl", "TES_COM_CountLineCodeExe",
    "TES_COM_CountLineComment", "TES_COM_CountStmt", "TES_COM_CountStmtDecl",
    "TES_COM_CountStmtExe", "TES_COM_RatioCommentToCode",
    "TES_COM_MaxCyclomatic", "TES_COM_MaxCyclomaticModified", "TES_COM_MaxCyclomaticStrict",
    "TES_COM_MaxEssential", "TES_COM_MaxNesting",
    "TES_COM_SumCyclomatic", "TES_COM_SumCyclomaticModified", "TES_COM_SumCyclomaticStrict",
    "TES_COM_SumEssential",
    "TES_COM_CountDeclClass", "TES_COM_CountDeclClassMethod", "TES_COM_CountDeclClassVariable",
    "TES_COM_CountDeclExecutableUnit", "TES_COM_CountDeclInstanceMethod",
    "TES_COM_CountDeclInstanceVariable", "TES_COM_CountDeclMethod",
    "TES_COM_CountDeclMethodDefault", "TES_COM_CountDeclMethodPrivate",
    "TES_COM_CountDeclMethodProtected", "TES_COM_CountDeclMethodPublic",
]

TES_PRO_COLS = [
    "TES_PRO_CommitCount", "TES_PRO_DistinctDevCount", "TES_PRO_OwnersContribution",
    "TES_PRO_MinorContributorCount", "TES_PRO_OwnersExperience",
    "TES_PRO_AllCommitersExperience",
]

ALL_COLS = TES_COM_COLS + TES_PRO_COLS  # 37 total


@tool
def get_test_complexity(dataset_path: str) -> list[dict]:
    """Get the complexity and ownership profile for every test (37 features).

    Returns per test:
    - TES_COM_ (31): Static complexity of the test file — cyclomatic complexity,
      line counts, nesting depth, class/method counts.
    - TES_PRO_ (6): Process metrics — commit count, distinct developers,
      ownership concentration, contributor experience.

    More complex tests with many contributors tend to be more failure-prone.
    """
    df = pd.read_csv(dataset_path)

    # latest build per test
    latest = df.sort_values("Build", ascending=False).groupby("Test").first()
    latest = latest.reset_index().copy()

    keep_cols = ["Test"] + [c for c in ALL_COLS if c in latest.columns]
    result = latest[keep_cols].rename(columns={"Test": "test"})

    if "TES_COM_SumCyclomatic" in result.columns:
        result = result.sort_values("TES_COM_SumCyclomatic", ascending=False)

    return result.to_dict("records")
