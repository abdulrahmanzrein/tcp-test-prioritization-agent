import pandas as pd
from langchain_core.tools import tool

@tool
def get_all_failure_rates(dataset_path):
    """Get every test in the dataset ranked by historical failure rate. Returns all tests with their failure rate (0.0 to 1.0). Use this first to get the full picture of which tests fail the most."""
    df = pd.read_csv(dataset_path)

    #lambda is a oneline built in function method
    failure_rates = df.groupby("Test").apply(
        lambda g: (g["Verdict"] != 0).sum() / len(g)
    ).reset_index()
    failure_rates.columns = ["test", "failure_rate"]  # name columns explicitly — reset_index creates unnamed column by default

    top_fails = failure_rates.sort_values("failure_rate", ascending=False)
    return top_fails.to_dict("records")


@tool
def get_test_risk_profile(dataset_path):
    """Get the most recent risk profile for every test using REC_ (execution history) and DET_COV_ (fault detection) features.
    Returns each test's latest build snapshot with: recent/total failure rates, last verdict,
    transition rates, failure age, coverage fault counts, and change/impact coverage scores.
    These are the strongest predictive signals — call this early to guide your ranking."""
    df = pd.read_csv(dataset_path)

    # get the latest build for each test — higher build id = more recent
    latest = df.sort_values("Build", ascending=False).groupby("Test").first()
    latest = latest.reset_index().copy()

    # only grab the columns that actually matter for ranking
    keep_cols = [
        "Test",
        # how often it fails, did it fail last time, how long ago, etc
        "REC_RecentFailRate", "REC_TotalFailRate",
        "REC_LastVerdict", "REC_LastFailureAge",
        "REC_RecentTransitionRate", "REC_TotalTransitionRate",
        "REC_Age",
        "REC_RecentAssertRate", "REC_TotalAssertRate",
        "REC_RecentExcRate", "REC_TotalExcRate",
        # execution time history
        "REC_LastExeTime",
        "REC_RecentAvgExeTime", "REC_RecentMaxExeTime",
        "REC_TotalAvgExeTime", "REC_TotalMaxExeTime",
        # how long since the last state change
        "REC_LastTransitionAge",
        # does the test tend to fail when specific files change
        "REC_MaxTestFileFailRate", "REC_MaxTestFileTransitionRate",
        # does the code this test covers have known bugs?
        "DET_COV_C_Faults", "DET_COV_IMP_Faults",
        # how much changed/impacted code does this test cover
        "COV_ChnScoreSum", "COV_ImpScoreSum",
        "COV_ChnCount", "COV_ImpCount",
        # was the test itself recently edited
        "TES_CHN_LinesAdded", "TES_CHN_LinesDeleted",
        "TES_CHN_AddedChangeScattering", "TES_CHN_DeletedChangeScattering",
        "TES_CHN_DMMSize", "TES_CHN_DMMComplexity", "TES_CHN_DMMInterfacing",
    ]

    # filter out any columns that dont exist in this dataset
    keep_cols = [c for c in keep_cols if c in latest.columns]
    result = latest[keep_cols].rename(columns={"Test": "test"})
    result = result.sort_values("REC_RecentFailRate", ascending=False)
    return result.to_dict("records")
