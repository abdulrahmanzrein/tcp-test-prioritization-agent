from langchain_core.tools import tool

from tcp_agent.data_cache import load_dataset


@tool
def get_test_risk_profile(dataset_path, test_ids=None):
    """Get the most recent risk profile for every test (or a subset) using REC_ (execution history) and DET_COV_ (fault detection) features.
    Returns each test's latest build snapshot with: recent/total failure rates, last verdict,
    transition rates, failure age, coverage fault counts, and change/impact coverage scores.
    These are the strongest predictive signals — call this early to guide your ranking.
    Optional: pass test_ids (list of ints) to get profiles for specific tests only."""
    df = load_dataset(dataset_path)

    # get the latest build for each test — higher build id = more recent
    latest = df.sort_values("Build", ascending=False).groupby("Test").first()
    latest = latest.reset_index().copy()

    if test_ids is not None:
        latest = latest[latest["Test"].isin(test_ids)]

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

    keep_cols = [c for c in keep_cols if c in latest.columns]
    result = latest[keep_cols].rename(columns={"Test": "test"})
    result = result.sort_values("REC_RecentFailRate", ascending=False)
    return result.to_dict("records")
