import pandas as pd
from langchain_core.tools import tool

@tool
def get_failed_builds(dataset_path, n=5):
    """Get the most recent builds that had test failures. Returns build IDs and how many tests failed in each. Use this to understand recent CI stability and find builds worth investigating."""
    df = pd.read_csv(dataset_path)

    # count how many tests failed per build
    fail_per_build = df.groupby("Build").apply(lambda g: (g["Verdict"] != 0).sum()).reset_index()
    fail_per_build.columns = ["build", "num_failures"]

    # only keep builds that actually had failures, sort by build id (higher = more recent)
    failed = fail_per_build[fail_per_build["num_failures"] > 0]
    return failed.sort_values("build", ascending=False).head(n).to_dict("records")

@tool
def get_build_failure_summary(dataset_path, build_id):
    """Get the list of tests that failed in a specific build. Use this after get_failed_builds to see exactly which tests broke in a particular build."""

    df = pd.read_csv(dataset_path)

    failed = df[(df["Build"] == build_id) & (df["Verdict"] != 0)] #gives all the failed tests of a specific build

    return {
        "build_id": build_id,
        "failed_tests": failed["Test"].tolist(),
        "num_failures": len(failed)
     }
