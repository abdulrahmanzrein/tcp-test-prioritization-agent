import pandas as pd


def get_failed_builds(dataset_path, n=5):
    """
    Returns the n most recent builds that had at least one test failure.
    """
    df = pd.read_csv(dataset_path)

    # count how many tests failed per build
    fail_per_build = df.groupby("Build").apply(lambda g: (g["Verdict"] != 0).sum()).reset_index()
    fail_per_build.columns = ["build", "num_failures"]

    # only keep builds that actually had failures, sort by build id (higher = more recent)
    failed = fail_per_build[fail_per_build["num_failures"] > 0]
    return failed.sort_values("build", ascending=False).head(n).to_dict("records")


def get_build_failure_summary(dataset_path, build_id):
    """
    For a specific build, return which tests failed.
    """
    
    df = pd.read_csv(dataset_path)


    failed = df[(df["Build"] == build_id) & (df["Verdict"] != 0)] #gives all the failed tests of a specific build

    return {
        "build_id": build_id,
        "failed_tests": failed["Test"].tolist(),
        "num_failures": len(failed)
     }
    
