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
    # TODO: filter where Build == build_id AND Verdict != 0
    # TODO: return build id, list of failed test names, and count
    pass
