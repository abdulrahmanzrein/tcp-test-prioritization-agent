from langchain_core.tools import tool

from tcp_agent.data_cache import load_dataset

# NOTE: these tools were removed from the active toolset.
# get_tests_for_changed_files is broken — the dataset uses numeric test IDs, not named tests,
# so filename matching never works. keeping here for reference if the agent ever connects to a
# real repo with named tests.
# get_high_coverage_tests is superseded by get_test_risk_profile which includes COV_ features.

@tool
def get_tests_for_changed_files(dataset_path, changed_files):
    """
    Given a list of changed filenames, return tests that likely cover them.
    Matches by looking for the filename (without extension) in the test name.
    """

    # NOTE: this dataset uses numeric test IDs not named tests so file matching won't work here.
    # the agent still ranks effectively using failure rate and recent build history instead.

    df = load_dataset(dataset_path)
    all_tests = df["Test"].unique().tolist() #create a list to store all test names

    matched_tests = []
    for file in changed_files:
        base_name = file.split("/")[-1].replace(".py", "")

        #find ALL tests whose name contains the basename
        matches = [t for t in all_tests if base_name.lower() in t.lower()]
        #add matches to the running list
        matched_tests.extend(matches)

    # removing duplicates via set
    return list(set(matched_tests))


@tool
def get_high_coverage_tests(dataset_path, n=10):
    """Get the top n tests with the highest code coverage scores. Tests that cover more code are more likely to catch regressions. Use this to find high-value safety-net tests."""
    df = load_dataset(dataset_path)

    covered_cols = df.filter(like="COV_")

    df = df.copy()
    df["covered_score"] = covered_cols.mean(axis=1)

    # group tests by name, average their coverage score, sort highest first, take top n, return as list of dicts
    return df.groupby("Test")["covered_score"].mean().sort_values(ascending=False).head(n).reset_index().to_dict("records")
