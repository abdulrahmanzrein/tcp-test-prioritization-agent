import pandas as pd
from langchain_core.tools import tool

@tool
def get_test_history(dataset_path, test_name):
    """Look up a single test's history. Returns how many builds it ran in and what percentage of those it failed. Use this to drill into a specific test you're suspicious about."""
    df = pd.read_csv(dataset_path)
    test_df = df[df["Test"] == test_name]
    
    if test_df.empty:
        return {"test": test_name, "runs": 0, "failure_rate": 0.0}
    
    runs = len(test_df) #how many builds a specific tests went through
    fails = (test_df["Verdict"] != 0).sum()

    return {
        "test": test_name,
        "runs": runs,
        "failure_rate": round(fails / runs, 3)
    }

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
def get_execution_times(dataset_path):
    """Get average execution time for every test. Returns all tests sorted by duration (slowest first). Use this to factor in test cost — fast tests that might fail should run before slow ones."""
    df = pd.read_csv(dataset_path)
    exec_times = df.groupby("Test")["Duration"].mean().reset_index()
    exec_times.columns = ["test", "avg_duration"]
    exec_times = exec_times.sort_values("avg_duration", ascending=False)
    return exec_times.to_dict("records")