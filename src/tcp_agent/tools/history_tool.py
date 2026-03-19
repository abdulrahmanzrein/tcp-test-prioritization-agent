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
def get_recent_failures(dataset_path, n=10):
    """Get the top n tests ranked by historical failure rate. Use this first to see which tests fail most often across all builds. This is usually the strongest signal for predicting future failures."""
    df = pd.read_csv(dataset_path)

    #lambda is a oneline built in function method
    failure_rates = df.groupby("Test").apply(
        lambda g: (g["Verdict"] != 0).sum() / len(g)
    ).reset_index()
    failure_rates.columns = ["test", "failure_rate"]  # name columns explicitly — reset_index creates unnamed column by default

    top_fails = failure_rates.sort_values("failure_rate", ascending=False)
    return top_fails.to_dict("records")