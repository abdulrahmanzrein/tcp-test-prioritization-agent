import pandas as pd


def get_test_history(dataset_path, test_name):
    """
    Returns failure rate and run count for a specific test.
    """
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


def get_recent_failures(dataset_path, n=10):
    """
    Returns the n tests with the highest failure rates.
    """
    df = pd.read_csv(dataset_path)

    #lambda is a oneline built in function method
    failure_rates = df.groupby("Test").apply(
        lambda g: (g["Verdict"] != 0).sum() / len(g)
    ).reset_index()
    failure_rates.columns = ["test", "failure_rate"]  # name columns explicitly — reset_index creates unnamed column by default

    top_fails = failure_rates.sort_values("failure_rate", ascending=False)
    return top_fails.to_dict("records")