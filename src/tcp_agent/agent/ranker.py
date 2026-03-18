import pandas as pd


def build_ranked_df(ranked, dataset_path):
    """
    Takes Claude's ranked output and merges it with real Verdict data
    so evaluation.py can score it.
    """
    # TODO: convert Claude's ranked list into a DataFrame
    # TODO: load the dataset and get the real Verdict and Duration per test
    # TODO: merge the two so each row has priority + Verdict + Duration
    # TODO: sort by priority ascending (1 = first to run)
    # TODO: return the merged DataFrame
    pass
