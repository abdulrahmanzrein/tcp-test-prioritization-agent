import pandas as pd


def get_tests_for_changed_files(dataset_path, changed_files):
    """
    Given a list of changed filenames, return tests that likely cover them.
    Matches by looking for the filename (without extension) in the test name.
    """
    # TODO: load the CSV
    # TODO: get unique test names
    # TODO: for each changed file, strip the extension to get the base name
    # TODO: find tests whose name contains the base name (case insensitive)
    # TODO: return list of matched test names
    pass


def get_high_coverage_tests(dataset_path, n=10):
    """
    Returns the n tests with the highest average coverage score.
    Uses COV_* columns from the dataset.
    """
    # TODO: load the CSV
    # TODO: find all columns that start with "COV_"
    # TODO: calculate the mean COV score per test
    # TODO: return top n tests sorted by coverage score
    pass
