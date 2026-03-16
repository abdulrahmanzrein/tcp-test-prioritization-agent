import pandas as pd
from imblearn.over_sampling import SMOTE

# features.py
# cleans the feature matrix and handles class imbalance
# the dataset has ~3% failure rate so without balancing the model just predicts pass every time
# SMOTE creates synthetic failure examples to fix this (justified by Mendoza et al. 2022)


def clean_features(X):
    #cleaning the dataset so columns with -1 don't get ignored and instead are read as 0
    X = X.replace(-1, 0)
    X = X.replace(-1.0, 0)
    return X

def apply_smote(X, y):
    print(f"before SMOTE - failures: {y.sum()}, passes: {(y == 0).sum()}")
    sm = SMOTE(random_state=0)
    X_resampled, y_resampled = sm.fit_resample(X, y)
    print(f"after SMOTE  — failures: {y_resampled.sum()}, passes: {(y_resampled==0).sum()}")
    return X_resampled, y_resampled

