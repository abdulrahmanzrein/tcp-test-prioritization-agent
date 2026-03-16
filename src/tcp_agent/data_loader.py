import pandas as pd


#load the file into a pandas dataframe
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {file_path}") 
    return df


def get_features_and_labels(df):
    #turns verdict into either 0 (pass) or 1 (fail)
    y = (df["Verdict"] != 0).astype(int)

    #drops the columns that are not features
    drop_cols = ["Build", "Test", "Verdict", "Duration", "DET_COV_C_Faults", "DET_COV_IMP_Faults"]
    X = df.drop(columns=drop_cols)

    # DET_COV_C_Faults and DET_COV_IMP_Faults tell you how many bugs the test actually found
    # we can't know this before running the test — using it would be cheating
    # C = in changed files, IMP = in files impacted by the change

    print(f"failure rate: {y.mean():.1%}  |  features: {X.shape[1]}")
    return X, y

def get_metadata(df):
    return df[["Build", "Test", "Duration", "Verdict"]].copy()





