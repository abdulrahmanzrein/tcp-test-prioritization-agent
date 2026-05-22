from tcp_agent.data_cache import load_dataset


#load the file into a pandas dataframe
def load_data(file_path):
    df = load_dataset(file_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {file_path}") 
    return df


def get_features_and_labels(df):
    #turns verdict into either 0 (pass) or 1 (fail)
    y = (df["Verdict"] != 0).astype(int)

    # drops identifiers, labels, and execution outcomes. DET_COV_*_Faults stay
    # in the feature matrix as historical previously-detected-fault features.

    drop_cols = ["Build", "Test", "Verdict", "Duration"]
    X = df.drop(columns=drop_cols)


    print(f"failure rate: {y.mean():.1%}  |  features: {X.shape[1]}")
    return X, y

def get_metadata(df):
    #were gonna need this info later for specifying specific tests
    return df[["Build", "Test", "Duration", "Verdict"]].copy()




