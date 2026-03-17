#How many of the top k tests acctually failed?
def precision_at_k(ranked_df, k=10):
    top_k = ranked_df.head(k) #top k rows
    failures_in_top_k = (top_k["Verdict"] != 0).sum() #all the rows that failed 
    return failures_in_top_k / k

#The earlier failures appear, the better. APFD converts that into a 0–1 score. Random order gets ~0.5. 
def apfd(ranked_df):
    n = len(ranked_df)
    failures = ranked_df[ranked_df["Verdict"] != 0] #rows that failed
    m = len(failures)

    if m == 0:
        return 1.0 #no fails equals perfect score

    positions = failures.index.to_series() + 1 #1 indexed position of rankedlist

    return 1 - (positions.sum() / (n * m)) + (1 / (2 * n))


# for each failure, calculate how much test time we saved by finding it early, that is what apfdc calculates
def apfdc(ranked_df):
    n = len(ranked_df)
    total_duration = ranked_df["Duration"].sum() #duration of each tests, we need to this weigh it later
    failures = ranked_df[ranked_df["Verdict"] != 0]
    m = len(failures)

    if m == 0 or total_duration == 0:
        return 1.0

    score = 0
    for i, row in failures.iterrows():
        duration_after = ranked_df.loc[i+1:, "Duration"].sum() #tests durations of tests after the fails
        score += duration_after + row["Duration"] / 2

    return score / (total_duration * m) #normalize it between 0 - 1.0

    


