import pandas as pd

def rank_tests(model, X_test, metadata_test):
    probs = model.predict_proba(X_test)[:,1] #take only the fail rates from X_test
    ranked = metadata_test.copy().reset_index(drop=True)
    ranked["fail_prob"] = probs #adds a new fail probability coloumn
    
    ranked = ranked.sort_values("fail_prob", ascending=False).reset_index(drop=True) #sorts value in order of most likely to fail first
    return ranked

