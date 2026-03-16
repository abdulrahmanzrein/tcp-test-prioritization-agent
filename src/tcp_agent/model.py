from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# model.py
# trains a Random Forest classifier to predict test failure probability


def train_model(X, y):
    # split into train and test sets before training
    # the model learns from X_train and y_train and uses that information and
    # tests it out on X_test and y_test
    # 80% training rows and 20# new tests rows
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    from tcp_agent.features import apply_smote
    X_train, y_train = apply_smote(X_train, y_train) #creating fake fails so the model can learn what a fail looks like

    model = RandomForestClassifier(n_estimators=100, random_state=7)
    model.fit(X_train, y_train)

    print(f"model trained on {len(X_train)} samples")
    return model, X_test, y_test
