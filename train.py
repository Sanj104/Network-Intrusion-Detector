import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def load_data():
    X_train = pd.read_csv("data/X_train_processed.csv")
    X_test = pd.read_csv("data/X_test_processed.csv")
    y_train = pd.read_csv("data/y_train.csv")
    y_test = pd.read_csv("data/y_test.csv")

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("\ny_train distribution:")
    print(y_train.squeeze().value_counts())


    print("\ny_test distribution:")
    print(y_test.squeeze().value_counts())

    return X_train, X_test, y_train.squeeze(), y_test.squeeze()


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    base_model = DecisionTreeClassifier(
        random_state=42
    )

    model = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=5
    )

    model.fit(X_train, y_train)

    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, report, cm

def save_model(model):
    joblib.dump(model, "model/best_model.pkl")

def main():

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train models
    logistic = train_logistic_regression(X_train, y_train)
    decision = train_decision_tree(X_train, y_train)
    forest = train_random_forest(X_train, y_train)

    # Evaluate models
    log_acc, log_report, log_cm = evaluate_model(logistic, X_test, y_test)

    dt_acc, dt_report, dt_cm = evaluate_model(decision, X_test, y_test)

    rf_acc, rf_report, rf_cm = evaluate_model(forest, X_test, y_test)

    # Print results
    print("Logistic Regression Accuracy:", log_acc)

    print("Decision Tree Accuracy:", dt_acc)

    print("Random Forest Accuracy:", rf_acc)

    results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        log_acc,
        dt_acc,
        rf_acc
    ]
})

    results.to_csv(
    "model/model_scores.csv",
    index=False
    )
    print("\n===== Model Comparison =====")
    print(results)
    print("\n===== Logistic Regression =====")
    print(log_report)

    print("\n===== Decision Tree =====")
    print(dt_report)

    print("\n===== Random Forest =====")
    print(rf_report)

    # Compare
    models = {
        "Logistic Regression": (logistic, log_acc),
        "Decision Tree": (decision, dt_acc),
        "Random Forest": (forest, rf_acc)
    }

    best_model_name = max(models, key=lambda x: models[x][1])

    best_model = models[best_model_name][0]

    print("\nBest Model:", best_model_name)
    with open("model/best_model_name.txt", "w") as f:
        f.write(best_model_name)

    save_model(best_model)

if __name__ == "__main__":
    main()