import mlflow.sklearn
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.ensemble import (
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
import pandas as pd
import joblib
import os


df = pd.read_csv(
    "data/processed/processed_with_target.csv"
)
drop_cols = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "TransactionStartTime"
]

X = df.drop(
    columns=drop_cols + ["is_high_risk"],
    errors="ignore"
)
X = pd.get_dummies(
    X,
    drop_first=True
)
y = df["is_high_risk"]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)
mlflow.set_experiment(
    "Credit_Risk_Model"
)

# logistic regression
with mlflow.start_run(
    run_name="LogisticRegression"
):

    model = LogisticRegression(
        max_iter=5000,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        preds
    )

    precision = precision_score(
        y_test,
        preds
    )

    recall = recall_score(
        y_test,
        preds
    )

    f1 = f1_score(
        y_test,
        preds
    )

    roc_auc = roc_auc_score(
        y_test,
        preds
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    mlflow.sklearn.log_model(
        model,
        "model"
    )

  # random forest
with mlflow.start_run(
    run_name="RandomForest"
):

    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    rf.fit(
        X_train,
        y_train
    )

    preds = rf.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        preds
    )

    precision = precision_score(
        y_test,
        preds
    )

    recall = recall_score(
        y_test,
        preds
    )

    f1 = f1_score(
        y_test,
        preds
    )

    roc_auc = roc_auc_score(
        y_test,
        preds
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc
    )

    mlflow.sklearn.log_model(
        rf,
        "model"
    )
params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10]
}
rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

grid = GridSearchCV(
    rf,
    params,
    cv=3,
    scoring="f1"
)

grid.fit(
    X_train,
    y_train
)

best_model = grid.best_estimator_

print(grid.best_params_)
os.makedirs("models", exist_ok=True)
joblib.dump(
    best_model,
    "models/best_model.pkl"
)
