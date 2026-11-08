"""Download the UCI Bank Marketing data, train five models, and save artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from ucimlrepo import fetch_ucirepo


ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "subscribed"
DROP_COLUMNS = ["duration"]

MODEL_SPECS = {
    "logistic_regression": ("Logistic Regression", LogisticRegression(max_iter=2000)),
    "decision_tree": ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    "knn": ("KNN", KNeighborsClassifier(n_neighbors=5)),
    "naive_bayes": ("Naive Bayes", GaussianNB()),
    "support_vector_machine": (
        "Support Vector Machine",
        LinearSVC(max_iter=5000, random_state=RANDOM_STATE),
    ),
}


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch UCI dataset 222 and return predictors and a binary target."""
    dataset = fetch_ucirepo(id=222)
    features = dataset.data.features.copy()
    targets = dataset.data.targets

    if targets is None or targets.empty:
        raise ValueError("UCI dataset 222 did not provide a target column.")

    target_values = targets.iloc[:, 0]
    target_name = str(targets.columns[0]).strip().lower()
    if target_name not in {"y", TARGET}:
        raise ValueError(f"Unexpected UCI target column: {targets.columns[0]!r}")

    features.columns = [str(column).strip().lower() for column in features.columns]
    features = features.drop(columns=DROP_COLUMNS, errors="ignore")

    normalized_target = target_values.astype("string").str.strip().str.lower()
    target = normalized_target.map({"no": 0, "yes": 1})
    if target.isna().any():
        raise ValueError("The UCI target contains missing or unrecognized labels.")
    target = target.astype("int64").rename(TARGET)
    features = features.reset_index(drop=True)
    target = target.reset_index(drop=True)
    return features, target


def create_pipeline(
    classifier: BaseEstimator,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            ("classifier", classifier),
        ]
    )


def positive_class_scores(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        positive_index = list(classifier.classes_).index(1)
        return pd.Series(model.predict_proba(features)[:, positive_index], index=features.index)
    return pd.Series(model.decision_function(features), index=features.index)


def calculate_metrics(
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    predictions = model.predict(features)
    scores = positive_class_scores(model, features)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "auc": float(roc_auc_score(target, scores)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(target, predictions)),
    }


def main() -> None:
    features, target = load_dataset()
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    if not numeric_columns and not categorical_columns:
        raise ValueError("The dataset has no usable predictor columns.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model_directory = ROOT / "model"
    data_directory = ROOT / "data"
    model_directory.mkdir(exist_ok=True)
    data_directory.mkdir(exist_ok=True)

    metric_rows = []
    model_metadata = {}
    for model_id, (display_name, classifier) in MODEL_SPECS.items():
        pipeline = create_pipeline(classifier, numeric_columns, categorical_columns)
        pipeline.fit(X_train, y_train)
        model_path = model_directory / f"{model_id}.joblib"
        joblib.dump(pipeline, model_path)
        metric_rows.append(
            {"model_id": model_id, "model": display_name, **calculate_metrics(pipeline, X_test, y_test)}
        )
        model_metadata[model_id] = {
            "name": display_name,
            "file": f"model/{model_id}.joblib",
        }
        print(f"Trained {display_name}; saved {model_path.relative_to(ROOT)}")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(ROOT / "model_metrics.csv", index=False, float_format="%.6f")

    test_data = X_test.copy()
    test_data[TARGET] = y_test
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    processed_data = features.copy()
    processed_data[TARGET] = target
    processed_data.to_csv(data_directory / "bank_marketing_processed.csv", index=False)

    metadata = {
        "dataset": "UCI Bank Marketing",
        "dataset_id": 222,
        "dataset_url": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "target": TARGET,
        "target_mapping": {"no": 0, "yes": 1},
        "feature_columns": features.columns.tolist(),
        "dropped_columns": DROP_COLUMNS,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "models": model_metadata,
        "metrics_file": "model_metrics.csv",
        "test_data_file": "test_data.csv",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print("\nHeld-out test metrics:")
    print(metrics.to_string(index=False))
    print("\nSaved test_data.csv, model_metrics.csv, model_metadata.json, model/ and data/.")


if __name__ == "__main__":
    main()
