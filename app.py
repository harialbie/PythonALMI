"""Streamlit interface for evaluating the saved Bank Marketing classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
METRIC_COLUMNS = ["accuracy", "auc", "precision", "recall", "f1", "mcc"]


def load_metadata() -> dict:
    metadata_path = ROOT / "model_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Model metadata is missing. Run `python train_models.py` before starting the app."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@st.cache_resource
def load_model(model_path: str) -> Pipeline:
    return joblib.load(model_path)


def normalize_target(target: pd.Series) -> pd.Series:
    normalized = target.astype("string").str.strip().str.lower()
    result = normalized.map({"no": 0, "yes": 1, "0": 0, "1": 1})
    if target.isna().any() or result.isna().any():
        raise ValueError("The subscribed column must contain only yes/no or 1/0 labels.")
    return result.astype("int64")


def positive_class_scores(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        positive_index = list(classifier.classes_).index(1)
        return pd.Series(model.predict_proba(features)[:, positive_index], index=features.index)
    return pd.Series(model.decision_function(features), index=features.index)


def evaluate_model(
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, float], pd.Series]:
    predictions = pd.Series(model.predict(features), index=features.index)
    scores = positive_class_scores(model, features)
    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "auc": float(roc_auc_score(target, scores)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(target, predictions)),
    }
    return metrics, predictions


def main() -> None:
    st.set_page_config(page_title="Bank Marketing Classification", page_icon="📊", layout="wide")
    st.title("Bank Marketing Classification")
    st.caption("BITS AIML Assignment 2 · Term deposit subscription prediction")

    try:
        metadata = load_metadata()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    model_options = metadata.get("models", {})
    if not model_options:
        st.error("No trained models are listed in model_metadata.json. Run the training script again.")
        st.stop()

    uploaded_file = st.file_uploader("Upload a labelled CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV with the training predictors and a `subscribed` target column.")
        st.write("The generated `test_data.csv` is available after running `python train_models.py`.")
        return

    try:
        uploaded_data = pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        return

    target_column = metadata["target"]
    feature_columns = metadata["feature_columns"]
    missing_columns = [column for column in [*feature_columns, target_column] if column not in uploaded_data]
    if missing_columns:
        st.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
        return
    if uploaded_data.empty:
        st.error("The uploaded CSV contains no rows.")
        return

    X = uploaded_data.loc[:, feature_columns]
    try:
        y = normalize_target(uploaded_data[target_column])
    except ValueError as error:
        st.error(str(error))
        return

    st.write(f"Loaded **{len(uploaded_data):,}** labelled rows.")
    selected_model_id = st.selectbox(
        "Model for detailed evaluation",
        options=list(model_options),
        format_func=lambda model_id: model_options[model_id]["name"],
    )

    loaded_models = {}
    for model_id, model_info in model_options.items():
        model_path = ROOT / model_info["file"]
        if not model_path.is_file():
            st.error(f"Saved model is missing: {model_info['file']}. Run `python train_models.py`.")
            st.stop()
        loaded_models[model_id] = load_model(str(model_path))

    comparison_rows = []
    predictions_by_model = {}
    for model_id, model in loaded_models.items():
        metrics, predictions = evaluate_model(model, X, y)
        predictions_by_model[model_id] = predictions
        comparison_rows.append(
            {"model": model_options[model_id]["name"], **metrics}
        )
    comparison = pd.DataFrame(comparison_rows)

    st.subheader("Model comparison on uploaded data")
    st.dataframe(
        comparison.style.format({column: "{:.4f}" for column in METRIC_COLUMNS}),
        use_container_width=True,
        hide_index=True,
    )

    selected_model = loaded_models[selected_model_id]
    selected_predictions = predictions_by_model[selected_model_id]
    selected_metrics = evaluate_model(selected_model, X, y)[0]
    st.subheader(f"Detailed evaluation: {model_options[selected_model_id]['name']}")
    metric_columns = st.columns(len(METRIC_COLUMNS))
    for column, metric_name in zip(metric_columns, METRIC_COLUMNS):
        column.metric(metric_name.upper(), f"{selected_metrics[metric_name]:.4f}")

    matrix = confusion_matrix(y, selected_predictions, labels=[0, 1])
    st.markdown("**Confusion matrix** (rows: actual, columns: predicted)")
    st.dataframe(
        pd.DataFrame(matrix, index=["Actual no", "Actual yes"], columns=["Predicted no", "Predicted yes"]),
        use_container_width=True,
    )

    report = classification_report(
        y,
        selected_predictions,
        labels=[0, 1],
        target_names=["no", "yes"],
        output_dict=True,
        zero_division=0,
    )
    st.markdown("**Classification report**")
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    with st.expander("Generated held-out test-set metrics"):
        metrics_path = ROOT / metadata.get("metrics_file", "model_metrics.csv")
        if metrics_path.is_file():
            st.dataframe(pd.read_csv(metrics_path), use_container_width=True, hide_index=True)
        else:
            st.info("Run `python train_models.py` to generate model_metrics.csv.")


if __name__ == "__main__":
    main()
