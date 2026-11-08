# Bank Marketing Classification — BITS AIML Assignment 2

## Problem statement

Predict whether a Portuguese bank customer will subscribe to a term deposit
(`yes`/`no`) from customer information and prior marketing campaign details.
Five classification algorithms are compared using Accuracy, AUC, Precision,
Recall, F1, and Matthews Correlation Coefficient (MCC).

## Dataset

- **Dataset:** Bank Marketing, UCI Machine Learning Repository, dataset ID 222
- **Official source:** <https://archive.ics.uci.edu/dataset/222/bank+marketing>
- **Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*
  [Dataset]. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C5K306>
- **Original data:** 45,211 instances, 16 input features, and target `y`
- **Target:** Renamed to `subscribed`; `no` maps to 0 and `yes` maps to 1
- **Leakage prevention:** `duration` is dropped because it is known only after
  the call and can reveal the outcome. The remaining 15 features are used.
- **Split:** 80% training / 20% test, stratified, `random_state=42`
- **Preprocessing:** Numeric missing values are median-imputed and standardized;
  categorical missing values are mode-imputed and one-hot encoded. Preprocessing
  is fitted inside each saved model pipeline to avoid test-set leakage.

The script downloads the official dataset at run time. It does not use a
fabricated or illustrative metric table.

## Models

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (5 neighbors)
4. Gaussian Naive Bayes
5. Linear Support Vector Machine

## Run locally

Python 3.10 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

`train_models.py` downloads UCI dataset 222, performs the stratified split,
trains all five models, evaluates them on the held-out test set, and creates:

- `test_data.csv` — labelled test rows for app evaluation
- `model_metrics.csv` — measured held-out metrics for all five models
- `model_metadata.json` — feature, split, and saved-model metadata
- `model/` — five serialized preprocessing-and-classifier pipelines
- `data/bank_marketing_processed.csv` — cleaned predictors and encoded target

The app loads these saved pipelines; it does not retrain on an uploaded file.
Upload a labelled CSV containing the 15 predictor columns and a `subscribed`
column (`yes`/`no` or `1`/`0`). The app reports all six metrics, a confusion
matrix, a classification report, and a comparison of all five models on the
uploaded rows.

## Results and observations

Use the measured values in `model_metrics.csv` to complete the final report.
Consider AUC, F1, and MCC alongside accuracy and positive-class (`yes`) recall;
the target classes are imbalanced, so accuracy alone is not sufficient.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.893730 | 0.751229 | 0.671378 | 0.179584 | 0.283371 | 0.310015 |
| Decision Tree | 0.830587 | 0.610120 | 0.294983 | 0.322306 | 0.308040 | 0.212003 |
| KNN | 0.887648 | 0.691511 | 0.556757 | 0.194707 | 0.288515 | 0.282595 |
| Naive Bayes | 0.846622 | 0.733961 | 0.368924 | 0.437618 | 0.400346 | 0.314642 |
| Support Vector Machine | 0.893288 | 0.749624 | 0.654485 | 0.186200 | 0.289919 | 0.310298 |

### Observations

- **Logistic Regression:** Accuracy is 0.893730 and AUC is 0.751229, the
  highest AUC. Its 0.179584 recall and 0.283371 F1 show that its high accuracy
  does not translate into strong detection of the minority `yes` class.
- **Decision Tree:** It reaches 0.322306 recall, but has the lowest AUC
  (0.610120), accuracy (0.830587), and MCC (0.212003). Its transparent
  decision rules are useful to interpret, but this split does not show strong
  generalization.
- **KNN:** After scaling, KNN attains 0.887648 accuracy and 0.691511 AUC.
  Recall is 0.194707 and F1 is 0.288515, so the distance-based model misses many
  actual subscribers.
- **Naive Bayes:** It has the strongest `yes` recall (0.437618), F1 (0.400346),
  and MCC (0.314642), despite lower accuracy (0.846622). Its conditional
  independence assumption is simplifying, but it yields the best balance for
  positive-class detection in this run.
- **Support Vector Machine:** The linear margin-based classifier achieves
  0.893288 accuracy and 0.749624 AUC, but recall (0.186200) and F1 (0.289919)
  are comparatively low.

**Overall winner for balanced positive-class performance: Naive Bayes.** It
leads in recall, F1, and MCC; Logistic Regression has the highest AUC and
accuracy. This reflects the trade-off on an imbalanced target and is why
accuracy alone is insufficient. These values are the measured held-out results
from the generated `model_metrics.csv`; rerunning with a different library
version may produce slightly different values.

## Repository and deployment

- GitHub repository: <https://github.com/harialbie/PythonALMI>
- Streamlit Community Cloud URL: pending deployment.
- BITS Virtual Lab screenshot: pending. Capture the successful
  `python train_models.py` run inside the BITS Virtual Lab; no screenshot from
  another environment is used or fabricated.

Before pushing or deploying, run the training script and include the generated
model artifacts so the app can load them. Deploy `app.py` to Streamlit Community
Cloud from the repository root.

## Project layout

```text
app.py
train_models.py
requirements.txt
README.md
test_data.csv                         # generated
model_metrics.csv                     # generated
model_metadata.json                   # generated
model/                                # generated saved pipelines
data/bank_marketing_processed.csv     # generated
```

This educational exercise is not financial advice.
