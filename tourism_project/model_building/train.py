"""
Stage 3 of the MLOps pipeline: train, evaluate, track and register the model.

Compares Logistic Regression, Random Forest and XGBoost under identical stratified
cross validation, logs every run to MLflow, tunes the decision threshold on out of
fold predictions, and uploads the winning pipeline to the Hugging Face model hub.
"""

import os
import sys
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, precision_recall_curve)
from xgboost import XGBClassifier
from huggingface_hub import HfApi, hf_hub_download, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

warnings.filterwarnings("ignore")

# --- Fill in your Hugging Face username ---------------------------------
HF_USERNAME = "AhmedBenHamouda"
# ------------------------------------------------------------------------

DATASET_REPO = f"{HF_USERNAME}/tourism-package-data"
MODEL_REPO = f"{HF_USERNAME}/tourism-package-model"     # created automatically
RANDOM_STATE = 42
N_SPLITS = 5
SCORING = "f1"                 # positive class F1, see the notebook for why
MODEL_DIR = "tourism_project/model_building"
MODEL_FILE = os.path.join(MODEL_DIR, "best_tourism_model_v1.joblib")
META_FILE = os.path.join(MODEL_DIR, "model_metadata.json")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    sys.exit("HF_TOKEN is not set.")

api = HfApi(token=HF_TOKEN)

# MLflow writes to a local file store by default and to a tracking server when
# MLFLOW_TRACKING_URI is set, which is what the GitHub Actions workflow provides.
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("tourism-wellness-package")

# ---------------------------------------------------------------- load
def load_split(filename):
    path = hf_hub_download(repo_id=DATASET_REPO, filename=filename,
                           repo_type="dataset", token=HF_TOKEN)
    return pd.read_csv(path)

Xtrain = load_split("Xtrain.csv")
Xtest = load_split("Xtest.csv")
ytrain = load_split("ytrain.csv").squeeze("columns")
ytest = load_split("ytest.csv").squeeze("columns")

print(f"Train {Xtrain.shape}, Test {Xtest.shape}")
print(f"Train positive rate {ytrain.mean() * 100:.2f}%")

categorical = Xtrain.select_dtypes(include=["object", "category"]).columns.tolist()
numeric = [c for c in Xtrain.columns if c not in categorical]
print(f"{len(categorical)} categorical, {len(numeric)} numeric features")

# Imbalance ratio used to weight the positive class in XGBoost.
scale_pos_weight = float((ytrain == 0).sum() / (ytrain == 1).sum())
print(f"scale_pos_weight = {scale_pos_weight:.3f}")


def make_preprocessor(scale_numeric=False):
    """One hot encode the categoricals. Scale the numerics only for the linear model."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if scale_numeric:
        return ColumnTransformer([
            ("cat", encoder, categorical),
            ("num", StandardScaler(), numeric),
        ])
    # Tree models need no scaling, so the numerics pass through untouched.
    return ColumnTransformer([("cat", encoder, categorical)], remainder="passthrough")


# ------------------------------------------------------- candidate models
candidates = {
    "LogisticRegression": (
        Pipeline([
            ("preprocessor", make_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced",
                                              random_state=RANDOM_STATE)),
        ]),
        {"classifier__C": [0.1, 1.0, 10.0]},
    ),
    "RandomForest": (
        Pipeline([
            ("preprocessor", make_preprocessor()),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE,
                                                  class_weight="balanced", n_jobs=-1)),
        ]),
        {"classifier__n_estimators": [300],
         "classifier__max_depth": [10, None],
         "classifier__min_samples_leaf": [1, 2]},
    ),
    "XGBoost": (
        Pipeline([
            ("preprocessor", make_preprocessor()),
            ("classifier", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss",
                                         scale_pos_weight=scale_pos_weight, n_jobs=-1)),
        ]),
        {"classifier__n_estimators": [300],
         "classifier__max_depth": [4, 6],
         "classifier__learning_rate": [0.1, 0.2],
         "classifier__subsample": [0.8],
         "classifier__colsample_bytree": [0.8],
         "classifier__min_child_weight": [1, 5],
         "classifier__reg_lambda": [1.0, 5.0]},
    ),
}

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def evaluate(estimator, X, y, threshold=0.5):
    """Return the full metric set at a given decision threshold."""
    proba = estimator.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


# ------------------------------------------------------------- training
results = {}
best_name, best_score, best_estimator = None, -1.0, None

for name, (pipeline, grid) in candidates.items():
    with mlflow.start_run(run_name=name):
        print(f"\n--- Tuning {name} ---")
        search = GridSearchCV(pipeline, grid, scoring=SCORING, cv=cv, n_jobs=-1, verbose=0)
        search.fit(Xtrain, ytrain)

        estimator = search.best_estimator_
        train_metrics = evaluate(estimator, Xtrain, ytrain)
        test_metrics = evaluate(estimator, Xtest, ytest)

        mlflow.log_param("model_family", name)
        mlflow.log_params({k.replace("classifier__", ""): v
                           for k, v in search.best_params_.items()})
        mlflow.log_metric("cv_f1", search.best_score_)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        results[name] = {"cv_f1": search.best_score_,
                         "best_params": search.best_params_,
                         "train": train_metrics, "test": test_metrics}

        print(f"  best params : {search.best_params_}")
        print(f"  cv f1       : {search.best_score_:.4f}")
        print(f"  train f1    : {train_metrics['f1']:.4f}")
        print(f"  test  f1    : {test_metrics['f1']:.4f}   auc {test_metrics['roc_auc']:.4f}")

        if search.best_score_ > best_score:
            best_score, best_name, best_estimator = search.best_score_, name, estimator

print("\n" + "=" * 62)
print("MODEL COMPARISON (selection metric: cross validated F1)")
print("=" * 62)
comparison = pd.DataFrame({
    name: {"cv_f1": r["cv_f1"], "train_f1": r["train"]["f1"],
           "test_f1": r["test"]["f1"], "test_recall": r["test"]["recall"],
           "test_precision": r["test"]["precision"], "test_roc_auc": r["test"]["roc_auc"]}
    for name, r in results.items()
}).T.round(4)
print(comparison.to_string())
print(f"\nSelected model: {best_name}")

# --------------------------------------------------- threshold tuning
# Tuned on out of fold predictions. Using training probabilities would be
# worthless because a boosted tree separates its own training data perfectly.
oof_proba = cross_val_predict(best_estimator, Xtrain, ytrain, cv=cv,
                              method="predict_proba", n_jobs=-1)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(ytrain, oof_proba)
f1_scores = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
best_threshold = float(thresholds[int(np.nanargmax(f1_scores[:-1]))])
print(f"\nTuned decision threshold (out of fold): {best_threshold:.3f}")

# ------------------------------------------------------ final evaluation
final_metrics = evaluate(best_estimator, Xtest, ytest, threshold=best_threshold)
default_metrics = evaluate(best_estimator, Xtest, ytest, threshold=0.5)

print("\nTest performance at threshold 0.500:")
for k, v in default_metrics.items():
    print(f"  {k:10s} {v:.4f}")
print(f"\nTest performance at tuned threshold {best_threshold:.3f}:")
for k, v in final_metrics.items():
    print(f"  {k:10s} {v:.4f}")

test_pred = (best_estimator.predict_proba(Xtest)[:, 1] >= best_threshold).astype(int)
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(ytest, test_pred))
print("\nClassification report:")
print(classification_report(ytest, test_pred, digits=4,
                            target_names=["Not purchased", "Purchased"]))

# ------------------------------------------------ feature importance
try:
    feature_names = best_estimator.named_steps["preprocessor"].get_feature_names_out()
    classifier = best_estimator.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importance = (pd.Series(classifier.feature_importances_, index=feature_names)
                      .sort_values(ascending=False))
        print("\nTop 15 features by importance:")
        print(importance.head(15).round(4).to_string())
except Exception as exc:
    print(f"Feature importance unavailable: {exc}")

# ------------------------------------------------------------- persist
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best_estimator, MODEL_FILE)

metadata = {
    "model_family": best_name,
    "decision_threshold": best_threshold,
    "selection_metric": SCORING,
    "cv_f1": float(best_score),
    "test_metrics_tuned": {k: float(v) for k, v in final_metrics.items()},
    "test_metrics_default": {k: float(v) for k, v in default_metrics.items()},
    "feature_order": list(Xtrain.columns),
    "categorical_features": categorical,
    "numeric_features": numeric,
    "category_levels": {c: sorted(Xtrain[c].dropna().unique().tolist()) for c in categorical},
}
with open(META_FILE, "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)

with mlflow.start_run(run_name=f"best-{best_name}"):
    mlflow.log_param("model_family", best_name)
    mlflow.log_param("decision_threshold", best_threshold)
    for k, v in final_metrics.items():
        mlflow.log_metric(f"final_{k}", v)
    # Logging a signature keeps MLflow from warning and records the input schema.
    signature = infer_signature(Xtrain, best_estimator.predict(Xtrain))
    mlflow.sklearn.log_model(best_estimator, "model", signature=signature)
    mlflow.log_artifact(META_FILE)

# ------------------------------------------------------------ register
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"\nModel repository already exists: {MODEL_REPO}")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, token=HF_TOKEN)
    print(f"\nCreated model repository: {MODEL_REPO}")

for path, name_in_repo in [(MODEL_FILE, "best_tourism_model_v1.joblib"),
                           (META_FILE, "model_metadata.json")]:
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=name_in_repo,
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message=f"Register {best_name} model",
    )
    print(f"Uploaded {name_in_repo} to {MODEL_REPO}")

print("\nTraining stage complete.")
