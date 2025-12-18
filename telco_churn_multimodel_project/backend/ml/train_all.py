from __future__ import annotations
import argparse
from pathlib import Path
import json

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

import shap
import matplotlib.pyplot as plt
import joblib

from backend.config import MODELS_DIR, REPORTS_DIR, EDA_DIR
from backend.ml.eda import generate_eda_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True, help="Path to Telco churn CSV")
    return p.parse_args()


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop ID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    # Fix TotalCharges (blank -> NaN)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Target mapping
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def split(df: pd.DataFrame):
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    numeric_features = [c for c in numeric_features if c in X.columns]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num, numeric_features),
            ("cat", cat, categorical_features),
        ]
    )


def make_model_pipelines(preprocessor: ColumnTransformer):
    pipelines = {}

    pipelines["logistic_regression"] = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", LogisticRegression(max_iter=2000, n_jobs=None)),
    ])

    pipelines["decision_tree"] = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ])

    pipelines["random_forest"] = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])

    pipelines["xgboost"] = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", XGBClassifier(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            tree_method="hist",
        )),
    ])

    return pipelines


def param_grids():
    return {
        "logistic_regression": {
            "classifier__C": [0.5, 1.0, 2.0],
        },
        "decision_tree": {
            "classifier__max_depth": [None, 6, 12],
            "classifier__min_samples_split": [2, 5, 10],
        },
        "random_forest": {
            "classifier__n_estimators": [150, 300],
            "classifier__max_depth": [None, 12],
            "classifier__min_samples_split": [2, 5],
        },
        "xgboost": {
            "classifier__max_depth": [4, 6],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__subsample": [0.8, 0.9],
        }
    }


def score_dict(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def train_and_eval_all(X_train, y_train, X_val, y_val, X_test, y_test, pipelines):
    grids = param_grids()
    results = {}
    fitted = {}

    for key, pipe in pipelines.items():
        grid = GridSearchCV(
            pipe,
            grids.get(key, {}),
            scoring="recall",
            cv=5,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        val_pred = best.predict(X_val)
        test_pred = best.predict(X_test)

        results[key] = {
            "best_params": grid.best_params_,
            "validation": score_dict(y_val, val_pred),
            "test": score_dict(y_test, test_pred),
        }
        fitted[key] = best

    return results, fitted


def shap_summary_for_log_reg(model, background: pd.DataFrame, top_n: int = 15, max_background: int = 100):
    """Compute SHAP summary for the logistic regression pipeline on a sampled background."""
    try:
        bg = shap.utils.sample(background, max_background, random_state=42)
        preprocessor = model.named_steps.get("preprocessor")
        classifier = model.named_steps.get("classifier")
        if preprocessor is None or classifier is None:
            raise ValueError("Pipeline missing expected steps for SHAP.")

        X_enc = preprocessor.transform(bg)
        if hasattr(X_enc, "toarray"):
            X_enc = X_enc.toarray()
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.Explainer(classifier.predict_proba, X_enc, feature_names=feature_names)
        values = explainer(X_enc)
        class1 = values[:, 1]  # class=1 contributions
        mean_abs = np.abs(class1.values).mean(axis=0)
        pairs = sorted(zip(feature_names, mean_abs), key=lambda t: t[1], reverse=True)[:top_n]
        return {
            "model": "logistic_regression",
            "sample_size": bg.shape[0],
            "top_features": [{"feature": f, "mean_abs_shap": float(v)} for f, v in pairs],
        }
    except Exception as e:
        print(f"SHAP summary generation skipped: {e}")
        return None


def save_models(fitted: dict):
    MODELS_DIR.mkdir(exist_ok=True)
    for key, model in fitted.items():
        joblib.dump(model, MODELS_DIR / f"{key}.joblib")


def plot_comparison(metrics: dict):
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    model_keys = sorted(metrics.keys())
    val_acc = [metrics[k]["validation"]["accuracy"] for k in model_keys]
    val_prec = [metrics[k]["validation"]["precision"] for k in model_keys]
    val_rec = [metrics[k]["validation"]["recall"] for k in model_keys]

    x = range(len(model_keys))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar([i - width for i in x], val_acc, width=width, label="Accuracy (val)")
    plt.bar(list(x), val_prec, width=width, label="Precision (val)")
    plt.bar([i + width for i in x], val_rec, width=width, label="Recall (val)")
    plt.xticks(list(x), model_keys, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.title("Model Comparison — Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "model_comparison.png")
    plt.close()


def main():
    args = parse_args()

    df = load_and_clean(args.data_path)

    # EDA after cleaning (before encoding/scaling, but after fixing TotalCharges + mapping Churn)
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    generate_eda_report(df.copy(), EDA_DIR)

    X_train, X_val, X_test, y_train, y_val, y_test = split(df)

    preprocessor = build_preprocessor(X_train)
    pipes = make_model_pipelines(preprocessor)

    metrics, fitted = train_and_eval_all(X_train, y_train, X_val, y_val, X_test, y_test, pipes)

    shap_summary = None
    if "logistic_regression" in fitted:
        shap_summary = shap_summary_for_log_reg(fitted["logistic_regression"], X_train)

    save_models(fitted)
    plot_comparison(metrics)

    REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    out = {
        "models": metrics,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    if shap_summary:
        (REPORTS_DIR / "shap_summary.json").write_text(json.dumps(shap_summary, indent=2), encoding="utf-8")

    print("Training complete. Models saved to ./models and reports saved to ./reports")
    print("EDA report saved to ./reports/eda/eda_report.html")


if __name__ == "__main__":
    main()
