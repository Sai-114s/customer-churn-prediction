"""
Model Training Module for Customer Churn Prediction.

Trains and validates three classification models using Scikit-learn Pipelines:
1. Logistic Regression (class_weight='balanced')
2. Random Forest Classifier (class_weight='balanced')
3. XGBoost Classifier (scale_pos_weight computed from class distribution)

Evaluates cross-validation performance strictly on the training partition
and serializes fitted model pipelines into the models/ directory.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src.preprocess import (
    load_raw_data,
    prepare_features_and_target,
    split_data,
    build_preprocessor,
)

# Project paths
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Compute scale_pos_weight for XGBoost based on training class distribution.
    scale_pos_weight = count(negative class) / count(positive class)
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    if pos_count == 0:
        raise ValueError("Positive class count is zero in training data.")
    return float(neg_count / pos_count)


def build_model_pipelines(y_train: pd.Series) -> Dict[str, Pipeline]:
    """
    Construct Scikit-learn Pipelines for Logistic Regression, Random Forest,
    and XGBoost, each encapsulating the ColumnTransformer preprocessor.

    Parameters
    ----------
    y_train : pd.Series
        Training target vector used to compute class balancing weights.

    Returns
    -------
    Dict[str, Pipeline]
        Dictionary of unfitted model pipelines.
    """
    scale_pos_weight = compute_scale_pos_weight(y_train)

    pipelines = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                        max_iter=1000,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=8,
                        min_samples_split=10,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.08,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return pipelines


def evaluate_cross_validation(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
) -> pd.DataFrame:
    """
    Perform Stratified K-Fold cross-validation on training data only.

    Parameters
    ----------
    pipelines : Dict[str, Pipeline]
        Candidate model pipelines.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    cv_splits : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    pd.DataFrame
        Summary table of cross-validation metrics across all models.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = []

    for name, pipe in pipelines.items():
        print(f"--> Running {cv_splits}-fold Stratified CV for {name}...")
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        results.append(
            {
                "Model": name,
                "CV Accuracy": np.mean(scores["test_accuracy"]),
                "CV Precision": np.mean(scores["test_precision"]),
                "CV Recall": np.mean(scores["test_recall"]),
                "CV F1-Score": np.mean(scores["test_f1"]),
                "CV ROC-AUC": np.mean(scores["test_roc_auc"]),
            }
        )

    return pd.DataFrame(results).set_index("Model")


def train_and_save_models(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path = MODELS_DIR,
) -> Dict[str, Path]:
    """
    Fit each pipeline on the complete training set and save using joblib.

    Parameters
    ----------
    pipelines : Dict[str, Pipeline]
        Model pipelines.
    X_train : pd.DataFrame
        Full training features.
    y_train : pd.Series
        Full training target.
    output_dir : Path
        Directory to save models.

    Returns
    -------
    Dict[str, Path]
        Mapping from model name to saved artifact path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    name_to_filename = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
    }

    for name, pipe in pipelines.items():
        print(f"--> Fitting {name} on full training set (5,634 samples)...")
        pipe.fit(X_train, y_train)

        filename = name_to_filename[name]
        save_path = output_dir / filename
        joblib.dump(pipe, save_path)
        saved_paths[name] = save_path
        print(f"    Saved {name} to {save_path.name}")

    return saved_paths


def run_training_pipeline() -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """
    Execute full training workflow:
    1. Load and split data.
    2. Build pipelines.
    3. Run cross-validation on training data.
    4. Fit and serialize all models.
    """
    print("=" * 70)
    print("Milestone 2: Model Training Pipeline")
    print("=" * 70)

    # 1. Load and split
    raw_df = load_raw_data(DATA_PATH)
    X, y = prepare_features_and_target(raw_df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"Dataset split:")
    print(f"  Training samples: {X_train.shape[0]} | Negative: {(y_train == 0).sum()}, Positive: {(y_train == 1).sum()}")
    print(f"  Test samples:     {X_test.shape[0]} (kept strictly untouched)")
    print(f"  XGBoost scale_pos_weight: {scale_pos_weight:.4f}")

    # 2. Build pipelines
    pipelines = build_model_pipelines(y_train)

    # 3. Cross-validation on training partition
    cv_summary = evaluate_cross_validation(pipelines, X_train, y_train, cv_splits=5)
    print("\n" + "=" * 70)
    print("5-Fold Stratified Cross-Validation Results (Training Set Only)")
    print("=" * 70)
    print(cv_summary.round(4).to_string())
    print("=" * 70 + "\n")

    # 4. Fit and save
    saved_paths = train_and_save_models(pipelines, X_train, y_train)

    print("\nModel training and serialization completed successfully!")
    return cv_summary, saved_paths


if __name__ == "__main__":
    run_training_pipeline()
