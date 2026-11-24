"""
Data Preprocessing Pipeline for Customer Churn Prediction.

This module provides reusable, leak-free data ingestion, cleaning, splitting,
and feature transformation using Scikit-learn ColumnTransformer.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define Canonical Feature Lists
NUMERICAL_FEATURES: List[str] = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_INPUT_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: str = "Churn"
TARGET_MAPPING: dict = {"No": 0, "Yes": 1}
REVERSE_TARGET_MAPPING: dict = {0: "No", 1: "Yes"}


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load raw CSV data and coerce TotalCharges to numeric,
    replacing blank spaces with NaN without pre-imputing.

    Parameters
    ----------
    filepath : str or Path
        Path to the Telco Customer Churn CSV dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with TotalCharges properly coerced to numeric.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    df = pd.read_csv(path)

    # Clean whitespace strings in TotalCharges and coerce to numeric
    # Do NOT impute here to prevent data leakage across train and test sets
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].astype(str).str.strip(),
            errors="coerce"
        )

    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    include_target: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Prepare feature matrix X and target vector y.
    Removes 'customerID' and maps binary target.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or cleaned DataFrame.
    include_target : bool, default=True
        Whether to extract and map the target column 'Churn'.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.Series]]
        X (features DataFrame) and y (target Series, if include_target=True).
    """
    df_copy = df.copy()

    # Drop customerID identifier if present
    if "customerID" in df_copy.columns:
        df_copy = df_copy.drop(columns=["customerID"])

    if include_target and TARGET_COLUMN in df_copy.columns:
        y = df_copy[TARGET_COLUMN].map(TARGET_MAPPING)
        if y.isnull().any():
            raise ValueError("Target mapping produced null values. Check target classes.")
        X = df_copy.drop(columns=[TARGET_COLUMN])
        return X, y
    else:
        # If target column exists but include_target is False, drop it
        if TARGET_COLUMN in df_copy.columns:
            df_copy = df_copy.drop(columns=[TARGET_COLUMN])
        return df_copy, None


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into stratified train and test partitions.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target vector.
    test_size : float, default=0.2
        Proportion of dataset to include in test split.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def build_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Construct a Scikit-learn ColumnTransformer for preprocessing.

    - Numerical pipeline:
        1. SimpleImputer(strategy='median')
        2. StandardScaler()
    - Categorical pipeline:
        1. SimpleImputer(strategy='most_frequent')
        2. OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    Parameters
    ----------
    numerical_features : list of str, optional
        Numerical feature names. Defaults to NUMERICAL_FEATURES.
    categorical_features : list of str, optional
        Categorical feature names. Defaults to CATEGORICAL_FEATURES.

    Returns
    -------
    ColumnTransformer
        Unfitted ColumnTransformer pipeline ready to fit on training data.
    """
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract transformed feature names from a fitted ColumnTransformer.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted ColumnTransformer instance.

    Returns
    -------
    List[str]
        List of output feature names.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual extraction
        names = list(NUMERICAL_FEATURES)
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        names.extend(list(cat_names))
        return names


if __name__ == "__main__":
    print("=" * 70)
    print("Customer Churn Preprocessing Pipeline Validation")
    print("=" * 70)

    # 1. Load data
    data_path = Path(__file__).resolve().parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    print(f"[1] Loading dataset from: {data_path}")
    raw_df = load_raw_data(data_path)
    print(f"    Raw records: {raw_df.shape[0]}, Raw columns: {raw_df.shape[1]}")
    print(f"    Missing TotalCharges values identified: {raw_df['TotalCharges'].isnull().sum()}")

    # 2. Extract features and target
    X, y = prepare_features_and_target(raw_df)
    print(f"[2] Prepared features X: {X.shape}, target y: {y.shape}")
    print(f"    Numerical features ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}")
    print(f"    Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
    print(f"    Class distribution: Class 0 (No) = {(y == 0).sum()} ({((y == 0).mean()*100):.2f}%), "
          f"Class 1 (Yes) = {(y == 1).sum()} ({((y == 1).mean()*100):.2f}%)")

    # 3. Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"[3] Train-Test Split completed:")
    print(f"    X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"    X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}")
    print(f"    y_train Churn rate: {y_train.mean():.4f}")
    print(f"    y_test  Churn rate: {y_test.mean():.4f}")
    assert np.isclose(y_train.mean(), y_test.mean(), atol=1e-3), "Stratification mismatch!"

    # Check missing values before transformation
    train_nulls = X_train["TotalCharges"].isnull().sum()
    test_nulls = X_test["TotalCharges"].isnull().sum()
    print(f"    Missing TotalCharges in X_train: {train_nulls}")
    print(f"    Missing TotalCharges in X_test:  {test_nulls}")

    # 4. Build and Fit Preprocessor strictly on training data
    preprocessor = build_preprocessor()
    print(f"[4] Fitting ColumnTransformer on X_train only (no data leakage)...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 5. Verify transformed shapes and absence of NaNs
    feature_names = get_feature_names_out(preprocessor)
    print(f"[5] Transformation verification:")
    print(f"    X_train_transformed shape: {X_train_transformed.shape}")
    print(f"    X_test_transformed shape:  {X_test_transformed.shape}")
    print(f"    Total engineered features: {len(feature_names)}")
    print(f"    NaNs in X_train_transformed: {np.isnan(X_train_transformed).sum()}")
    print(f"    NaNs in X_test_transformed:  {np.isnan(X_test_transformed).sum()}")

    # 6. Test handling of unseen categories
    print(f"[6] Testing handling of unseen categorical values...")
    unseen_sample = X_test.iloc[0:1].copy()
    unseen_sample.loc[unseen_sample.index[0], "PaymentMethod"] = "CryptoWallet"
    unseen_sample.loc[unseen_sample.index[0], "Contract"] = "Decade-to-Decade"
    transformed_unseen = preprocessor.transform(unseen_sample)
    print(f"    Successfully transformed sample with unseen categories without errors.")
    print(f"    Unseen sample output shape: {transformed_unseen.shape}")

    print("\nPre-processing pipeline verification: SUCCESS!")
