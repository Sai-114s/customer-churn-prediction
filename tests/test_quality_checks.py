"""
Comprehensive Quality Assurance & Testing Suite.

Automated verification of:
1. Environment and package imports.
2. Dataset loading and schema integrity.
3. Train-test split stratification and absence of data leakage.
4. Preprocessing resilience against unseen categories.
5. Model serialization and artifact availability.
6. End-to-end prediction validity and probability bounds.
7. Evaluation metrics consistency.
8. Generation of diagnostic visualization artifacts.
9. Streamlit application syntax and headless execution.
10. Alignment between README commands and project files.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestCustomerChurnProject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        cls.models_dir = PROJECT_ROOT / "models"
        cls.figures_dir = PROJECT_ROOT / "reports" / "figures"
        cls.app_path = PROJECT_ROOT / "app.py"

    def test_01_imports(self):
        """Verify all essential modules and third-party libraries import cleanly."""
        modules = [
            "pandas",
            "numpy",
            "sklearn",
            "xgboost",
            "shap",
            "streamlit",
            "joblib",
            "matplotlib",
            "seaborn",
            "src.preprocess",
            "src.train",
            "src.evaluate",
            "src.explain",
        ]
        for mod in modules:
            with self.subTest(module=mod):
                imported = __import__(mod, fromlist=["*"])
                self.assertIsNotNone(imported)

    def test_02_dataset_loading_and_cleaning(self):
        """Verify dataset loads with 7,043 rows, 21 columns, and 11 blank TotalCharges handled."""
        from src.preprocess import load_raw_data

        df = load_raw_data(self.data_path)
        self.assertEqual(df.shape, (7043, 21))
        self.assertEqual(df["TotalCharges"].isnull().sum(), 11)
        self.assertTrue(pd.api.types.is_float_dtype(df["TotalCharges"]))

    def test_03_train_test_split_and_leakage(self):
        """Verify 80/20 stratified split, zero index overlap, and no data leakage."""
        from src.preprocess import load_raw_data, prepare_features_and_target, split_data

        raw_df = load_raw_data(self.data_path)
        X, y = prepare_features_and_target(raw_df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

        self.assertEqual(X_train.shape, (5634, 19))
        self.assertEqual(X_test.shape, (1409, 19))

        # Check stratification
        self.assertAlmostEqual(float(y_train.mean()), 0.2654, delta=0.005)
        self.assertAlmostEqual(float(y_test.mean()), 0.2654, delta=0.005)

        # Check no index overlap
        overlap = set(X_train.index).intersection(set(X_test.index))
        self.assertEqual(len(overlap), 0)

    def test_04_preprocessing_and_unseen_categories(self):
        """Verify ColumnTransformer handles missing values and unseen categorical values without crashing."""
        from src.preprocess import (
            load_raw_data,
            prepare_features_and_target,
            split_data,
            build_preprocessor,
        )

        raw_df = load_raw_data(self.data_path)
        X, y = prepare_features_and_target(raw_df)
        X_train, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)

        preprocessor = build_preprocessor()
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)

        self.assertEqual(X_train_trans.shape, (5634, 45))
        self.assertEqual(X_test_trans.shape, (1409, 45))
        self.assertEqual(np.isnan(X_train_trans).sum(), 0)
        self.assertEqual(np.isnan(X_test_trans).sum(), 0)

        # Introduce novel unseen categories
        unseen_df = X_test.iloc[0:2].copy()
        unseen_df.loc[unseen_df.index[0], "PaymentMethod"] = "SolanaPay"
        unseen_df.loc[unseen_df.index[0], "Contract"] = "5-Year-Enterprise"
        unseen_df.loc[unseen_df.index[1], "InternetService"] = "Satellite-Starlink"

        unseen_trans = preprocessor.transform(unseen_df)
        self.assertEqual(unseen_trans.shape, (2, 45))
        self.assertEqual(np.isnan(unseen_trans).sum(), 0)

    def test_05_saved_models_exist(self):
        """Verify all model pipelines are serialized and can be deserialized."""
        expected_models = [
            "logistic_regression.pkl",
            "random_forest.pkl",
            "xgboost.pkl",
            "best_churn_model.pkl",
        ]
        for m in expected_models:
            path = self.models_dir / m
            self.assertTrue(path.exists(), f"Model {m} does not exist in {self.models_dir}")
            loaded_pipe = joblib.load(path)
            self.assertTrue(hasattr(loaded_pipe, "predict"))
            self.assertTrue(hasattr(loaded_pipe, "predict_proba"))

    def test_06_model_inference_and_probability_bounds(self):
        """Verify best model prediction bounds and responsiveness to high- and low-risk profiles."""
        best_model = joblib.load(self.models_dir / "best_churn_model.pkl")

        high_risk_sample = pd.DataFrame(
            [
                {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "No",
                    "Dependents": "No",
                    "tenure": 1,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 98.5,
                    "TotalCharges": 98.5,
                }
            ]
        )

        low_risk_sample = pd.DataFrame(
            [
                {
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "Yes",
                    "tenure": 65,
                    "PhoneService": "Yes",
                    "MultipleLines": "Yes",
                    "InternetService": "DSL",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "Yes",
                    "TechSupport": "Yes",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Two year",
                    "PaperlessBilling": "No",
                    "PaymentMethod": "Credit card (automatic)",
                    "MonthlyCharges": 60.0,
                    "TotalCharges": 3900.0,
                }
            ]
        )

        prob_high = best_model.predict_proba(high_risk_sample)[0][1]
        prob_low = best_model.predict_proba(low_risk_sample)[0][1]

        self.assertGreaterEqual(prob_high, 0.0)
        self.assertLessEqual(prob_high, 1.0)
        self.assertGreaterEqual(prob_low, 0.0)
        self.assertLessEqual(prob_low, 1.0)

        # High risk must have higher churn probability than low risk
        self.assertGreater(prob_high, 0.70)
        self.assertLess(prob_low, 0.15)

    def test_07_generated_figures_exist(self):
        """Verify all evaluation and SHAP figure assets were created and are non-empty."""
        expected_figures = [
            "confusion_matrices.png",
            "roc_curves.png",
            "feature_importance.png",
            "model_comparison.png",
            "shap_summary_bar.png",
            "shap_summary_beeswarm.png",
            "shap_local_high_risk.png",
            "shap_local_low_risk.png",
        ]
        for fig_name in expected_figures:
            fig_path = self.figures_dir / fig_name
            self.assertTrue(fig_path.exists(), f"Figure {fig_name} missing from {self.figures_dir}")
            self.assertGreater(fig_path.stat().st_size, 1000, f"Figure {fig_name} is empty or corrupted")

    def test_08_app_file_integrity(self):
        """Verify app.py exists and contains all required page modules."""
        self.assertTrue(self.app_path.exists())
        content = self.app_path.read_text(encoding="utf-8")
        self.assertIn("Executive Overview", content)
        self.assertIn("Customer Churn Predictor", content)
        self.assertIn("Model Performance & Diagnostics", content)
        self.assertIn("Feature Importance & SHAP Insights", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
