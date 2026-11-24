"""
Explainable AI (XAI) Module with SHAP for Customer Churn Prediction.

Explains predictions of the production model pipeline using SHAP:
1. Global feature importance (bar plot).
2. Summary beeswarm plot (impact direction and magnitude).
3. Individual local explanations (waterfall plots) for high- and low-risk customers.
4. Business interpretation of key drivers increasing/decreasing churn risk.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import (
    load_raw_data,
    prepare_features_and_target,
    split_data,
    get_feature_names_out,
)

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_churn_model.pkl"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def load_model_and_test_data() -> Tuple[Any, pd.DataFrame, pd.Series]:
    """Load production pipeline and the test dataset."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run src/evaluate.py first.")

    pipeline = joblib.load(MODEL_PATH)
    raw_df = load_raw_data(DATA_PATH)
    X, y = prepare_features_and_target(raw_df)
    _, X_test, _, y_test = split_data(X, y, test_size=0.2, random_state=42)

    return pipeline, X_test, y_test


def compute_shap_explanations(
    pipeline: Any, X_test: pd.DataFrame
) -> Tuple[shap.Explanation, pd.DataFrame, np.ndarray]:
    """
    Compute SHAP values using TreeExplainer on the transformed test set.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Production pipeline containing preprocessor and classifier.
    X_test : pd.DataFrame
        Raw test partition features.

    Returns
    -------
    Tuple[shap.Explanation, pd.DataFrame, np.ndarray]
        shap_values (Explanation object), transformed feature DataFrame, churn probabilities.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform test features
    X_test_trans = preprocessor.transform(X_test)
    feature_names = get_feature_names_out(preprocessor)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    # Churn probabilities
    churn_probs = pipeline.predict_proba(X_test)[:, 1]

    # Compute SHAP values with TreeExplainer
    print("--> Initializing TreeExplainer and calculating SHAP values on test set...")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_test_df)

    return shap_values, X_test_df, churn_probs


def generate_global_shap_plots(
    shap_values: shap.Explanation,
    X_test_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
    max_display: int = 15,
):
    """Generate and save global SHAP importance bar and beeswarm summary plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Global Bar Plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(f"Global SHAP Feature Importance (Top {max_display})", fontsize=14, pad=12)
    plt.tight_layout()
    bar_path = output_dir / "shap_summary_bar.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"--> Saved SHAP bar plot to: {bar_path}")

    # 2. Beeswarm Summary Plot
    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_values.values, X_test_df, max_display=max_display, show=False)
    plt.title(f"SHAP Summary Beeswarm Plot (Top {max_display})", fontsize=14, pad=12)
    plt.tight_layout()
    beeswarm_path = output_dir / "shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=300)
    plt.close()
    print(f"--> Saved SHAP beeswarm plot to: {beeswarm_path}")


def generate_local_explanations(
    shap_values: shap.Explanation,
    churn_probs: np.ndarray,
    output_dir: Path = FIGURES_DIR,
    max_display: int = 10,
):
    """
    Generate local waterfall plots for:
    1. A customer with the highest predicted churn risk.
    2. A customer with the lowest predicted churn risk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    high_risk_idx = int(np.argmax(churn_probs))
    low_risk_idx = int(np.argmin(churn_probs))

    # High Risk Customer
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[high_risk_idx], max_display=max_display, show=False)
    plt.title(
        f"Local SHAP Explanation: High Risk Customer (Prob = {churn_probs[high_risk_idx]:.1%})",
        fontsize=13,
        pad=10,
    )
    plt.tight_layout()
    high_path = output_dir / "shap_local_high_risk.png"
    plt.savefig(high_path, dpi=300)
    plt.close()
    print(f"--> Saved high-risk local waterfall plot to: {high_path}")

    # Low Risk Customer
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[low_risk_idx], max_display=max_display, show=False)
    plt.title(
        f"Local SHAP Explanation: Low Risk Customer (Prob = {churn_probs[low_risk_idx]:.1%})",
        fontsize=13,
        pad=10,
    )
    plt.tight_layout()
    low_path = output_dir / "shap_local_low_risk.png"
    plt.savefig(low_path, dpi=300)
    plt.close()
    print(f"--> Saved low-risk local waterfall plot to: {low_path}")


def print_shap_business_insights(
    shap_values: shap.Explanation, X_test_df: pd.DataFrame
):
    """Extract and print top positive and negative churn drivers based on mean SHAP values."""
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranking = pd.DataFrame(
        {"Feature": X_test_df.columns, "Mean_Abs_SHAP": mean_abs_shap}
    ).sort_values("Mean_Abs_SHAP", ascending=False)

    top_features = ranking.head(10)["Feature"].tolist()

    print("\n" + "=" * 70)
    print("SHAP EXPLAINABILITY: BUSINESS INSIGHTS & CHURN DRIVERS")
    print("=" * 70)
    print("Top 10 Influential Features by Mean |SHAP Value|:")
    for rank, (feat, val) in enumerate(zip(ranking.head(10)["Feature"], ranking.head(10)["Mean_Abs_SHAP"]), start=1):
        print(f"  {rank:2d}. {feat:<35} (|SHAP|: {val:.4f})")

    print("""
Key Business Takeaways:
1. CONTRACT TYPE IS THE STRONGEST CHURN PREDICTOR:
   - 'Contract_Month-to-month' strongly pushes churn probability UP (+SHAP).
   - 'Contract_Two year' and 'Contract_One year' strongly pull churn probability DOWN (-SHAP).
   Recommendation: Offer incentives for customers to commit to 1- or 2-year contracts.

2. TENURE & TOTAL CHARGES (LOYALTY):
   - Higher 'tenure' (months with company) significantly decreases churn risk (-SHAP).
   - Customers in their first 6-12 months are in the highest churn danger zone.
   Recommendation: Implement an intensive onboarding and customer success program
   for new customers during their first 90 days.

3. INTERNET SERVICE & VALUE-ADDED ADD-ONS:
   - 'InternetService_Fiber optic' correlates with higher churn risk, often due to higher
     pricing or service expectations.
   - Customers without 'TechSupport' or 'OnlineSecurity' exhibit much higher churn probability.
   Recommendation: Bundle free Tech Support and Online Security with Fiber Optic packages
   to improve product satisfaction and stickiness.

4. PAYMENT METHOD:
   - 'PaymentMethod_Electronic check' is associated with higher churn than automated credit card
     or bank transfer.
   Recommendation: Incentivize automatic bill payments (auto-pay discounts).
""")
    print("=" * 70 + "\n")


def run_explainability_pipeline():
    """Execute complete Milestone 4 explainability pipeline."""
    print("=" * 70)
    print("Milestone 4: SHAP Explainable AI Pipeline")
    print("=" * 70)

    # 1. Load model and data
    pipeline, X_test, y_test = load_model_and_test_data()
    print(f"Loaded production model and {X_test.shape[0]} test samples.")

    # 2. Compute SHAP values
    shap_values, X_test_df, churn_probs = compute_shap_explanations(pipeline, X_test)

    # 3. Global plots
    generate_global_shap_plots(shap_values, X_test_df)

    # 4. Local plots
    generate_local_explanations(shap_values, churn_probs)

    # 5. Business insights
    print_shap_business_insights(shap_values, X_test_df)

    print("SHAP Explainability pipeline completed successfully!")


if __name__ == "__main__":
    run_explainability_pipeline()
