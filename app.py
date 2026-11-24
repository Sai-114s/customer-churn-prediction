"""
Customer Churn Prediction Dashboard.

Interactive business analytics and real-time prediction dashboard built with Streamlit.
Features:
1. Executive Dashboard: Key telecom metrics, churn distribution, and baseline KPIs.
2. Real-Time Churn Predictor: Interactive inputs for all 19 customer features, probability gauge, risk tiers, and retention recommendations.
3. Model Diagnostics: Comprehensive comparison of Logistic Regression, Random Forest, and XGBoost with ROC curves and confusion matrices.
4. Explainable AI (SHAP): Global feature importances, beeswarm plots, and customer-level waterfall explanations.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_churn_model.pkl"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Page configuration
st.set_page_config(
    page_title="Telco Churn Intelligence Hub",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern enterprise look
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #DC2626;
        font-weight: 700;
    }
    .risk-med {
        color: #D97706;
        font-weight: 700;
    }
    .risk-low {
        color: #16A34A;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_production_model():
    """Load the trained production pipeline from models/best_churn_model.pkl."""
    if not MODEL_PATH.exists():
        st.error(f"Production model not found at {MODEL_PATH}. Please run src/train.py and src/evaluate.py first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    """Load raw dataset for dashboard aggregations."""
    df = pd.read_csv(DATA_PATH)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    return df


# Load assets
model = load_production_model()
raw_data = load_dataset()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/clouds/200/telecommunication.png", width=110)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "📊 Executive Overview",
        "🔮 Customer Churn Predictor",
        "📈 Model Performance & Diagnostics",
        "💡 Feature Importance & SHAP Insights",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Production Model:**  
    `XGBoost Classifier`  
    **ROC-AUC:** `0.8440` | **Recall:** `79.68%`  
    **Imbalance Ratio:** `2.77:1`
    """
)
st.sidebar.caption("© IBM Telco Customer Churn System")


# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if page == "📊 Executive Overview":
    st.markdown('<div class="main-header">Telecom Churn Executive Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Strategic portfolio overview, customer retention health, and predictive model benchmarks.</div>',
        unsafe_allow_html=True,
    )

    total_customers = len(raw_data)
    churn_count = (raw_data["Churn"] == "Yes").sum()
    retained_count = (raw_data["Churn"] == "No").sum()
    churn_rate = (churn_count / total_customers) * 100
    retention_rate = 100 - churn_rate

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Customers", value=f"{total_customers:,}")
    with col2:
        st.metric(label="Churn Rate", value=f"{churn_rate:.2f}%", delta=f"-{churn_rate:.1f}% Risk", delta_color="inverse")
    with col3:
        st.metric(label="Retention Rate", value=f"{retention_rate:.2f}%", delta=f"{retention_rate:.1f}% Loyal")
    with col4:
        st.metric(label="Model ROC-AUC", value="0.8440", delta="XGBoost Champion")

    st.markdown("---")

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.subheader("Customer Churn Breakdown")
        churn_dist = raw_data["Churn"].value_counts().rename(index={"No": "Retained", "Yes": "Churned"})
        st.bar_chart(churn_dist, color="#2563EB")

        st.markdown(
            """
            > **Executive Summary:**  
            > Out of **7,043** total customers, **1,869 (26.54%)** have churned, representing significant annual recurring revenue (ARR) loss. Retaining customers through early, high-precision intervention costs **5x to 7x less** than acquiring replacements.
            """
        )

    with col_right:
        st.subheader("Contract Type vs Churn Propensity")
        contract_churn = pd.crosstab(raw_data["Contract"], raw_data["Churn"], normalize="index") * 100
        st.dataframe(
            contract_churn.round(2).rename(columns={"No": "Retained (%)", "Yes": "Churned (%)"}),
            width=800,
        )

        st.subheader("Tenure Distribution by Churn Status")
        tenure_summary = raw_data.groupby("Churn")["tenure"].describe()[["mean", "50%", "min", "max"]]
        tenure_summary.index = ["Retained Customers", "Churned Customers"]
        st.table(tenure_summary.round(1))


# ==============================================================================
# PAGE 2: REAL-TIME CUSTOMER CHURN PREDICTOR
# ==============================================================================
elif page == "🔮 Customer Churn Predictor":
    st.markdown('<div class="main-header">Real-Time Customer Churn Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Enter customer profile details across all 19 features to forecast churn probability and risk tier.</div>',
        unsafe_allow_html=True,
    )

    # Preset profiles
    st.markdown("##### ⚡ Quick-Load Benchmark Profiles")
    preset_cols = st.columns(3)
    load_high = preset_cols[0].button("⚠️ Load High-Risk Profile (Fiber, M2M, New)")
    load_low = preset_cols[1].button("🛡️ Load Low-Risk Profile (2-Year, Loyal, Add-ons)")
    reset_btn = preset_cols[2].button("🔄 Reset to Default Profile")

    # Set default values based on button clicks
    if load_high:
        defaults = {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 95.50, "TotalCharges": 191.00
        }
    elif load_low:
        defaults = {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
            "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
            "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
            "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.00, "TotalCharges": 3900.00
        }
    else:
        defaults = {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 50.00, "TotalCharges": 600.00
        }

    with st.form("churn_prediction_form"):
        col1, col2, col3 = st.columns(3)

        # Column 1: Demographics
        with col1:
            st.markdown("### 👤 Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(defaults["gender"]))
            senior = st.selectbox("Senior Citizen", ["No", "Yes"], index=defaults["SeniorCitizen"])
            partner = st.selectbox("Partner", ["No", "Yes"], index=["No", "Yes"].index(defaults["Partner"]))
            dependents = st.selectbox("Dependents", ["No", "Yes"], index=["No", "Yes"].index(defaults["Dependents"]))
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(defaults["PhoneService"]))
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(defaults["MultipleLines"]))

        # Column 2: Connectivity & Services
        with col2:
            st.markdown("### 🌐 Internet & Add-ons")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(defaults["InternetService"]))
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["OnlineSecurity"]))
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["OnlineBackup"]))
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["DeviceProtection"]))
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["TechSupport"]))
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["StreamingTV"]))
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(defaults["StreamingMovies"]))

        # Column 3: Account & Billing
        with col3:
            st.markdown("### 💳 Account & Billing")
            tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=int(defaults["tenure"]), step=1)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(defaults["Contract"]))
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=["Yes", "No"].index(defaults["PaperlessBilling"]))
            payment = st.selectbox("Payment Method", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ], index=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ].index(defaults["PaymentMethod"]))

            monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=float(defaults["MonthlyCharges"]), step=0.5)
            suggested_total = round(tenure * monthly_charges, 2)
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=float(defaults["TotalCharges"]),
                step=10.0,
                help=f"Suggested estimate based on tenure * monthly charges: ~${suggested_total:,.2f}",
            )

        submit_btn = st.form_submit_button("⚡ Predict Churn Risk", type="primary", use_container_width=True)

    if submit_btn:
        try:
            # Construct feature dictionary strictly matching training schema
            input_dict = {
                "gender": [gender],
                "SeniorCitizen": [1 if senior == "Yes" else 0],
                "Partner": [partner],
                "Dependents": [dependents],
                "tenure": [tenure],
                "PhoneService": [phone_service],
                "MultipleLines": [multiple_lines],
                "InternetService": [internet_service],
                "OnlineSecurity": [online_security],
                "OnlineBackup": [online_backup],
                "DeviceProtection": [device_protection],
                "TechSupport": [tech_support],
                "StreamingTV": [streaming_tv],
                "StreamingMovies": [streaming_movies],
                "Contract": [contract],
                "PaperlessBilling": [paperless],
                "PaymentMethod": [payment],
                "MonthlyCharges": [monthly_charges],
                "TotalCharges": [total_charges],
            }

            input_df = pd.DataFrame(input_dict)

            # Predict probability and class using production pipeline
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            # Determine Risk Tier
            if probability >= 0.60:
                risk_tier = "HIGH RISK"
                risk_color = "#DC2626"
                risk_icon = "🚨"
                status_msg = "Critical churn hazard. Immediate retention intervention advised."
            elif probability >= 0.30:
                risk_tier = "MEDIUM RISK"
                risk_color = "#D97706"
                risk_icon = "⚠️"
                status_msg = "Moderate churn vulnerability. Proactive engagement recommended."
            else:
                risk_tier = "LOW RISK"
                risk_color = "#16A34A"
                risk_icon = "🛡️"
                status_msg = "Customer displays high loyalty metrics. Maintain standard relationship."

            st.markdown("---")
            st.markdown("### Prediction Results")

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(
                    label="Churn Prediction",
                    value="YES (Will Churn)" if prediction == 1 else "NO (Retained)",
                    delta="Churn Expected" if prediction == 1 else "Loyal Customer",
                    delta_color="inverse" if prediction == 1 else "normal",
                )
            with res_col2:
                st.metric(label="Churn Probability", value=f"{probability:.1%}")
            with res_col3:
                st.markdown(
                    f"""
                    <div style="background-color: {risk_color}15; border-left: 5px solid {risk_color}; padding: 10px 15px; border-radius: 6px;">
                        <span style="font-size: 1.2rem; font-weight: bold; color: {risk_color};">{risk_icon} {risk_tier}</span><br>
                        <span style="font-size: 0.9rem; color: #374151;">{status_msg}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.progress(float(probability))

            # Automated Prescription Recommendations
            st.markdown("#### 📋 Retention Action Plan")
            rec_items = []
            if contract == "Month-to-month":
                rec_items.append("Offer a 15% discount incentive to upgrade from **Month-to-month** to a **1-Year or 2-Year Contract**.")
            if internet_service == "Fiber optic" and tech_support == "No":
                rec_items.append("Bundle complimentary **Tech Support** and **Online Security** to improve fiber optic customer experience.")
            if payment == "Electronic check":
                rec_items.append("Incentivize enrollment in **Automatic Bank Transfer / Credit Card Autopay** with a $5 bill credit.")
            if tenure <= 6:
                rec_items.append("Enroll customer in the **90-Day VIP Onboarding Program** with milestone loyalty check-ins.")
            if not rec_items:
                rec_items.append("Customer exhibits solid retention indicators. Continue periodic satisfaction surveys.")

            for r in rec_items:
                st.markdown(f"- {r}")

        except Exception as e:
            st.error(f"Prediction failed with error: {str(e)}")


# ==============================================================================
# PAGE 3: MODEL PERFORMANCE & DIAGNOSTICS
# ==============================================================================
elif page == "📈 Model Performance & Diagnostics":
    st.markdown('<div class="main-header">Model Performance & Diagnostic Suite</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Objective verification on the untouched 20% test split (1,409 samples). No data leakage.</div>',
        unsafe_allow_html=True,
    )

    # Benchmark Table
    st.subheader("1. Test Set Performance Comparison")
    perf_data = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost (Champion)"],
        "Accuracy": ["73.81%", "75.51%", "75.59%"],
        "Precision": ["50.43%", "52.55%", "52.65%"],
        "Recall": ["78.34%", "79.95%", "79.68%"],
        "F1-Score": ["61.36%", "63.41%", "63.40%"],
        "ROC-AUC": ["0.8415", "0.8434", "0.8440"],
    }
    st.table(pd.DataFrame(perf_data).set_index("Model"))

    st.markdown(
        """
        > **ML Engineering Decision Rationale:**  
        > **XGBoost** was selected as the champion production model because it achieved the highest **ROC-AUC (0.8440)** and a formidable **79.68% Recall**, capturing nearly 8 out of every 10 churning customers while providing flexible probability calibration across risk thresholds.
        """
    )

    st.markdown("---")
    st.subheader("2. Diagnostic Visualizations")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrices", "ROC Curves", "Metric Comparison Chart"])

    with tab1:
        if (FIGURES_DIR / "confusion_matrices.png").exists():
            st.image(str(FIGURES_DIR / "confusion_matrices.png"), caption="Confusion Matrices across all three models on 1,409 test samples.")
        else:
            st.info("Confusion matrix image not found. Run `src/evaluate.py` to generate it.")

    with tab2:
        if (FIGURES_DIR / "roc_curves.png").exists():
            st.image(str(FIGURES_DIR / "roc_curves.png"), caption="Combined ROC Curves demonstrating superior discrimination vs random guess.")
        else:
            st.info("ROC curves image not found. Run `src/evaluate.py` to generate it.")

    with tab3:
        if (FIGURES_DIR / "model_comparison.png").exists():
            st.image(str(FIGURES_DIR / "model_comparison.png"), caption="Grouped bar chart comparing key evaluation metrics.")
        else:
            st.info("Model comparison chart not found. Run `src/evaluate.py` to generate it.")


# ==============================================================================
# PAGE 4: FEATURE IMPORTANCE & SHAP INSIGHTS
# ==============================================================================
elif page == "💡 Feature Importance & SHAP Insights":
    st.markdown('<div class="main-header">Explainable AI (SHAP) Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Uncovering the global and individual drivers of churn using Shapley Additive Explanations.</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Global SHAP Feature Importance")
        if (FIGURES_DIR / "shap_summary_bar.png").exists():
            st.image(str(FIGURES_DIR / "shap_summary_bar.png"), caption="Mean absolute SHAP value impact per feature.")
        else:
            st.info("Run `src/explain.py` to generate SHAP bar plot.")

    with col_b:
        st.subheader("Tree-Based Feature Importance")
        if (FIGURES_DIR / "feature_importance.png").exists():
            st.image(str(FIGURES_DIR / "feature_importance.png"), caption="Top 15 features in XGBoost and Random Forest.")
        else:
            st.info("Run `src/evaluate.py` to generate feature importance plot.")

    st.markdown("---")
    st.subheader("SHAP Beeswarm Summary Plot")
    st.markdown(
        "Each point represents an individual customer. Red indicates a high feature value, blue indicates a low feature value. Points to the right increase churn risk; points to the left decrease churn risk."
    )
    if (FIGURES_DIR / "shap_summary_beeswarm.png").exists():
        st.image(str(FIGURES_DIR / "shap_summary_beeswarm.png"), caption="SHAP Summary Beeswarm Plot.")
    else:
        st.info("Run `src/explain.py` to generate SHAP beeswarm plot.")

    st.markdown("---")
    st.subheader("Local Customer Waterfall Explanations")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("##### High-Risk Customer Profile")
        if (FIGURES_DIR / "shap_local_high_risk.png").exists():
            st.image(str(FIGURES_DIR / "shap_local_high_risk.png"), caption="Feature contributions driving high churn risk.")
        else:
            st.info("Run `src/explain.py` to generate high-risk waterfall plot.")

    with col_l2:
        st.markdown("##### Low-Risk Customer Profile")
        if (FIGURES_DIR / "shap_local_low_risk.png").exists():
            st.image(str(FIGURES_DIR / "shap_local_low_risk.png"), caption="Feature contributions pulling churn risk down.")
        else:
            st.info("Run `src/explain.py` to generate low-risk waterfall plot.")
