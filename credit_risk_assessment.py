from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="🏦",
    layout="centered"
)

# Resolve bundled assets relative to the app location.
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "credit_risk_production_model.pkl"

# Cache model artifact loading for performance optimization
@st.cache_resource
def load_model_artifact(path: Path):
    return joblib.load(path)

try:
    model_artifact = load_model_artifact(MODEL_PATH)
    pipeline = model_artifact['pipeline']
    threshold = model_artifact['threshold']
except (FileNotFoundError, OSError, EOFError, ImportError, ValueError, KeyError):
    st.error(f"Failed to load model file: {MODEL_PATH}")
    st.stop()

# Header Section
st.title("🏦 Credit Risk Assessment")
st.markdown(
    "Interactive Machine Learning Model for Applicant Default Risk Prediction. "
    "Adjust applicant details below to generate a real-time underwriting decision."
)
st.divider()

# Input Form in Columns
st.subheader("📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    income = st.number_input("Annual Income (R)", min_value=0, value=75000, step=1000)
    home_ownership = st.selectbox("Home Ownership", ["OWN", "MORTGAGE", "RENT", "OTHER"])
    emp_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=60.0, value=8.0, step=0.5)
    default_on_file = st.selectbox("Historical Default on File", ["N", "Y"])
    cred_hist_length = st.number_input("Credit History Length (years)", min_value=0.0, max_value=60.0, value=10.0, step=0.5)

with col2:
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount (R)", min_value=0.0, value=15000.0, step=500.0)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.5, step=0.1)
    loan_percent_income = st.number_input("Loan-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.20, step=0.01)

st.divider()

# Prediction Action
if st.button("Evaluate Credit Application", type="primary", use_container_width=True):
    input_payload = {
        'person_age': [age],
        'person_income': [income],
        'person_home_ownership': [home_ownership],
        'person_emp_length': [emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [default_on_file],
        'cb_person_cred_hist_length': [cred_hist_length]
    }

    input_df = pd.DataFrame(input_payload)

    # Generate Prediction Probability
    proba = pipeline.predict_proba(input_df)[0, 1]
    is_declined = proba >= threshold

    # Display Decision Results
    st.subheader("📊 Evaluation Decision")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        if is_declined:
            st.error("❌ DECLINED")
        else:
            st.success("✅ APPROVED")

    with metric_col2:
        st.metric(
            label="Probability of Default",
            value=f"{proba:.1%}",
            delta=f"Threshold: {threshold:.1%}",
            delta_color="inverse"
        )

    # Risk Gauge Bar
    st.write("**Risk Probability Scale:**")
    st.progress(min(float(proba), 1.0))

    if is_declined:
        st.warning(
            f"Applicant default probability ({proba:.2%}) exceeds the risk policy threshold ({threshold:.2%})."
        )
    else:
        st.info(
            f"Applicant default probability ({proba:.2%}) is within acceptable risk limits (Threshold: {threshold:.2%})."
        )