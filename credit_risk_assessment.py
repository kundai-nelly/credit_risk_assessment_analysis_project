from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Path is relative to this file so the app runs from any clone location
MODEL_PATH = Path(__file__).resolve().parent / "credit_risk_production_model.pkl"

# Loads the model
model_artifact = joblib.load(MODEL_PATH)
pipeline = model_artifact['pipeline']
threshold = model_artifact['threshold']

st.title("🏦 Credit Risk Assessment")
st.write("Enter the applicant's details below:")

# Inputs required fields
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Annual Income (R)", min_value=0, value=75000)
home_ownership = st.selectbox("Home Ownership", ["OWN", "MORTGAGE", "RENT", "OTHER"])
emp_length = st.number_input("Employment Length (years)", min_value=0.0, value=8.0)
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.selectbox("Loan Grade", ["A","B","C","D","E","F","G"])
loan_amnt = st.number_input("Loan Amount (R)", min_value=0.0, value=15000.0)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.5)
loan_percent_income = st.number_input("Loan-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.2)
default_on_file = st.selectbox("Default on File", ["N", "Y"])
cred_hist_length = st.number_input("Credit History Length (years)", min_value=0.0, value=10.0)

if st.button("Predict"):
    data = {
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
    df = pd.DataFrame(data)
    proba = pipeline.predict_proba(df)[0, 1]
    pred = int(proba >= threshold)
    decision = "✅ APPROVED" if pred == 0 else "❌ DECLINED"
    st.subheader(f"Decision: {decision}")
    st.write(f"Probability of Default: {proba:.2%}")