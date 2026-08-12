import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="centered")

# ---- load saved model artifacts ----
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    model_name = joblib.load("model_name.pkl")
    return model, scaler, encoders, feature_columns, model_name


model, scaler, encoders, feature_columns, model_name = load_artifacts()

st.title("💰 Loan Default Predictor")
st.caption(f"Using **{model_name}** (chosen automatically as the best-performing model)")

st.write("Enter the applicant's details below to predict whether they are likely to default on the loan.")

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        annual_income = st.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
        employment_type = st.selectbox("Employment Type", encoders["Employment_Type"].classes_)
        education = st.selectbox("Education", encoders["Education"].classes_)

    with col2:
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=150000, step=1000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "Age": age,
                "Annual_Income": annual_income,
                "Employment_Type": encoders["Employment_Type"].transform([employment_type])[0],
                "Education": encoders["Education"].transform([education])[0],
                "Loan_Amount": loan_amount,
                "Credit_Score": credit_score,
            }
        ]
    )

    # keep column order identical to training
    input_df = input_df[feature_columns]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error("⚠️ **High risk: this applicant is likely to default.**")
    else:
        st.success("✅ **Low risk: this applicant is likely to repay.**")

    if proba is not None:
        st.metric("Estimated default probability", f"{proba * 100:.1f}%")

st.divider()
st.caption(
    "This tool is for educational/demo purposes only and should not be used "
    "for real lending decisions without further validation."
)
