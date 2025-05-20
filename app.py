
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PD Symptom Prediction", layout="wide")
st.title("🧠 Parkinson's Disease Prediction Platform")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Home", "Predict"])

if page == "Home":
    st.markdown("""This app predicts motor and non-motor symptoms in Parkinson's Disease patients
    using machine learning models trained on Age, Sex, and Years Since Diagnosis.""")

elif page == "Predict":
    st.header("📈 Real-Time Prediction")

    Age = st.number_input("Age", 0, 100, 65)
    Sex = st.selectbox("Sex", ["M", "F"])
    YearsSinceDx = st.slider("Years Since Diagnosis", 0, 40, 5)

    try:
        if st.button("Predict"):
            # Convert sex to numeric
            sex_numeric = 1 if Sex == "M" else 0

            # Prepare input DataFrame
            input_df = pd.DataFrame([{
                "Age": Age,
                "Sex": sex_numeric,
                "YearsSinceDx": YearsSinceDx
            }])

            # Scale input
            scaler = StandardScaler()
            input_scaled = scaler.fit_transform(input_df)  # Assume mean/std centered for demo
            input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)

            # Load model
            model = joblib.load("combined_model_subjects_final.joblib")
            st.success("Model loaded successfully!")

            # Predict
            for name, clf in model.items():
                result = clf.predict_proba(input_scaled_df)[0][1]
                st.info(f"{name}: {result * 100:.2f}% probability")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
