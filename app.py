
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="PD Symptom Prediction", layout="wide")
st.title("🧠 Parkinson's Disease Prediction Platform")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Home", "Predict"])

if page == "Home":
    st.markdown("""
    This app uses machine learning to predict Parkinson's Disease motor and non-motor symptoms.
    Upload patient data or manually input parameters to see real-time predictions.
    """)

elif page == "Predict":
    st.header("📈 Real-Time Prediction")

    Medication = st.selectbox("Medication Status", ["on", "off"])
    Kinetic = st.slider("Kinetic", 0, 10, 5)
    Task = st.selectbox("Task", ["Rest1", "Rest2", "4MW", "4MW-C", "MB1"])
    Age = st.number_input("Age", 0, 100, 65)
    Sex = st.selectbox("Sex", ["M", "F"])
    YearsSinceDx = st.slider("Years Since Diagnosis", 0, 40, 5)

    if st.button("Predict"):
        try:
            data = pd.DataFrame([{
                "Medication": 1 if Medication == "on" else 0,
                "Kinetic": Kinetic,
                "Task": Task,
                "Age": Age,
                "Sex": 1 if Sex == "M" else 0,
                "YearsSinceDx": YearsSinceDx
            }])

            # Encode Task manually
            task_dummies = pd.get_dummies(data["Task"], prefix="Task")
            for col in ["Task_4MW", "Task_4MW-C", "Task_MB1", "Task_Rest1", "Task_Rest2"]:
                if col not in task_dummies.columns:
                    task_dummies[col] = 0
            task_dummies = task_dummies[["Task_4MW", "Task_4MW-C", "Task_MB1", "Task_Rest1", "Task_Rest2"]]

            # Standardize numerical values
            num = data[["Medication", "Kinetic", "Age", "Sex", "YearsSinceDx"]]
            num_scaled = (num - num.mean()) / num.std()

            final_input = pd.concat([task_dummies, num_scaled], axis=1)

            model = joblib.load("combined_model.joblib")
            st.success("Model loaded successfully!")

            for name, clf in model.items():
                result = clf.predict_proba(final_input)[0][1]
                st.info(f"{name}: {result*100:.2f}% probability")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
