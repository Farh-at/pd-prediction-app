
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
    Task = st.selectbox("Task", ["Rest1", "Rest2", "4MW", "4MW-C", "MB1", "Hotspot1", "Hotspot2", "Hotspot1-C", "Hotspot2-C", "MB10"])
    Age = st.number_input("Age", 0, 100, 65)
    Sex = st.selectbox("Sex", ["M", "F"])
    YearsSinceDx = st.slider("Years Since Diagnosis", 0, 40, 5)

    if st.button("Predict"):
        try:
            # Input as DataFrame
            input_df = pd.DataFrame([{
                "Medication": 1 if Medication == "on" else 0,
                "Kinetic": Kinetic,
                "Task": Task,
                "Age": Age,
                "Sex": 1 if Sex == "M" else 0,
                "YearsSinceDx": YearsSinceDx
            }])

            # Load encoder and encode Task
    if st.button("Predict"):
        try:
            # Input DataFrame
            input_df = pd.DataFrame([{
                "Medication": 1 if Medication == "on" else 0,
                "Kinetic": Kinetic,
                "Task": Task,
                "Age": Age,
                "Sex": 1 if Sex == "M" else 0,
                "YearsSinceDx": YearsSinceDx
            }])

            # Dynamically re-train encoder on known task categories
            task_categories = [
                "Rest1", "Rest2", "4MW", "4MW-C", "MB1",
                "Hotspot1", "Hotspot2", "Hotspot1-C", "Hotspot2-C", "MB10"
            ]
            encoder = OneHotEncoder(handle_unknown='ignore', categories=[task_categories])
            encoder.fit(pd.DataFrame(task_categories, columns=["Task"]))

            task_encoded = encoder.transform(input_df[["Task"]]).toarray()
            task_columns = encoder.get_feature_names_out(["Task"])
            task_df = pd.DataFrame(task_encoded, columns=task_columns)

            # Normalize numeric data
            numeric = input_df[["Medication", "Kinetic", "Age", "Sex", "YearsSinceDx"]]
            numeric_scaled = (numeric - numeric.mean()) / numeric.std()

            # Merge final input
            final_input = pd.concat([task_df, numeric_scaled.reset_index(drop=True)], axis=1)

            # Load model
            model = joblib.load("combined_model.joblib")
            st.success("Model loaded successfully!")

            # Predict
            for name, clf in model.items():
                result = clf.predict_proba(final_input)[0][1]
                st.info(f"{name}: {result * 100:.2f}% probability")
        
        except Exception as e:
            st.error(f"Prediction Error: {e}")
