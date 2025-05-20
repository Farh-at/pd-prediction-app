
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="PD Symptom Prediction", layout="wide")
st.title("🧠 Parkinson's Disease Prediction Platform")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Home", "Predict"])

if page == "Home":
    st.markdown("""This app predicts motor and non-motor symptoms in Parkinson's Disease patients
    using machine learning models trained on clinical and task-based data.""")

elif page == "Predict":
    st.header("📈 Real-Time Prediction")

    Medication = st.selectbox("Medication Status", ["on", "off"])
    Kinetic = st.slider("Kinetic", 0, 10, 5)
    Task = st.text_input("Task Name (as in dataset)", "Rest1")
    Age = st.number_input("Age", 0, 100, 65)
    Sex = st.selectbox("Sex", ["M", "F"])
    YearsSinceDx = st.slider("Years Since Diagnosis", 0, 40, 5)

    try:
        if st.button("Predict"):
            # Prepare input dataframe
            input_df = pd.DataFrame([{
                "Medication": 1 if Medication == "on" else 0,
                "Kinetic": Kinetic,
                "Task": Task,
                "Age": Age,
                "Sex": 1 if Sex == "M" else 0,
                "YearsSinceDx": YearsSinceDx
            }])

            # Load full task list from tasks.csv
            tasks_df = pd.read_csv("tasks.csv")
            all_tasks = sorted(tasks_df["Task"].dropna().unique().tolist())

            # Dynamically fit encoder based on full training task set
            encoder = OneHotEncoder(handle_unknown='ignore', categories=[all_tasks])
            encoder.fit(pd.DataFrame(all_tasks, columns=["Task"]))

            # Encode task
            task_encoded = encoder.transform(input_df[["Task"]]).toarray()
            task_columns = encoder.get_feature_names_out(["Task"])
            task_df = pd.DataFrame(task_encoded, columns=task_columns)

            # Normalize numeric fields
            numeric = input_df[["Medication", "Kinetic", "Age", "Sex", "YearsSinceDx"]]
            numeric_scaled = (numeric - numeric.mean()) / numeric.std()

            # Merge inputs
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
