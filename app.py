
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go

# results = {
#     "UPDRSIII_On": on_prob,
#     "UPDRSIII_Off": off_prob,
#     "NFOGQ": nfogq_prob
# }

# Display numeric results
for name, prob in results.items():
    st.info(f"{name}: {prob*100:.2f}% probability")

# Now render a bar chart
fig = go.Figure(
    [go.Bar(x=list(results.keys()), y=[p*100 for p in results.values()])]
)
fig.update_layout(
    title="Symptom Prediction Probabilities",
    yaxis_title="Probability (%)",
    xaxis_title="Symptom",
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig, use_container_width=True)


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

            # Load all known task types from tasks.csv
            tasks_df = pd.read_csv("tasks.csv")
            all_tasks = sorted(tasks_df["Task"].dropna().unique().tolist())

            # Fit encoder
            encoder = OneHotEncoder(handle_unknown='ignore', categories=[all_tasks])
            encoder.fit(pd.DataFrame(all_tasks, columns=["Task"]))

            # Encode task input
            task_encoded = encoder.transform(input_df[["Task"]]).toarray()
            task_columns = encoder.get_feature_names_out(["Task"])
            task_df = pd.DataFrame(task_encoded, columns=task_columns)

            # Normalize numeric data
            numeric = input_df[["Medication", "Kinetic", "Age", "Sex", "YearsSinceDx"]]
            numeric_scaled = (numeric - numeric.mean()) / numeric.std()

            # Combine task and numeric features
            final_input = pd.concat([task_df, numeric_scaled.reset_index(drop=True)], axis=1)

            # Load model
            model = joblib.load("combined_model.joblib")
            st.success("Model loaded successfully!")

            # Ensure no missing columns, fill all NaNs
            expected_features = model[next(iter(model))].feature_names_in_
            for col in expected_features:
                if col not in final_input.columns:
                    final_input[col] = 0

            final_input = final_input[expected_features].fillna(0)

            # Predict
            for name, clf in model.items():
                result = clf.predict_proba(final_input)[0][1]
                st.info(f"{name}: {result * 100:.2f}% probability")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
