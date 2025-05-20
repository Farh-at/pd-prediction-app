
# Parkinson's Disease Prediction App 🧠

This app predicts motor and non-motor symptoms (UPDRSIII On, Off, and NFOGQ scores) in Parkinson's Disease patients using machine learning models trained on clinical and task-based data.

## 🚀 Features
- Real-time prediction interface using `Streamlit`
- Trained Logistic Regression models for symptom classification
- OneHotEncoder trained on task types
- Fully deployable to [Streamlit Cloud](https://streamlit.io/cloud)

## 📁 Files Included
- `app.py`: Streamlit frontend
- `combined_model.joblib`: Trained ML models
- `task_encoder.joblib`: OneHotEncoder for task categories
- `*.csv`: Training data files
- `Abu New.py`: Training and preprocessing script
- `requirements.txt`: All Python dependencies

## 🧠 Usage

### 🔗 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### ☁️ Deploy to Streamlit Cloud
1. Upload all files to a public GitHub repo
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub and deploy using `app.py`

## 🛠 Requirements
- Python >= 3.8
- scikit-learn >= 1.3
