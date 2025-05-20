# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 23:32:32 2025

@author:ABUBAKAR
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump

# Set formatting option
pd.options.display.float_format = '{:.2f}'.format

def fill_missing_values(df):
    
 
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    return df

def explore_dataframe(df, name="DataFrame"):
    print(f"\nExploring {name}:")
    print(f"Shape: {df.shape}")
    display(df.head())  # Displaying first 5 rows):
    print("\nInfo:")
    df.info()
    print("\nMissing values:\n", df.isnull().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

def preprocess_data(df):
    df['Sex'] = df['Sex'].replace({'M': 1, 'F': 0}).fillna(0).astype(int)
    df['Medication'] = df['Medication'].replace({'on': 1, 'off': 0}).fillna(0).astype(int)
    
    numeric_cols = ['Visit', 'Test', 'Kinetic', 'Age', 'YearsSinceDx', 'UPDRSIII_On', 'UPDRSIII_Off', 'NFOGQ']
    df[numeric_cols] = df[numeric_cols].fillna(0).astype(int)
    
    mode_cols = ['Sex', 'Medication', 'Age', 'YearsSinceDx', 'UPDRSIII_On', 'UPDRSIII_Off', 'NFOGQ']
    df[mode_cols] = df[mode_cols].fillna(df[mode_cols].mode().iloc[0])
    
    return df

# Loading datasets
data_files = {
    "tdcsfog": 'tdcsfog_metadata.csv',
    "defog": 'defog_metadata.csv',
    "events": 'events.csv',
    "tasks": 'tasks.csv',
    "subjects": 'subjects.csv'
}

# Reading Data
datasets = {name: pd.read_csv(f"C:/Users/Administrator/Desktop/New folder/{filename}") for name, filename in data_files.items()}

# Preprocessing data
for name, df in datasets.items():
    datasets[name] = fill_missing_values(df)
    explore_dataframe(df, name)

# Merging datasets
datasets = [datasets['tdcsfog'], datasets['defog'], datasets['events'], datasets['tasks'], datasets['subjects']]
merged_data = pd.concat(datasets, axis=1, join='outer')

# Removing duplicate columns
merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

# Merging rows with the same column names
merged_data['Subject'] = merged_data['Subject'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

# Dropping 'Id' and 'Subject' columns
merged_data = merged_data.drop(['Id', 'Subject'], axis=1)

# Replacing values in 'Sex' and 'Medication' columns
merged_data['Sex'] = merged_data['Sex'].replace({'M': 1, 'F': 0})
merged_data['Medication'] = merged_data['Medication'].replace({'on': 1, 'off': 0})

# Fill missing values for numeric columns
numeric_columns = ['Visit', 'Test', 'Kinetic', 'Age', 'YearsSinceDx', 'UPDRSIII_On', 'UPDRSIII_Off', 'NFOGQ']
merged_data[numeric_columns] = merged_data[numeric_columns].fillna(0).astype(int)

# Plotting histograms
def plot_histograms(df, columns, colors, bins=6):
    fig, ax = plt.subplots()
    for col, color in zip(columns, colors):
        ax.hist(df[col], bins=bins, color=color, alpha=0.7, label=col)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {", ".join(columns)}')
    ax.legend()
    plt.show()

plot_histograms(merged_data, ['UPDRSIII_On', 'UPDRSIII_Off', 'NFOGQ'], ['lightblue', 'yellow', 'lightpink'])

# Correlation Matrix
correlation_matrix = merged_data[['UPDRSIII_On', 'UPDRSIII_Off', 'NFOGQ', 'Visit', 'Test', 'Medication', 'Kinetic', 'Age', 'Sex', 'YearsSinceDx']].corr()

# Heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='Blues', annot=True, fmt=".2f", annot_kws={"fontsize": 10})
plt.title('Correlation Matrix Heatmap')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Define features and target
X = merged_data[['Medication', 'Kinetic', 'Task', 'Age', 'Sex', 'YearsSinceDx']]
y = merged_data[['UPDRSIII_On', 'UPDRSIII_Off', 'NFOGQ']]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encoding categorical features
encoder = OneHotEncoder(handle_unknown='ignore')  
categorical_features = ['Task']
X_train_encoded = encoder.fit_transform(X_train[categorical_features])
X_test_encoded = encoder.transform(X_test[categorical_features])

# Standardizing numerical features
numeric_features = ['Medication', 'Kinetic', 'Age', 'Sex', 'YearsSinceDx']
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[numeric_features]), columns=numeric_features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test[numeric_features]), columns=numeric_features)

from sklearn.impute import SimpleImputer

def preprocess_features_with_imputation(X_train, X_test, categorical_features, numeric_features):
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')  # Use 'mean' for numeric columns, you can also use 'median' or 'most_frequent' as needed
    
    # Apply imputer to numeric features
    X_train[numeric_features] = imputer.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = imputer.transform(X_test[numeric_features])
    
    # Apply imputer to categorical features (use most_frequent for categorical)
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features])
    X_test[categorical_features] = categorical_imputer.transform(X_test[categorical_features])
    
    # Now you can encode categorical features and scale numeric ones
    encoder = OneHotEncoder(handle_unknown='ignore')
    scaler = StandardScaler()

    X_train_encoded = encoder.fit_transform(X_train[categorical_features])
    X_test_encoded = encoder.transform(X_test[categorical_features])
    
    X_train_scaled = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled = scaler.transform(X_test[numeric_features])

    X_train_processed = np.hstack([X_train_encoded.toarray(), X_train_scaled])
    X_test_processed = np.hstack([X_test_encoded.toarray(), X_test_scaled])
    
    return pd.DataFrame(X_train_processed), pd.DataFrame(X_test_processed)

# Update the preprocessing pipeline
categorical_features = ['Task']
numeric_features = ['Medication', 'Kinetic', 'Age', 'Sex', 'YearsSinceDx']
X_train_processed, X_test_processed = preprocess_features_with_imputation(X_train, X_test, categorical_features, numeric_features)
# Logistic Regression Model
logistic_regression_on = LogisticRegression(max_iter=5000)
logistic_regression_on.fit(X_train_processed, y_train['UPDRSIII_On'])

logistic_regression_off = LogisticRegression(max_iter=5000)
logistic_regression_off.fit(X_train_processed, y_train['UPDRSIII_Off'])

logistic_regression_nfogq = LogisticRegression(max_iter=5000)
logistic_regression_nfogq.fit(X_train_processed, y_train['NFOGQ'])

# Metrics Calculation
def calculate_metrics(models, X_train, X_test, y_train, y_test):
    metrics_results = []
    
    for class_name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        metrics_results.append({
            'Class': class_name,
            'Train Accuracy': accuracy_score(y_train[class_name], y_train_pred),
            'Test Accuracy': accuracy_score(y_test[class_name], y_test_pred),
            'F1 Score': f1_score(y_test[class_name], y_test_pred, average='weighted')
        })
    
    return metrics_results

# Creating the model for each target
combined_model = {
    'UPDRSIII_On': logistic_regression_on,
    'UPDRSIII_Off': logistic_regression_off,
    'NFOGQ': logistic_regression_nfogq
}

metrics_results = calculate_metrics(combined_model, X_train_processed, X_test_processed, y_train, y_test)

for metrics in metrics_results:
    print(f"Metrics for '{metrics['Class']}': Train Accuracy: {metrics['Train Accuracy']}, Test Accuracy: {metrics['Test Accuracy']}, F1 Score: {metrics['F1 Score']}")