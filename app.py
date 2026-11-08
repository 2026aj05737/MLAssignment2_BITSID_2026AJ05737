import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classifier Evaluation App", layout="wide")

st.title("Machine Learning Model Performance Evaluator")
st.write("Upload test dataset (CSV) and select a trained classification model to view metrics.")

# Sidebar Controls
st.sidebar.header("User Settings")

uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

model_options = {
    'Logistic Regression': 'model/logistic_regression.joblib',
    'Decision Tree': 'model/decision_tree.joblib',
    'KNN': 'model/knn.joblib',
    'Naive Bayes': 'model/naive_bayes.joblib',
    'Support Vector Machine': 'model/support_vector_machine.joblib'
}

selected_model_name = st.sidebar.selectbox("Select ML Model", list(model_options.keys()))

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Test Data Preview")
    st.dataframe(data.head())
    
    if 'target' not in data.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
    else:
        X_test = data.drop(columns=['target'])
        y_test = data['target']
        
        # Load Model
        model = joblib.load(model_options[selected_model_name])
        
        # Scale if required by model type
        if selected_model_name in ['Logistic Regression', 'KNN', 'Support Vector Machine']:
            scaler = joblib.load('scaler.joblib')
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test
            
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None

        # Display Metrics
        st.subheader(f"Evaluation Metrics: {selected_model_name}")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("AUC", f"{roc_auc_score(y_test, y_proba):.4f}" if y_proba is not None else "N/A")
        col3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        col4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        col5.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
        col6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")
        
        # Display Confusion Matrix & Report
        st.subheader("Confusion Matrix & Classification Report")
        col_left, col_right = st.columns(2)
        
        with col_left:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
        with col_right:
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
else:
    st.info("Please upload a `test_data.csv` file from the sidebar to proceed.")