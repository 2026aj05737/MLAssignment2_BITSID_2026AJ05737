# Machine Learning Classifier Evaluation Web Application

## Problem Statement
The goal of this assignment is to build, evaluate, and compare multiple machine learning classification algorithms on a tabular dataset containing at least 12 features and 500 instances. The dataset is used to perform binary classification to predict target outcomes (such as wine quality classification based on physicochemical properties).

The assignment requires saving trained model artifacts (`.joblib`), creating an interactive web interface using **Streamlit**, and deploying the web application on **Streamlit Community Cloud** to allow dynamic model evaluation via user-uploaded test datasets.

---

## Dataset Description
- **Dataset Used:** Red Wine Quality Dataset (`winequality-red.csv`)
- **Number of Instances:** 1,599 instances
- **Number of Features:** 11 numerical physicochemical attributes + 1 engineered target feature
- **Features Included:**
  1. `fixed acidity`
  2. `volatile acidity`
  3. `citric acid`
  4. `residual sugar`
  5. `chlorides`
  6. `free sulfur dioxide`
  7. `total sulfur dioxide`
  8. `density`
  9. `pH`
  10. `sulphates`
  11. `alcohol`
- **Target Variable:** `target` (Binary: `1` if `quality` $\ge$ 6, `0` otherwise)

---

## Links & Deliverables
- **GitHub Repository:** [Insert your GitHub Repo URL here]
- **Streamlit Web Application:** [Insert your Streamlit App URL here]

---

## Model Performance Comparison Table

Below are the key performance metrics evaluated on an 80/20 train-test split:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7438 | 0.8124 | 0.7612 | 0.7485 | 0.7548 | 0.4851 |
| **Decision Tree** | 0.7625 | 0.7610 | 0.7711 | 0.7778 | 0.7744 | 0.5231 |
| **KNN** | 0.7594 | 0.8195 | 0.7622 | 0.7836 | 0.7728 | 0.5162 |
| **Naive Bayes** | 0.7281 | 0.7963 | 0.7200 | 0.7895 | 0.7532 | 0.4539 |
| **Support Vector Machine** | 0.7812 | 0.8450 | 0.7857 | 0.8012 | 0.7934 | 0.5604 |

---

## Model Performance Observations

| ML Model Name | Performance Observation |
| :--- | :--- |
| **Logistic Regression** | Establishes a strong linear baseline. Feature scaling was applied to enable proper coefficient learning. |
| **Decision Tree** | Captures non-linear feature interactions effectively, though slightly prone to variance compared to SVM. |
| **KNN** | Benefits significantly from feature standardization, achieving solid recall and F1 metrics. |
| **Naive Bayes** | Executes extremely fast; however, strong feature independence assumptions result in slightly lower overall accuracy and precision. |
| **Support Vector Machine** | Achieves the overall best performance across most metrics by constructing optimal separation hyperplanes in scaled feature space. |
| **Overall Best Model** | **Support Vector Machine (SVM)** achieved the highest F1 Score (0.7934), AUC (0.8450), and MCC (0.5604). |

---

## Repository Structure

```text
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── scaler.joblib
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
