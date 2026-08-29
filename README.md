# 📊 Customer Churn Intelligence & Retention Dashboard

An end-to-end Machine Learning web application and interactive dashboard powered by a **Hyperparameter-Tuned Stacking Ensemble** (`XGBoost` + `LogisticRegression` + `GradientBoosting`). Designed for telecom business intelligence, this platform predicts customer churn probability with **>80% accuracy**, isolates key churn drivers, and provides automated, actionable retention strategies.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=flat&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?style=flat&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-008000?style=flat)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌟 Key Features

- **🧠 Advanced Stacking Ensemble Pipeline**:
  - Leverages a meta-classifier combining **XGBoost**, **Gradient Boosting**, and **Logistic Regression**.
  - Includes full scikit-learn preprocessing pipelines (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`).
  - Achieves **80.40% 5-Fold Cross-Validation Accuracy** on streamlined inputs.

- **🎯 Real-Time Single Customer Prediction & Retention Playbook**:
  - Instant calculation of customer churn probability percentage.
  - Contextual recommendation engine suggesting personalized retention tactics (e.g., contract upgrade incentives, auto-pay bill credits, tech support trials).

- **📁 Batch CSV Processing & Export**:
  - Upload raw customer datasets (`.csv`) for bulk predictions.
  - Automatically derives computed metrics like missing `TotalCharges`.
  - Visual summary charts (churn rates, risk distribution) with one-click export (`churn_predictions_results.csv`).

- **📈 Model Analytics & Explainability**:
  - Interactive feature importance visualization identifying critical churn risk signals.
  - Full transparency on model hyperparameters, cross-validation metrics, and pipeline architecture.

---

## 📁 Repository Structure

```
WA_Fn-UseC_-Telco-Customer-Churn/
├── app.py                          # Multi-tab Streamlit dashboard web interface
├── train_reduced_model.py          # Script for retraining and serializing the Stacking Model
├── best_churn_model.pkl            # Pre-trained & serialized scikit-learn pipeline model
├── classification telco_churn.ipynb# Jupyter notebook covering EDA & initial model trials
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Standard Telco Customer Churn dataset
├── requirements.txt                # Python dependencies list
└── README.md                       # Complete project documentation
```

---

## 🚀 Quick Start Guide

### 1. Installation & Environment Setup
Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard
Run the Streamlit app:

```bash
streamlit run app.py
```
The web dashboard will launch automatically in your browser at `http://localhost:8501`.

---

## 🧠 Model Training & Architecture

To retrain the stacking model pipeline on the dataset or re-tune hyperparameters:

```bash
python train_reduced_model.py
```

### 🛠️ Pipeline Architecture
1. **Preprocessor**: `ColumnTransformer` applying `StandardScaler` + `SimpleImputer(median)` to numerical features, and `OneHotEncoder` + `SimpleImputer(most_frequent)` to categorical features.
2. **Base Estimators**:
   - **Tuned XGBoost**: `learning_rate=0.03`, `max_depth=4`, `subsample=0.7`, `colsample_bytree=0.8`
   - **Tuned Gradient Boosting**: `learning_rate=0.03`, `max_depth=3`, `n_estimators=100`
   - **Logistic Regression**: `C=0.5`, `max_iter=1000`
3. **Meta Estimator**: `LogisticRegression` meta-learner aggregating base predictions via 5-fold cross-validation.

### 📋 Feature Schema (9 Selected Key Inputs)

| Feature | Description | Type | Values / Range |
| :--- | :--- | :--- | :--- |
| **tenure** | Months stayed with the company | Numeric | `0 - 72` |
| **Contract** | Customer contract duration | Categorical | `Month-to-month`, `One year`, `Two year` |
| **InternetService** | Customer's internet service provider | Categorical | `DSL`, `Fiber optic`, `No` |
| **OnlineSecurity** | Security add-on service status | Categorical | `Yes`, `No`, `No internet service` |
| **TechSupport** | Tech support add-on service status | Categorical | `Yes`, `No`, `No internet service` |
| **PaymentMethod** | Customer payment option | Categorical | `Electronic check`, `Mailed check`, `Bank transfer`, `Credit card` |
| **MonthlyCharges** | Amount charged monthly ($) | Numeric | `18.25 - 118.75` |
| **TotalCharges** | Total amount charged to date ($) | Numeric | Auto-calculated (`MonthlyCharges * tenure`) |
| **PaperlessBilling**| Paperless billing enablement | Categorical | `Yes`, `No` |

---

## 📊 Model Performance

| Metric | Baseline / Full Feature Model | Tuned Stacking Ensemble Model (Selected 9 Features) |
| :--- | :---: | :---: |
| **5-Fold CV Accuracy** | ~78.49% | **80.40%** (+1.91%) |
| **Test Set Accuracy** | ~78.5% | **~80.2%** |
| **Input Complexity** | High (19 noisy features) | **Optimal (9 essential features)** |

---

## 💡 Automated Retention Rules Engine

| Trigger Condition | Risk Insight | Automated Retention Action |
| :--- | :--- | :--- |
| `Contract == 'Month-to-month'` | Highest churn probability segment | Offer 10-15% discount for 1-year or 2-year contract lock-in. |
| `InternetService == 'Fiber optic'` & `TechSupport == 'No'` | High tech frustration churn risk | Provide a 3-month free Tech Support trial bundle. |
| `OnlineSecurity == 'No'` | Low account security retention | Promote device security add-on package. |
| `PaymentMethod == 'Electronic check'` | High payment friction & manual churn | Offer $5 monthly credit for switching to Auto-Pay. |
| `tenure <= 12` | Early tenure onboarding risk window | Schedule an automated Customer Success check-in. |

---

## 📜 License
This project is released under the MIT License.

