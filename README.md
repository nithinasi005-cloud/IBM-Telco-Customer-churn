# 📊 Customer Churn Intelligence & Retention Dashboard

An end-to-end Machine Learning web application built with **Python**, **XGBoost**, **scikit-learn**, and **Streamlit** to predict customer churn probability and generate actionable customer retention strategies.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)

---

## 🌟 Key Features

1. **Feature-Reduced Machine Learning Pipeline**:
   - Analyzed feature importances and reduced inputs from **19 noisy features down to 8 key features**.
   - Achieved **79.21% test accuracy** (outperforming the original 19-feature model's 78.49%).
   
2. **Single Customer Prediction & Automated Retention Advice**:
   - Instant churn probability calculation.
   - Generates personalized recommendations (e.g. contract upgrades, tech support bundling, auto-pay incentives) based on customer risk drivers.

3. **Batch CSV Prediction & Export**:
   - Upload any CSV dataset of customer records.
   - Generates predictions and churn risk scores in bulk with interactive visual summary charts.
   - Export predictions directly as `churn_predictions_results.csv`.

4. **Model Analytics & Insights Tab**:
   - Interactive visual breakdown of feature importance rankings.
   - Full model performance transparency.

---

## 📁 Repository Structure

```
WA_Fn-UseC_-Telco-Customer-Churn/
├── app.py                          # Streamlit Dashboard (Multi-tab web UI)
├── best_churn_model.pkl            # Serialized XGBoost Pipeline (8-feature model)
├── train_reduced_model.py          # Script to retrain model on 8 key features
├── classification telco_churn.ipynb# Jupyter Notebook for EDA & model experimentation
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset
├── requirements.txt                # Project Python dependencies
└── README.md                       # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation
Clone the repository and install required packages:

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard
Run the following command to start the web application locally:

```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧠 Model Training & Feature Engineering

To retrain the model pipeline or update hyperparameters:

```bash
python train_reduced_model.py
```

### Top 8 Selected Features

| Feature | Description | Type |
| :--- | :--- | :--- |
| **Contract** | Month-to-month, One year, Two year | Categorical |
| **InternetService** | Fiber optic, DSL, No | Categorical |
| **tenure** | Number of months stayed with company (0 - 72) | Numeric |
| **TechSupport** | Yes, No, No internet service | Categorical |
| **OnlineSecurity** | Yes, No, No internet service | Categorical |
| **PaymentMethod** | Electronic check, Mailed check, Bank transfer, Credit card | Categorical |
| **MonthlyCharges** | Monthly subscription fee amount | Numeric |
| **PaperlessBilling**| Yes, No | Categorical |

---

## 📊 Performance Comparison

| Model Pipeline | Features | Test Accuracy | User Input Complexity |
| :--- | :---: | :---: | :---: |
| **Full Model** | 19 | 78.49% | High (19 inputs) |
| **Reduced Model (Ours)** | **8** | **79.21%** | **Optimal (8 inputs)** |

---

## 📜 License
This project is open-source under the MIT License.
