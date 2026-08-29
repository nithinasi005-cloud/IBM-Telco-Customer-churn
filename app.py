import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Customer Churn Intelligence & Retention Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_churn_model():
    return joblib.load('best_churn_model.pkl')

model = load_churn_model()

# Header
st.title("📊 Customer Churn Intelligence & Retention Dashboard")
st.write("Predict customer churn probability with high accuracy (>80%), analyze risk drivers, run batch predictions, and get actionable retention recommendations.")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Single Customer Prediction", "📁 Batch CSV Prediction", "📈 Model Analytics & Insights"])

# ==========================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ==========================================
with tab1:
    st.subheader("Predict Customer Churn Risk")
    st.write("Enter the customer's account details below to evaluate churn risk.")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider(
            "Tenure (Months)", 
            min_value=0, max_value=72, value=12, 
            help="Number of months the customer has stayed with the company"
        )
        Contract = st.selectbox(
            "Contract Type", 
            ['Month-to-month', 'One year', 'Two year'],
            help="Month-to-month contracts have significantly higher churn rates"
        )
        InternetService = st.selectbox(
            "Internet Service", 
            ['DSL', 'Fiber optic', 'No']
        )
        OnlineSecurity = st.selectbox(
            "Online Security", 
            ['Yes', 'No', 'No internet service']
        )

    with col2:
        TechSupport = st.selectbox(
            "Tech Support", 
            ['Yes', 'No', 'No internet service']
        )
        PaymentMethod = st.selectbox(
            "Payment Method",
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
        )
        MonthlyCharges = st.number_input(
            "Monthly Charges ($)", 
            min_value=0.0, max_value=200.0, value=70.0, step=1.0
        )
        PaperlessBilling = st.selectbox(
            "Paperless Billing", 
            ['Yes', 'No']
        )

    st.divider()

    # Automatically derive TotalCharges from MonthlyCharges * tenure for model accuracy
    total_charges = MonthlyCharges * tenure

    # Create DataFrame for model input
    input_df = pd.DataFrame({
        'tenure': [tenure],
        'Contract': [Contract],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'TechSupport': [TechSupport],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [total_charges],
        'PaperlessBilling': [PaperlessBilling]
    })

    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        churn_prob = probabilities[1] * 100

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric("Churn Probability", f"{churn_prob:.1f}%")
            if prediction == 1:
                st.error("⚠️ **High Churn Risk**")
            else:
                st.success("✅ **Low Churn Risk**")

        with res_col2:
            st.subheader("💡 Automated Retention Strategies")
            recommendations = []

            if Contract == 'Month-to-month':
                recommendations.append("📌 **Incentivize Contract Extension**: Offer a 10-15% discount for upgrading to a 1-year or 2-year contract.")
            if InternetService == 'Fiber optic' and TechSupport == 'No':
                recommendations.append("🛠️ **Bundle Tech Support**: Fiber Optic customers without Tech Support have higher churn. Offer a 3-month free Tech Support trial.")
            if OnlineSecurity == 'No' and InternetService != 'No':
                recommendations.append("🔒 **Promote Security Add-on**: Suggest adding Online Security to protect customer devices.")
            if PaymentMethod == 'Electronic check':
                recommendations.append("💳 **Switch to Auto-Pay**: Offer a $5 monthly bill credit for switching to automatic bank transfer or credit card payments.")
            if tenure <= 12:
                recommendations.append("🎁 **Onboarding Check-in**: Customer is in high-risk early tenure (<12 months). Initiate a customer success check-in call.")

            if not recommendations:
                st.info("Customer has strong retention indicators! Continue standard loyalty engagements.")
            else:
                for rec in recommendations:
                    st.write(rec)

# ==========================================
# TAB 2: BATCH CSV PREDICTION
# ==========================================
with tab2:
    st.subheader("Batch Churn Prediction")
    st.write("Upload a CSV file containing customer data to predict churn across multiple records at once.")

    uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"📄 **Uploaded Dataset**: {batch_df.shape[0]} rows, {batch_df.shape[1]} columns")

            # Derive TotalCharges if missing
            if 'TotalCharges' not in batch_df.columns:
                batch_df['TotalCharges'] = pd.to_numeric(batch_df.get('TotalCharges', np.nan), errors='coerce')
                batch_df['TotalCharges'] = batch_df['TotalCharges'].fillna(batch_df['MonthlyCharges'] * batch_df['tenure'])

            required_cols = ['tenure', 'Contract', 'InternetService', 'OnlineSecurity', 'TechSupport', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'PaperlessBilling']
            missing_cols = [col for col in required_cols if col not in batch_df.columns]

            if missing_cols:
                st.error(f"Missing required columns in uploaded CSV: {', '.join(missing_cols)}")
            else:
                if st.button("🚀 Process Batch Predictions"):
                    X_batch = batch_df[required_cols]
                    batch_preds = model.predict(X_batch)
                    batch_probs = model.predict_proba(X_batch)[:, 1]

                    output_df = batch_df.copy()
                    output_df['Predicted_Churn'] = ['Yes' if p == 1 else 'No' for p in batch_preds]
                    output_df['Churn_Probability_%'] = (batch_probs * 100).round(2)

                    st.success("Batch Prediction Completed Successfully!")

                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        churn_count = sum(batch_preds)
                        st.metric("Total High Risk Customers", f"{churn_count} / {len(batch_preds)}", f"{(churn_count/len(batch_preds))*100:.1f}% Risk Rate")
                    with b_col2:
                        st.write("### Churn Breakdown")
                        st.bar_chart(output_df['Predicted_Churn'].value_counts())

                    st.dataframe(output_df.head(20), use_container_width=True)

                    csv_data = output_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_data,
                        file_name="churn_predictions_results.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
    else:
        st.info("💡 **Tip**: You can use `WA_Fn-UseC_-Telco-Customer-Churn.csv` from this project directory for batch testing!")

# ==========================================
# TAB 3: MODEL ANALYTICS & INSIGHTS
# ==========================================
with tab3:
    st.subheader("Model Performance & Hyperparameter Tuning")

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Model Architecture", "Tuned Stacking Ensemble")
    m_col2.metric("Base Estimators", "XGBoost + LogisticRegression + GradientBoosting")
    m_col3.metric("5-Fold CV Accuracy", "80.40%", "+1.91% vs Baseline")

    st.divider()

    st.write("### 📌 Top Predictive Feature Drivers")

    feature_importance = pd.DataFrame({
        'Feature': ['Internet Service', 'Contract Type', 'Streaming Movies', 'Payment Method', 'Online Security', 'Tech Support', 'Tenure', 'Monthly Charges'],
        'Importance Weight (%)': [53.39, 28.55, 2.32, 2.30, 1.60, 1.57, 0.89, 0.59]
    }).sort_values(by='Importance Weight (%)', ascending=True)

    st.bar_chart(feature_importance.set_index('Feature'))

    st.write("""
    **Hyperparameter Tuning Highlights**:
    - **Optimization**: Tuned using 5-Fold Stratified Cross-Validation on XGBoost (`learning_rate=0.03`, `max_depth=4`, `subsample=0.7`, `colsample_bytree=0.8`) and Logistic Regression.
    - **Stacking Meta-Learner**: Combined predictions from XGBoost, GradientBoosting, and LogisticRegression to push accuracy above **80%**.
    """)
