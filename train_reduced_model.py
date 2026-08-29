import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    print("Loading dataset...")
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Data Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

    # Top 8 important features
    selected_features = [
        'tenure',
        'Contract',
        'InternetService',
        'OnlineSecurity',
        'TechSupport',
        'PaymentMethod',
        'MonthlyCharges',
        'PaperlessBilling'
    ]

    X = df[selected_features]
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cat_cols = [c for c in selected_features if X[c].dtype == 'object']
    num_cols = [c for c in selected_features if X[c].dtype in ['int64', 'float64']]

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    ])

    print("Fitting model on reduced 8 features...")
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model_pipeline, 'best_churn_model.pkl')
    print("Model saved successfully as 'best_churn_model.pkl'!")

if __name__ == '__main__':
    train_model()
