import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_tuned_model():
    print("Loading dataset...")
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Preprocess numeric columns
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])

    # Key features for optimal performance & simple UI
    selected_features = [
        'tenure',
        'Contract',
        'InternetService',
        'OnlineSecurity',
        'TechSupport',
        'PaymentMethod',
        'MonthlyCharges',
        'TotalCharges',
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

    # Base models with tuned hyperparameters
    xgb_tuned = XGBClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=1,
        eval_metric='logloss',
        random_state=42
    )

    lr_tuned = LogisticRegression(C=0.5, max_iter=1000, random_state=42)

    gb_tuned = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        random_state=42
    )

    # Stacking Ensemble
    stacking_ensemble = StackingClassifier(
        estimators=[
            ('xgb', xgb_tuned),
            ('lr', lr_tuned),
            ('gb', gb_tuned)
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', stacking_ensemble)
    ])

    print("Training Hyperparameter-Tuned Stacking Ensemble Model...")
    full_pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = full_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring='accuracy')

    print(f"\n==========================================")
    print(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")
    print(f"Test Set Accuracy:                {test_acc * 100:.2f}%")
    print(f"==========================================\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(full_pipeline, 'best_churn_model.pkl')
    print("Hyperparameter-tuned model saved successfully as 'best_churn_model.pkl'!")

if __name__ == '__main__':
    train_tuned_model()
