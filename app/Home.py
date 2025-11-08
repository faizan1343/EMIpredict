# app/Home.py
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import engineer_features, get_feature_columns
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# --- AUTO-FIND DATA FILE ---
def find_data_file():
    root = Path(__file__).resolve().parents[1]
    matches = list(root.rglob("emi_clean.parquet"))
    if not matches:
        st.error("emi_clean.parquet NOT FOUND in project!")
        st.write("Run this in PowerShell to find it:")
        st.code("cd E:\\Labmentix\\projects\\EMIpredict\nGet-ChildItem -Recurse -Include 'emi_clean.parquet'")
        st.stop()
    return matches[0]

DATA_PATH = find_data_file()

@st.cache_resource
def train_models():
    st.write("Loading data and training models... (first run only)")

    df = pd.read_parquet(DATA_PATH)
    df = engineer_features(df)
    X = df[get_feature_columns()]

    # --- LABEL ENCODE TARGET ---
    le = LabelEncoder()
    y_clf = le.fit_transform(df["emi_eligibility"])  # ← CRITICAL FIX
    y_reg = df["max_monthly_emi"]

    cat_cols = X.select_dtypes("category").columns.tolist()
    num_cols = X.select_dtypes("number").columns.tolist()
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ])

    train_idx = X.sample(frac=0.8, random_state=42).index
    X_train = X.loc[train_idx]
    y_train_clf = y_clf[train_idx]
    y_train_reg = y_reg.loc[train_idx]

    # Train Classifier
    clf_pipe = Pipeline([
        ('prep', prep),
        ('clf', XGBClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42))
    ])
    clf_pipe.fit(X_train, y_train_clf)

    # Train Regressor
    reg_pipe = Pipeline([
        ('prep', prep),
        ('reg', XGBRegressor(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42))
    ])
    reg_pipe.fit(X_train, y_train_reg)

    st.success("Models trained and cached in memory!")
    return clf_pipe, reg_pipe, le  # ← return encoder too

clf_model, reg_model, label_encoder = train_models()

st.set_page_config(page_title="EMIPredict AI", layout="centered")
st.title("EMIPredict AI")
st.caption("**Production XGBoost** – 99.78% Accuracy | ₹313 RMSE")

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 25, 60, 38)
        salary = st.number_input("Monthly Salary (₹)", 15000, 300000, 75000)
        rent = st.number_input("Monthly Rent (₹)", 0, 50000, 15000)
        credit = st.slider("Credit Score", 300, 850, 720)
        employment_years = st.slider("Years of Employment", 0, 40, 5)
        dependents = st.slider("Dependents", 0, 5, 0)
        bank_balance = st.number_input("Bank Balance (₹)", 0, 500000, 50000)
        emergency_fund = st.number_input("Emergency Fund (₹)", 0, 100000, 20000)
    with col2:
        amount = st.number_input("Requested Amount (₹)", 10000, 2000000, 400000)
        tenure = st.slider("Tenure (months)", 3, 84, 36)
        scenario = st.selectbox("EMI Type", [
            "Personal Loan EMI", "Vehicle EMI", "Education EMI",
            "Home Appliances EMI", "E-commerce Shopping EMI"
        ])
        family_size = st.slider("Family Size", 1, 10, 4)
        travel_expenses = st.number_input("Monthly Travel (₹)", 0, 10000, 3000)
    submitted = st.form_submit_button("Predict EMI Risk")

if submitted:
    # Validate inputs to prevent division by zero
    if tenure <= 0:
        st.error("Tenure must be greater than 0!")
        st.stop()
    if salary <= 0:
        st.error("Monthly salary must be greater than 0!")
        st.stop()
    
    requested_emi = amount / tenure
    input_df = pd.DataFrame([{
        'age': age, 'monthly_salary': salary, 'monthly_rent': rent, 'credit_score': credit,
        'requested_amount': amount, 'requested_tenure': tenure, 'emi_scenario': scenario,
        'years_of_employment': employment_years, 'dependents': dependents,
        'bank_balance': bank_balance, 'emergency_fund': emergency_fund,
        'family_size': family_size, 'travel_expenses': travel_expenses,
        'max_monthly_emi': 0, 'requested_emi': requested_emi,
        'gender': 'Male', 'marital_status': 'Married', 'education': 'Graduate',
        'employment_type': 'Private', 'company_type': 'MNC', 'house_type': 'Rented',
        'existing_loans': 'No', 'current_emi_amount': 0, 'school_fees': 0, 'college_fees': 0,
        'groceries_utilities': 12000, 'other_monthly_expenses': 5000
    }])
    
    try:
        X_input = engineer_features(input_df)[get_feature_columns()]
        
        # Validate that no infinity or NaN values remain
        numeric_cols = X_input.select_dtypes(include=['number']).columns
        if X_input[numeric_cols].isin([np.inf, -np.inf]).any().any() or X_input[numeric_cols].isna().any().any():
            st.error("Error: Invalid values detected in features. Please check your inputs.")
            st.stop()
        
        pred_label = clf_model.predict(X_input)[0]
        eligibility = label_encoder.inverse_transform([pred_label])[0]  # ← DECODE
        conf = float(clf_model.predict_proba(X_input).max())  # Convert to Python float
        max_emi = float(reg_model.predict(X_input)[0])  # Convert to Python float
        
        color = {"Eligible": "green", "High_Risk": "orange", "Not_Eligible": "red"}[eligibility]
        st.markdown(f"### <span style='color:{color}'>**{eligibility}**</span>", unsafe_allow_html=True)
        st.progress(conf)
        st.caption(f"Confidence: **{conf:.1%}**")
        
        st.metric("Max Safe EMI (AI)", f"₹{max_emi:,.0f}")
        st.metric("Your EMI", f"₹{requested_emi:,.0f}")
        st.write("---")
        if requested_emi > max_emi:
            st.error("EMI exceeds safe limit!")
        else:
            st.success("Within safe range")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.exception(e)