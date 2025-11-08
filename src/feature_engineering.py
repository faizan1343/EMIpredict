# src/feature_engineering.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path

ROOT = Path(r"E:\Labmentix\projects\EMIpredict")
DATA_PATH = ROOT / "data" / "processed" / "emi_clean.parquet"

def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create high-signal, interpretable features.
    Vectorized, no loops, production-safe.
    Handles division by zero and infinity values.
    """
    df = df.copy()
    
    # Small epsilon to prevent division by zero
    eps = 1e-8
    
    # === 1. Core Financial Ratios ===
    total_expenses = (
        df['current_emi_amount'] + df['monthly_rent'] +
        df['school_fees'] + df['college_fees'] +
        df['groceries_utilities'] + df['other_monthly_expenses']
    )
    
    # Safe division: replace zero denominators with epsilon
    monthly_salary_safe = df['monthly_salary'].replace(0, eps).clip(lower=eps)
    
    df['dti_ratio'] = total_expenses / monthly_salary_safe
    df['emi_to_income'] = df['current_emi_amount'] / monthly_salary_safe
    df['rent_to_income'] = df['monthly_rent'] / monthly_salary_safe
    
    # === 2. Affordability & Capacity ===
    # Safe division for tenure
    requested_tenure_safe = df['requested_tenure'].replace(0, eps).clip(lower=eps)
    df['requested_emi'] = df['requested_amount'] / requested_tenure_safe
    df['affordability_gap'] = df['max_monthly_emi'] - df['requested_emi']
    
    # Safe division for max_monthly_emi (can be 0 in prediction scenarios)
    max_monthly_emi_safe = df['max_monthly_emi'].replace(0, eps).clip(lower=eps)
    df['emi_affordability_ratio'] = df['requested_emi'] / max_monthly_emi_safe
    
    # === 3. Stability & Risk Scores ===
    # Handle case where all years_of_employment might be 0
    years_max = df['years_of_employment'].max()
    if years_max == 0:
        years_max = 1.0  # Avoid division by zero
    
    df['stability_score'] = (
        0.4 * (df['years_of_employment'] / years_max) +
        0.6 * (df['credit_score'] / 850)
    )
    
    df['existing_loan_flag'] = (df['existing_loans'] == 'Yes').astype(int)
    df['has_dependents'] = (df['dependents'] > 0).astype(int)
    
    # === 4. Interaction Features ===
    df['salary_x_tenure'] = df['monthly_salary'] * df['requested_tenure']
    df['amount_per_month'] = df['requested_amount'] / requested_tenure_safe
    
    # === 5. Binning & Risk Buckets ===
    # Replace inf values before binning (temporary copy for binning)
    dti_ratio_for_binning = df['dti_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Create dti_bucket with safe handling
    try:
        dti_bucket = pd.cut(
            dti_ratio_for_binning,
            bins=[-np.inf, 0.3, 0.5, 0.7, np.inf],
            labels=['Low', 'Medium', 'High', 'Critical'],
            duplicates='drop'
        )
        # Fill NaN with 'Critical' (worst case) - convert to string, replace, convert back
        if dti_bucket.isna().any():
            dti_bucket = dti_bucket.astype(str)
            dti_bucket = dti_bucket.replace('nan', 'Critical')
            dti_bucket = pd.Categorical(dti_bucket, categories=['Low', 'Medium', 'High', 'Critical'])
        df['dti_bucket'] = dti_bucket
    except Exception:
        # Fallback: assign all to 'Critical' if binning fails
        df['dti_bucket'] = pd.Categorical(['Critical'] * len(df), 
                                         categories=['Low', 'Medium', 'High', 'Critical'])
    
    # Create credit_bucket with safe handling
    try:
        credit_bucket = pd.cut(
            df['credit_score'],
            bins=[300, 580, 670, 740, 850],
            labels=['Poor', 'Fair', 'Good', 'Excellent'],
            duplicates='drop'
        )
        # Fill NaN with 'Poor' (worst case)
        if credit_bucket.isna().any():
            credit_bucket = credit_bucket.astype(str)
            credit_bucket = credit_bucket.replace('nan', 'Poor')
            credit_bucket = pd.Categorical(credit_bucket, categories=['Poor', 'Fair', 'Good', 'Excellent'])
        df['credit_bucket'] = credit_bucket
    except Exception:
        # Fallback: assign all to 'Poor' if binning fails
        df['credit_bucket'] = pd.Categorical(['Poor'] * len(df),
                                            categories=['Poor', 'Fair', 'Good', 'Excellent'])
    
    # === 6. Outlier Capping (99th percentile) ===
    # For prediction with single row, we'll use reasonable caps instead of quantiles
    reasonable_caps = {
        'dti_ratio': 10.0,  # 10x salary is extremely high
        'emi_to_income': 2.0,  # 200% of income
        'rent_to_income': 1.0,  # 100% of income
        'emi_affordability_ratio': 5.0  # 5x max EMI
    }
    
    for col in ['dti_ratio', 'emi_to_income', 'rent_to_income', 'emi_affordability_ratio']:
        # Replace inf values first
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Calculate cap: use quantile if we have enough data, otherwise use reasonable cap
        finite_values = df[col].dropna()
        if len(finite_values) > 1:
            # Use quantile for multiple rows
            cap = finite_values.quantile(0.99)
            if pd.notna(cap) and cap > 0:
                max_cap = reasonable_caps.get(col, 100.0)
                cap = min(cap, max_cap)  # Don't exceed reasonable maximum
                df[col] = df[col].clip(upper=cap, lower=-cap if col != 'emi_affordability_ratio' else 0)
        else:
            # For single row or insufficient data, use reasonable cap
            cap = reasonable_caps.get(col, 100.0)
            df[col] = df[col].clip(upper=cap, lower=-cap if col != 'emi_affordability_ratio' else 0)
        
        # Fill any remaining NaN with 0 or a reasonable default
        df[col] = df[col].fillna(0)
    
    # === 7. Final cleanup: Replace any remaining inf/NaN in numeric columns ===
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # Fill NaN with median for that column, or 0 if all are NaN
            if df[col].isna().any():
                fill_val = df[col].median()
                if pd.isna(fill_val):
                    fill_val = 0
                df[col] = df[col].fillna(fill_val)
    
    return df

def get_feature_columns() -> list:
    """Return final feature list for modeling"""
    return [
        # Raw
        'age', 'monthly_salary', 'years_of_employment', 'credit_score',
        'requested_amount', 'requested_tenure', 'bank_balance', 'emergency_fund',
        # Categorical
        'gender', 'marital_status', 'education', 'employment_type',
        'company_type', 'house_type', 'emi_scenario',
        # Flags
        'existing_loan_flag', 'has_dependents',
        # Ratios
        'dti_ratio', 'emi_to_income', 'rent_to_income',
        'emi_affordability_ratio', 'affordability_gap',
        # Derived
        'stability_score', 'salary_x_tenure', 'amount_per_month',
        # Buckets
        'dti_bucket', 'credit_bucket'
    ]

if __name__ == "__main__":
    df = load_data()
    df_feat = engineer_features(df)
    print(f"Features engineered: {df_feat.shape[1]} columns")
    print("Sample features:")
    print(df_feat[get_feature_columns()].head(3).T)
    
    # Save
    out_path = ROOT / "data" / "processed" / "emi_features.parquet"
    df_feat[get_feature_columns() + ['emi_eligibility', 'max_monthly_emi']].to_parquet(out_path, index=False)
    print(f"Feature set saved to: {out_path}")