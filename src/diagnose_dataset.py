# src/diagnose_dataset.py


import pandas as pd
import numpy as np
import os
from pathlib import Path

# === CONFIG ===
ROOT_DIR = Path(r"E:\Labmentix\projects\EMIpredict")
DATA_PATH = ROOT_DIR / "emi_prediction_dataset.csv"

def diagnose():
    print(f"Loading dataset from:\n{DATA_PATH}\n")
    assert DATA_PATH.exists(), "Dataset not found!"
    
    # Load with explicit dtypes to save memory
    df = pd.read_csv(DATA_PATH)
    
    print(f"Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB\n")
    
    # === 1. Data Types & Sample ===
    print("Sample (5 rows):")
    print(df.head(5).to_string(index=False))
    print("\nData Types:")
    print(df.dtypes)
    
    # === 2. Missing Values ===
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values!")
    else:
        print(missing[missing > 0])
    
    # === 3. Duplicates ===
    dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {dupes}")
    
    # === 4. Categorical Consistency ===
    cat_cols = [
        'gender', 'marital_status', 'education', 'employment_type',
        'company_type', 'house_type', 'existing_loans', 'emi_scenario', 'emi_eligibility'
    ]
    
    print("\nCategorical Value Distribution:")
    for col in cat_cols:
        if col in df.columns:
            unique = df[col].astype(str).str.strip().unique()
            print(f"{col}: {len(unique)} unique → {list(unique)[:10]}{'...' if len(unique)>10 else ''}")
    
    # === 5. Target Variables ===
    print("\nTarget: emi_eligibility")
    print(df['emi_eligibility'].value_counts())
    
    print("\nTarget: max_monthly_emi")
    print(df['max_monthly_emi'].describe())
    
    # === 6. Numeric Outliers (Quick Check) ===
    num_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nNumeric columns: {len(num_cols)}")
    print("99th percentile (outlier check):")
    print(df[num_cols].quantile(0.99))
    
    return df

if __name__ == "__main__":
    df = diagnose()