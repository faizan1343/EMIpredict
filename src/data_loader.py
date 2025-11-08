# src/data_loader.py

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = Path(r"E:\Labmentix\projects\EMIpredict")
DATA_PATH = ROOT_DIR / "emi_prediction_dataset.csv"
PROCESSED_PATH = ROOT_DIR / "data" / "processed"
PROCESSED_PATH.mkdir(exist_ok=True)

# Columns that should be numeric
NUMERIC_COLS = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'family_size', 'dependents', 'school_fees', 'college_fees',
    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
    'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
    'requested_amount', 'requested_tenure', 'max_monthly_emi'
]

CAT_COLS = [
    'gender', 'marital_status', 'education', 'employment_type',
    'company_type', 'house_type', 'existing_loans', 'emi_scenario', 'emi_eligibility'
]

def robust_parse_numeric(series):
    """Convert to numeric, coerce errors, log bad values"""
    original = series.copy()
    numeric = pd.to_numeric(series, errors='coerce')
    bad_mask = original.notna() & numeric.isna()
    if bad_mask.any():
        bad_values = original[bad_mask].unique()[:10]
        print(f"Warning: Non-numeric values found and coerced to NaN: {bad_values}")
    return numeric

def load_and_preprocess() -> pd.DataFrame:
    print("Loading dataset with robust parsing...")
    
    # Read CSV without dtype enforcement first
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Raw shape: {df.shape}")

    # === 1. Coerce numeric columns ===
    for col in NUMERIC_COLS:
        if col in df.columns:
            print(f"Parsing {col}...")
            df[col] = robust_parse_numeric(df[col])

    # === 2. Drop rows where critical fields are NaN after coercion ===
    critical_cols = ['age', 'monthly_salary', 'credit_score', 'requested_amount', 'requested_tenure']
    before_drop = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    print(f"Dropped {before_drop - len(df)} rows with unparseable critical fields")

    # === 3. Standardize categoricals ===
    print("Standardizing categoricals...")
    
    # Gender mapping
    gender_map = {
        'M': 'Male', 'm': 'Male', 'male': 'Male', 'MALE': 'Male',
        'F': 'Female', 'f': 'Female', 'female': 'Female', 'FEMALE': 'Female'
    }
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.strip().replace(gender_map)

    # Strip whitespace from all cat cols
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})

    # === 4. Impute missing values ===
    print("Imputing missing values...")
    
    # Numeric: median
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # Categorical: mode or 'Unknown'
    for col in CAT_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
            df[col] = df[col].fillna(fill_val)

    # === 5. Optimize dtypes ===
    df['family_size'] = df['family_size'].astype('int8')
    df['dependents'] = df['dependents'].astype('int8')
    df['requested_tenure'] = df['requested_tenure'].astype('int32')
    
    float_cols = [c for c in NUMERIC_COLS if c not in ['family_size', 'dependents', 'requested_tenure']]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype('float32')

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # === 6. Final validation ===
    assert df.isnull().sum().sum() == 0, "Nulls remain!"
    df.drop_duplicates(inplace=True)
    
    print(f"Final clean shape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"emi_eligibility distribution:\n{df['emi_eligibility'].value_counts()}")

    return df

def create_stratified_splits(df, test_size=0.15, val_size=0.15, random_state=42):
    from sklearn.model_selection import train_test_split
    
    X = df.drop(['emi_eligibility', 'max_monthly_emi'], axis=1)
    y_class = df['emi_eligibility']
    y_reg = df['max_monthly_emi']
    
    X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
        X, y_class, y_reg,
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=y_class
    )
    
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_class_temp, y_reg_temp,
        test_size=relative_val,
        random_state=random_state,
        stratify=y_class_temp
    )
    
    print(f"Splits → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return (X_train, X_val, X_test,
            y_class_train, y_class_val, y_class_test,
            y_reg_train, y_reg_val, y_reg_test)

if __name__ == "__main__":
    df = load_and_preprocess()
    splits = create_stratified_splits(df)
    
    # Save clean data
    clean_path = PROCESSED_PATH / "emi_clean.parquet"
    df.to_parquet(clean_path, index=False)
    print(f"Clean data saved to: {clean_path}")