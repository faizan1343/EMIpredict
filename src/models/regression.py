# src/models/regression.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow.sklearn
from mlflow.models import infer_signature

from src.mlflow_config import init_mlflow
from src.feature_engineering import engineer_features, get_feature_columns

init_mlflow()

# ------------------------------------------------------------------
# Load & Engineer
# ------------------------------------------------------------------
df = pd.read_parquet("data/processed/emi_clean.parquet")
df = engineer_features(df)
X = df[get_feature_columns()]
y = df["max_monthly_emi"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# Preprocessor
# ------------------------------------------------------------------
cat_cols = X.select_dtypes("category").columns.tolist()
num_cols = X.select_dtypes("number").columns.tolist()

prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        n_jobs=-1, random_state=42
    ),
}

# ------------------------------------------------------------------
# Train + Log
# ------------------------------------------------------------------
for name, reg in models.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([("prep", prep), ("reg", reg)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, pred))
        mae  = mean_absolute_error(y_val, pred)
        r2   = r2_score(y_val, pred)

        # Log metrics
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # ---- CRITICAL FIX ----
        # 1. Use artifact_path="model"
        # 2. Add signature + input example
        signature = infer_signature(X_val, pred)
        input_example = X_val.iloc[:1]

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",                 # <-- critical
            signature=infer_signature(X_val, pred),   # or pred_proba for clf
            input_example=X_val.iloc[:1]
        )
        # ---------------------

        print(f"{name} → RMSE {rmse:.1f} | MAE {mae:.1f} | R² {r2:.4f}")