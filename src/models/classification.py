# src/models/classification.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow.sklearn
from mlflow.models import infer_signature

from src.mlflow_config import init_mlflow
from src.feature_engineering import engineer_features, get_feature_columns

init_mlflow()

df = pd.read_parquet("data/processed/emi_clean.parquet")
df = engineer_features(df)
X = df[get_feature_columns()]
y = df["emi_eligibility"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

cat_cols = X.select_dtypes("category").columns.tolist()
num_cols = X.select_dtypes("number").columns.tolist()

prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        n_jobs=-1, random_state=42, eval_metric="mlogloss"
    ),
}

for name, clf in models.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        prob = pipe.predict_proba(X_val)

        acc = accuracy_score(y_val, pred)
        f1  = f1_score(y_val, pred, average="macro")
        auc = roc_auc_score(y_val, prob, multi_class="ovr")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("auc_ovr", auc)

        # ---- CRITICAL FIX ----
        signature = infer_signature(X_val, prob)
        input_example = X_val.iloc[:1]

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",                 # <-- critical
            signature=infer_signature(X_val, pred),   # or pred_proba for clf
            input_example=X_val.iloc[:1]
        )
        # ---------------------

        print(f"{name} → Acc {acc:.4f} | F1 {f1:.4f} | AUC {auc:.4f}")