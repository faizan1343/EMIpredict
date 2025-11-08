# src/models/baseline.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from src.mlflow_config import init_mlflow
from src.feature_engineering import engineer_features, get_feature_columns

# ------------------------------------------------------------------
init_mlflow()
df = pd.read_parquet("data/processed/emi_clean.parquet")
df = engineer_features(df)

X = df[get_feature_columns()]
y = df['emi_eligibility']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cat_cols = X.select_dtypes(include='category').columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])

model = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
])

with mlflow.start_run(run_name="Logistic_Baseline"):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)
    
    report = classification_report(y_val, preds, output_dict=True)
    mlflow.log_metric("accuracy",  report['accuracy'])
    mlflow.log_metric("f1_macro", report['macro avg']['f1-score'])
    mlflow.log_metric("auc_ovr",  roc_auc_score(y_val, probs, multi_class='ovr'))
    
    mlflow.sklearn.log_model(model, "logistic_baseline")
    print("\nLogistic Baseline logged!")
    print(f"Accuracy : {report['accuracy']:.4f}")
    print(f"F1-macro : {report['macro avg']['f1-score']:.4f}")
    print(f"AUC-OVR  : {roc_auc_score(y_val, probs, multi_class='ovr'):.4f}")