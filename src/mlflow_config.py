# src/mlflow_config.py
import mlflow
from pathlib import Path

def init_mlflow(experiment_name: str = "EMIPredict_AI"):
    tracking_uri = Path(r"E:\Labmentix\projects\EMIpredict\mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri.replace("file:///", "file:///"))  # Windows fix
    mlflow.set_experiment(experiment_name)
    print(f"MLflow ready → {tracking_uri}")