import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import joblib
import os
import yaml


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("NYC-Taxi-Trip-Duration")

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def register_latest_model():
    params = load_params()
    target_col = params["training"]["target_col"]

    file_path = "mlops_data/processed/eta_features.parquet"

    if not os.path.exists(file_path):
        print(f"❌ File not found at {file_path}. Checking directories...")
        if os.path.exists("mlops_data"):
             print(f"Files in mlops_data: {os.listdir('mlops_data')}")
        raise FileNotFoundError(f"Could not find file at: {file_path}")

    df = pd.read_parquet(file_path).sample(100) # Sample is enough for signature

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {df.columns}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model at: {model_path}")
    model = joblib.load(model_path)

    with mlflow.start_run(run_name="Model_Registration") as run:
        mlflow.log_param("model_type", type(model).__name__)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X.head(1),
            signature=infer_signature(X, y),
            registered_model_name="NYC_Taxi_Predictor"
        )

if __name__ == "__main__":
    register_latest_model()