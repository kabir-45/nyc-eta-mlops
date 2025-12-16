import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import joblib
import os

# 1. Connect to the MLflow Server (Port 5001)
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("NYC_Taxi_Registry_Demo")


def register_latest_model():
    print("🔍 Debugging paths...")
    print(f"Current Working Directory: {os.getcwd()}")

    # Check if directory exists
    data_path = "data/processed"
    if os.path.exists(data_path):
        print(f"Files in '{data_path}': {os.listdir(data_path)}")
    else:
        print(f"❌ Directory '{data_path}' does not exist!")

    # 2. Load Data (NO TRY/EXCEPT BLOCK - Let it crash if it fails!)
    print("🚀 Attempting to load data...")
    # Common names: 'train.parquet', 'train_data.parquet', 'processed_data.parquet'
    # adjusting strictly to what your pipeline likely produced:
    file_path = "data/processed/train.parquet"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file at: {file_path}")

    df = pd.read_parquet(file_path).sample(1000)

    X = df.drop(columns=["ETA"])  # Ensure this column matches your data
    y = df["ETA"]

    # 3. Load the Model
    model_path = "models/production_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model at: {model_path}")

    model = joblib.load(model_path)

    # 4. Register
    print("🚀 Registering Model...")
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", type(model).__name__)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X.head(1),
            signature=infer_signature(X, y),
            registered_model_name="TaxiPredictor"
        )

        print(f"✅ Model registered! Run ID: {run.info.run_id}")


if __name__ == "__main__":
    register_latest_model()