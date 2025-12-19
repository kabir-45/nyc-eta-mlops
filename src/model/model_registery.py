import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import joblib
import os

dagshub_uri = os.getenv("MLFLOW_TRACKING_URI")
if not dagshub_uri:
    dagshub_uri = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(dagshub_uri)
mlflow.set_experiment(experiment_id="0")


def register_latest_model():
    print("🔍 CHECKING PATHS...")

    # 1. Update Path to match your actual structure: 'mlops_data/processed'
    folder = "mlops_data/processed"
    filename = "eta_features.parquet"  # Using the file you listed

    file_path = os.path.join(folder, filename)
    print(f"📄 Loading data from: {file_path}")

    if not os.path.exists(file_path):
        # Fallback for debugging if paths are still tricky
        print(f"❌ Error: File still not found at {os.path.abspath(file_path)}")
        return

    # 2. Load Data
    df = pd.read_parquet(file_path).sample(1000)

    # Ensure column names match what your model expects
    # If your model was trained on 'duration_minutes', rename or drop appropriately
    # Adjust this drop if your target column name is different!
    if "duration_minutes" in df.columns:
        y = df["duration_minutes"]
        X = df.drop(columns=["duration_minutes"])
    elif "ETA" in df.columns:
        y = df["ETA"]
        X = df.drop(columns=["ETA"])
    else:
        # Fallback: assume last column is target if names don't match
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    # 3. Load Model
    model_path = "models/production_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return

    model = joblib.load(model_path)

    # 4. Register
    print("🚀 Registering with MLflow...")
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X.head(1),
            signature=infer_signature(X, y),
            registered_model_name="TaxiPredictor"
        )
        print(f"✅ SUCCESS! Run ID: {run.info.run_id}")


if __name__ == "__main__":
    register_latest_model()