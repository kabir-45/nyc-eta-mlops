import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import joblib
import os
import dagshub

def register_latest_model():
    dagshub.init(repo_owner='kabir-45', repo_name='nyc-eta-mlops', mlflow=True)
    mlflow.set_experiment(experiment_id="0")

    folder = "mlops_data/processed" # feature engineered data
    filename = "eta_features.parquet"  # using the file

    file_path = os.path.join(folder, filename)
    print(f"📄 Loading data from: {file_path}")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {os.path.abspath(file_path)}")
        return

    df = pd.read_parquet(file_path).sample(1000)

    if "ETA" in df.columns:
        y = df["ETA"]
        X = df.drop(columns=["ETA"])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    # Load Model
    model_path = "models/production_model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = joblib.load(model_path)

    print(" Registering with MLflow...")
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X.head(1),
            signature=infer_signature(X, y),
            registered_model_name="TaxiPredictor"
        )
        print(f" SUCCESS! Run ID: {run.info.run_id}")


if __name__ == "__main__":
    register_latest_model()