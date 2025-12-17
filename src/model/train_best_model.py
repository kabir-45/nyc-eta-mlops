import sys
import os
import pandas as pd
import joblib
import mlflow
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import yaml


# Load Params Helper
def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def train_production_model():
    params = load_params()

    # Paths
    processed_data_path = params['preprocessing']['processed_path']  # e.g. mlops_data/processed/eta_features.parquet
    model_path = params['training']['model_path']  # The model saved by selection
    target_col = params['training']['target_col']

    # ➤ FIX: Use local DB file for CI/CD compatibility
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("NYC-Taxi-Trip-Duration")

    with mlflow.start_run(run_name="production_model_training"):

        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data not found at {processed_data_path}")

        df = pd.read_parquet(processed_data_path)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Load the best model found during selection
        if not os.path.exists(model_path):
            # Fallback: If best_model.pkl isn't there, check for a 'models/' dir
            raise FileNotFoundError(f"Best model not found at {model_path}. Run model_selection.py first.")

        print(f"🚀 Loading best model from {model_path}...")
        loaded_model = joblib.load(model_path)

        # Clone and Retrain
        production_model = clone(loaded_model)
        production_model.fit(X, y)

        train_preds = production_model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, train_preds))
        r2 = r2_score(y, train_preds)

        print(f"✅ Production Model Trained. R2: {r2:.4f}")

        mlflow.log_metric("production_rmse", rmse)
        mlflow.log_metric("production_r2", r2)
        mlflow.sklearn.log_model(production_model, "model")  # Name artifact 'model' for consistency

        # Save the final artifact
        prod_path = "models/best_model.pkl"  # Ensuring consistent naming
        joblib.dump(production_model, prod_path)
        print(f"💾 Saved final production model to {prod_path}")


if __name__ == "__main__":
    train_production_model()