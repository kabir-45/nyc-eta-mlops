import sys
import os
import pandas as pd
import joblib
import mlflow
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import yaml
import dagshub


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def train_production_model():
    dagshub.init(repo_owner='kabir-45', repo_name='nyc-eta-mlops', mlflow=True)
    mlflow.set_experiment(experiment_id="0")

    # Load Params
    params = load_params()
    processed_data_path = params['preprocessing']['processed_path']
    model_path = "models/best_model.pkl"  # Hardcode or ensure param matches this
    target_col = params['training']['target_col']

    # Start MLflow Run
    with mlflow.start_run(run_name="train_production") as run:
        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data not found at {processed_data_path}")

        df = pd.read_parquet(processed_data_path)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Best model not found at {model_path}. Run model_selection first.")

        print(f"Loading best model from {model_path}...")
        loaded_model = joblib.load(model_path)

        # Retrain on Full Data
        production_model = clone(loaded_model)
        production_model.fit(X, y)

        # Calculate metrics
        train_preds = production_model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, train_preds))
        r2 = r2_score(y, train_preds)

        print(f"✅ Production Model Trained. R2: {r2:.4f}")

        # LOG TO MLFLOW
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("production_rmse", rmse)

        # Log the model artifact as "model"
        mlflow.sklearn.log_model(
            sk_model=production_model,
            artifact_path="model",
            registered_model_name="NYC_Taxi_Predictor"
        )

        # Save Local File
        prod_path = "models/production_model.pkl"
        joblib.dump(production_model, prod_path)
        print(f"💾 Saved final production model to {prod_path}")
        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    train_production_model()