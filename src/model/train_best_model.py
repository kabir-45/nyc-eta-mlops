import sys
import os
import pandas as pd
import joblib
import mlflow
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils import load_params


def train_production_model():
    params = load_params()

    # Paths
    processed_data_path = params['preprocessing']['processed_path']
    model_path = params['training']['model_path']  # The one saved by selection
    target_col = params['training']['target_col']

    # MLflow Setup
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("eta_production_training")

    with mlflow.start_run(run_name="production_model_training"):

        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data not found at {processed_data_path}")

        df = pd.read_parquet(processed_data_path)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Best model not found at {model_path}. Run model_selection.py first.")


        loaded_model = joblib.load(model_path)
        production_model = clone(loaded_model)
        production_model.fit(X, y)
        train_preds = production_model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, train_preds))
        r2 = r2_score(y, train_preds)
        mlflow.log_metric("production_rmse", rmse)
        mlflow.log_metric("production_r2", r2)
        mlflow.sklearn.log_model(production_model, "production_model")

        prod_path = "models/production_model.pkl"
        joblib.dump(production_model, prod_path)


if __name__ == "__main__":
    train_production_model()