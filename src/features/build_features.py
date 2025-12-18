import os
import sys
import pandas as pd
import mlflow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.features.build_features_transformer import BuildFeatures
from src.utils import load_params

params = load_params()

def run_feature_pipeline(raw_path: str, processed_path: str):
    dagshub_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not dagshub_uri:
        dagshub_uri = "sqlite:///mlflow.db"

    mlflow.set_tracking_uri(dagshub_uri)
    mlflow.set_experiment("eta_feature_engineering")

    with mlflow.start_run(run_name="build_features"):

        try:
            df = pd.read_parquet(raw_path)
        except Exception:
            df = pd.read_csv(raw_path)

        mlflow.log_metric("raw_rows", len(df))
        mlflow.log_metric("raw_columns", df.shape[1])

        transformer = BuildFeatures()
        transformer.fit(df)

        mlflow.log_params({f"{k}_low": v for k, v in transformer.lower_bounds.items()})
        mlflow.log_params({f"{k}_high": v for k, v in transformer.upper_bounds.items()})

        processed_df = transformer.transform(df)

        mlflow.log_metric("processed_rows", len(processed_df))
        mlflow.log_metric("processed_columns", processed_df.shape[1])

        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        processed_df.to_parquet(processed_path, index=False)
        mlflow.log_param("processed_path", processed_path)
        return processed_path


if __name__ == "__main__":
    raw_path = params['ingestion']['raw_data_path']
    processed_path = params['preprocessing']['processed_path']

    run_feature_pipeline(raw_path, processed_path)