import mlflow
from mlflow.tracking import MlflowClient
import os
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
model_name = "NYC_Taxi_ETA_Model"


def promote_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # 1. Get the Experiment
    experiment = client.get_experiment_by_name("NYC-Taxi-Trip-Duration")
    if experiment is None:
        print("❌ Error: Experiment 'NYC-Taxi-Trip-Duration' not found.")
        return

    # 2. Get the latest run (Candidate Model)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1
    )

    if not runs:
        print("❌ Error: No runs found in the experiment.")
        return

    latest_run = runs[0]  # The variable that was causing issues
    new_rmse = latest_run.data.metrics.get("rmse")

    if new_rmse is None:
        print("❌ Error: Could not find 'rmse' metric in the latest run.")
        return

    new_model_uri = f"runs:/{latest_run.info.run_id}/model"

    print(f"🆕 Candidate Model RMSE: {new_rmse}")

    # 3. Get the current Production Model (if exists)
    current_rmse = float('inf')  # Default to infinity so first model always wins
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            current_prod_version = versions[0]
            current_run_id = current_prod_version.run_id
            current_run = client.get_run(current_run_id)
            current_rmse = current_run.data.metrics.get("rmse")
            print(f"🏭 Current Production RMSE: {current_rmse}")
        else:
            print("ℹ️ No model currently in Production.")
    except Exception as e:
        print(f"ℹ️ No Production model found (Exception: {e}). Promoting candidate...")

    # 4. Compare Logic (Lower RMSE is better)
    if new_rmse < current_rmse:
        print(f"🚀 Improvement detected! ({new_rmse} < {current_rmse}). Promoting...")
        register_and_transition(new_model_uri, client)
    else:
        print(f"📉 No improvement ({new_rmse} >= {current_rmse}). Rejecting candidate.")


def register_and_transition(model_uri, client):
    mv = mlflow.register_model(model_uri, model_name)

    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"✅ Model version {mv.version} is now in Production!")


if __name__ == "__main__":
    promote_model()