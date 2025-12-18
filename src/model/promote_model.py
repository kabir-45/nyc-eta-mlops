import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os

dagshub_uri = os.getenv("MLFLOW_TRACKING_URI")
if not dagshub_uri:
    dagshub_uri = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(dagshub_uri)
mlflow.set_experiment("NYC-Taxi-Trip-Duration")


def promote_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("NYC-Taxi-Trip-Duration")
    recent_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        max_results=1,
        order_by=["start_time DESC"]
    )

    if not recent_runs:
        print("Error: No finished runs found.")
        return

    challenger_run = recent_runs[0]
    challenger_id = challenger_run.info.run_id
    challenger_r2 = challenger_run.data.metrics.get("r2_score", 0)
    print(f"🥊 Challenger (New Model) R2 Score: {challenger_r2:.4f}")

    # 2. Get the current Production model (The "Champion")
    model_name = "NYC_Taxi_Predictor"
    champion_r2 = -1.0  # Default if no model exists
    champion_version = None

    try:
        # Check if model is registered at all
        registered_models = client.search_registered_models(f"name='{model_name}'")
        if registered_models:
            # Look for version tagged "Production"
            latest_versions = client.get_latest_versions(model_name, stages=["Production"])
            if latest_versions:
                champion_version = latest_versions[0]
                champion_run_id = champion_version.run_id
                champion_run = client.get_run(champion_run_id)
                champion_r2 = champion_run.data.metrics.get("r2_score", -1.0)
                print(f"🏆 Champion (Current Prod) R2 Score: {champion_r2:.4f}")
            else:
                print("ℹ️ No model currently in Production. Challenger wins by default.")
    except Exception as e:
        print(f"⚠️ Warning during registry check: {e}")

    if challenger_r2 > champion_r2:
        print("✅ Challenger Wins! Promoting to Production...")

        # Register the model first
        model_uri = f"runs:/{challenger_id}/model"
        reg_model = mlflow.register_model(model_uri, model_name)

        # Move to Production Stage
        client.transition_model_version_stage(
            name=model_name,
            version=reg_model.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"🚀 Model version {reg_model.version} is now in Production!")

    else:
        print(f"❌ Challenger Failed. ({challenger_r2:.4f} <= {champion_r2:.4f})")
        print("Keeping the existing model in Production.")


if __name__ == "__main__":
    promote_model()