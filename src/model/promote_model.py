import mlflow
from mlflow.tracking import MlflowClient
import dagshub


def promote_model():
    # 1. Initialize Connection
    dagshub.init(repo_owner='kabir-45', repo_name='nyc-eta-mlops', mlflow=True)
    mlflow.set_experiment(experiment_id="0")

    client = MlflowClient()
    model_name = "NYC_Taxi_Predictor"

    # 2. Find the best run in Experiment 0 (The Challenger)
    print("🔍 Searching for runs in Experiment ID 0...")
    recent_runs = client.search_runs(
        experiment_ids=["0"],
        filter_string="status = 'FINISHED'",
        max_results=1,
        order_by=["start_time DESC"]
    )

    if not recent_runs:
        print("❌ Error: No finished runs found.")
        return

    challenger_run = recent_runs[0]
    challenger_id = challenger_run.info.run_id
    challenger_r2 = challenger_run.data.metrics.get("r2_score", 0)
    print(f"🥊 Challenger (New Model) R2 Score: {challenger_r2:.4f}")

    # 3. Get the current Production model (The Champion)
    champion_r2 = -1.0

    try:
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if latest_versions:
            champion_version = latest_versions[0]
            champion_run = client.get_run(champion_version.run_id)
            champion_r2 = champion_run.data.metrics.get("r2_score", -1.0)
            print(f"🏆 Champion (Current Prod) R2 Score: {champion_r2:.4f}")
        else:
            print("ℹ️ No model currently in Production. Challenger wins by default.")
    except Exception:
        print("ℹ️ Model not registered yet. Challenger wins by default.")

    # 4. Compare and Promote
    if challenger_r2 > champion_r2:
        print("✅ Challenger Wins! Promoting to Production...")

        # --- CRITICAL FIX START ---
        # Check if this Run ID is ALREADY registered as a model version
        # This prevents the "Unable to find logged_model" crash
        existing_versions = client.search_model_versions(f"run_id='{challenger_id}'")

        if existing_versions:
            target_version = existing_versions[0].version
            print(f"✅ Found existing Model Version: {target_version}")
        else:
            print("⚠️ No version found. Registering new model version...")
            model_uri = f"runs:/{challenger_id}/model"
            reg_model = mlflow.register_model(model_uri, model_name)
            target_version = reg_model.version
            print(f"✅ Successfully registered Version: {target_version}")
        # --- CRITICAL FIX END ---

        # Move to Production Stage
        print(f"🚀 Transitioning Version {target_version} to Production...")
        client.transition_model_version_stage(
            name=model_name,
            version=target_version,
            stage="Production",
            archive_existing_versions=True
        )
        print("🏆 Success! Model is now in Production.")

    else:
        print(f"❌ Challenger Failed. ({challenger_r2:.4f} <= {champion_r2:.4f})")
        print("Keeping the existing model in Production.")


if __name__ == "__main__":
    promote_model()