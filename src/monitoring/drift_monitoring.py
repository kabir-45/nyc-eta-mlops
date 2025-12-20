import sys
import os
import pandas as pd
import mlflow
import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils import load_params


def detect_drift():
    reference_path = "mlops_data/processed/eta_features.parquet"
    current_path = "mlops_data/processed/eta_featuresV2.parquet"

    if not os.path.exists(reference_path) or not os.path.exists(current_path):
        return

    reference_data = pd.read_parquet(reference_path)

    current_data = pd.read_parquet(current_path)

    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]

    report = Report(metrics=[
        DataDriftPreset(),
    ],include_tests=True)

    report.run(reference_data=reference_data, current_data=current_data)

    report_path = "drift_report.html"
    report.save_html(report_path)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("eta_drift_monitoring")

    with mlflow.start_run(run_name="drift_check"):
        mlflow.log_artifact(report_path)

        result = report.as_dict()
        drift_share = result['metrics'][0]['result']['drift_share']
        dataset_drift = result['metrics'][0]['result']['dataset_drift']

        mlflow.log_metric("drift_share", drift_share)
        mlflow.log_param("dataset_drift_detected", dataset_drift)

if __name__ == "__main__":
    detect_drift()