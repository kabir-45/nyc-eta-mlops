import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("eta_prediction")


def load_dataset(path="mlops_data/processed/eta_features.parquet"):
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    return pd.read_parquet(path)

def build_preprocessor(df_columns):
    potential_cats = [
        'VendorID', 'passenger_count', 'store_and_fwd_flag',
        'PULocationID', 'DOLocationID', 'day_of_week', 'pickup_hour', 'RatecodeID'
    ]

    potential_nums = [
        'trip_distance'
    ]

    cat_cols = [c for c in potential_cats if c in df_columns]
    num_cols = [c for c in potential_nums if c in df_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ],
        remainder='passthrough'
    )

    return preprocessor

def evaluate_model(model_name, model, preprocessor, x_train, x_val, y_train, y_val, x_test, y_test):
    results = {}
    results["model_name"] = model_name

    # Build pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, x_val, y_val, cv=kf, scoring="r2")

    results["cv_r2"] = cv_scores.mean()

    # ---- Fit on training set ----
    pipeline.fit(x_train, y_train)

    # ---- Evaluate on test set ----
    preds = pipeline.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results["test_rmse"] = rmse
    results["test_r2"] = r2
    results["model"] = pipeline

    return results

def run_model_selection():
    df = load_dataset()
    if df is None: return

    # Target
    target_col = "ETA"
    if target_col not in df.columns:
        print(f"Error: Target {target_col} not found in dataset.")
        return

    y = df[target_col]
    X = df.drop(columns=[target_col])

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    x_val, y_val = x_train, y_train

    models = {
        "LinearRegression": LinearRegression(),
        "XGBoost": XGBRegressor(objective="reg:squarederror", n_jobs=-1),
        "CatBoost": CatBoostRegressor(verbose=False, allow_writing_files=False),
        "LightGBM": LGBMRegressor(n_jobs=-1, verbose=-1)
    }

    # Pass columns to builder to ensure safety
    preprocessor = build_preprocessor(X.columns)

    best_model_info = None
    best_rmse = float("inf")
    results_table = []

    # MLflow experiment
    mlflow.set_experiment("eta_model_selection")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            result = evaluate_model(
                name, model, preprocessor,
                x_train, x_val, y_train, y_val,
                x_test, y_test
            )

            # Log metrics
            mlflow.log_metric("cv_r2", result["cv_r2"])
            mlflow.log_metric("test_rmse", result["test_rmse"])
            mlflow.log_metric("test_r2", result["test_r2"])

            # Log model artifact (scikit-learn flavor handles pipelines)
            mlflow.sklearn.log_model(result["model"], artifact_path="model")

            results_table.append(result)

            if result["test_rmse"] < best_rmse:
                best_rmse = result["test_rmse"]
                best_model_info = result

    os.makedirs("models", exist_ok=True)
    save_path = "models/best_model.pkl"
    joblib.dump(best_model_info["model"], save_path)

    return best_model_info


if __name__ == "__main__":
    run_model_selection()