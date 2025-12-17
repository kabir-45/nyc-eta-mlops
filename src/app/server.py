import sys
import os
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class TaxiTripInput(BaseModel):
    VendorID: int = Field(1, example=1)
    tpep_pickup_datetime: str = Field(..., example="2025-01-01 12:00:00")
    passenger_count: float = Field(1.0, example=1.0)
    trip_distance: float = Field(..., example=2.5)
    RatecodeID: float = Field(1.0, example=1.0)
    PULocationID: int = Field(..., example=100)
    DOLocationID: int = Field(..., example=102)
    payment_type: int = Field(1, example=1)

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    model_path = "models/production_model.pkl"
    if os.path.exists(model_path):
        models["production"] = joblib.load(model_path)
    else:
        models["production"] = None
    yield
    models.clear()


app = FastAPI(title="NYC Taxi ETA Predictor", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)
@app.get("/")
def read_root():
    return {"Hello": "World"}

def preprocess_input(data: TaxiTripInput) -> pd.DataFrame:
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])

    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.weekday

    return df


@app.post("/predict")
def predict_eta(trip: TaxiTripInput):
    if models["production"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        input_df = preprocess_input(trip)

        # Predict
        prediction = models["production"].predict(input_df)[0]

        return {
            "predicted_duration_minutes": round(float(prediction), 2),
            "estimated_arrival": (
                    input_df['tpep_pickup_datetime'].iloc[0] +
                    pd.Timedelta(minutes=float(prediction))
            ).strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)