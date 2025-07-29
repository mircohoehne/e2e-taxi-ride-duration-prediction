from pathlib import Path

import joblib
import polars as pl
from fastapi import FastAPI
from pydantic import BaseModel

from e2e_taxi_ride_duration_prediction.preprocessing import create_pickup_dropoff_pairs

pl.Config.set_engine_affinity("streaming")

app = FastAPI()


class TaxiRideRequest(BaseModel):
    PULocationID: int
    DOLocationID: int
    trip_distance: float


class TaxiRidePrediction(BaseModel):
    predicted_duration: float


@app.post("/predict")
def predict_duration(request: TaxiRideRequest):
    lf = create_pickup_dropoff_pairs(
        pl.LazyFrame(
            {
                "PULocationID": [request.PULocationID],
                "DOLocationID": [request.DOLocationID],
                "trip_distance": [request.trip_distance],
            }
        )
    )

    root = Path(__file__).parents[2]
    with open(
        (root / "models/baseline_taxi_duration_model_and_vectorizer.joblib"), "rb"
    ) as f:
        model, dict_vectorizer = joblib.load(f)

    X_dicts = lf.collect().to_dicts()
    X_test = dict_vectorizer.transform(X_dicts)
    prediction = model.predict(X_test)

    return TaxiRidePrediction(predicted_duration=prediction[0])
