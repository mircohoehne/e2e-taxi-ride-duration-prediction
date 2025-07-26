from fastapi.testclient import TestClient

from e2e_taxi_ride_duration_prediction.serving.main import app


def test_predict_endpoint():
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"PULocationID": 132, "DOLocationID": 148, "trip_distance": 3.1},
    )
    assert response.status_code == 200
    assert "predicted_duration" in response.json()


def test_predict_missing_fields():
    client = TestClient(app)
    response = client.post("/predict", json={"PULocationID": 132})
    assert response.status_code == 422


def test_predict_invalid_types():
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"PULocationID": "invalid", "DOLocationID": 148, "trip_distance": 3.1},
    )
    assert response.status_code == 422
