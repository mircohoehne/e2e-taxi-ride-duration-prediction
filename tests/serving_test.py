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
