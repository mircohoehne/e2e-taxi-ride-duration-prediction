_default:
    @just --list

# Development setup with all dependencies including dev tools
dev:
    uv sync --dev
    @printf "Install pre-commit hooks[y/N]? " && read ans && [ "$$ans" = "y" ] && pre-commit install || true

# Build Docker image
docker-build:
    docker build -t taxi-ride-duration-prediction-api -f e2e_taxi_ride_duration_prediction/serving/dockerfile . --load

# Run Docker container
docker-run:
    docker run --rm -p 8000:8000 taxi-ride-duration-prediction-api

# Start FastAPI Server (default port 8000)
serve:
    uv run fastapi run e2e_taxi_ride_duration_prediction/serving/main.py --host 0.0.0.0 --port 8000

# Run setup, train model and serve the model
serve-fresh: setup train serve

# Serve baseline model training flow with prefect
serve-prefect: start-prefect
    uv run prefect flow serve scripts/train_model.py:main --name taxi-model-baseline-training

# Setup without dev dependencies
setup:
    uv sync --no-dev

# Start Prefect server in background and set prefect config to API URL
start-prefect:
    uv run prefect server start -b && prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

# Start mlflow server
mlflow:
    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root /mlruns --host 0.0.0.0

# Run pytest
test:
    uv run pytest

# Report modules missing tests with line numbers
test-miss:
    uv run pytest --cov-report term-missing:skip-covered --cov=e2e_taxi_ride_duration_prediction --cov-branch

# Report test coverage
test-cov:
    uv run pytest --cov=e2e_taxi_ride_duration_prediction --cov-branch

# Train baseline model with data from Jan/Feb 2025, Validate on Mar 2025 data
train:
    uv run scripts/train_model.py

# Start prefect workflow for baseline model training
train-prefect:
    uv run prefect deployment run 'main/taxi-model-baseline-training'
