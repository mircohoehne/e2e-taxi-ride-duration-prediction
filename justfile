_default:
    @just --list

# Development setup with all dependencies including dev tools
dev:
    uv sync --dev
    @printf "Install pre-commit hooks[y/N]? " && read ans && [ "$$ans" = "y" ] && pre-commit install || true

# Setup without dev dependencies
setup:
    uv sync --no-dev

# Train baseline model with data from Jan/Feb 2025, Validate on Mar 2025 data
train:
    uv run scripts/train_model.py

# Start FastAPI Server (default port 8000)
serve:
    uv run fastapi run e2e_taxi_ride_duration_prediction/serving/main.py --host 0.0.0.0 --port 8000

# Build Docker image
docker-build:
    docker build -t taxi-ride-duration-prediction-api -f e2e_taxi_ride_duration_prediction/serving/dockerfile . --load

# Run Docker container
docker-run:
    docker run --rm -p 8000:8000 taxi-ride-duration-prediction-api

# Run pytest
test:
    uv run pytest

# Run setup, train model and serve the model
serve-fresh: setup train serve

# Start Prefect server
start-prefect:
    uv run prefect server start

# Serve baseline model training flow with prefect
serve-prefect:
    uv run prefect flow serve scripts/train_model.py:main --name taxi-model-baseline-training

# Start prefect workflow for baseline model training
train-prefect:
    uv run prefect deployment run 'main/taxi-model-baseline-training'

# Report modules missing tests with line numbers
test-miss:
    uv run pytest --cov-report term-missing:skip-covered --cov=e2e_taxi_ride_duration_prediction --cov-branch

# Report test coverage
test-cov:
    uv run pytest --cov=e2e_taxi_ride_duration_prediction --cov-branch
