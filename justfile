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
    docker build -t taxi-ride-duration-prediction-api -f e2e_taxi_ride_duration_prediction/serving/dockerfile .

# Run Docker container
docker-run:
    docker run --rm -p 8000:8000 taxi-ride-duration-prediction-api

# Run pytest
test:
    uv run pytest

# Run setup, train model and serve the model
serve-fresh: setup train serve
