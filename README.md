# NYC-Taxi End-to-End MLOps Demo

[![CI Pipeline](https://github.com/mircohoehne/e2e-taxi-ride-duration-prediction/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mircohoehne/e2e-taxi-ride-duration-prediction/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/mircohoehne/e2e-taxi-ride-duration-prediction/graph/badge.svg?token=A4INWVJQDR)](https://codecov.io/github/mircohoehne/e2e-taxi-ride-duration-prediction)

> **TL;DR**: End-to-end MLOps project that ingests NYC taxi data, preprocesses with Polars, trains a baseline model, tracks runs in MLflow, serves predictions with FastAPI, and generates an Evidently drift report. CI that runs lint, formatting, checks for secrets, big files and runs tests. Manual CD trigger that uploads the containerized model to GitHub Container Registry. The project is built with a focus on MLOps practices, from data ingestion to model deployment and monitoring. Modeling is not the focus of this project, but rather the MLOps practices and engineering side of things, therefore the model is very basic, which may change in the future.

## Key features

| Feature                | Description                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| End-to-end pipeline    | Ingestion, preprocessing, training; modularized as Prefect tasks/flows for easy orchestration |
| Experiment tracking    | MLflow tracking with autolog; extended where autolog wasn't sufficient                        |
| Serving                | FastAPI app with '/predict' endpoint, Pydantic validation, containerized; Swagger at '/docs'  |
| Justfile               | For easy command runs and smooth DX; run `just` in the repo root to see all commands          |
| Tests and typing       | Typed Python; pytest across modules; coverage reported in CI                                  |
| CI                     | GitHub Actions: linting, tests, type checker[^1], and Codecov coverage                        |
| CD (containerized API) | Manually-triggered GitHub Action publishes ghcr.io/mircohoehne/taxi-api:latest                |
| Monitoring             | Evidently script producing HTML drift and regression metric reports                           |
| Pre-commit hooks       | Formatting, linting, testing, secret detection, conventional commits, and more                |
| IaC demo               | Terraform spins up an EC2 instance and exposes the containerized API endpoint                 |

[^1]: Typchecking is done with [ty](https://docs.astral.sh/ty/) (a fast Typechecker written in Rust) which only has a pre-release version. In Production I would use a robust typechecker like `mypy`.

## Quick Start

Prerequisites:

- uv
- just (optional, you can look at the `justfile` and run the commands manually)

To see available just commands, run `just` in the root directory

### Run published Container

You don't need to clone the repo. Just run

```bash
docker run -p 8000:8000 ghcr.io/mircohoehne/taxi-api:latest
```

and hit the API at `http://localhost:8000/predict` with a POST request containing the required JSON payload.
For example:

```bash
curl -X POST "http://localhost:8000/predict" \
      -H "Content-Type: application/json" \
      -d '{
    "PULocationID": 161,
    "DOLocationID": 236,
    "trip_distance": 2.5
  }'
```

### Local

Clone the repo and run:

```bash
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
just serve-fresh
```

This will download the data, preprocess it, train the baseline model, and start an FastAPI server on port 8000.
Then you can test the API with the same command as above.

### Cloud (AWS)

Prerequisites:

- aws cli configured with access to an AWS account

Clone the repo and run:

```bash
terraform init
terraform apply

```

The script will deploy the API to an AWS EC2 instance and show you the public IP address of the instance. You can then access the API and hit the api with the following command, replacing `<<public-ip>>` with the actual public IP address of the EC2 instance:

```bash
curl -X POST "http://<<public-ip>>:8000/predict" \
      -H "Content-Type: application/json" \
      -d '{
    "PULocationID": 161,
    "DOLocationID": 236,
    "trip_distance": 2.5
  }'
```

To destroy the infrastructure after using it, run:

```bash
terraform destroy
```

## Project layout

```
e2e-taxi-ride-duration-prediction/
├── .github/
│   └── workflows/
│       └── ci.yml                    # Continuous Integration workflow
│       ├── cd.yml                    # Continuous Deployment workflow
|-- data/
├── e2e_taxi_ride_duration_prediction/
│   ├── serving/
│   │   ├── dockerfile                # Docker configuration for API serving
│   │   └── main.py                   # FastAPI application with prediction endpoint
│   ├── __init__.py
│   ├── ingestion.py                  # Data download pipeline
│   ├── mlflow_utils.py               # MLflow setup utilities
│   ├── models.py                     # Model Protocol definition for typing
│   ├── monitoring.py                 # Evidently drift detection and monitoring
│   ├── preprocessing.py              # Data preprocessing
│   └── training.py                   # Model training and evaluation
├── mlruns/                           # MLflow experiment tracking artifacts
│   ├── mlflow.db                     # SQLite database for MLflow metadata
│   └── models/                       # MLflow model registry
├── models/                           # Saved model artifacts
├── notebooks/
│   ├── 00_demo.ipynb                 # Project demonstration notebook
│   ├── 01_baseline.ipynb             # Baseline model development
│   ├── 02_monitoring.ipynb           # Monitoring and drift analysis
│   └── 99_scratch.ipynb              # Experimental/scratch work
├── reports/                          # Generated monitoring reports (HTML)
├── scripts/
│   ├── prefect_deployment.py         # Prefect workflow deployment
│   └── train_model.py                # Training script for production
├── terraform/
│   └── main.tf                       # Infrastructure as Code for AWS deployment
├── tests/
│   ├── conftest.py                   # Pytest configuration and fixtures
│   ├── *_test.py                     # Unit tests for all modules
├── docker-compose.yaml               # Multi-service container orchestration
├── justfile                          # Command runner for development workflow
├── pyproject.toml                    # Python project configuration and dependencies
├── README.md                         # Project documentation
└── uv.lock                           # Dependency lock file
```

## Model tracking

To setup local model tracking with mlflow, just import the setup function from `mlflow_utils.py` and call it in your training script (with optional parameters for tracking URI, experiment name and autolog parameters). Then run an mlflow run with the context manager to log your runs.

## Data / Model Monitoring

For a demonstration of the monitoring you can refer to the following notebook: [02_monitoring.ipynb](notebooks/02_monitoring.ipynb), which also includes a sample report.
Alternatively Monitoring can be deployed as a Prefect task and run on a schedule.

## Orchestration with Prefect

1. Start a prefect server with `prefect server start`
2. For local development set the prefect API URL with `prefect config set PREFECT_API_URL="http://localhost:<<port>>/api"`
3. Serve the flow you want to execute, for example `uv run prefect flow serve scripts/train_model.py:main --name taxi-model-baseline-training` for the main function of the train_model.py script.
4. Run the flow with

## Future Improvements

- add complexer models
- add automatic hyperparameter tuning
- reduce container size
- create s3 Bucket for data / mlflow artifacts
- Move evidently server, prefect server and mlflow server to cloud
- add docstrings and generate docs with mkdocs
