![Status](<https://img.shields.io/badge/Status-Work_in_progress_(Level_2)-yellow>)
[![CI Pipeline](https://github.com/mircohoehne/e2e-taxi-ride-duration-prediction/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mircohoehne/e2e-taxi-ride-duration-prediction/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/mircohoehne/e2e-taxi-ride-duration-prediction/graph/badge.svg?token=A4INWVJQDR)](https://codecov.io/github/mircohoehne/e2e-taxi-ride-duration-prediction)

An end-to-end MLOps project showcasing a complete machine learning pipeline using the NYC Taxi dataset. The goal is to predict taxi ride durations based on trip data and demonstrate modern MLOps practices, from data ingestion to model deployment and monitoring.

## MLOps Zoomcamp Reviewer

If you are a reviewer for the MLOps Zoomcamp, I tried to make your life easier by providing a [demo notebook](notebooks/00_demo.ipynb).

## Project Overview

This repository will implement a production-grade ML system with the following key features:

- Monthly data ingestion (Prefect)
- Data storage and versioning (DVC, later S3/DB)
- Feature engineering and preprocessing (polars, scikit-learn)
- Model training and evaluation (scikit-learn, XGBoost)
- Model registry integration (MLFlow)
- CI/CD pipelines for automation (Github Actions)
- Monitoring, alerting, and automatic retraining (Evidently, Prefect)
- Infrastructure-as-Code (IaC) (Terraform, AWS)

### Already implemented

- **Data ingestion pipeline** (`e2e_taxi_ride_duration_prediction/ingestion.py`)
  - NYC taxi data download and processing with Polars
  - Memory-efficient streaming processing for large datasets
  - Automatic data caching as parquet files

- **Data preprocessing** (`e2e_taxi_ride_duration_prediction/preprocessing.py`)
  - Feature engineering for pickup/dropoff locations
  - Categorical encoding and data validation
  - Duration calculation from timestamps

- **Model training** (`e2e_taxi_ride_duration_prediction/training.py`)
  - Baseline LinearRegression model implementation
  - Model serialization with joblib
  - Training metrics logging and evaluation

- **Model serving** (`e2e_taxi_ride_duration_prediction/serving/`)
  - FastAPI REST API for model predictions
  - Docker containerization for deployment
  - GitHub Container Registry integration

- **Development workflow**
  - Just commands for common tasks (`justfile`)
  - Comprehensive test suite with pytest
  - Code quality tools (ruff formatting/linting)
  - Pre-commit hooks for consistency

- **CI/CD pipeline**
  - GitHub Actions for automated testing
  - Docker image building and publishing
  - Code quality checks on pull requests and pushes on main branch

## Approach

The development will be iterative and roughly follow the [Machine Learning operations maturity model by Microsoft](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model) which defines the following levels (Table from the linked site):

| Level | Description                     | Highlights                                                                                                                                                                                                               | Technology                                                                                                                                                             |
| ----- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | No MLOps                        | - Difficult to manage full machine learning model lifecycle<br> - The teams are disparate and releases are painful<br> - Most systems exist as "black boxes," little feedback during/post deployment<br>                 | - Manual builds and deployments<br> - Manual testing of model and application<br> - No centralized tracking of model performance<br> - Training of model is manual<br> |
| 1     | DevOps but no MLOps             | - Releases are less painful than No MLOps, but rely on Data Team for every new model<br> - Still limited feedback on how well a model performs in production<br> - Difficult to trace/reproduce results<br>              | - Automated builds<br> - Automated tests for application code<br>                                                                                                      |
| 2     | Automated Training              | - Training environment is fully managed and traceable<br> - Easy to reproduce model<br> - Releases are manual, but low friction<br>                                                                                      | - Automated model training - Centralized tracking of model training performance<br> - Model management<br>                                                             |
| 3     | Automated Model Deployment      | - Releases are low friction and automatic<br> - Full traceability from deployment back to original data<br> - Entire environment managed: train > test > production<br>                                                  | - Integrated A/B testing of model performance for deployment<br> - Automated tests for all code<br> - Centralized tracking of model training performance<br>           |
| 4     | Full MLOps Automated Operations | - Full system automated and easily monitored<br> - Production systems are providing information on how to improve and, in some cases, automatically improve with new models<br> - Approaching a zero-downtime system<br> | - Automated model training and testing<br> - Verbose, centralized metrics from deployed model<br>                                                                      |

> Note: The first development iteration begins at Level 1. Level 0 is described for context only and is not implemented in any way.

## Quick Start

### Docker / Podman

```bash
docker run --network=host ghcr.io/mircohoehne/taxi-api:latest
```

or

```bash
podman run --network=host ghcr.io/mircohoehne/taxi-api:latest
```

### Clone Repo and run locally

Prerequisites:

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [just](https://github.com/casey/just) (command runner)

```bash
# Clone the repository
git clone https://github.com/smircs/e2e-taxi-ride-duration-prediction.git
cd e2e-taxi-ride-duration-prediction

# See all available commands
just

# Run a complete demo (setup + train + test + serve)
just serve-fresh
```

### Test API

```bash
# Example:
curl -X POST "http://localhost:8000/predict" \
      -H "Content-Type: application/json" \
      -d '{
    "PULocationID": 161,
    "DOLocationID": 236,
    "trip_distance": 2.5
  }'


```

## Roadmap

- [x] Level 1: DevOps, but no MLOps (Local)
- [ ] Level 2: Automated Training
- [ ] Level 3: Automated Model Deployment (Switch to Cloud / emulate with LocalStack)
- [ ] Level 4: Full MLOps Automated Operations

## Other resources used to build and learn about MLOps:

[mlops-zoomcamp - DataTalksClub](https://github.com/DataTalksClub/mlops-zoomcamp)<br>
[Designing Machine Learning Systems - Chip Huyen](https://bookgoodies.com/a/1098107969)
