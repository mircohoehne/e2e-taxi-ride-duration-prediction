# NYC-Taxi End-to-End MLOps Demo

![Status](<https://img.shields.io/badge/Status-Work_in_progress_(Level_2)-yellow>)
[![CI Pipeline](https://github.com/mircohoehne/e2e-taxi-ride-duration-prediction/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mircohoehne/e2e-taxi-ride-duration-prediction/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/mircohoehne/e2e-taxi-ride-duration-prediction/graph/badge.svg?token=A4INWVJQDR)](https://codecov.io/github/mircohoehne/e2e-taxi-ride-duration-prediction)

> **TL;DR**: End-to-end MLOps project that ingests NYC taxi data, preprocesses with Polars, trains a baseline model, tracks runs in MLflow, serves predictions with FastAPI, and generates an Evidently drift report. CI that runs lint, formatting, checks for secrets, big files and runs tests. Manual CD trigger that uploads the containerized model to GitHub Container Registry. The project is built with a focus on MLOps practices, from data ingestion to model deployment and monitoring. Modeling is not the focus of this project, but rather the MLOps practices and infrastructure, therefore the model is very basic, which may change in the future.

<!-- TODO: insert gif/loom here -->

## Key features

| Feature                | Description                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- |
| End to end pipeline    | Ingestion, preprocessing, training, all modular and available as Prefect tasks and flows for easy orchestration     |
| Experiment tracking    | MLflow tracking with autolog and extended where autolog wasn't sufficient                                           |
| Serving                | FastAPI app with '/predict' endpoint, Pydantic validation, containerized, swagger at '/docs'                        |
| Jusfile                | For easy command runs and smooth DX, run `just` in root directory to see all commands                               |
| Tests and typing       | Typed Python, pytest across all modules, coverage reported in CI                                                    |
| CI                     | GitHub Actions that run linting, tests and report code coverage with Codecov                                        |
| CD (containerized API) | Manually triggered GitHub Action that containerizes the API and publishes it to ghcr.io/mircohoehne/taxi-api:latest |
| Monitoring             | Evidently script, that produces an HTML Drift- and Regression-Metric-Report                                         |
| IaC demo               | Terraform script that spins up an EC2 instance and exposes the containerized api endpoint                           |

## Architecture

## Quick Start

Prerequisites:

- uv
- just (optional, you can look at the `justfile` and run the commands manually)

To see available just commands, run `just` in the root directory

### Run published Container

### Local (Docker)

### Cloud (AWS)

## Project layout

## Reproducing Experiments

## Monitoring drift and performance

## What would be next steps?

- make container slimmer
- add mypy type enforcement
- change mlflow sqlite to a more robust database like Postgres
- create s3 Bucket for data / mlflow artifacts

## Learnings / Future Improvements

### Container size

Currently, the serving image installs all base dependencies, including libraries like MLflow. In future projects I would structure the project in different dependency groups from the beginning, keeping only the libraries necessary for the serving in the base dependency group and extend them for training, development, etc.
