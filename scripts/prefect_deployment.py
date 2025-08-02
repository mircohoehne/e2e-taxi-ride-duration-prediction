"""Prefect deployment configuration for taxi duration model training."""

from scripts.train_model import main

if __name__ == "__main__":
    main.serve(
        name="taxi-model-baseline-training",
        tags=["ml", "training", "taxi"],
        description="Manual training of NYC taxi ride duration prediction model",
        version="1.0.0",
        pause_on_shutdown=False,
    )
