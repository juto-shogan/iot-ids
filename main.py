"""Entry point for the IoT IDS training and evaluation pipeline."""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data_loader import load_train_test
from src.evaluate import evaluate_all_models, save_metrics
from src.feature_engineering import DLFeatureReducer, FeatureSelector
from src.preprocessing import preprocess_train_test
from src.train_dl import train_dl_model
from src.train_ml import train_ml_models
from src.utils import (
    PROJECT_ROOT,
    build_run_metadata,
    ensure_directories,
    load_config,
    save_json,
    setup_logging,
)
from src.visualize import plot_confusion_matrices, plot_model_comparison


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="IoT IDS pipeline runner")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_pipeline(config_path: str = "config.yaml") -> dict:
    """Run full IDS workflow from loading data to saving artifacts."""
    setup_logging()
    config = load_config(PROJECT_ROOT / config_path)

    random_seed = int(config.get("random_seed", 42))
    _set_seed(random_seed)

    models_dir = PROJECT_ROOT / config["paths"]["models_dir"]
    outputs_dir = PROJECT_ROOT / config["paths"]["outputs_dir"]
    ensure_directories(models_dir=models_dir, outputs_dir=outputs_dir)

    train_file = PROJECT_ROOT / config["paths"]["train_file"]
    test_file = PROJECT_ROOT / config["paths"]["test_file"]

    start_time = time.perf_counter()

    logging.info("Loading UNSW-NB15 training/testing datasets.")
    x_train, y_train, x_test, y_test = load_train_test(train_file, test_file)

    logging.info("Applying preprocessing pipeline.")
    x_train_processed, x_test_processed, _ = preprocess_train_test(
        x_train=x_train,
        x_test=x_test,
        preprocessor_path=models_dir / "preprocessor.joblib",
    )

    logging.info("Performing feature selection.")
    selector = FeatureSelector(threshold=float(config["preprocessing"].get("variance_threshold", 0.0)))
    x_train_selected = selector.fit_transform(x_train_processed)
    x_test_selected = selector.transform(x_test_processed)
    selector.save(models_dir / "feature_selector.joblib")

    logging.info("Training traditional ML models.")
    ml_models = train_ml_models(
        x_train=x_train_selected,
        y_train=y_train,
        models_dir=models_dir,
        svm_max_samples=config["ml"].get("svm_max_samples", 30000),
        random_seed=random_seed,
    )

    dl_enabled = bool(config["dl"].get("enabled", True))
    if not dl_enabled:
        raise ValueError("DL model is disabled in config; current pipeline expects DL enabled.")

    logging.info("Preparing DL-specific reduced features.")
    reducer = DLFeatureReducer(max_features=config["dl"].get("max_features", 256))
    x_train_dl = reducer.fit_transform(x_train_selected)
    x_test_dl = reducer.transform(x_test_selected)
    reducer.save(models_dir / "dl_reducer.joblib")

    logging.info("Training deep learning model.")
    dl_model, dl_threshold = train_dl_model(
        x_train=x_train_dl,
        y_train=y_train,
        models_dir=models_dir,
        epochs=int(config["dl"].get("epochs", 20)),
        batch_size=int(config["dl"].get("batch_size", 256)),
        max_samples=config["dl"].get("max_samples", 60000),
    )

    logging.info("Evaluating all models.")
    results = evaluate_all_models(
        trained_ml_models=ml_models,
        dl_model=dl_model,
        dl_threshold=dl_threshold,
        x_test=x_test_dl,
        y_test=y_test,
    )

    logging.info("Saving metrics and visualizations.")
    save_metrics(results, outputs_dir)
    plot_confusion_matrices(results, outputs_dir)
    plot_model_comparison(results, outputs_dir)

    elapsed_sec = time.perf_counter() - start_time
    metadata = build_run_metadata(
        config=config,
        extra={
            "elapsed_seconds": round(elapsed_sec, 2),
            "x_train_shape": list(x_train.shape),
            "x_test_shape": list(x_test.shape),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "dl_threshold": float(dl_threshold),
        },
    )
    save_json(metadata, outputs_dir / "run_metadata.json")

    logging.info("Pipeline complete. Artifacts saved in models/ and outputs/.")
    return results


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(config_path=args.config)
