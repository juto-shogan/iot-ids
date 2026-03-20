"""Entry point for the IoT IDS training and evaluation pipeline."""

from __future__ import annotations

import logging

from src.data_loader import load_train_test
from src.evaluate import evaluate_all_models, save_metrics
from src.feature_engineering import FeatureSelector
from src.preprocessing import preprocess_train_test
from src.train_dl import train_dl_model
from src.train_ml import train_ml_models
from src.utils import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, ensure_directories, setup_logging
from src.visualize import plot_confusion_matrices, plot_model_comparison


def run_pipeline() -> dict:
    """Run full IDS workflow from loading data to saving artifacts."""
    setup_logging()
    ensure_directories()

    train_file = DATA_DIR / "UNSW_NB15_training-set.csv"
    test_file = DATA_DIR / "UNSW_NB15_testing-set.csv"

    logging.info("Loading UNSW-NB15 training/testing datasets.")
    x_train, y_train, x_test, y_test = load_train_test(train_file, test_file)

    logging.info("Applying preprocessing pipeline.")
    x_train_processed, x_test_processed, _ = preprocess_train_test(
        x_train=x_train,
        x_test=x_test,
        preprocessor_path=MODELS_DIR / "preprocessor.joblib",
    )

    logging.info("Performing feature selection.")
    selector = FeatureSelector(threshold=0.0)
    x_train_selected = selector.fit_transform(x_train_processed)
    x_test_selected = selector.transform(x_test_processed)

    logging.info("Training traditional ML models.")
    ml_models = train_ml_models(
        x_train=x_train_selected,
        y_train=y_train,
        models_dir=MODELS_DIR,
        svm_max_samples=30000,
    )

    logging.info("Training deep learning model.")
    dl_model = train_dl_model(
        x_train=x_train_selected,
        y_train=y_train,
        models_dir=MODELS_DIR,
        epochs=20,
        batch_size=256,
    )

    logging.info("Evaluating all models.")
    results = evaluate_all_models(
        trained_ml_models=ml_models,
        dl_model=dl_model,
        x_test=x_test_selected,
        y_test=y_test,
    )

    logging.info("Saving metrics and visualizations.")
    save_metrics(results, OUTPUTS_DIR)
    plot_confusion_matrices(results, OUTPUTS_DIR)
    plot_model_comparison(results, OUTPUTS_DIR)

    logging.info("Pipeline complete. Artifacts saved in models/ and outputs/.")
    return results


if __name__ == "__main__":
    run_pipeline()
