"""Run the Gemma 3 Bloom's Taxonomy dataset evaluation."""

import argparse
import json
from pathlib import Path

from src.bloom_classifier import BloomClassifier
from src.evaluation import evaluate_from_csv
from src.model_service import ModelService


def _serialise_metrics(metrics: dict | None) -> dict | None:
    """Convert NumPy values to JSON-compatible data without changing metrics."""
    if metrics is None:
        return None
    return {
        **metrics,
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--predictions", type=Path, default=Path("experiments/predictions.csv"))
    parser.add_argument("--report", type=Path, default=Path("experiments/evaluation_report.json"))
    args = parser.parse_args()

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    model_service = ModelService(temperature=0.0)
    if not model_service.is_ready:
        raise RuntimeError(model_service.load_error or "Gemma 3 model could not be loaded.")

    result_df, metrics = evaluate_from_csv(
        str(args.input_csv),
        BloomClassifier(model_service),
        output_path=str(args.predictions),
    )
    valid_count = int((result_df["prediction_status"] == "Valid").sum())
    report = {
        "total_dataset_records": len(result_df),
        "valid_predictions": valid_count,
        "invalid_predictions": len(result_df) - valid_count,
        "metrics": _serialise_metrics(metrics),
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()