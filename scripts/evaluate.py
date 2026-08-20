"""Evaluate a frozen watermark calibration on a labeled test JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.calibration import DEFAULT_TOKENIZATION_BATCH_SIZE


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for frozen test-set evaluation.

    Returns:
        An argument parser containing test, calibration, and output options.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen watermark threshold on a test JSONL file."
    )
    parser.add_argument("--test", required=True, type=Path, help="Labeled test JSONL")
    parser.add_argument(
        "--calibration",
        required=True,
        type=Path,
        help="Frozen calibration JSON",
    )
    parser.add_argument("--key", type=Path)
    parser.add_argument(
        "--predictions-output",
        required=True,
        type=Path,
        help="Per-row prediction JSONL output",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional aggregate metrics JSON output",
    )
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=DEFAULT_TOKENIZATION_BATCH_SIZE,
        help="Number of response texts encoded per tokenizer call",
    )
    return parser


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-row evaluation records as JSONL.

    Args:
        path: Destination JSONL path.
        rows: Prediction records to serialize, one per line.

    Returns:
        None.
    """
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write aggregate evaluation metrics as formatted JSON.

    Args:
        path: Destination JSON path.
        value: Metrics mapping to serialize.

    Returns:
        None.
    """
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Apply a frozen threshold and report per-row and aggregate results.

    Args:
        argv: Optional command-line arguments; defaults to ``sys.argv``.

    Returns:
        ``0`` on success or ``2`` when evaluation input is invalid.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    from src.calibration import CalibrationError, read_jsonl
    from src.detection import (
        evaluate_predictions,
        load_calibration,
        summarize_predictions,
    )
    from src.watermark import load_key

    try:
        key = load_key(args.key) if args.key else load_key()
        calibration = load_calibration(args.calibration, key)
        tokenizer_name = calibration.get("tokenizer")
        if not isinstance(tokenizer_name, str) or not tokenizer_name:
            raise CalibrationError("calibration requires tokenizer metadata")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        test_rows = read_jsonl(args.test)
        if not test_rows:
            raise CalibrationError("test JSONL is empty")

        predictions = list(
            tqdm(
                evaluate_predictions(
                    test_rows,
                    calibration,
                    key,
                    tokenizer,
                    tokenization_batch_size=args.tokenization_batch_size,
                ),
                total=len(test_rows),
                desc="Evaluating",
                unit="row",
            )
        )
        metrics = summarize_predictions(predictions)
        _write_jsonl(args.predictions_output, predictions)
        if args.metrics_output is not None:
            _write_json(args.metrics_output, metrics)
    except (CalibrationError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"evaluate: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
