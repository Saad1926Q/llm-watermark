"""Evaluate a frozen watermark calibration on labeled response text."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from itertools import islice
from pathlib import Path
from typing import Any

from src.calibration import (
    DEFAULT_TOKENIZATION_BATCH_SIZE,
    CalibrationError,
    ScoreResult,
    _score_token_ids,
    _tokenize_texts,
    key_fingerprint,
)


def load_calibration(path: Path, key: bytes) -> dict[str, Any]:
    """Load a fitted calibration artifact and verify its key.

    Args:
        path: Calibration JSON file to load.
        key: Secret watermark key expected by the artifact.

    Returns:
        The decoded calibration mapping.

    Raises:
        CalibrationError: If the file is invalid or the key fingerprint differs.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            artifact = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot load calibration JSON {path}: {exc}") from exc

    if not isinstance(artifact, dict):
        raise CalibrationError("calibration JSON must contain an object")
    if artifact.get("key_fingerprint") != key_fingerprint(key):
        raise CalibrationError("calibration key fingerprint does not match supplied key")
    return artifact


def _prediction_from_result(
    row: Mapping[str, Any],
    calibration: Mapping[str, Any],
    result: ScoreResult,
    *,
    row_number: int,
) -> dict[str, Any]:
    """Build one prediction from a precomputed score."""
    kind = row.get("kind")

    threshold = float(calibration["threshold"])

    score = result.score if result.sufficient else None

    predicted_kind = (
        None if score is None else "watermarked" if score >= threshold else "unwatermarked"
    )

    prediction = dict(row)

    prediction.update(
        {
            "row_number": row_number,
            "predicted_kind": predicted_kind,
            "score": score,
            "threshold": threshold,
            "correct": None if predicted_kind is None else predicted_kind == kind,
            "sufficient": result.sufficient,
            "ones": result.ones,
            "total_bits": result.total_bits,
            "scored_contexts": result.scored_contexts,
            "required_tokens": result.required_tokens,
            "used_tokens": result.used_tokens,
        }
    )

    return prediction


def evaluate_row(
    row: Mapping[str, Any],
    calibration: Mapping[str, Any],
    key: bytes,
    tokenizer: Any,
    *,
    row_number: int,
) -> dict[str, Any]:
    """Score one labeled test row and record its classification.

    Args:
        row: Test row containing response text and its actual ``kind``.
        calibration: Frozen threshold and scoring configuration.
        key: Secret watermark key used to score the response.
        tokenizer: Tokenizer used to encode response text.
        row_number: One-based row number in the test JSONL file.

    Returns:
        A copy of the input row with score, prediction, correctness, and
        token-count fields appended.

    Raises:
        CalibrationError: If the row label or response text is invalid.
    """
    kind = row.get("kind")
    if kind not in {"unwatermarked", "watermarked"}:
        raise CalibrationError("row kind must be 'unwatermarked' or 'watermarked'")

    text = row.get("text")

    if not isinstance(text, str):
        raise CalibrationError("row text must be a string")

    token_ids = _tokenize_texts(
        [text],
        tokenizer,
        required_tokens=calibration["required_tokens"],
    )[0]

    result = _score_token_ids(
        token_ids,
        key,
        required_tokens=calibration["required_tokens"],
        layers=calibration["layers"],
    )
    return _prediction_from_result(
        row,
        calibration,
        result,
        row_number=row_number,
    )


def evaluate_predictions(
    test_rows: Iterable[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    key: bytes,
    tokenizer: Any,
    *,
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
) -> Iterator[dict[str, Any]]:
    """Batch-tokenize and score labeled test rows in input order."""
    if type(tokenization_batch_size) is not int or tokenization_batch_size < 1:
        raise CalibrationError("tokenization_batch_size must be a positive integer")

    rows = iter(test_rows)
    row_number = 0
    while batch_rows := list(islice(rows, tokenization_batch_size)):
        texts: list[str] = []

        for row in batch_rows:
            kind = row.get("kind")

            if kind not in {"unwatermarked", "watermarked"}:
                raise CalibrationError("row kind must be 'unwatermarked' or 'watermarked'")

            text = row.get("text")

            if not isinstance(text, str):
                raise CalibrationError("row text must be a string")

            texts.append(text)

        token_batches = _tokenize_texts(
            texts,
            tokenizer,
            required_tokens=calibration["required_tokens"],
        )

        for row, token_ids in zip(batch_rows, token_batches, strict=True):
            row_number += 1

            result = _score_token_ids(
                token_ids,
                key,
                required_tokens=calibration["required_tokens"],
                layers=calibration["layers"],
            )

            yield _prediction_from_result(
                row,
                calibration,
                result,
                row_number=row_number,
            )


def summarize_predictions(
    predictions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate test-set predictions into FPR and TPR metrics.

    Args:
        predictions: Per-row evaluation records produced by ``evaluate_row``.

    Returns:
        Counts of test rows and sufficient classes, plus FPR and TPR values.
    """
    # Unwatermarked responses are the negative controls.
    controls = 0
    # Watermarked responses are the positive examples.
    watermarked = 0
    # Controls incorrectly classified as watermarked.
    false_positives = 0
    # Watermarked responses correctly classified as watermarked.
    true_positives = 0
    # Too-short responses are excluded from FPR and TPR.
    insufficient = 0
    total = 0

    for prediction in predictions:
        total += 1
        if not prediction["sufficient"]:
            insufficient += 1
            continue

        # Whether this response was classified as watermarked.
        positive = prediction["predicted_kind"] == "watermarked"
        if prediction["kind"] == "unwatermarked":
            controls += 1
            false_positives += positive
        else:
            watermarked += 1
            true_positives += positive

    return {
        "test_rows": total,
        "test_insufficient_rows": insufficient,
        "test_unwatermarked_rows": controls,
        "test_watermarked_rows": watermarked,
        "test_fpr": false_positives / controls if controls else None,
        "test_tpr": true_positives / watermarked if watermarked else None,
    }


def evaluate_rows(
    test_rows: Iterable[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    key: bytes,
    tokenizer: Any,
    *,
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
) -> dict[str, Any]:
    """Evaluate a frozen calibration artifact on labeled test rows.

    Args:
        test_rows: Iterable of labeled rows containing response text.
        calibration: Frozen threshold and scoring configuration.
        key: Secret watermark key used to score responses.
        tokenizer: Tokenizer used to encode response text.
        tokenization_batch_size: Number of response texts encoded per call.

    Returns:
        Aggregate test-set counts, false-positive rate, and true-positive rate.
    """
    predictions = evaluate_predictions(
        test_rows,
        calibration,
        key,
        tokenizer,
        tokenization_batch_size=tokenization_batch_size,
    )
    return summarize_predictions(predictions)
