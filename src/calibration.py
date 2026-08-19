"""Fixed-prefix watermark scoring and threshold fitting."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.watermark import TOURNAMENT_LAYERS, score_tokens

DEFAULT_REQUIRED_TOKENS = 200


class CalibrationError(ValueError):
    """Raised when calibration data or metadata is invalid."""


def key_fingerprint(key: bytes) -> str:
    """Return a stable identifier for a watermark key.

    Args:
        key: Raw watermark key bytes.

    Returns:
        A lowercase hexadecimal BLAKE2b fingerprint.
    """
    return hashlib.blake2b(key, digest_size=16).hexdigest()


def _text_token_ids(row: Mapping[str, object], tokenizer: Any) -> list[int]:
    """Tokenize one response row without adding special tokens.

    Args:
        row: Mapping containing the response text under ``text``.
        tokenizer: Tokenizer used to encode the response text.

    Returns:
        A list of non-negative token IDs.

    Raises:
        CalibrationError: If the row text or tokenizer output is invalid.
    """
    text = row.get("text")
    if not isinstance(text, str):
        raise CalibrationError("row text must be a string")

    try:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError("tokenizer did not return input_ids") from exc

    if not isinstance(token_ids, list) or any(
        type(token_id) is not int or token_id < 0 for token_id in token_ids
    ):
        raise CalibrationError("tokenizer input_ids must be a list of non-negative integers")
    return token_ids


@dataclass(frozen=True)
class ScoreResult:
    """Score one fixed-length response prefix."""

    ones: int
    total_bits: int
    scored_contexts: int
    required_tokens: int
    used_tokens: int

    @property
    def sufficient(self) -> bool:
        """Return whether the response contains the required token prefix.

        Returns:
            ``True`` when enough tokens were available; otherwise ``False``.
        """
        return self.used_tokens >= self.required_tokens

    @property
    def score(self) -> float:
        """Return the mean keyed watermark bit score.

        Returns:
            The number of keyed one-bits divided by the total scored bits.

        Raises:
            CalibrationError: If no watermark bits were scored.
        """
        if not self.total_bits:
            raise CalibrationError("response has no watermark bits")
        return self.ones / self.total_bits


def score_row(
    row: Mapping[str, object],
    key: bytes,
    tokenizer: Any,
    *,
    required_tokens: int = DEFAULT_REQUIRED_TOKENS,
    layers: int = TOURNAMENT_LAYERS,
) -> ScoreResult:
    """Tokenize and score the first fixed-length response prefix.

    Args:
        row: Mapping containing the response text under ``text``.
        key: Secret watermark key used to score token choices.
        tokenizer: Tokenizer used to encode the response text.
        required_tokens: Number of response tokens to score.
        layers: Number of watermark layers to score per context.

    Returns:
        A ``ScoreResult`` containing keyed one-bit counts and token usage.

    Raises:
        CalibrationError: If the configuration or tokenizer output is invalid.
    """
    if type(required_tokens) is not int or required_tokens < 1:
        raise CalibrationError("required_tokens must be a positive integer")
    if type(layers) is not int or layers < 1:
        raise CalibrationError("layers must be a positive integer")

    token_ids = _text_token_ids(row, tokenizer)
    if len(token_ids) < required_tokens:
        return ScoreResult(
            ones=0,
            total_bits=0,
            scored_contexts=0,
            required_tokens=required_tokens,
            used_tokens=len(token_ids),
        )

    scored_token_ids = token_ids[:required_tokens]

    ones, total_bits, contexts = score_tokens(scored_token_ids, key, layers=layers)

    return ScoreResult(
        ones=ones,
        total_bits=total_bits,
        scored_contexts=contexts,
        required_tokens=required_tokens,
        used_tokens=len(scored_token_ids),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON objects from a JSONL file.

    Args:
        path: JSONL file to read.

    Returns:
        A list containing one dictionary for each non-empty line.

    Raises:
        CalibrationError: If the file cannot be read or contains invalid JSON.
    """
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise CalibrationError(f"cannot read JSONL file {path}: {exc}") from exc

    rows: list[dict[str, Any]] = []
    with handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibrationError(f"invalid JSON on line {number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise CalibrationError(f"line {number} is not a JSON object")
            rows.append(value)
    return rows


def _fit(scores: list[float], target_fpr: float) -> tuple[float, float]:
    """Select the smallest threshold meeting a target false-positive budget.

    Args:
        scores: Scores from sufficient unwatermarked development rows.
        target_fpr: Maximum allowed fraction classified as positive.

    Returns:
        A ``(threshold, observed_fpr)`` tuple.

    Raises:
        CalibrationError: If ``scores`` is empty or ``target_fpr`` is invalid.
    """
    if not scores:
        raise CalibrationError("no sufficient development unwatermarked rows to fit threshold")
    if (
        isinstance(target_fpr, bool)
        or not isinstance(target_fpr, (int, float))
        or not math.isfinite(target_fpr)
        or not 0 <= target_fpr <= 1
    ):
        raise CalibrationError("target_fpr must be a finite number in [0, 1]")

    ordered = sorted(scores)
    allowed = math.floor(float(target_fpr) * len(ordered) + 1e-12)
    for threshold in sorted(set(ordered)):
        count = sum(score >= threshold for score in ordered)
        if count <= allowed:
            return threshold, count / len(ordered)
    return math.nextafter(ordered[-1], math.inf), 0.0


def fit_calibration(
    development_rows: Iterable[Mapping[str, Any]],
    key: bytes,
    tokenizer: Any,
    *,
    target_fpr: float,
    required_tokens: int = DEFAULT_REQUIRED_TOKENS,
    layers: int = TOURNAMENT_LAYERS,
) -> dict[str, Any]:
    """Fit a threshold using unwatermarked development response text only.

    Args:
        development_rows: Labeled development rows containing response text.
        key: Secret watermark key used to score responses.
        tokenizer: Tokenizer used to encode response text.
        target_fpr: Maximum allowed development false-positive rate.
        required_tokens: Number of response tokens to score per row.
        layers: Number of watermark layers to score per context.

    Returns:
        A JSON-serializable calibration artifact containing the fitted threshold
        and the scoring metadata required for evaluation.

    Raises:
        CalibrationError: If row metadata, configuration, or scores are invalid.
    """
    if type(required_tokens) is not int or required_tokens < 1:
        raise CalibrationError("required_tokens must be a positive integer")
    if type(layers) is not int or layers < 1:
        raise CalibrationError("layers must be a positive integer")

    scores: list[float] = []
    insufficient = 0
    metadata: tuple[str, str] | None = None
    total = 0

    for row in development_rows:
        total += 1
        kind = row.get("kind")
        if kind not in {"unwatermarked", "watermarked"}:
            raise CalibrationError("row kind must be 'unwatermarked' or 'watermarked'")

        model_name = row.get("model")
        tokenizer_name = row.get("tokenizer")
        if not isinstance(model_name, str) or not model_name:
            raise CalibrationError("row model must be a non-empty string")
        if not isinstance(tokenizer_name, str) or not tokenizer_name:
            raise CalibrationError("row tokenizer must be a non-empty string")

        pair = (model_name, tokenizer_name)

        if metadata is None:
            metadata = pair
        elif pair != metadata:
            raise CalibrationError("development rows use inconsistent model/tokenizer metadata")

        if kind == "watermarked":
            continue

        result = score_row(
            row,
            key,
            tokenizer,
            required_tokens=required_tokens,
            layers=layers,
        )
        if result.sufficient:
            scores.append(result.score)
        else:
            insufficient += 1

    threshold, observed = _fit(scores, target_fpr)
    return {
        "threshold": threshold,
        "target_fpr": float(target_fpr),
        "observed_development_fpr": observed,
        "required_tokens": required_tokens,
        "layers": layers,
        "model": metadata[0] if metadata else None,
        "tokenizer": metadata[1] if metadata else None,
        "key_fingerprint": key_fingerprint(key),
        "development_rows": total,
        "development_sufficient_rows": len(scores),
        "development_insufficient_rows": insufficient,
    }
