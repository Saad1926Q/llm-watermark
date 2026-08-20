from __future__ import annotations

from typing import Any

from src import detection
from src.calibration import ScoreResult, key_fingerprint
from src.detection import evaluate_row, evaluate_rows
from src.watermark import TOURNAMENT_LAYERS


class FakeTokenizer:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
    def __call__(
        self,
        text: str | list[str],
        *,
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[int] | list[list[int]]]:
        assert add_special_tokens is False
        assert padding is False

        def encode(value: str) -> list[int]:
            token_ids = [ord(value[0]), *range(100, 100 + len(value) - 1)]
            return token_ids[:max_length] if truncation and max_length is not None else token_ids

        if isinstance(text, str):
            return {"input_ids": encode(text)}
        self.batch_sizes.append(len(text))
        return {"input_ids": [encode(value) for value in text]}


def _calibration(key: bytes, *, threshold: float = 0.5) -> dict[str, Any]:
    return {
        "threshold": threshold,
        "required_tokens": 10,
        "layers": TOURNAMENT_LAYERS,
        "model": "test-model",
        "tokenizer": "test-tokenizer",
        "key_fingerprint": key_fingerprint(key),
    }


def _row(prompt_id: str, kind: str) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "kind": kind,
        "text": ("w" if kind == "watermarked" else "u") * 10,
    }


def test_evaluate_row_preserves_identity_and_prediction() -> None:
    key = bytes(range(32))
    tokenizer = FakeTokenizer()
    calibration = _calibration(key)

    result = evaluate_row(
        _row("test-row", "unwatermarked"),
        calibration,
        key,
        tokenizer,
        row_number=7,
    )

    assert result["row_number"] == 7
    assert result["prompt_id"] == "test-row"
    assert result["kind"] == "unwatermarked"
    assert result["sufficient"] is True
    assert result["predicted_kind"] == (
        "watermarked" if result["score"] >= result["threshold"] else "unwatermarked"
    )
    assert result["correct"] == (result["predicted_kind"] == result["kind"])


def test_evaluate_rows_uses_separate_test_rows(monkeypatch) -> None:
    key = bytes(range(32))
    calibration = _calibration(key)
    tokenizer = FakeTokenizer()

    def fake_score_token_ids(
        token_ids,
        _key,
        *,
        required_tokens,
        layers,
    ):
        assert _key is key
        assert layers == TOURNAMENT_LAYERS
        score = 0.9 if token_ids[0] == ord("w") else 0.1
        return ScoreResult(
            ones=round(score * 10),
            total_bits=10,
            scored_contexts=1,
            required_tokens=required_tokens,
            used_tokens=required_tokens,
        )

    monkeypatch.setattr(detection, "_score_token_ids", fake_score_token_ids)
    metrics = evaluate_rows(
        [
            _row("test-control-1", "unwatermarked"),
            _row("test-watermarked", "watermarked"),
            _row("test-control-2", "unwatermarked"),
        ],
        calibration,
        key,
        tokenizer,
        tokenization_batch_size=2,
    )
    assert tokenizer.batch_sizes == [2, 1]

    assert metrics == {
        "test_rows": 3,
        "test_insufficient_rows": 0,
        "test_unwatermarked_rows": 2,
        "test_watermarked_rows": 1,
        "test_fpr": 0.0,
        "test_tpr": 1.0,
    }


def test_evaluate_predictions_preserves_batch_order() -> None:
    key = bytes(range(32))
    tokenizer = FakeTokenizer()
    rows = [
        _row("first", "unwatermarked"),
        _row("second", "watermarked"),
        _row("third", "unwatermarked"),
    ]

    predictions = list(
        detection.evaluate_predictions(
            rows,
            _calibration(key),
            key,
            tokenizer,
            tokenization_batch_size=2,
        )
    )

    assert [prediction["prompt_id"] for prediction in predictions] == [
        "first",
        "second",
        "third",
    ]
    assert [prediction["row_number"] for prediction in predictions] == [1, 2, 3]
    assert tokenizer.batch_sizes == [2, 1]
