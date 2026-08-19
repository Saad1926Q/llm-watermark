from __future__ import annotations

from typing import Any

from src import detection
from src.calibration import ScoreResult, key_fingerprint
from src.detection import evaluate_row, evaluate_rows
from src.watermark import TOURNAMENT_LAYERS


class FakeTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(100, 100 + len(text)))}


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
        "text": "x" * 10,
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

    def fake_score_row(row, _key, _tokenizer, *, required_tokens, layers):
        assert _tokenizer is tokenizer
        assert layers == TOURNAMENT_LAYERS
        score = 0.9 if row["kind"] == "watermarked" else 0.1
        return ScoreResult(
            ones=round(score * 10),
            total_bits=10,
            scored_contexts=1,
            required_tokens=required_tokens,
            used_tokens=required_tokens,
        )

    monkeypatch.setattr(detection, "score_row", fake_score_row)
    metrics = evaluate_rows(
        [_row("test-control", "unwatermarked"), _row("test-watermarked", "watermarked")],
        calibration,
        key,
        tokenizer,
    )

    assert metrics == {
        "test_rows": 2,
        "test_insufficient_rows": 0,
        "test_unwatermarked_rows": 1,
        "test_watermarked_rows": 1,
        "test_fpr": 0.0,
        "test_tpr": 1.0,
    }
