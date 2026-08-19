from __future__ import annotations

from typing import Any

import pytest

from src.calibration import CalibrationError, ScoreResult, fit_calibration, score_row
from src.watermark import CONTEXT_WIDTH, TOURNAMENT_LAYERS


class FakeTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(100, 100 + len(text)))}


def _row(
    prompt_id: str,
    *,
    kind: str = "unwatermarked",
    text: str | None = None,
) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "kind": kind,
        "text": text if text is not None else f"response for {prompt_id}",
        "model": "test-model",
        "tokenizer": "test-tokenizer",
    }


def test_score_row_tokenizes_text_and_uses_exact_required_prefix() -> None:
    key = bytes(range(32))
    tokenizer = FakeTokenizer()
    required_tokens = 10
    prefix_text = "x" * required_tokens

    prefix_result = score_row(
        {"text": prefix_text},
        key,
        tokenizer,
        required_tokens=required_tokens,
        layers=TOURNAMENT_LAYERS,
    )
    longer_result = score_row(
        {"text": prefix_text + "y" * 100},
        key,
        tokenizer,
        required_tokens=required_tokens,
        layers=TOURNAMENT_LAYERS,
    )

    assert prefix_result == longer_result
    assert prefix_result.sufficient is True
    assert prefix_result.used_tokens == required_tokens
    assert prefix_result.scored_contexts == required_tokens - CONTEXT_WIDTH
    assert prefix_result.total_bits == prefix_result.scored_contexts * TOURNAMENT_LAYERS


def test_short_response_is_insufficient() -> None:
    key = bytes(range(32))
    tokenizer = FakeTokenizer()

    result = score_row(
        {"text": "x" * 9},
        key,
        tokenizer,
        required_tokens=10,
        layers=TOURNAMENT_LAYERS,
    )

    assert result.sufficient is False
    assert result.used_tokens == 9
    assert result.scored_contexts == 0
    assert result.total_bits == 0
    with pytest.raises(CalibrationError, match="no watermark bits"):
        _ = result.score


def test_threshold_uses_only_development_unwatermarked_scores(monkeypatch) -> None:
    key = bytes(range(32))
    tokenizer = FakeTokenizer()
    score_values = {
        "dev-negative-low": 0.2,
        "dev-negative-high": 0.8,
        "dev-watermarked": 1.0,
    }

    def fake_score_row(row, _key, _tokenizer, *, required_tokens, layers):
        assert _tokenizer is tokenizer
        assert required_tokens == 10
        assert layers == TOURNAMENT_LAYERS
        score = score_values[row["prompt_id"]]
        return ScoreResult(
            ones=round(score * 10),
            total_bits=10,
            scored_contexts=1,
            required_tokens=required_tokens,
            used_tokens=required_tokens,
        )

    monkeypatch.setattr("src.calibration.score_row", fake_score_row)
    development_rows = [
        _row("dev-negative-low"),
        _row("dev-negative-high"),
        _row("dev-watermarked", kind="watermarked"),
    ]

    first = fit_calibration(
        development_rows,
        key,
        tokenizer,
        target_fpr=0.5,
        required_tokens=10,
        layers=TOURNAMENT_LAYERS,
    )

    score_values["dev-watermarked"] = 0.0
    second = fit_calibration(
        development_rows,
        key,
        tokenizer,
        target_fpr=0.5,
        required_tokens=10,
        layers=TOURNAMENT_LAYERS,
    )

    assert first["threshold"] == pytest.approx(0.8)
    assert second["threshold"] == first["threshold"]
    assert second["development_rows"] == 3
    assert second["development_sufficient_rows"] == 2
    assert second["development_insufficient_rows"] == 0
