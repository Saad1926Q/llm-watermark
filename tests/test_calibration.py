from __future__ import annotations

from typing import Any

import pytest

from src.calibration import (
    CalibrationError,
    ScoreResult,
    _score_token_ids,
    _tokenize_texts,
    fit_calibration,
)
from src.watermark import CONTEXT_WIDTH, TOURNAMENT_LAYERS


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
            token_ids = [ord(character) * 1000 + index for index, character in enumerate(value)]
            return token_ids[:max_length] if truncation and max_length is not None else token_ids

        if isinstance(text, str):
            return {"input_ids": encode(text)}

        self.batch_sizes.append(len(text))
        return {"input_ids": [encode(value) for value in text]}


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


def test_batched_tokenization_and_scoring_use_exact_required_prefix() -> None:
    key = bytes(range(32))
    tokenizer = FakeTokenizer()
    required_tokens = 10
    prefix_text = "x" * required_tokens
    token_batches = _tokenize_texts(
        [prefix_text, prefix_text + "y" * 100],
        tokenizer,
        required_tokens=required_tokens,
    )

    prefix_result, longer_result = [
        _score_token_ids(
            token_ids,
            key,
            required_tokens=required_tokens,
            layers=TOURNAMENT_LAYERS,
        )
        for token_ids in token_batches
    ]

    assert prefix_result == longer_result
    assert prefix_result.sufficient is True
    assert prefix_result.used_tokens == required_tokens
    assert prefix_result.scored_contexts == required_tokens - CONTEXT_WIDTH
    assert prefix_result.total_bits == prefix_result.scored_contexts * TOURNAMENT_LAYERS
    assert tokenizer.batch_sizes == [2]


def test_short_response_is_insufficient() -> None:
    key = bytes(range(32))
    tokenizer = FakeTokenizer()
    token_ids = _tokenize_texts(
        ["x" * 9],
        tokenizer,
        required_tokens=10,
    )[0]

    result = _score_token_ids(
        token_ids,
        key,
        required_tokens=10,
        layers=TOURNAMENT_LAYERS,
    )
    assert result.sufficient is False
    assert result.used_tokens == 9
    assert result.scored_contexts == 0
    assert result.total_bits == 0
    with pytest.raises(CalibrationError, match="no watermark bits"):
        _ = result.score


def test_threshold_batches_only_development_unwatermarked_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = bytes(range(32))
    tokenizer = FakeTokenizer()
    development_rows = [
        _row("dev-negative-low", text="a" * 10),
        _row("dev-negative-high", text="b" * 10),
        _row("dev-watermarked", kind="watermarked", text="c" * 10),
    ]
    score_by_first_token = {
        ord("a") * 1000: 0.2,
        ord("b") * 1000: 0.8,
    }

    def fake_score_token_ids(token_ids, _key, *, required_tokens, layers):
        assert _key is key
        assert required_tokens == 10
        assert layers == TOURNAMENT_LAYERS
        score = score_by_first_token[token_ids[0]]
        return ScoreResult(
            ones=round(score * 10),
            total_bits=10,
            scored_contexts=1,
            required_tokens=required_tokens,
            used_tokens=required_tokens,
        )

    monkeypatch.setattr("src.calibration._score_token_ids", fake_score_token_ids)
    artifact = fit_calibration(
        development_rows,
        key,
        tokenizer,
        target_fpr=0.5,
        required_tokens=10,
        layers=TOURNAMENT_LAYERS,
        tokenization_batch_size=1,
    )

    assert artifact["threshold"] == pytest.approx(0.8)
    assert artifact["development_rows"] == 3
    assert artifact["development_sufficient_rows"] == 2
    assert artifact["development_insufficient_rows"] == 0
    assert tokenizer.batch_sizes == [1, 1]
