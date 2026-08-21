from __future__ import annotations

import pytest

from scripts.generate_data import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_CONFIG,
    _build_parser,
    _generation_row,
    _generation_rows,
    _render_prompt,
)
from src.generate import GenerationResult


def test_render_prompt_uses_eli5_question() -> None:
    assert _render_prompt({"question": "  Why is the sky blue?  ", "answer": "..."}) == (
        "Why is the sky blue?"
    )


def test_render_prompt_rejects_missing_eli5_question() -> None:
    with pytest.raises(ValueError, match="non-empty question"):
        _render_prompt({"answer": "No question"})


def test_generation_row_contains_only_scoring_metadata() -> None:
    row = _generation_row(
        prompt_id="development-0000",
        kind="unwatermarked",
        result=GenerationResult(generated_token_ids=[10, 11], text="response"),
        model_name="test-model",
        tokenizer_name="test-tokenizer",
    )

    assert set(row) == {
        "prompt_id",
        "kind",
        "text",
        "model",
        "tokenizer",
    }
    assert "split" not in row
    assert "required_tokens" not in row


def test_generation_rows_only_pair_test_prompts() -> None:
    normal = GenerationResult(generated_token_ids=[10], text="ordinary")
    watermarked = GenerationResult(generated_token_ids=[20], text="watermarked")

    development_rows = _generation_rows(
        prompt_id="development-0000",
        normal_result=normal,
        watermarked_result=None,
        model_name="test-model",
        tokenizer_name="test-tokenizer",
    )
    test_rows = _generation_rows(
        prompt_id="test-0000",
        normal_result=normal,
        watermarked_result=watermarked,
        model_name="test-model",
        tokenizer_name="test-tokenizer",
    )

    assert [row["kind"] for row in development_rows] == ["unwatermarked"]
    assert [row["kind"] for row in test_rows] == ["unwatermarked", "watermarked"]



def test_generation_parser_exposes_batch_size() -> None:
    parser = _build_parser()

    assert parser.parse_args(["--model", "test-model"]).batch_size == 1
    defaults = parser.parse_args(["--model", "test-model"])
    assert defaults.dataset == DEFAULT_DATASET
    assert defaults.dataset_config == DEFAULT_DATASET_CONFIG
    assert parser.parse_args(["--model", "test-model", "--batch-size", "4"]).batch_size == 4
