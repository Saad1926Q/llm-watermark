from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src import generate


class FakeTokenizer:
    chat_template = None
    eos_token_id = None

    def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, torch.Tensor]:
        del prompt
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return " ".join(str(token_id) for token_id in token_ids)


class CacheModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        past_key_values: object | None = None,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "use_cache": use_cache,
                "past_key_values": past_key_values,
            }
        )
        logits = torch.zeros((1, input_ids.shape[1], 32), dtype=torch.float32)
        return SimpleNamespace(logits=logits, past_key_values=object())


def test_generate_watermarked_reuses_cache_and_warper(monkeypatch: pytest.MonkeyPatch) -> None:
    model = CacheModel()
    tokenizer = FakeTokenizer()
    chosen_tokens = iter([4, 5, 6, 7, 8, 9])
    warpers: list[object] = []

    def choose_token(*args: object, **kwargs: object) -> int:
        del args
        warpers.append(kwargs["_warper"])
        return next(chosen_tokens)

    monkeypatch.setattr(generate, "sample_normally", choose_token)
    monkeypatch.setattr(generate, "sample_watermarked_token", choose_token)

    result = generate.generate_watermarked(
        model,
        tokenizer,
        "prompt",
        bytes(range(32)),
        device=torch.device("cpu"),
        max_new_tokens=6,
    )

    assert result.generated_token_ids == [4, 5, 6, 7, 8, 9]
    assert [call["input_ids"].shape[1] for call in model.calls] == [3, 1, 1, 1, 1, 1]
    assert [call["attention_mask"].shape[1] for call in model.calls] == [3, 4, 5, 6, 7, 8]
    assert model.calls[0]["past_key_values"] is None
    assert all(call["past_key_values"] is not None for call in model.calls[1:])
    assert all(call["use_cache"] is True for call in model.calls)
    assert len(warpers) == 6
    assert all(warper is warpers[0] for warper in warpers)


def test_generate_watermarked_requires_model_cache() -> None:
    class NoCacheModel(CacheModel):
        def __call__(self, **kwargs: object) -> SimpleNamespace:
            input_ids = kwargs["input_ids"]
            assert isinstance(input_ids, torch.Tensor)
            logits = torch.zeros((1, input_ids.shape[1], 32), dtype=torch.float32)
            return SimpleNamespace(logits=logits, past_key_values=None)

    with pytest.raises(ValueError, match="past_key_values"):
        generate.generate_watermarked(
            NoCacheModel(),
            FakeTokenizer(),
            "prompt",
            bytes(range(32)),
            device=torch.device("cpu"),
            max_new_tokens=1,
        )
