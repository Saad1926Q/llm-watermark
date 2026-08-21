from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src import generate


class FakeTokenizer:
    chat_template = None
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "right"

    def __call__(
        self,
        prompt: str | list[str],
        *,
        return_tensors: str,
        padding: bool = False,
    ) -> dict[str, torch.Tensor]:
        del padding
        assert return_tensors == "pt"
        prompts = [prompt] if isinstance(prompt, str) else prompt
        token_rows = [
            [11, 12] if value == "short" else [21, 22, 23] if value == "long" else [10, 11, 12]
            for value in prompts
        ]
        width = max(len(row) for row in token_rows)
        input_rows = [[0] * (width - len(row)) + row for row in token_rows]
        mask_rows = [[0] * (width - len(row)) + [1] * len(row) for row in token_rows]
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
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
        logits_to_keep: int,
        past_key_values: object | None = None,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "use_cache": use_cache,
                "past_key_values": past_key_values,
                "logits_to_keep": logits_to_keep,
            }
        )
        logits = torch.zeros((input_ids.shape[0], logits_to_keep, 32), dtype=torch.float32)
        return SimpleNamespace(logits=logits, past_key_values=object())


class NormalCacheModel(CacheModel):
    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        logits_to_keep: int,
        past_key_values: object | None = None,
    ) -> SimpleNamespace:
        output = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            past_key_values=past_key_values,
        )
        output.logits.fill_(-torch.inf)
        output.logits[..., 31] = 0
        return output


def test_generate_normal_batch_preserves_order_and_batch_shape() -> None:
    model = NormalCacheModel()
    device = torch.device("cpu")
    results = generate.generate_normal_batch(
        model,
        FakeTokenizer(),
        ["short", "long"],
        generate.make_generators([101, 202], device=device),
        device=device,
        max_new_tokens=2,
        temperature=0.7,
        top_p=0.8,
    )

    assert [result.generated_token_ids for result in results] == [[31, 31], [31, 31]]
    assert [call["input_ids"].tolist() for call in model.calls] == [
        [[0, 11, 12], [21, 22, 23]],
        [[31], [31]],
    ]
    assert [call["attention_mask"].tolist() for call in model.calls] == [
        [[0, 1, 1], [1, 1, 1]],
        [[0, 1, 1, 1], [1, 1, 1, 1]],
    ]
    assert all(call["use_cache"] is True for call in model.calls)
    assert all(call["logits_to_keep"] == 1 for call in model.calls)


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_generate_normal_batch_requires_positive_max_tokens(max_new_tokens: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        generate.generate_normal_batch(
            object(),
            FakeTokenizer(),
            ["prompt"],
            generate.make_generators([1], device=torch.device("cpu")),
            device=torch.device("cpu"),
            max_new_tokens=max_new_tokens,
        )


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_generate_watermarked_batch_requires_positive_max_tokens(max_new_tokens: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        generate.generate_watermarked_batch(
            object(),
            FakeTokenizer(),
            ["prompt"],
            bytes(range(32)),
            device=torch.device("cpu"),
            max_new_tokens=max_new_tokens,
        )


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


def test_generate_watermarked_batch_keeps_independent_row_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CacheModel()
    tokenizer = FakeTokenizer()
    selected_tokens = iter([2, 5, 6, 2])

    def choose_token(*args: object, **kwargs: object) -> int:
        del args, kwargs
        return next(selected_tokens)

    monkeypatch.setattr(generate, "_select_next_token", choose_token)

    results = generate.generate_watermarked_batch(
        model,
        tokenizer,
        ["short", "long"],
        bytes(range(32)),
        device=torch.device("cpu"),
        max_new_tokens=4,
    )

    assert [result.generated_token_ids for result in results] == [[2], [5, 6, 2]]
    assert [call["input_ids"].tolist() for call in model.calls] == [
        [[0, 11, 12], [21, 22, 23]],
        [[2], [5]],
        [[0], [6]],
    ]
    assert [call["attention_mask"].shape[1] for call in model.calls] == [3, 4, 5]
    assert tokenizer.padding_side == "right"


def test_per_prompt_generators_are_batch_size_independent() -> None:
    prompts = ["short", "long"]
    key = bytes(range(32))
    device = torch.device("cpu")

    batched_normal = generate.generate_normal_batch(
        CacheModel(),
        FakeTokenizer(),
        prompts,
        device=device,
        max_new_tokens=5,
        generators=generate.make_generators([101, 202], device=device),
    )
    separate_normal = [
        generate.generate_normal_batch(
            CacheModel(),
            FakeTokenizer(),
            [prompt],
            device=device,
            max_new_tokens=5,
            generators=generate.make_generators([seed], device=device),
        )[0]
        for prompt, seed in zip(prompts, [101, 202], strict=True)
    ]
    assert [result.generated_token_ids for result in batched_normal] == [
        result.generated_token_ids for result in separate_normal
    ]

    batched_watermarked = generate.generate_watermarked_batch(
        CacheModel(),
        FakeTokenizer(),
        prompts,
        key,
        device=device,
        max_new_tokens=5,
        generators=generate.make_generators([101, 202], device=device),
    )
    separate_watermarked = [
        generate.generate_watermarked_batch(
            CacheModel(),
            FakeTokenizer(),
            [prompt],
            key,
            device=device,
            max_new_tokens=5,
            generators=generate.make_generators([seed], device=device),
        )[0]
        for prompt, seed in zip(prompts, [101, 202], strict=True)
    ]
    assert [result.generated_token_ids for result in batched_watermarked] == [
        result.generated_token_ids for result in separate_watermarked
    ]
