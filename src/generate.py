"""Manual autoregressive generation with the keyed watermark sampler."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers.generation.logits_process import TopPLogitsWarper

from src.watermark import (
    CONTEXT_WIDTH,
    TOURNAMENT_LAYERS,
    sample_normally,
    sample_watermarked_token,
)


@dataclass(frozen=True)
class GenerationResult:
    """Token-exact result from one autoregressive generation call.

    Attributes:
        generated_token_ids: Exact token IDs appended by generation.
        text: Decoded generated continuation, excluding the prompt.
    """

    generated_token_ids: list[int]
    text: str


def make_generators(
    seeds: Sequence[int],
    *,
    device: torch.device,
) -> list[torch.Generator]:
    """Create one deterministic sampling generator for each prompt seed."""
    return [torch.Generator(device=device).manual_seed(seed) for seed in seeds]


def _validate_generators(
    generators: Sequence[torch.Generator] | None,
    batch_size: int,
) -> Sequence[torch.Generator] | None:
    """Validate that per-prompt generators match the prompt batch."""
    if generators is None:
        return None
    if len(generators) != batch_size:
        raise ValueError("generators must contain one generator per prompt")
    return generators

def encode_prompt(tokenizer: Any, prompt: str) -> Any:
    """Tokenize a prompt, applying the tokenizer's chat template when available.

    Args:
        tokenizer: Tokenizer used for model inputs.
        prompt: User prompt to encode.

    Returns:
        A tokenizer batch encoding containing the input token IDs.
    """
    return tokenizer(_render_prompt(tokenizer, prompt), return_tensors="pt")


def _render_prompt(tokenizer: Any, prompt: str) -> str:
    """Render one prompt with the tokenizer's optional chat template."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return prompt


def encode_prompts(tokenizer: Any, prompts: Sequence[str]) -> Any:
    """Tokenize a prompt batch with left padding for decoder-only models."""
    rendered_prompts = [_render_prompt(tokenizer, prompt) for prompt in prompts]
    if not rendered_prompts:
        raise ValueError("prompts cannot be empty")

    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError("batch generation requires a pad token or eos token")
        tokenizer.pad_token = eos_token

    original_padding_side = getattr(tokenizer, "padding_side", None)

    if original_padding_side is not None:
        tokenizer.padding_side = "left"

    try:
        return tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
        )
    finally:
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side


def generate_normal_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    generators: Sequence[torch.Generator],
    *,
    device: torch.device,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> list[GenerationResult]:
    """Generate ordinary stochastic continuations for a prompt batch."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    prompt_list = list(prompts)
    if not prompt_list:
        return []
    if len(generators) != len(prompt_list):
        raise ValueError("generators must contain one generator per prompt")

    encoded = encode_prompts(tokenizer, prompt_list)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)

    batch_size = len(prompt_list)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is None:
        raise ValueError("batch generation requires a pad token or eos token")

    generated_token_ids: list[list[int]] = [[] for _ in prompt_list]
    finished = [False] * batch_size
    warper = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            raise ValueError("model did not return past_key_values with use_cache=True")

        for step in range(max_new_tokens):
            next_token_ids = torch.full(
                (batch_size, 1),
                pad_token_id,
                dtype=input_ids.dtype,
                device=device,
            )

            for row in range(batch_size):
                if finished[row]:
                    continue

                next_token_id = sample_normally(
                    outputs.logits[row, -1, :],
                    temperature=temperature,
                    top_p=top_p,
                    _warper=warper,
                    _generator=generators[row],
                )
                next_token_ids[row, 0] = next_token_id
                generated_token_ids[row].append(next_token_id)

                if eos_token_id is not None and next_token_id == eos_token_id:
                    finished[row] = True

            if all(finished) or step == max_new_tokens - 1:
                break

            attention_mask = torch.cat(
                (
                    attention_mask,
                    attention_mask.new_ones((batch_size, 1)),
                ),
                dim=1,
            )
            outputs = model(
                input_ids=next_token_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = getattr(outputs, "past_key_values", None)
            if past_key_values is None:
                raise ValueError("model did not return past_key_values with use_cache=True")

    return [
        GenerationResult(
            token_ids,
            tokenizer.decode(token_ids, skip_special_tokens=True),
        )
        for token_ids in generated_token_ids
    ]


def generate_normal(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    generator: torch.Generator,
) -> GenerationResult:
    """Generate one ordinary continuation using the batch implementation."""
    return generate_normal_batch(
        model,
        tokenizer,
        [prompt],
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        generators=[generator],
    )[0]


def _select_next_token(
    next_logits: torch.Tensor,
    generated_token_ids: list[int],
    seen_contexts: set[tuple[int, ...]],
    key: bytes,
    *,
    temperature: float,
    top_p: float,
    layers: int,
    warper: TopPLogitsWarper,
    _generator: torch.Generator | None = None,
) -> int:
    """Select one normal or watermarked token for a response row."""
    if len(generated_token_ids) < CONTEXT_WIDTH:
        return sample_normally(
            next_logits,
            temperature=temperature,
            top_p=top_p,
            _warper=warper,
            _generator=_generator,
        )

    context = generated_token_ids[-CONTEXT_WIDTH:]
    context_key = tuple(context)
    is_repeated_context = context_key in seen_contexts
    seen_contexts.add(context_key)

    if is_repeated_context:
        return sample_normally(
            next_logits,
            temperature=temperature,
            top_p=top_p,
            _warper=warper,
            _generator=_generator,
        )

    return sample_watermarked_token(
        next_logits,
        context,
        key,
        temperature=temperature,
        top_p=top_p,
        layers=layers,
        _warper=warper,
        _generator=_generator,
    )


def generate_watermarked_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    key: bytes,
    *,
    device: torch.device,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    layers: int = TOURNAMENT_LAYERS,
    generators: Sequence[torch.Generator] | None = None,
) -> list[GenerationResult]:
    """Generate independent watermarked continuations for a prompt batch."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    prompt_list = list(prompts)
    if not prompt_list:
        return []
    prompt_generators = _validate_generators(generators, len(prompt_list))

    encoded = encode_prompts(tokenizer, prompt_list)
    input_ids = encoded["input_ids"].to(device)

    attention_mask = encoded.get("attention_mask")

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)

    batch_size = len(prompt_list)

    eos_token_id = tokenizer.eos_token_id

    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    if pad_token_id is None:
        pad_token_id = eos_token_id

    if pad_token_id is None:
        raise ValueError("batch generation requires a pad token or eos token")

    generated_token_ids: list[list[int]] = [[] for _ in prompt_list]
    seen_contexts: list[set[tuple[int, ...]]] = [set() for _ in prompt_list]

    finished = [False] * batch_size

    warper = TopPLogitsWarper(
        top_p=top_p,
        min_tokens_to_keep=1,
    )

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
        past_key_values = getattr(outputs, "past_key_values", None)

        if past_key_values is None:
            raise ValueError("model did not return past_key_values with use_cache=True")

        for step in range(max_new_tokens):
            next_token_ids = torch.full(
                (batch_size, 1),
                pad_token_id,
                dtype=input_ids.dtype,
                device=device,
            )

            for row in range(batch_size):
                if finished[row]:
                    continue

                next_token_id = _select_next_token(
                    outputs.logits[row, -1, :],
                    generated_token_ids[row],
                    seen_contexts[row],
                    key,
                    temperature=temperature,
                    top_p=top_p,
                    layers=layers,
                    warper=warper,
                    _generator=(
                        prompt_generators[row] if prompt_generators is not None else None
                    ),
                )
                next_token_ids[row, 0] = next_token_id
                generated_token_ids[row].append(next_token_id)

                if eos_token_id is not None and next_token_id == eos_token_id:
                    finished[row] = True

            if all(finished) or step == max_new_tokens - 1:
                break

            attention_mask = torch.cat(
                (
                    attention_mask,
                    attention_mask.new_ones((batch_size, 1)),
                ),
                dim=1,
            )
            outputs = model(
                input_ids=next_token_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = getattr(outputs, "past_key_values", None)
            if past_key_values is None:
                raise ValueError("model did not return past_key_values with use_cache=True")

    return [
        GenerationResult(
            token_ids,
            tokenizer.decode(token_ids, skip_special_tokens=True),
        )
        for token_ids in generated_token_ids
    ]


def generate_watermarked(
    model: Any,
    tokenizer: Any,
    prompt: str,
    key: bytes,
    *,
    device: torch.device,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    layers: int = TOURNAMENT_LAYERS,
    generator: torch.Generator | None = None,
) -> GenerationResult:
    """Generate one watermarked continuation using the batch implementation."""
    return generate_watermarked_batch(
        model,
        tokenizer,
        [prompt],
        key,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        layers=layers,
        generators=[generator] if generator is not None else None,
    )[0]
