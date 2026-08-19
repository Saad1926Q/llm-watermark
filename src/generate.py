"""Manual autoregressive generation with the keyed watermark sampler."""

from __future__ import annotations

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


def encode_prompt(tokenizer: Any, prompt: str) -> Any:
    """Tokenize a prompt, applying the tokenizer's chat template when available.

    Args:
        tokenizer: Tokenizer used for model inputs.
        prompt: User prompt to encode.

    Returns:
        A tokenizer batch encoding containing the input token IDs.
    """
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    return tokenizer(prompt, return_tensors="pt")


def generate_normal(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> GenerationResult:
    """Generate an ordinary stochastic continuation and preserve exact IDs.

    Args:
        model: Evaluation-mode causal language model.
        tokenizer: Tokenizer corresponding to ``model``.
        prompt: Text to use as the generation prefix.
        device: Device on which model inputs are stored.
        max_new_tokens: Maximum number of tokens to append to ``prompt``.
        temperature: Sampling temperature passed to Transformers.
        top_p: Nucleus sampling threshold passed to Transformers.

    Returns:
        Generated IDs and decoded generated continuation.

    Raises:
        ValueError: If ``max_new_tokens`` is negative.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    encoded = encode_prompt(tokenizer, prompt)
    input_ids = encoded["input_ids"].to(device)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(input_ids=input_ids, **generation_kwargs)

    generated_token_ids = [
        int(token_id) for token_id in output_ids[0, input_ids.shape[1] :].tolist()
    ]
    text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return GenerationResult(generated_token_ids, text)


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
) -> GenerationResult:
    """Generate a response-only watermarked continuation and preserve exact IDs.

    The first ``CONTEXT_WIDTH`` response tokens are sampled normally so they
    establish a response context. Subsequent tokens use only the generated
    response suffix as context. A context is sampled normally if it repeats
    within this response, matching the detector's repeated-context masking.

    Args:
        model: Evaluation-mode causal language model returning next-token logits.
        tokenizer: Tokenizer corresponding to ``model``.
        prompt: Text to use as the generation prefix.
        key: Secret watermark key used by the sampler.
        device: Device on which model inputs and sampled tokens are stored.
        max_new_tokens: Maximum number of tokens to append to ``prompt``.
        temperature: Sampling temperature applied to model logits.
        top_p: Nucleus sampling threshold.
        layers: Number of keyed tournament-sampling layers.

    Returns:
        Generated IDs and decoded generated continuation.

    Raises:
        ValueError: If ``max_new_tokens`` is negative or the model does not
            return a KV cache.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    encoded = encode_prompt(tokenizer, prompt)
    input_ids = encoded["input_ids"].to(device)
    generated_token_ids: list[int] = []

    if max_new_tokens == 0:
        text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        return GenerationResult(generated_token_ids, text)

    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)

    eos_token_id = tokenizer.eos_token_id
    seen_contexts: set[tuple[int, ...]] = set()
    warper = TopPLogitsWarper(
        top_p=top_p,
        min_tokens_to_keep=1,
    )

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            raise ValueError("model did not return past_key_values with use_cache=True")

        for step in range(max_new_tokens):
            next_logits = outputs.logits[0, -1, :]

            if len(generated_token_ids) < CONTEXT_WIDTH:
                next_token_id = sample_normally(
                    next_logits,
                    temperature=temperature,
                    top_p=top_p,
                    _warper=warper,
                )
            else:
                context = generated_token_ids[-CONTEXT_WIDTH:]
                context_key = tuple(context)
                is_repeated_context = context_key in seen_contexts
                seen_contexts.add(context_key)

                if is_repeated_context:
                    next_token_id = sample_normally(
                        next_logits,
                        temperature=temperature,
                        top_p=top_p,
                        _warper=warper,
                    )
                else:
                    next_token_id = sample_watermarked_token(
                        next_logits,
                        context,
                        key,
                        temperature=temperature,
                        top_p=top_p,
                        layers=layers,
                        _warper=warper,
                    )

            generated_token_ids.append(next_token_id)

            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            if step == max_new_tokens - 1:
                break

            next_token = torch.tensor(
                [[next_token_id]],
                dtype=input_ids.dtype,
                device=device,
            )
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ),
                dim=1,
            )
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = getattr(outputs, "past_key_values", None)
            if past_key_values is None:
                raise ValueError("model did not return past_key_values with use_cache=True")

    text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return GenerationResult(generated_token_ids, text)
