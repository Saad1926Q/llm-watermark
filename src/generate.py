"""Manual autoregressive generation with the keyed watermark sampler."""

import torch

from src.watermark import (
    CONTEXT_WIDTH,
    TOURNAMENT_LAYERS,
    sample_normally,
    sample_watermarked_token,
)


def encode_prompt(tokenizer, prompt: str):
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


def _context_tokens(input_ids: torch.Tensor, fill_token_id: int) -> list[int]:
    """Return the latest context tokens, left-padding short prompts.

    Args:
        input_ids: A batch containing the tokenized input sequence.
        fill_token_id: Token ID used to left-pad short sequences.

    Returns:
        Exactly ``CONTEXT_WIDTH`` token IDs for watermark sampling.
    """
    token_ids = input_ids[0].tolist()
    if len(token_ids) >= CONTEXT_WIDTH:
        return token_ids[-CONTEXT_WIDTH:]

    padding = [fill_token_id] * (CONTEXT_WIDTH - len(token_ids))
    return padding + token_ids


def generate_watermarked_text(
    model,
    tokenizer,
    prompt: str,
    key: bytes,
    *,
    device: torch.device,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    layers: int = TOURNAMENT_LAYERS,
) -> str:
    """Generate one watermarked continuation with manual token sampling.

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
        The decoded generated continuation, excluding the prompt.

    Raises:
        ValueError: If ``max_new_tokens`` is negative or sampler inputs are
            invalid.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    encoded = encode_prompt(tokenizer, prompt)
    input_ids = encoded["input_ids"].to(device)

    prompt_length = input_ids.shape[1]

    eos_token_id = tokenizer.eos_token_id
    fill_token_id = eos_token_id if eos_token_id is not None else 0

    seen_contexts: set[tuple[int, ...]] = set()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Only the final logits row predicts the next token.
            next_logits = model(input_ids=input_ids).logits[0, -1, :]

            context = _context_tokens(input_ids, fill_token_id)
            context_key = tuple(context)
            is_repeated_context = context_key in seen_contexts
            seen_contexts.add(context_key)

            if is_repeated_context:
                next_token_id = sample_normally(
                    next_logits,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                next_token_id = sample_watermarked_token(
                    next_logits,
                    context,
                    key,
                    temperature=temperature,
                    top_p=top_p,
                    layers=layers,
                )

            next_token = torch.tensor(
                [[next_token_id]],
                dtype=input_ids.dtype,
                device=device,
            )

            input_ids = torch.cat((input_ids, next_token), dim=1)

            if eos_token_id is not None and next_token_id == eos_token_id:
                break

    generated_ids = input_ids[0, prompt_length:]

    return tokenizer.decode(generated_ids, skip_special_tokens=True)
