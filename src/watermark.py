"""Keyed token sampling and baseline detection for SynthID-style watermarks.

The implementation modifies next-token sampling and detects the resulting
statistical pattern without loading the language model.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers.generation.logits_process import TopPLogitsWarper

KEY_SIZE_BYTES = 32
CONTEXT_WIDTH = 4
TOURNAMENT_LAYERS = 3
DEFAULT_KEY_PATH = Path(__file__).parent.parent / "keys" / "watermark.key"

_UINT32_MAX = 2**32 - 1


def _validate_key(key: bytes) -> None:
    """Validate that a key contains the required number of raw bytes.

    Args:
        key: Raw watermark key bytes.

    Returns:
        None.

    Raises:
        ValueError: If ``key`` is not exactly ``KEY_SIZE_BYTES`` bytes.
    """
    if len(key) != KEY_SIZE_BYTES:
        raise ValueError(f"watermark key must be {KEY_SIZE_BYTES} bytes, got {len(key)} bytes")


def load_key(path: Path = DEFAULT_KEY_PATH) -> bytes:
    """Load and validate a watermark key from disk.

    Args:
        path: File containing the raw watermark key.

    Returns:
        The validated key bytes.

    Raises:
        FileNotFoundError: If the key file does not exist.
        ValueError: If the key is not exactly 32 bytes.
    """

    key = path.read_bytes()
    _validate_key(key)
    return key


def _as_uint32(value: int, name: str) -> int:
    """Convert a value to an unsigned 32-bit integer.

    Args:
        value: Integer-like value to normalize.
        name: Human-readable name used in validation errors.

    Returns:
        The normalized integer.

    Raises:
        ValueError: If the value is outside the unsigned 32-bit range.
    """
    value = int(value)
    if not 0 <= value <= _UINT32_MAX:
        raise ValueError(f"{name} must be between 0 and {_UINT32_MAX}, got {value}")
    return value


def _normalize_context(context_tokens: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize a fixed-width watermark context.

    Args:
        context_tokens: Context token IDs to normalize.

    Returns:
        The context as a tuple of unsigned 32-bit token IDs.

    Raises:
        ValueError: If the context width or token IDs are invalid.
    """
    if len(context_tokens) != CONTEXT_WIDTH:
        raise ValueError(
            f"watermark context must contain {CONTEXT_WIDTH} tokens, got {len(context_tokens)}"
        )
    return tuple(_as_uint32(token, "context token ID") for token in context_tokens)


def _validate_layers(layers: int) -> None:
    """Validate the number of tournament layers.

    Args:
        layers: Number of tournament layers.

    Returns:
        None.

    Raises:
        ValueError: If ``layers`` is less than one.
    """
    if layers < 1:
        raise ValueError(f"layers must be at least 1, got {layers}")


def _watermark_bit_from_context(
    key: bytes,
    context_tokens: tuple[int, ...],
    layer: int,
    token_id: int,
) -> int:
    """Compute one keyed watermark bit for a normalized candidate token.

    Args:
        key: Secret watermark key.
        context_tokens: Validated fixed-width context token IDs.
        layer: Tournament layer number.
        token_id: Candidate or observed token ID.

    Returns:
        A deterministic bit, either ``0`` or ``1``.
    """
    payload = struct.pack(">6I", *context_tokens, layer, token_id)
    digest = hashlib.blake2b(payload, key=key, digest_size=8).digest()
    return digest[0] & 1


def watermark_bit(
    key: bytes,
    context_tokens: Sequence[int],
    layer: int,
    token_id: int,
) -> int:
    """Return the deterministic keyed watermark bit for one token choice.

    Args:
        key: Secret watermark key.
        context_tokens: The preceding four token IDs.
        layer: Tournament layer number.
        token_id: Candidate or observed token ID.

    Returns:
        A deterministic bit, either ``0`` or ``1``.

    Raises:
        ValueError: If the key, context, layer, or token ID is invalid.
    """

    _validate_key(key)
    context = _normalize_context(context_tokens)
    layer = _as_uint32(layer, "layer")
    token_id = _as_uint32(token_id, "token ID")
    return _watermark_bit_from_context(key, context, layer, token_id)


def _sampling_probabilities(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    *,
    warper: TopPLogitsWarper | None = None,
) -> torch.Tensor:
    """Apply temperature and top-p filtering to a logits vector.

    Args:
        logits: One-dimensional next-token logits.
        temperature: Positive sampling temperature.
        top_p: Nucleus sampling threshold in ``(0, 1]``.

    Returns:
        A one-dimensional probability tensor over the vocabulary.

    Raises:
        ValueError: If the logits shape or sampling parameters are invalid.
    """
    if logits.ndim != 1:
        raise ValueError(f"logits must be one-dimensional, got shape {tuple(logits.shape)}")
    if logits.numel() == 0:
        raise ValueError("logits cannot be empty")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    # Transformers logits warpers expect [batch_size, vocabulary_size],
    # while this helper receives one [vocabulary_size] logits vector.
    scores = logits.detach().float().unsqueeze(0)
    scores = scores / temperature

    # Top-p does not inspect token values, but the warper still requires
    # an input_ids tensor so it can use the standard generation interface.
    input_ids = torch.empty(
        (1, 0),
        dtype=torch.long,
        device=logits.device,
    )

    if warper is None:
        warper = TopPLogitsWarper(
            top_p=top_p,
            min_tokens_to_keep=1,
        )

    filtered_scores = warper(input_ids, scores)

    # Remove the artificial batch dimension before converting scores
    # into probabilities for torch.multinomial().
    return torch.softmax(filtered_scores.squeeze(0), dim=-1)


def sample_normally(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    _warper: TopPLogitsWarper | None = None,
) -> int:
    """Sample one token from the filtered model distribution without watermarking.

    Args:
        logits: One-dimensional next-token logits.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        The sampled token ID.
    """
    probabilities = _sampling_probabilities(
        logits,
        temperature,
        top_p,
        warper=_warper,
    )
    return int(torch.multinomial(probabilities, 1).item())


def sample_watermarked_token(
    logits: torch.Tensor,
    context_tokens: Sequence[int],
    key: bytes,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    layers: int = TOURNAMENT_LAYERS,
    _warper: TopPLogitsWarper | None = None,
) -> int:
    """Sample one token using keyed tournament sampling.

    Args:
        logits: One-dimensional next-token logits.
        context_tokens: The preceding four token IDs.
        key: Secret watermark key.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        layers: Number of tournament layers.

    Returns:
        The selected token ID.

    Raises:
        ValueError: If the inputs are invalid.
    """

    _validate_key(key)
    context = _normalize_context(context_tokens)
    _validate_layers(layers)

    probabilities = _sampling_probabilities(
        logits,
        temperature,
        top_p,
        warper=_warper,
    )

    candidate_count = 1 << layers

    sampled = torch.multinomial(probabilities, candidate_count, replacement=True)

    candidates = [int(token_id) for token_id in sampled.tolist()]

    tie_breaks = torch.randint(
        0,
        2,
        (candidate_count - 1,),
        device=logits.device,
    ).tolist()

    tie_index = 0

    for layer in range(layers):
        if layer > 0:
            permutation = torch.randperm(len(candidates), device=logits.device).tolist()

            candidates = [candidates[int(index)] for index in permutation]

        winners: list[int] = []
        for pair_start in range(0, len(candidates), 2):
            left = candidates[pair_start]
            right = candidates[pair_start + 1]

            left_bit = _watermark_bit_from_context(
                key,
                context,
                layer,
                _as_uint32(left, "candidate token ID"),
            )

            right_bit = _watermark_bit_from_context(
                key,
                context,
                layer,
                _as_uint32(right, "candidate token ID"),
            )

            if left_bit > right_bit:
                winner = left

            elif right_bit > left_bit:
                winner = right

            else:
                winner = left if tie_breaks[tie_index] == 0 else right

            tie_index += 1
            winners.append(winner)

        candidates = winners

    return candidates[0]


def score_tokens(
    token_ids: Sequence[int],
    key: bytes,
    *,
    layers: int = TOURNAMENT_LAYERS,
) -> tuple[int, int, int]:
    """Count keyed watermark bits in a token sequence.

    Args:
        token_ids: Token IDs to score.
        key: Secret watermark key.
        layers: Number of watermark bits scored per token.

    Returns:
        A ``(ones, total_bits, scored_contexts)`` tuple. The first four tokens
        are skipped, and repeated contexts contribute no additional bits.

    Raises:
        ValueError: If the key, token IDs, or layer count is invalid.
    """
    _validate_key(key)
    _validate_layers(layers)
    tokens = tuple(_as_uint32(token_id, "token ID") for token_id in token_ids)

    ones = 0
    total_bits = 0
    scored_contexts = 0
    seen_contexts: set[tuple[int, ...]] = set()

    for position in range(CONTEXT_WIDTH, len(tokens)):
        context = tokens[position - CONTEXT_WIDTH : position]
        if context in seen_contexts:
            continue

        seen_contexts.add(context)
        token_id = tokens[position]
        for layer in range(layers):
            ones += _watermark_bit_from_context(key, context, layer, token_id)
            total_bits += 1
        scored_contexts += 1

    return ones, total_bits, scored_contexts
