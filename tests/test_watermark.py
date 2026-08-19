from __future__ import annotations

import torch

from src.watermark import CONTEXT_WIDTH, sample_watermarked_token, score_tokens, watermark_bit


def test_watermark_bit_is_deterministic_and_uses_all_inputs() -> None:
    key = bytes(range(32))
    other_key = bytes(range(1, 33))
    context = (11, 22, 33, 44)
    token_ids = range(128)

    def bits(test_key: bytes, test_context: tuple[int, ...], layer: int) -> tuple[int, ...]:
        return tuple(
            watermark_bit(test_key, test_context, layer, token_id) for token_id in token_ids
        )

    baseline = bits(key, context, 1)
    assert baseline == bits(key, context, 1)
    assert set(baseline) <= {0, 1}
    assert baseline != bits(other_key, context, 1)
    assert baseline != bits(key, (11, 22, 33, 45), 1)
    assert baseline != bits(key, context, 2)


def _different_bit_candidates(key: bytes, context: tuple[int, ...]) -> tuple[int, int]:
    candidates: dict[int, int] = {}
    for token_id in range(256):
        bit = watermark_bit(key, context, 0, token_id)
        candidates.setdefault(bit, token_id)
        if 0 in candidates and 1 in candidates:
            return candidates[0], candidates[1]
    raise AssertionError("candidate search did not find both keyed bits")


def test_tournament_prefers_candidate_with_higher_keyed_bit(monkeypatch) -> None:
    key = bytes(range(32))
    context = (11, 22, 33, 44)
    low, high = _different_bit_candidates(key, context)
    logits = torch.zeros(2)
    samples = iter((torch.tensor([low, high]), torch.tensor([high, low])))

    def fake_multinomial(probabilities, num_samples, replacement):
        assert probabilities.ndim == 1
        assert num_samples == 2
        assert replacement is True
        return next(samples)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    assert (
        sample_watermarked_token(logits, context, key, layers=1, temperature=1.0, top_p=1.0) == high
    )
    assert (
        sample_watermarked_token(logits, context, key, layers=1, temperature=1.0, top_p=1.0) == high
    )


def test_score_tokens_counts_repeated_context_once() -> None:
    key = bytes(range(32))
    token_ids = [1, 2, 3, 4, 10, 1, 2, 3, 4, 11]

    full = score_tokens(token_ids, key, layers=3)
    without_repeated_context = score_tokens(token_ids[:-1], key, layers=3)

    assert full == without_repeated_context
    assert full[2] == len(token_ids) - CONTEXT_WIDTH - 1
    assert full[1] == full[2] * 3
