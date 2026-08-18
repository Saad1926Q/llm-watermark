import argparse
import json
import secrets
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generate import encode_prompt, generate_watermarked_text
from src.watermark import (
    DEFAULT_KEY_PATH,
    KEY_SIZE_BYTES,
    TOURNAMENT_LAYERS,
    calculate_statistics,
    load_key,
    score_tokens,
)

DEFAULT_SEED = 42


def _read_prompts(path: Path) -> list[str]:
    """Read non-empty prompts from a one-prompt-per-line text file.

    Args:
        path: Text file containing one prompt on each line.

    Returns:
        Stripped, non-empty prompt strings in file order.
    """
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _different_key(key: bytes) -> bytes:
    """Generate a random key guaranteed not to equal the watermark key.

    Args:
        key: Existing watermark key to avoid returning.

    Returns:
        A random key with the same length as ``key``.
    """
    candidate = secrets.token_bytes(KEY_SIZE_BYTES)
    while candidate == key:
        candidate = secrets.token_bytes(KEY_SIZE_BYTES)
    return candidate


def _generate_normal_text(
    model,
    tokenizer,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate one ordinary stochastic continuation with ``model.generate``.

    Args:
        model: Evaluation-mode causal language model.
        tokenizer: Tokenizer corresponding to ``model``.
        prompt: Text to use as the generation prefix.
        device: Device on which model inputs are stored.
        max_new_tokens: Maximum number of tokens to append to ``prompt``.
        temperature: Sampling temperature passed to Transformers.
        top_p: Nucleus sampling threshold passed to Transformers.

    Returns:
        The decoded generated continuation, excluding the prompt.
    """
    encoded = encode_prompt(tokenizer, prompt)

    input_ids = encoded["input_ids"].to(device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
    }

    if tokenizer.eos_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(input_ids=input_ids, **generation_kwargs)

    generated_ids = output_ids[0, input_ids.shape[1] :]

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def _score_text(text: str, tokenizer, key: bytes) -> dict[str, float | int] | None:
    """Score one generated continuation, returning ``None`` when too short.

    Args:
        text: Generated continuation to tokenize and score.
        tokenizer: Tokenizer used during generation.
        key: Secret watermark key used for scoring.

    Returns:
        A statistics dictionary, or ``None`` if no token has a complete
        watermark context.
    """
    encoded = tokenizer(text, add_special_tokens=False)
    ones, total_bits = score_tokens(encoded["input_ids"], key)
    if total_bits == 0:
        return None

    score, z_score, p_value = calculate_statistics(ones, total_bits)

    return {
        "ones": ones,
        "total_bits": total_bits,
        "score": score,
        "z_score": z_score,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare normal and watermarked generation on prompt fixtures."
    )

    parser.add_argument("--model", required=True, help="Hugging Face causal LM name or path")

    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/prompts.txt"),
        help="One prompt per line",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("comparison.jsonl"),
        help="JSONL output path",
    )

    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--layers", type=int, default=TOURNAMENT_LAYERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--key-path", type=Path, default=DEFAULT_KEY_PATH)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--device-map",
        help='Transformers device map, such as "auto"; requires Accelerate',
    )

    args = parser.parse_args()

    prompts = _read_prompts(args.prompts)

    if not prompts:
        parser.error("the prompt file must contain at least one non-empty line")

    device = torch.device(args.device)

    key = load_key(args.key_path)
    wrong_key = _different_key(key)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype="auto",
            device_map=args.device_map,
        )
    else:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
        )
        model.to(device)

    model.eval()
    input_device = torch.device(str(model.get_input_embeddings().weight.device))

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with args.output_file.open("w", encoding="utf-8") as result_file:
        for index, prompt in enumerate(prompts):
            torch.manual_seed(args.seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            normal_text = _generate_normal_text(
                model,
                tokenizer,
                prompt,
                device=input_device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            torch.manual_seed(args.seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            watermarked_text = generate_watermarked_text(
                model,
                tokenizer,
                prompt,
                key,
                device=input_device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                layers=args.layers,
            )

            result = {
                "prompt": prompt,
                "normal_text": normal_text,
                "watermarked_text": watermarked_text,
                "normal_score": _score_text(normal_text, tokenizer, key),
                "watermarked_score": _score_text(watermarked_text, tokenizer, key),
                "wrong_key_score": _score_text(watermarked_text, tokenizer, wrong_key),
            }

            result_file.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(f"[{index + 1}/{len(prompts)}] generated")

    print(f"Saved comparison results to {args.output_file}")


if __name__ == "__main__":
    main()
