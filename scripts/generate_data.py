"""Generate separate development and test JSONL files for calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from src.generate import GenerationResult, generate_normal_batch, generate_watermarked_batch
from src.watermark import DEFAULT_KEY_PATH, TOURNAMENT_LAYERS, load_key

DEFAULT_SEED = 42
DEFAULT_DATASET = "databricks/databricks-dolly-15k"
DEFAULT_OUTPUT_DIR = Path("data/calibration")


def _render_prompt(row: dict[str, Any]) -> str:
    """Render a dataset instruction and optional context into a prompt.

    Args:
        row: Dataset row containing ``instruction`` and optional ``context``.

    Returns:
        The text prompt passed to the language model.

    Raises:
        ValueError: If the row has no non-empty instruction.
    """
    instruction = str(row.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("Dataset row is missing a non-empty instruction")

    context = str(row.get("context", "")).strip()
    if context:
        return f"{instruction}\n\nContext:\n{context}"
    return instruction


def _load_model(
    model_name: str,
    *,
    device: Any,
    device_map: str | None,
) -> tuple[Any, Any, Any]:
    """Load a causal LM, tokenizer, and the model's actual input device.

    Args:
        model_name: Hugging Face model name or local model path.
        device: Fallback device used when ``device_map`` is not supplied.
        device_map: Optional Transformers device-map configuration.

    Returns:
        A ``(model, tokenizer, input_device)`` tuple.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device_map,
        )
    else:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model.to(device)

    model.eval()
    input_device = torch.device(str(model.get_input_embeddings().weight.device))
    return model, tokenizer, input_device


def _generation_row(
    *,
    prompt_id: str,
    kind: str,
    result: GenerationResult,
    model_name: str,
    tokenizer_name: str,
) -> dict[str, Any]:
    """Build one JSON-serializable response row.

    Args:
        prompt_id: Stable identifier for the source prompt.
        kind: Ground-truth response type.
        result: Generated response and its token IDs.
        model_name: Model name recorded in the row metadata.
        tokenizer_name: Tokenizer name recorded in the row metadata.

    Returns:
        A JSON-compatible calibration row containing response text and metadata.
    """
    return {
        "prompt_id": prompt_id,
        "kind": kind,
        "text": result.text,
        "model": model_name,
        "tokenizer": tokenizer_name,
    }


def _generation_rows(
    *,
    prompt_id: str,
    normal_result: GenerationResult,
    watermarked_result: GenerationResult | None,
    model_name: str,
    tokenizer_name: str,
) -> tuple[dict[str, Any], ...]:
    """Build one development row or a paired test response set.

    Args:
        prompt_id: Stable identifier for the source prompt.
        normal_result: Ordinary generated response.
        watermarked_result: Watermarked response, or ``None`` for development.
        model_name: Model name recorded in each row.
        tokenizer_name: Tokenizer name recorded in each row.

    Returns:
        A tuple containing one ordinary row and, for test data, one watermarked row.
    """
    rows = [
        _generation_row(
            prompt_id=prompt_id,
            kind="unwatermarked",
            result=normal_result,
            model_name=model_name,
            tokenizer_name=tokenizer_name,
        )
    ]

    if watermarked_result is not None:
        rows.append(
            _generation_row(
                prompt_id=prompt_id,
                kind="watermarked",
                result=watermarked_result,
                model_name=model_name,
                tokenizer_name=tokenizer_name,
            )
        )

    return tuple(rows)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for response dataset generation.

    Returns:
        An argument parser containing model, split, sampling, and output options.
    """
    parser = argparse.ArgumentParser(
        description="Generate ordinary development and paired evaluation responses."
    )
    parser.add_argument("--model", required=True, help="Hugging Face causal LM name or path")
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--development-count",
        type=int,
        default=1200,
        help="Number of ordinary-only development prompts",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=400,
        help="Number of test prompts; each gets ordinary and watermarked responses",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts processed per generation batch",
    )
    parser.add_argument("--layers", type=int, default=TOURNAMENT_LAYERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        help="Model input device (defaults to CUDA when available, otherwise CPU)",
    )
    parser.add_argument(
        "--device-map",
        help='Transformers device map, such as "auto"; requires Accelerate',
    )
    return parser


def main() -> None:
    """Generate ordinary development and paired test JSONL files.

    Returns:
        None.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.development_count < 0 or args.test_count < 0:
        parser.error("split counts must be non-negative")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.layers < 1:
        parser.error("--layers must be positive")

    import torch
    from datasets import load_dataset

    key = load_key(args.key)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, input_device = _load_model(
        args.model,
        device=torch.device(device_name),
        device_map=args.device_map,
    )

    dataset = load_dataset(args.dataset, split=args.dataset_split)
    total_count = args.development_count + args.test_count
    if total_count > len(dataset):
        raise ValueError(f"requested {total_count} rows but dataset has only {len(dataset)}")

    selected = dataset.shuffle(seed=args.seed).select(range(total_count))

    tokenizer_name = str(getattr(tokenizer, "name_or_path", args.model))

    development_path = args.output_dir / "development.jsonl"

    test_path = args.output_dir / "test.jsonl"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (
        development_path.open("w", encoding="utf-8") as development_file,
        test_path.open("w", encoding="utf-8") as test_file,
        tqdm(total=total_count, desc="Generating", unit="prompt") as progress,
    ):
        splits = (
            ("development", 0, args.development_count, development_file, False),
            (
                "test",
                args.development_count,
                args.test_count,
                test_file,
                True,
            ),
        )

        for split_name, split_start, split_count, output_file, is_test in splits:
            for batch_offset in range(0, split_count, args.batch_size):
                batch_end = min(batch_offset + args.batch_size, split_count)

                source_indices = range(
                    split_start + batch_offset,
                    split_start + batch_end,
                )

                prompts = [
                    _render_prompt(selected[source_index]) for source_index in source_indices
                ]

                batch_seed = args.seed + split_start + batch_offset

                torch.manual_seed(batch_seed)

                normal_results = generate_normal_batch(
                    model,
                    tokenizer,
                    prompts,
                    device=input_device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                watermarked_results: list[GenerationResult] = []

                if is_test:
                    torch.manual_seed(batch_seed)
                    watermarked_results = generate_watermarked_batch(
                        model,
                        tokenizer,
                        prompts,
                        key,
                        device=input_device,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        layers=args.layers,
                    )

                for local_index, (source_index, normal_result) in enumerate(
                    zip(source_indices, normal_results, strict=True)
                ):
                    split_index = source_index - split_start
                    prompt_id = f"{split_name}-{split_index:04d}"

                    watermarked_result = watermarked_results[local_index] if is_test else None

                    rows = _generation_rows(
                        prompt_id=prompt_id,
                        normal_result=normal_result,
                        watermarked_result=watermarked_result,
                        model_name=args.model,
                        tokenizer_name=tokenizer_name,
                    )

                    for generated_row in rows:
                        output_file.write(json.dumps(generated_row, ensure_ascii=False) + "\n")

                progress.update(len(prompts))

    print(f"Saved development data to {development_path}")
    print(f"Saved test data to {test_path}")


if __name__ == "__main__":
    main()
