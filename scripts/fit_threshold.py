"""Fit a conservative empirical watermark threshold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.calibration import DEFAULT_REQUIRED_TOKENS
from src.watermark import TOURNAMENT_LAYERS


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for threshold calibration.

    Returns:
        An argument parser containing development-data and fitting options.
    """
    parser = argparse.ArgumentParser(
        description="Fit a calibrated upper-tail mean-score threshold."
    )
    parser.add_argument("--development", required=True, type=Path, help="Development JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Output calibration JSON")
    parser.add_argument("--key", type=Path)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--required-tokens", type=int, default=DEFAULT_REQUIRED_TOKENS)
    parser.add_argument("--layers", type=int, default=TOURNAMENT_LAYERS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Fit a threshold from development rows only.

    Args:
        argv: Optional command-line arguments; defaults to ``sys.argv``.

    Returns:
        ``0`` on success or ``2`` when calibration input is invalid.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from src.calibration import CalibrationError, fit_calibration, read_jsonl
    from src.watermark import load_key

    try:
        key = load_key(args.key) if args.key else load_key()
        development_rows = read_jsonl(args.development)
        if not development_rows:
            raise CalibrationError("development JSONL is empty")

        tokenizer_name = development_rows[0].get("tokenizer")
        if not isinstance(tokenizer_name, str) or not tokenizer_name:
            raise CalibrationError("development rows require tokenizer metadata")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        artifact = fit_calibration(
            development_rows,
            key,
            tokenizer,
            target_fpr=args.target_fpr,
            required_tokens=args.required_tokens,
            layers=args.layers,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (CalibrationError, OSError, ValueError) as exc:
        print(f"fit-threshold: {exc}", file=sys.stderr)
        return 2

    print(
        f"wrote {args.output}: threshold={artifact['threshold']:.12g} "
        f"development_fpr={artifact['observed_development_fpr']:.12g} "
        f"insufficient_rows={artifact['development_insufficient_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
