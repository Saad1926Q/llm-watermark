import argparse
import secrets
from pathlib import Path

KEY_SIZE_BYTES = 32
DEFAULT_KEY_PATH = Path(__file__).parent / "keys" / "watermark.key"


def generate_key(path: Path = DEFAULT_KEY_PATH, overwrite: bool = False) -> None:
    """Generate and save a 256-bit watermark key."""
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "wb" if overwrite else "xb"

    with path.open(mode) as key_file:
        key_file.write(secrets.token_bytes(KEY_SIZE_BYTES))

    path.chmod(0o600)


def main():
    parser = argparse.ArgumentParser(description="Generate the watermark secret key.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the existing key",
    )
    args = parser.parse_args()

    try:
        generate_key(overwrite=args.force)
    except FileExistsError:
        parser.error(f"key already exists at {DEFAULT_KEY_PATH}; use --force to replace it")

    print(f"Generated watermark key at {DEFAULT_KEY_PATH}")


if __name__ == "__main__":
    main()
