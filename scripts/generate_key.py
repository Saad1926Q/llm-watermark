import argparse
import secrets
from pathlib import Path

from src.watermark import DEFAULT_KEY_PATH, KEY_SIZE_BYTES


def generate_key(path: Path = DEFAULT_KEY_PATH, overwrite: bool = False) -> None:
    """Generate and save a cryptographically random 256-bit watermark key.

    Args:
        path: Destination path for the raw key bytes.
        overwrite: Whether to replace an existing key file.

    Returns:
        None.

    Raises:
        FileExistsError: If the destination exists and ``overwrite`` is false.
        OSError: If the key cannot be written or its permissions changed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "wb" if overwrite else "xb"

    with path.open(mode) as key_file:
        key_file.write(secrets.token_bytes(KEY_SIZE_BYTES))

    path.chmod(0o600)


def main() -> None:
    """Parse key-generation options and write the default key.

    Returns:
        None.
    """
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
