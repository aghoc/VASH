#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

MODEL_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
MODEL_CHECKSUMS = {
    "yolo26m": "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7",
    "yolo26s": "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, model: str) -> None:
    actual = sha256(path)
    expected = MODEL_CHECKSUMS[model]
    if actual != expected:
        raise RuntimeError(f"{model} checksum mismatch: expected {expected}, found {actual}")


def restore(model: str, output: Path, *, verify_only: bool = False) -> None:
    if not verify_only and not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(".pt.part")
        try:
            with (
                urllib.request.urlopen(f"{MODEL_RELEASE}/{model}.pt", timeout=30) as response,
                partial.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            verify(partial, model)
            partial.replace(output)
        finally:
            partial.unlink(missing_ok=True)
    if not output.is_file():
        raise FileNotFoundError(output)
    verify(output, model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a pinned official YOLO26 checkpoint")
    parser.add_argument("--model", choices=MODEL_CHECKSUMS, default="yolo26m")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output or Path("models/pretrained") / f"{args.model}.pt"
    restore(args.model, output, verify_only=args.verify_only)
    print(f"verified {output}: sha256={sha256(output)}")


if __name__ == "__main__":
    main()
