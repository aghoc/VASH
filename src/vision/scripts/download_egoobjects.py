#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

OFFICIAL_FILES = (
    "EgoObjectsV1_images.zip",
    "EgoObjectsV1_unified_train.json",
    "EgoObjectsV1_unified_eval.json",
    "EgoObjectsV1_unified_metadata.json",
)
META_PAGE = "https://ai.meta.com/datasets/egoobjects-downloads/"
HF_MIRROR_REPOSITORY = "Myszz/EgoObjects"
HF_MIRROR_REVISION = "b9b6b37cddcddf28b467f657edc81b7824eb906d"
HF_MIRROR_FILES = {
    "Meta_EgoObjects.zip": 39_826_913_851,
    "Meta_EgoObjects_Train.json": 72_487_045,
    "Meta_EgoObjects_Test.json": 161_152_266,
    "Meta_EgoObjects_Val.json": 85_217,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path) -> None:
    partial = output.with_suffix(output.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and response.status == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as target:
            shutil.copyfileobj(response, target, length=8 * 1024 * 1024)
    partial.replace(output)


def mirror_urls() -> dict[str, str]:
    base = f"https://huggingface.co/datasets/{HF_MIRROR_REPOSITORY}/resolve"
    return {
        name: f"{base}/{HF_MIRROR_REVISION}/data/{name}?download=true"
        for name in HF_MIRROR_FILES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify EgoObjects v1 data")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--urls", type=Path, help="JSON mapping official filename to signed URL")
    source.add_argument(
        "--huggingface-mirror",
        action="store_true",
        help="Use the pinned public recovery mirror because Meta's download view is empty",
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw/egoobjects"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.huggingface_mirror:
        urls = mirror_urls()
        expected_sizes: dict[str, int] = HF_MIRROR_FILES
        source_info = {
            "kind": "huggingface_recovery_mirror",
            "repository": HF_MIRROR_REPOSITORY,
            "revision": HF_MIRROR_REVISION,
            "official_page": META_PAGE,
        }
    elif args.urls:
        urls = json.loads(args.urls.read_text(encoding="utf-8"))
        expected_sizes = {}
        source_info = {"kind": "official_signed_urls", "official_page": META_PAGE}
    else:
        print(f"Obtain/accept the dataset terms at: {META_PAGE}")
        print("Use --huggingface-mirror for the pinned recovery mirror, or create a JSON map:")
        print(json.dumps({name: "SIGNED_URL" for name in OFFICIAL_FILES}, indent=2))
        raise SystemExit(2)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"source": source_info, "files": {}}
    file_manifest: dict[str, object] = manifest["files"]  # type: ignore[assignment]
    for name, url in urls.items():
        destination = args.output / name
        expected_size = expected_sizes.get(name)
        if not args.verify_only and (
            not destination.exists() or (expected_size and destination.stat().st_size != expected_size)
        ):
            download(str(url), destination)
        if not destination.is_file():
            raise FileNotFoundError(destination)
        size = destination.stat().st_size
        if expected_size is not None and size != expected_size:
            raise RuntimeError(f"{name}: expected {expected_size} bytes, found {size}")
        file_manifest[name] = {"bytes": size, "sha256": sha256(destination)}
    (args.output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
