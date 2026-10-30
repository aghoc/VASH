from __future__ import annotations

from pathlib import Path


def select_class_balanced_images(
    images: list[Path], label_dir: Path, limit: int, class_count: int
) -> list[Path]:
    """Select deterministically while rotating through classes with remaining images."""
    if limit >= len(images):
        return images
    buckets: dict[int, list[Path]] = {class_id: [] for class_id in range(class_count)}
    unlabeled: list[Path] = []
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        class_ids = {
            int(line.split(maxsplit=1)[0])
            for line in label.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not class_ids:
            unlabeled.append(image)
        for class_id in sorted(class_ids):
            if class_id in buckets:
                buckets[class_id].append(image)

    selected: list[Path] = []
    selected_set: set[Path] = set()
    offsets = {class_id: 0 for class_id in buckets}
    while len(selected) < limit:
        added = False
        for class_id in range(class_count):
            bucket = buckets[class_id]
            offset = offsets[class_id]
            while offset < len(bucket) and bucket[offset] in selected_set:
                offset += 1
            offsets[class_id] = offset
            if offset < len(bucket):
                image = bucket[offset]
                offsets[class_id] += 1
                selected.append(image)
                selected_set.add(image)
                added = True
                if len(selected) == limit:
                    return selected
        if not added:
            break
    for image in [*unlabeled, *images]:
        if image not in selected_set:
            selected.append(image)
            selected_set.add(image)
            if len(selected) == limit:
                break
    return selected
