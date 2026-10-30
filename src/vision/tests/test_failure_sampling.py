from pathlib import Path

from vagt_orca.evaluation import select_class_balanced_images


def test_failure_sampling_covers_all_classes_and_is_deterministic(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    images = []
    for class_id in range(15):
        image = tmp_path / f"class_{class_id:02d}.jpg"
        images.append(image)
        (labels / f"{image.stem}.txt").write_text(
            f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )

    first = select_class_balanced_images(images, labels, 15, 15)
    second = select_class_balanced_images(images, labels, 15, 15)

    assert first == second
    assert len(first) == 15
    assert {int((labels / f"{image.stem}.txt").read_text().split()[0]) for image in first} == set(
        range(15)
    )
