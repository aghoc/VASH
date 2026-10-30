import json
from pathlib import Path

from vagt_orca.config import PROJECT_ROOT
from vagt_orca.datasets.egoobjects import convert_egoobjects, load_class_mapping, normalize_category
from vagt_orca.datasets.validate import validate_yolo_dataset


def test_conversion_and_integrity(tmp_path: Path) -> None:
    images = tmp_path / "raw"
    images.mkdir()
    records = []
    annotations = []
    for index, video_id in enumerate(("a", "b", "c"), 1):
        filename = f"{video_id}_{index}.jpg"
        (images / filename).write_bytes(b"fixture")
        records.append(
            {"id": index, "file_name": filename, "width": 100, "height": 80, "video_id": video_id}
        )
        annotations.append(
            {"id": index, "image_id": index, "category_id": 9, "bbox": [-5, 10, 50, 40]}
        )
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": records,
                "annotations": annotations,
                "categories": [{"id": 9, "name": "water bottle"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "yolo"
    report = convert_egoobjects(
        [annotation_path], images, output, split_ratios=(1, 1, 1), link_mode="symlink"
    )
    assert report["stats"]["images"] == 3
    assert report["stats"]["annotations"] == 3
    integrity = validate_yolo_dataset(output / "data.yaml")
    assert integrity["valid"], integrity["errors"]


def test_validator_detects_invalid_bbox(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
    (tmp_path / "images/train/a.jpg").write_bytes(b"fixture")
    (tmp_path / "labels/train/a.txt").write_text("0 1.2 0.5 0.3 0.3\n", encoding="utf-8")
    (tmp_path / "data.yaml").write_text(
        f"path: {tmp_path}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: bottle\n",
        encoding="utf-8",
    )
    report = validate_yolo_dataset(tmp_path / "data.yaml")
    assert not report["valid"]
    assert any("outside normalized range" in error for error in report["errors"])


def test_validator_detects_bbox_edges_outside_image(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
    (tmp_path / "images/train/a.jpg").write_bytes(b"fixture")
    (tmp_path / "labels/train/a.txt").write_text("0 0.9 0.5 0.3 0.3\n", encoding="utf-8")
    (tmp_path / "data.yaml").write_text(
        f"path: {tmp_path}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: bottle\n",
        encoding="utf-8",
    )
    report = validate_yolo_dataset(tmp_path / "data.yaml")
    assert not report["valid"]
    assert any("bbox edges outside image" in error for error in report["errors"])


def test_official_egoobjects_aliases_cover_project_taxonomy() -> None:
    _, aliases = load_class_mapping(PROJECT_ROOT / "configs/classes.yaml")
    assert aliases[normalize_category("tin can")] == 1
    assert aliases[normalize_category("tea cup")] == 2
    assert aliases[normalize_category("basketball")] == 4
    assert aliases[normalize_category("mobile phone")] == 5
    assert aliases[normalize_category("house/car key")] == 12


def test_conversion_supports_egoobjects_v1_url_and_separate_metadata(tmp_path: Path) -> None:
    images = tmp_path / "raw"
    images.mkdir()
    (images / "frame.jpg").write_bytes(b"fixture")
    (images / "train-frame.jpg").write_bytes(b"fixture")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"categories": [{"id": 67, "name": "bottle"}]}), encoding="utf-8"
    )
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 10,
                        "url": "frame.jpg",
                        "width": 100,
                        "height": 80,
                        "group_id": "sequence-a",
                        "subset": "val",
                    },
                    {
                        "id": 11,
                        "url": "train-frame.jpg",
                        "width": 100,
                        "height": 80,
                        "group_id": "sequence-train",
                    },
                ],
                "annotations": [
                    {"id": 1, "image_id": 10, "category_id": 67, "bbox": [10, 5, 30, 20]},
                    {"id": 2, "image_id": 10, "bbox": [10, 5, 30, 20]},
                    {"id": 3, "image_id": 11, "category_id": 67, "bbox": [10, 5, 30, 20]},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "yolo"
    report = convert_egoobjects([metadata, annotations], images, output)
    assert report["stats"]["images"] == 2
    assert report["stats"]["missing_category_ids"] == 1
    assert sum(report["split_image_counts"].values()) == 2
    assert report["split_policy"] == "sequence_safe_group_resplit_with_class_coverage"


def test_group_split_covers_rare_class_without_sequence_leakage(tmp_path: Path) -> None:
    images = tmp_path / "raw"
    images.mkdir()
    records = []
    annotations = []
    for index in range(1, 7):
        filename = f"coin-{index}.jpg"
        (images / filename).write_bytes(b"fixture")
        records.append(
            {
                "id": index,
                "file_name": filename,
                "width": 100,
                "height": 80,
                "group_id": f"sequence-{index}",
            }
        )
        annotations.append(
            {"id": index, "image_id": index, "category_id": 142, "bbox": [10, 5, 30, 20]}
        )
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": records,
                "annotations": annotations,
                "categories": [{"id": 142, "name": "coin"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "yolo"
    report = convert_egoobjects([annotation_path], images, output, split_ratios=(8, 1, 1))
    assert set(report["split_image_counts"]) == {"train", "val", "test"}
    assert report["classes_with_fewer_than_three_groups"] == []
