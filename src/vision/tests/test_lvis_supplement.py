import json
from pathlib import Path

from vagt_orca.datasets.lvis import _load_lvis, prepare


def _lvis_fixture(path: Path, image_url: str, image_id: int) -> None:
    path.write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 1, "name": "credit_card"},
                    {"id": 2, "name": "Lego"},
                ],
                "images": [
                    {
                        "id": image_id,
                        "file_name": f"{image_id}.jpg",
                        "coco_url": image_url,
                        "width": 100,
                        "height": 80,
                    }
                ],
                "annotations": [
                    {"id": 1, "image_id": image_id, "category_id": 1, "bbox": [10, 8, 20, 16]},
                    {"id": 2, "image_id": image_id, "category_id": 2, "bbox": [40, 20, 10, 10]},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_prepare_lvis_supplement(tmp_path: Path) -> None:
    source_image = tmp_path / "source.jpg"
    source_image.write_bytes(b"fixture")
    train_json = tmp_path / "train.json"
    val_json = tmp_path / "val.json"
    _lvis_fixture(train_json, source_image.as_uri(), 1)
    _lvis_fixture(val_json, source_image.as_uri(), 2)
    output = tmp_path / "dataset"
    report = prepare(train_json, val_json, output, 10, 10, 1, 42)
    assert report["annotations"] == {"card": 2, "small_block": 2}
    assert report["images"] == 2
    labels = list((output / "labels").glob("*/*.txt"))
    assert len(labels) == 2
    assert {line.split()[0] for label in labels for line in label.read_text().splitlines()} == {
        "11",
        "14",
    }


def test_drink_can_and_coin_subclasses_are_kept_without_watering_can(tmp_path: Path) -> None:
    source = tmp_path / "lvis.json"
    source.write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 1, "name": "beer_can"},
                    {"id": 2, "name": "penny_(coin)"},
                    {"id": 3, "name": "watering_can"},
                ],
                "images": [{"id": 10}],
                "annotations": [
                    {"image_id": 10, "category_id": category_id} for category_id in (1, 2, 3)
                ],
            }
        )
    )
    _, annotations = _load_lvis(source)
    assert {item["category_name"] for item in annotations[10]} == {"beer_can", "penny_(coin)"}
