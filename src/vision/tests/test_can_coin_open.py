import runpy
from pathlib import Path

import cv2
import numpy as np
import pytest

MODULE = runpy.run_path(str(Path(__file__).parents[1] / "datasets/prepare_can_coin_open.py"))


def test_rf_sequences_do_not_follow_leaky_original_split():
    group = MODULE["rf_group"]
    assert group("20220422131017669_jpg.rf.aaa.jpg") == ("train", "rf_video_20220422")
    assert group("20220422130958000_jpg.rf.bbb.jpg")[1] == "rf_video_20220422"
    assert group("IMG_20220427_154053_Burst01_jpg.rf.aaa.jpg") == group(
        "IMG_20220427_154058_Burst19_jpg.rf.bbb.jpg"
    )
    assert group("IMG_2647_JPG_jpg.rf.aaa.jpg") == group("IMG_2648_JPG_jpg.rf.bbb.jpg")
    assert group("IMG_2647_JPG_jpg.rf.aaa.jpg")[0] in {"val", "test"}
    with pytest.raises(ValueError):
        group("unknown.jpg")


def test_bbox_clips_edges_and_rejects_invalid_boxes():
    box = MODULE["box_line"]
    assert box(13, [-0.1, 0.0, 1.1, 1.0]) == "13 0.50000000 0.50000000 1.00000000 1.00000000"
    with pytest.raises(ValueError):
        box(13, [0.6, 0.0, 0.4, 1.0])


def test_openimages_mapping_excludes_ambiguous_container_classes():
    mapping = MODULE["OI_MAPPING"]
    assert mapping["Tin can"] == 1
    assert mapping["Coin"] == 13
    assert "Watering can" not in mapping
    assert "Waste container" not in mapping


def test_decoded_duplicates_prefer_holdout_over_training(tmp_path):
    frozen = tmp_path / "frozen"
    for split in ("val", "test"):
        (frozen / "images" / split).mkdir(parents=True)
    root = tmp_path / "new"
    records = []
    for split in ("train", "test"):
        image = root / "images" / split / f"{split}.png"
        image.parent.mkdir(parents=True)
        cv2.imwrite(str(image), np.full((32, 32, 3), 127, dtype=np.uint8))
        label = root / "labels" / split / f"{split}.txt"
        label.parent.mkdir(parents=True)
        label.write_text("13 0.5 0.5 0.5 0.5\n")
        records.append(
            {"image": str(image), "split": split, "source": "Open Images", "group": split}
        )
    kept, removed = MODULE["deduplicate"](records, frozen)
    assert [r["split"] for r in kept] == ["test"]
    assert [r["split"] for r in removed] == ["train"]
    assert not (root / "images" / "train" / "train.png").exists()
    assert not (root / "labels" / "train" / "train.txt").exists()
