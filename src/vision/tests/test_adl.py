from pathlib import Path

from vagt_orca.datasets.adl import AdlBox, load_adl_annotations, yolo_line


def test_load_adl_annotations_filters_ambiguous_classes(tmp_path: Path) -> None:
    source = tmp_path / "objects.txt"
    source.write_text(
        "1 10 20 30 40 00000030 0 bottle\n"
        "2 1 2 8 9 00000030 0 knife/spoon/fork\n"
        "3 3 4 13 14 00000060 0 cell_phone\n",
        encoding="utf-8",
    )
    frames = load_adl_annotations(source)
    assert sorted(frames) == [30, 60]
    assert frames[30][0].class_id == 0
    assert frames[60][0].class_id == 5


def test_yolo_line_clamps_box() -> None:
    assert yolo_line(AdlBox(2, -10, 10, 110, 90), 100, 100) == (
        "2 0.50000000 0.50000000 1.00000000 0.80000000"
    )
