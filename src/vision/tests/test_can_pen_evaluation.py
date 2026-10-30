import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "evaluate_can_pen", Path(__file__).resolve().parents[1] / "training/evaluate_can_pen.py"
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_duplicate_predictions_and_wrong_class_do_not_inflate_recall():
    records = [
        {"targets": [[0, 0, 10, 10, 0], [20, 0, 30, 10, 0]],
         "predictions": [[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 10, 0.8, 0],
                         [20, 0, 30, 10, 0.9, 1]]},
        {"targets": [], "predictions": [[0, 0, 10, 10, 0.7, 0]]},
        {"targets": [], "predictions": [[0, 0, 10, 10, 0.3, 0]]},
    ]
    result = evaluation.measure(records, 0, 0.5)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 1)
    assert result["precision"] == 1 / 3
    assert result["recall"] == 0.5
    assert result["negative_images"] == 2
    assert result["negative_fp_images"] == 1
