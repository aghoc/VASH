#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

from vagt_orca.config import load_yaml
from vagt_orca.datasets.validate import validate_yolo_dataset
from vagt_orca.gpu import gpu_summary, require_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-only FP32 YOLO26s fine-tuning")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--project", type=Path, default=Path("artifacts/training"))
    parser.add_argument(
        "--model-output", type=Path, default=Path("artifacts/models/yolo26s_best.pt")
    )
    parser.add_argument("--name", default="yolo26s_egoobjects_15")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    config = load_yaml(args.config)
    device = int(config["device"])
    require_cuda(device)
    print(f"GPU: {gpu_summary(device)}")
    integrity = validate_yolo_dataset(args.data)
    if not integrity["valid"]:
        raise RuntimeError(f"Dataset integrity failed: {integrity['errors'][:10]}")
    if integrity["missing_classes"]:
        raise RuntimeError(f"Dataset has no annotations for classes {integrity['missing_classes']}")
    incomplete_splits = {
        split: missing
        for split, missing in integrity["missing_classes_by_split"].items()
        if missing
    }
    if incomplete_splits:
        raise RuntimeError(f"Dataset splits do not cover every class: {incomplete_splits}")
    from ultralytics import YOLO

    model = YOLO(str(args.resume or config["base_model"]))

    def record_class_metrics(trainer) -> None:
        box = trainer.validator.metrics.box
        observed = {int(index) for index in box.ap_class_index}
        names = trainer.data["names"]
        record = {
            "epoch": trainer.epoch + 1,
            "phase": "epoch" if trainer.validator.training else "best_checkpoint_validation",
            "per_class_map": {
                names[index]: float(box.maps[index]) if index in observed else None
                for index in range(len(names))
            },
        }
        output = Path(trainer.save_dir) / "class_metrics.jsonl"
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"CLASS_METRICS {json.dumps(record)}", flush=True)

    model.add_callback("on_fit_epoch_end", record_class_metrics)
    augment = config.get("augment", {})
    schedule = {}
    for key in ("warmup_epochs", "patience", "close_mosaic", "lrf", "save_period", "freeze", "multi_scale"):
        if key in config:
            schedule[key] = config[key]
    started = time.time()
    result = model.train(
        data=str(args.data),
        epochs=args.epochs or int(config["epochs"]),
        imgsz=int(config["image_size"]),
        batch=args.batch or int(config["batch"]),
        workers=args.workers if args.workers is not None else int(config["workers"]),
        device=device,
        amp=False,  # Phase 3 explicitly requires FP32 fine-tuning.
        seed=int(config["seed"]),
        deterministic=bool(config["deterministic"]),
        optimizer=str(config["optimizer"]),
        lr0=float(config["lr0"]),
        weight_decay=float(config["weight_decay"]),
        project=str(args.project.resolve()),
        name=args.name,
        resume=bool(args.resume),
        plots=True,
        val=True,
        verbose=False,
        **augment,
        **schedule,
    )
    save_dir = Path(result.save_dir)
    best_checkpoint = save_dir / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise RuntimeError(f"Training completed without best checkpoint at {best_checkpoint}")
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, args.model_output)
    with (save_dir / "results.csv").open(encoding="utf-8") as handle:
        epoch_results = list(csv.DictReader(handle))
    best_epoch = max(epoch_results, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    metadata = {
        "gpu": gpu_summary(device),
        "data": str(args.data.resolve()),
        "config": config,
        "configured_epochs": args.epochs or int(config["epochs"]),
        "epochs": len(epoch_results),
        "best_epoch": int(best_epoch["epoch"]),
        "batch": args.batch or int(config["batch"]),
        "elapsed_seconds": round(time.time() - started, 3),
        "run_directory": str(save_dir.resolve()),
        "best_checkpoint": str(args.model_output.resolve()),
        "metrics": {
            key: float(value)
            for key, value in result.results_dict.items()
            if isinstance(value, int | float)
        },
    }
    metadata_path = args.model_output.with_suffix(".training.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
