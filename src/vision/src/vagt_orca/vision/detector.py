from __future__ import annotations

from pathlib import Path

import numpy as np

from vagt_orca.gpu import prepare_onnxruntime_cuda, require_cuda
from vagt_orca.types import Detection


class YoloDetector:
    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        device: int = 0,
        fp16: bool = False,
        class_confidence_thresholds: dict[str, float] | None = None,
        focus_profile: dict | None = None,
        detection_profile: dict | None = None,
    ) -> None:
        self.device = require_cuda(device)
        if Path(model_path).suffix.lower() == ".onnx":
            prepare_onnxruntime_cuda()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed; run `uv sync --extra dev`") from exc
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.detection_profile = dict(detection_profile or {})
        self.class_aliases = dict(self.detection_profile.get("aliases", {}))
        self.project_class_ids = dict(self.detection_profile.get("project_class_ids", {}))
        self.allowed_classes = self.detection_profile.get("allowed_classes")
        excluded = self.detection_profile.get("excluded_classes", [])
        self.excluded_classes = set(excluded)
        self.unmapped_class_id_offset = self.detection_profile.get("unmapped_class_id_offset", 0)
        if type(self.unmapped_class_id_offset) is not int or self.unmapped_class_id_offset < 0:
            raise ValueError("unmapped_class_id_offset must be a non-negative integer")
        native_names = self.model.names
        unknown_aliases = set(self.class_aliases) - set(native_names.values())
        if unknown_aliases:
            raise ValueError(f"Unknown native class aliases: {sorted(unknown_aliases)}")
        available_names = {self.class_aliases.get(name, name) for name in native_names.values()}
        if len(self.excluded_classes) != len(excluded) or self.excluded_classes - available_names:
            raise ValueError("excluded_classes must be unique supported canonical class names")
        if self.excluded_classes and self.allowed_classes is not None:
            raise ValueError("Choose allowed_classes or excluded_classes, not both")
        if set(self.project_class_ids) - available_names:
            raise ValueError("Unknown canonical project class mapping")
        ids = list(self.project_class_ids.values())
        if any(type(cid) is not int or cid < 0 for cid in ids) or len(set(ids)) != len(ids):
            raise ValueError("Project class IDs must be unique non-negative integers")
        self.prediction_options = {
            key: self.detection_profile[key] for key in ("rect", "end2end")
            if key in self.detection_profile
        }
        if self.allowed_classes is not None:
            allowed = set(self.allowed_classes)
            if not allowed or len(allowed) != len(self.allowed_classes) or allowed - available_names:
                raise ValueError("allowed_classes must be unique supported canonical class names")
            if set(self.project_class_ids) != allowed:
                raise ValueError("project_class_ids must map every allowed class exactly once")
            self.prediction_options["classes"] = [
                cid for cid, name in native_names.items()
                if self.class_aliases.get(name, name) in allowed
            ]
            available_names = allowed
        else:
            # All native classes remain visible. Keep their identities separate
            # from project IDs so e.g. COCO person=0 cannot become bottle=0.
            output_ids = [
                self.project_class_ids.get(
                    self.class_aliases.get(name, name), cid + self.unmapped_class_id_offset
                )
                for cid, name in native_names.items()
            ]
            if len(set(output_ids)) != len(output_ids):
                raise ValueError("Mapped and unmapped class IDs collide; set an unmapped offset")
            if self.excluded_classes:
                available_names -= self.excluded_classes
                if not available_names:
                    raise ValueError("excluded_classes cannot remove every class")
                self.prediction_options["classes"] = [
                    cid for cid, name in native_names.items()
                    if self.class_aliases.get(name, name) not in self.excluded_classes
                ]
        self.class_confidence_thresholds = dict(class_confidence_thresholds or {})
        if set(self.class_confidence_thresholds) - available_names:
            raise ValueError("Unknown class confidence threshold")
        if any(not 0.0 <= value <= 1.0 for value in self.class_confidence_thresholds.values()):
            raise ValueError("class confidence thresholds must be in [0, 1]")
        self.iou = iou
        self.image_size = image_size
        self.quantize = 16 if fp16 else 32
        self.focus_profile = dict(focus_profile or {})
        if self.focus_profile:
            from vagt_orca.vision.focus_profile import attach_focus_profile

            def install_profile(predictor):
                if not getattr(predictor.model, '_focus_profile', False):
                    attach_focus_profile(
                        predictor.model, views=self.focus_profile.get('views', 'single'),
                        specialist=self.focus_profile.get('specialist_model'),
                        specialist_classes=self.focus_profile.get('specialist_classes', ()),
                        strategy=self.focus_profile.get('strategy', 'route'),
                        small_primary_size=self.focus_profile.get('small_primary_size', 0),
                    )

            self.model.add_callback('on_predict_start', install_profile)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=min([self.confidence, *self.class_confidence_thresholds.values()]),
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            quantize=self.quantize,
            verbose=False,
            **({'rect': False} if self.focus_profile else {}) | self.prediction_options,
        )
        output: list[Detection] = []
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            ids = boxes.id.detach().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
            for bbox, class_id, confidence, track_id in zip(xyxy, classes, confidences, ids):
                cid = int(class_id)
                native_name = str(names[cid])
                class_name = self.class_aliases.get(native_name, native_name)
                if self.allowed_classes is not None and class_name not in self.allowed_classes:
                    continue
                if class_name in self.excluded_classes:
                    continue
                if confidence < self.class_confidence_thresholds.get(class_name, self.confidence):
                    continue
                output.append(
                    Detection(
                        class_id=self.project_class_ids.get(
                            class_name, cid + self.unmapped_class_id_offset
                        ),
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox=tuple(float(value) for value in bbox),
                        track_id=int(track_id) if track_id is not None else None,
                    )
                )
        return output
