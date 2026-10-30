"""Validate single/full+crop profiles using the exact deployed backend wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from vagt_orca.deployment import sha256
from vagt_orca.gpu import gpu_summary, require_cuda
from vagt_orca.vision.focus_profile import attach_focus_profile
from vagt_orca.vision.multiview import VIEW_MODES


class ViewValidator(DetectionValidator):
    view_mode = 'single'
    specialist = None
    specialist_classes = ()
    strategy = 'route'
    small_primary_size = 0

    def init_metrics(self, model):
        super().init_metrics(model)
        attach_focus_profile(model, self.view_mode, self.specialist, self.specialist_classes, self.strategy, self.small_primary_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--views', choices=VIEW_MODES, default='single')
    parser.add_argument('--specialist', type=Path)
    parser.add_argument('--specialist-classes', default='0,2,5,6,8,9,10')
    parser.add_argument('--strategy', choices=('route', 'union'), default='route')
    parser.add_argument('--small-primary-size', type=int, default=0)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    require_cuda(0)
    ViewValidator.view_mode = args.views
    ViewValidator.specialist = args.specialist
    ViewValidator.specialist_classes = tuple(int(value) for value in args.specialist_classes.split(',') if value)
    ViewValidator.strategy = args.strategy
    ViewValidator.small_primary_size = args.small_primary_size
    result = YOLO(str(args.model)).val(
        validator=ViewValidator, data=str(args.data), device=0, imgsz=640,
        split=args.split, batch=args.batch, quantize=32, amp=False, rect=False,
        project=str(args.project.resolve()), name=args.split,
        plots=False, save_json=False, workers=8,
    )
    summary = {'model': str(args.model.resolve()), 'model_sha256': sha256(args.model),
               'data': str(args.data.resolve()), 'split': args.split, 'views': args.views,
               'specialist': str(args.specialist.resolve()) if args.specialist else None,
               'specialist_sha256': sha256(args.specialist) if args.specialist else None,
               'specialist_classes': ViewValidator.specialist_classes if args.specialist else [],
               'strategy': args.strategy if args.specialist else None,
               'small_primary_size': args.small_primary_size,
               'image_size': 640, 'batch': args.batch, 'rect': False,
               'precision': 'fp16' if args.model.suffix == '.engine' else 'fp32',
               'map50_95': float(result.box.map), 'map50': float(result.box.map50),
               'per_class_map': {result.names[int(cid)]: float(result.box.ap[row])
                                 for row, cid in enumerate(result.box.ap_class_index)},
               'per_class': {result.names[int(cid)]: {
                   'map50_95': float(result.box.ap[row]), 'map50': float(result.box.ap50[row]),
                   'precision': float(result.box.p[row]), 'recall': float(result.box.r[row]),
                   'instances': int(result.nt_per_class[int(cid)])}
                   for row, cid in enumerate(result.box.ap_class_index)},
               'speed_ms_per_image': {key: float(value) for key, value in result.speed.items()},
               'gpu': gpu_summary(0)}
    output = Path(result.save_dir) / 'metrics_summary.json'
    output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
