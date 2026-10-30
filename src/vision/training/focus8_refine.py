"""Refine eight class outputs while keeping shared features and box heads fixed."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from vagt_orca.gpu import gpu_summary, require_cuda

TARGET_IDS = (0, 2, 5, 6, 7, 8, 9, 10)
TARGET_NAMES = ('bottle', 'cup', 'phone', 'remote', 'pen', 'spoon', 'fork', 'toothbrush')


class FocusTrainer(DetectionTrainer):
    """Adapt classification without changing the geometry or feature extractor."""

    refine_mode = 'rows'

    def build_optimizer(self, model, *args, **kwargs):
        selected = []
        head_prefix = f'model.{len(model.model) - 1}.'
        for name, parameter in model.named_parameters():
            classifier = name.startswith(head_prefix) and ('.cv3.' in name or '.one2one_cv3.' in name)
            final = classifier and re.search(r'\.(?:one2one_)?cv3\.\d+\.2\.(?:weight|bias)$', name)
            parameter.requires_grad_(bool(final if self.refine_mode == 'rows' else classifier))
            if parameter.requires_grad:
                selected.append(name)
                if final:
                    mask = torch.zeros_like(parameter)
                    mask[list(TARGET_IDS)] = 1
                    parameter.register_hook(lambda grad, mask=mask: grad * mask)
        assert selected
        self.refinement_parameters = selected
        optimizer = super().build_optimizer(model, *args, **kwargs)
        # Decay would move the seven protected classifier rows even with zero gradients.
        for group in optimizer.param_groups:
            group['weight_decay'] = 0.0
        (self.save_dir / 'refinement_parameters.json').write_text(json.dumps({
            'mode': self.refine_mode, 'target_ids': TARGET_IDS,
            'trainable_names': selected, 'all_batchnorm_statistics_fixed': True,
            'shared_backbone_neck_and_box_heads_fixed': True,
        }, indent=2) + '\n')
        return optimizer

    def _model_train(self):
        super()._model_train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    def validate(self):
        previous_best = self.best_fitness
        metrics, fitness = super().validate()
        if metrics is not None:
            box = self.validator.metrics.box
            values = {self.validator.names[int(index)]: float(box.ap[row])
                      for row, index in enumerate(box.ap_class_index)}
            fitness = sum(values[name] for name in TARGET_NAMES) / len(TARGET_NAMES)
            self.best_fitness = fitness if previous_best is None else max(previous_best, fitness)
            record = {'epoch': self.epoch + 1, 'target_macro': fitness, 'per_class_map': values}
            with (self.save_dir / 'focus_class_metrics.jsonl').open('a') as output:
                output.write(json.dumps(record) + '\n')
            print('FOCUS_METRICS', json.dumps(record), flush=True)
        return metrics, fitness


class ClassifierTrainer(FocusTrainer):
    refine_mode = 'classifier'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('rows', 'classifier'), required=True)
    parser.add_argument('--base', default='artifacts/models/yolo26s_focus8_v5.pt')
    parser.add_argument('--data', default='data/processed/public_v7_focus8_targeted_15/data.yaml')
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    require_cuda(0)
    assert torch.cuda.is_available()
    started = time.monotonic()
    model = YOLO(args.base)
    result = model.train(
        trainer=FocusTrainer if args.mode == 'rows' else ClassifierTrainer,
        data=args.data, epochs=args.epochs, imgsz=640, batch=16, workers=8,
        device=0, amp=False, optimizer='AdamW', lr0=args.lr, lrf=0.2,
        weight_decay=0.0, warmup_epochs=0.25, warmup_bias_lr=0.0,
        seed=20261015, deterministic=True, patience=0, close_mosaic=0,
        save_period=1, plots=True, verbose=False, project=str(Path('artifacts/training').resolve()), name=args.name,
        hsv_h=0.01, hsv_s=0.25, hsv_v=0.2, degrees=4, translate=0.03,
        scale=0.15, shear=0, perspective=0, flipud=0, fliplr=0.5, mosaic=0, mixup=0,
    )
    run = Path(result.save_dir)
    output = Path('artifacts/models') / f'{args.name}_best.pt'
    shutil.copy2(run / 'weights/best.pt', output)
    metadata = {'mode': args.mode, 'base': args.base, 'data': args.data,
                'epochs': args.epochs, 'learning_rate': args.lr,
                'gpu': gpu_summary(0), 'elapsed_seconds': time.monotonic() - started,
                'run_directory': str(run.resolve()), 'checkpoint': str(output.resolve()),
                'model_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'selection': 'Eight target class validation mean; final selection uses three domains',
                'deployment_accepted': False}
    output.with_suffix('.training.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
