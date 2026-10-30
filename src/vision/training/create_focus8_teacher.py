"""Remap seven official COCO heads into the deployed fifteen-class taxonomy."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('models/pretrained/yolo26s.pt'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/models/yolo26s_focus8_coco_teacher_v8.pt'))
    args = parser.parse_args()
    source = args.source
    destination = args.output
    model = YOLO(str(source))
    reference = YOLO('artifacts/models/yolo26s_focus8_v5.pt')
    names = reference.names
    mapping = {'bottle': 'bottle', 'cup': 'cup', 'phone': 'cell phone', 'remote': 'remote',
               'spoon': 'spoon', 'fork': 'fork', 'toothbrush': 'toothbrush'}
    row_map = {cid: next(index for index, label in model.names.items() if label == mapping[name])
               for cid, name in names.items() if name in mapping}
    head = model.model.model[-1]
    assert head.nc == 80 and head.end2end
    for branch in (head.cv3, head.one2one_cv3):
        for scale in branch:
            previous = scale[-1]
            replacement = torch.nn.Conv2d(previous.in_channels, len(names), 1)
            with torch.no_grad():
                replacement.weight.zero_()
                replacement.bias.fill_(-30)
                for target, old in row_map.items():
                    replacement.weight[target].copy_(previous.weight[old])
                    replacement.bias[target].copy_(previous.bias[old])
            scale[-1] = replacement
    head.nc = len(names)
    head.no = head.nc + head.reg_max * 4
    model.model.names = copy.deepcopy(names)
    model.model.yaml['nc'] = len(names)
    model.save(str(destination))
    metadata = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'checkpoint': str(destination), 'model_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                'source_classes': 80, 'output_classes': names, 'exact_classifier_row_map': row_map,
                'unmapped_logits': -30, 'shared_features_and_boxes': 'Official COCO pretrained weights unchanged',
                'new_training': False, 'deployment_accepted': False}
    destination.with_suffix('.remap.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
