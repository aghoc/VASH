"""Validated GPU specialist routing/merging for the requested object classes."""
from __future__ import annotations

from types import MethodType

import torch
from torchvision.ops import batched_nms

from vagt_orca.vision.multiview import TARGET_IDS, end2end_tensor, predict_views


def merge_specialist(primary, specialist, class_ids, strategy='route', iou=0.55, small_primary_size=0):
    if primary.device.type != 'cuda' or specialist.device != primary.device:
        raise RuntimeError('Specialist inference requires the same CUDA device')
    ids = primary.new_tensor(class_ids)
    output = primary.new_zeros((len(primary), 300, 6))
    for index, (base, extra) in enumerate(zip(primary, specialist)):
        chosen = (base[:, 5:6] == ids).any(-1)
        extra = extra[(extra[:, 5:6] == ids).any(-1) & (extra[:, 4] >= 0.001)]
        if strategy == 'route':
            kept = torch.cat((base[~chosen], extra))
            if small_primary_size:
                # COCO and egocentric data differ most on very small toothbrushes.
                small = base[(base[:, 5] == 10) & (base[:, 4] >= 0.001)
                             & ((base[:, 2:4] - base[:, :2]).amax(-1) <= small_primary_size)]
                teeth = torch.cat((extra[extra[:, 5] == 10], small))
                selected = batched_nms(teeth[:, :4].float(), teeth[:, 4].float(), teeth[:, 5].long(), iou)
                kept = torch.cat((kept[kept[:, 5] != 10], teeth[selected]))
        elif strategy == 'union':
            focus = torch.cat((base[chosen], extra))
            focus = focus[focus[:, 4] >= 0.001]
            selected = batched_nms(focus[:, :4].float(), focus[:, 4].float(), focus[:, 5].long(), iou)
            kept = torch.cat((base[~chosen], focus[selected]))
        else:
            raise ValueError(strategy)
        kept = kept[kept[:, 4] >= 0.001]
        kept = kept[kept[:, 4].argsort(descending=True)[:300]]
        output[index, :len(kept)] = kept
    return output


def attach_focus_profile(backend, views='single', specialist=None, specialist_classes=(), strategy='route', small_primary_size=0):
    if views == 'single' and not specialist:
        return
    assert not getattr(backend, '_focus_profile', False), 'Profile already attached'
    assert set(specialist_classes) <= set(TARGET_IDS)
    original = backend.forward
    extra = None
    if specialist:
        from ultralytics.nn.autobackend import AutoBackend
        extra = AutoBackend(str(specialist), device=backend.device, fp16=backend.fp16, verbose=False)
        assert extra.names == backend.names, (extra.names, backend.names)
        assert extra.end2end and backend.end2end, 'Only matching end-to-end detection heads are supported'
        extra.eval()

    def forward(self, images, *args, **kwargs):
        primary = predict_views(lambda view: original(view, *args, **kwargs), images, views)
        if extra is None:
            return primary
        secondary = end2end_tensor(extra(images))
        return merge_specialist(primary, secondary, specialist_classes, strategy, small_primary_size=small_primary_size)

    backend.forward = MethodType(forward, backend)
    backend._focus_profile = True
    # A strong reference keeps the specialist backend and its CUDA buffers alive.
    backend._focus_specialist = extra
