"""Extra contextual views for the eight requested classes, on the same GPU backend."""

from __future__ import annotations

from types import MethodType

import torch
from torch.nn import functional
from torchvision.ops import batched_nms

TARGET_IDS = (0, 2, 5, 6, 7, 8, 9, 10)
VIEW_MODES = ('single', 'center', 'corners')


def view_windows(height: int, width: int, mode: str):
    if mode not in VIEW_MODES:
        raise ValueError(f'Unknown view mode: {mode}')
    if mode == 'single':
        return []
    ch, cw = round(height * 0.6), round(width * 0.6)
    if mode == 'center':
        y, x = (height - ch) // 2, (width - cw) // 2
        return [(x, y, x + cw, y + ch)]
    return [(x, y, x + cw, y + ch)
            for y in (0, height - ch) for x in (0, width - cw)]


def end2end_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def predict_views(infer, images, mode: str, merge_iou: float = 0.55, max_det: int = 300):
    """Apply one full view plus crops, retaining full-view-only non-target classes."""
    if images.device.type != 'cuda':
        raise RuntimeError('Multi-view inference requires CUDA; CPU fallback is forbidden')
    base = end2end_tensor(infer(images))
    if mode == 'single':
        return base
    # TensorRT reuses its output buffer on the next call. Preserve full-view boxes
    # before running the cropped view through that same engine.
    base = base.clone()
    assert base.ndim == 3 and base.shape[-1] == 6, base.shape
    height, width = images.shape[-2:]
    candidates = [base]
    for x1, y1, x2, y2 in view_windows(height, width, mode):
        crop = functional.interpolate(images[:, :, y1:y2, x1:x2], size=(height, width),
                                      mode='bilinear', align_corners=False)
        output = end2end_tensor(infer(crop)).clone()
        focus = (output[..., 5:6] == output.new_tensor(TARGET_IDS)).any(-1)
        # Internal crop borders can truncate objects. The full view covers these objects.
        valid = focus & (output[..., 4] >= 0.001)
        if x1 > 0:
            valid &= output[..., 0] > 2
        if y1 > 0:
            valid &= output[..., 1] > 2
        if x2 < width:
            valid &= output[..., 2] < width - 2
        if y2 < height:
            valid &= output[..., 3] < height - 2
        output[..., [0, 2]] = output[..., [0, 2]] * ((x2 - x1) / width) + x1
        output[..., [1, 3]] = output[..., [1, 3]] * ((y2 - y1) / height) + y1
        output[..., 4] *= valid
        candidates.append(output)
    merged = torch.cat(candidates, dim=1)
    result = base.new_zeros((len(images), max_det, 6))
    for index, boxes in enumerate(merged):
        boxes = boxes[boxes[:, 4] >= 0.001]
        if not len(boxes):
            continue
        focus = (boxes[:, 5:6] == boxes.new_tensor(TARGET_IDS)).any(-1)
        targets, protected = boxes[focus], boxes[~focus]
        selected = batched_nms(targets[:, :4].float(), targets[:, 4].float(),
                               targets[:, 5].long(), merge_iou)
        kept = torch.cat((targets[selected], protected))
        kept = kept[kept[:, 4].argsort(descending=True)[:max_det]]
        result[index, :len(kept)] = kept
    return result


def attach_multiview(backend, mode: str):
    """Attach to an Ultralytics AutoBackend for identical validation and runtime."""
    if mode not in VIEW_MODES:
        raise ValueError(mode)
    if mode == 'single':
        return
    assert [backend.names[index] for index in TARGET_IDS] == [
        'bottle', 'cup', 'phone', 'remote', 'pen', 'spoon', 'fork', 'toothbrush'], backend.names
    if getattr(backend, '_focus_multiview_mode', None):
        assert backend._focus_multiview_mode == mode
        return
    original = backend.forward

    def forward(self, images, *args, **kwargs):
        return predict_views(lambda view: original(view, *args, **kwargs), images, mode)

    backend.forward = MethodType(forward, backend)
    backend._focus_multiview_mode = mode
