"""Test learned weights with reference BatchNorm running statistics; validation required."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from vagt_orca.deployment import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from ultralytics import YOLO

    reference = YOLO(str(args.reference)).model.float().cpu()
    candidate = YOLO(str(args.candidate)).model.float().cpu()
    assert reference.names == candidate.names
    old, new = reference.state_dict(), candidate.state_dict()
    assert old.keys() == new.keys() and all(old[k].shape == new[k].shape for k in old)
    replaced = []
    for name, module in candidate.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for suffix in ("running_mean", "running_var", "num_batches_tracked"):
                key = name+"."+suffix
                if key in old:
                    replaced.append({"tensor": key, "l2_change": float((new[key].float()-old[key].float()).norm())})
                    new[key] = old[key].clone()
    candidate.load_state_dict(new, strict=True)
    checkpoint = torch.load(args.candidate, map_location="cpu", weights_only=False)
    checkpoint.update({"model": copy.deepcopy(candidate).half(), "ema": None, "optimizer": None,
                       "epoch": -1, "updates": None})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {"method": "Keep learned parameters; restore only reference BatchNorm running statistics",
              "reference": str(args.reference.resolve()), "reference_sha256": sha256(args.reference),
              "candidate": str(args.candidate.resolve()), "candidate_sha256": sha256(args.candidate),
              "output": str(args.output.resolve()), "output_sha256": sha256(args.output), "replaced": replaced}
    args.output.with_suffix(".bn_stats.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key != "replaced"}, indent=2))


if __name__ == "__main__":
    main()
