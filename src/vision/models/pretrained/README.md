# Official pretrained checkpoints

The active model is the unmodified official **YOLO26m**. Binary checkpoints are
excluded from Git. From the vision project root, restore and verify it with:

```bash
.venv/bin/python scripts/download_models.py
```

For historical YOLO26s experiments, use `--model yolo26s` explicitly.
`--verify-only` checks an existing checkpoint without downloading it.
The release URLs and SHA-256 digests are pinned in the script and were checked
against the official Ultralytics v8.4.0 release metadata on 2026-10-30.

TensorRT engines are generated on the target Jetson; see
[the current runbook](../../deployment/JETSON_RUNBOOK.md).
