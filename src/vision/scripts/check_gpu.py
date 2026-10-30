#!/usr/bin/env python3
from __future__ import annotations

import json

from vagt_orca.gpu import gpu_summary

if __name__ == "__main__":
    print(json.dumps(gpu_summary(), indent=2))
