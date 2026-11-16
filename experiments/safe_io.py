"""Overwrite-protected result writing (Stage 10.0).

Phase 10 rule: previous results are never silently overwritten. If the target file already exists,
the new results are written to '<name>.<UTC timestamp>.json' instead, unless the environment
variable CASPIAN_ALLOW_OVERWRITE=1 is set explicitly.
"""
import json
import os
from datetime import datetime, timezone


def safe_write_json(path: str, obj) -> str:
    target = path
    if os.path.exists(path) and os.environ.get("CASPIAN_ALLOW_OVERWRITE") != "1":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base, ext = os.path.splitext(path)
        target = f"{base}.{stamp}{ext}"
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return target
