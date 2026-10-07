# -*- coding: utf-8 -*-
"""看各 split 的 GT `components[].type` 取值分布，以及 `canon_type` 之后落在 IC_TYPES 的比例。"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))
sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json          # noqa: E402
from schematic_parser.metrics import IC_TYPES, canon_type               # noqa: E402

root = Path(DEFAULT_DATA_ROOT)
print("DATA_ROOT =", root)
print("IC_TYPES =", sorted(IC_TYPES))
print()
for split in sorted(p.name for p in root.iterdir() if p.is_dir()):
    d = root / split
    js = sorted(d.glob("*/*.json"))
    raw, can = Counter(), Counter()
    n_ic, n_all = 0, 0
    for j in js:
        try:
            g = load_json(j)
        except Exception:                                    # noqa: BLE001
            continue
        for _k, v in (g.get("components") or {}).items():
            t = (v or {}).get("type")
            raw[str(t)] += 1
            ct = canon_type(t)
            can[str(ct)] += 1
            n_all += 1
            if ct in IC_TYPES:
                n_ic += 1
    print("== %-22s json=%d  元件 %d  其中 IC %d (%.3f)"
          % (split, len(js), n_all, n_ic, n_ic / max(1, n_all)))
    print("   原始 type:", raw.most_common(12))
    print("   canon   :", can.most_common(12))
