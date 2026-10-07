# -*- coding: utf-8 -*-
"""把官方 GT 与我们的预测各打印一小段，确认 schema 完全对齐。

    python work\tools\diag\_schemadump.py [pred_dir]
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser import data as D  # noqa: E402


def brief(o, n=2):
    s = json.dumps(o, ensure_ascii=False)
    return s if len(s) <= 900 else s[:900] + " ..."


def main():
    pred_dir = sys.argv[1] if len(sys.argv) > 1 else r"Y:\workspace\EDA\work\stage2_v14"

    cases = D.discover()
    gt = None
    for c in cases:
        if c.case_id == "0001":
            gt = c
            break
    print("=== GT case ===", gt.png_path)
    doc = D.load_json(gt.json_path)
    print("顶层键:", list(doc.keys()))
    print("components 键数:", len(doc.get("components") or {}))
    k = list(doc["components"])[0]
    print("components[%r] =" % k, brief(doc["components"][k]))
    print("pins 键数:", len(doc.get("pins") or {}))
    if doc.get("pins"):
        pk = list(doc["pins"])[0]
        print("pins[%r] =" % pk, brief(doc["pins"][pk]))
    print("nets 键数:", len(doc.get("nets") or {}))
    if doc.get("nets"):
        nk = list(doc["nets"])[0]
        print("nets[%r] 的键 =" % nk, list(doc["nets"][nk].keys()))
        print("nets[%r] =" % nk, brief(doc["nets"][nk]))

    print()
    print("=== 我们的预测 ===")
    p = os.path.join(pred_dir, "0001.json")
    if not os.path.exists(p):
        print("没有", p)
        return
    pd = D.load_json(p)
    print("顶层键:", list(pd.keys()))
    print("components 键数:", len(pd.get("components") or {}))
    for k in list(pd["components"])[:2]:
        print("components[%r] =" % k, brief(pd["components"][k]))
    print("pins 键数:", len(pd.get("pins") or {}))
    for k in list(pd.get("pins") or {})[:1]:
        print("pins[%r] =" % k, brief(pd["pins"][k]))
    print("nets 键数:", len(pd.get("nets") or {}))
    for k in list(pd.get("nets") or {})[:1]:
        print("nets[%r] 的键 =" % k, list(pd["nets"][k].keys()))
        print("nets[%r] =" % k, brief(pd["nets"][k]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
