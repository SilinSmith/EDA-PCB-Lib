# -*- coding: utf-8 -*-
"""每张图的「位号可读性」分布 —— 用来判断哪些 case 压根没标注位号。

信号：GT 元件的键有多少出现在我们的预测里（跳掉 ∅NNN，它们靠坐标匹配）。
逐图给出 key_hit / gt，并统计 box(IC) 那一档。

用法：
    python work\tools\diag\_percase.py --pred work\stage2_v13
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eda-pcb"))
sys.path.insert(0, str(REPO / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json  # noqa: E402
from stage2_eval import build_index, resolve  # noqa: E402

UKEY = "\u2205"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(REPO / "work" / "stage2_v13"))
    a = ap.parse_args()
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    rows = []
    for f in sorted(Path(a.pred).glob("*.json")):
        if f.name.startswith("_"):
            continue
        case, _ = resolve(f.stem, exact, digits)
        if case is None:
            continue
        gt = load_json(case.json_path)
        pred = json.loads(f.read_text(encoding="utf-8"))
        gc, pc = gt.get("components") or {}, pred.get("components") or {}
        named = {k: v for k, v in gc.items() if not k.startswith(UKEY)}
        boxn = {k: v for k, v in named.items() if (v.get("type") or "") == "box"}
        hit = sum(1 for k in named if k in pc)
        bhit = sum(1 for k in boxn if k in pc)
        rows.append((f.stem, len(named), hit, len(boxn), bhit))
    rows.sort(key=lambda r: (r[2] / max(1, r[1])))
    tot_n = sum(r[1] for r in rows)
    tot_h = sum(r[2] for r in rows)
    print("有名字的 GT 元件 %d，键命中 %d (%.3f)"
          % (tot_n, tot_h, tot_h / max(1, tot_n)))
    print("box 元件 %d，键命中 %d (%.3f)\n"
          % (sum(r[3] for r in rows), sum(r[4] for r in rows),
             sum(r[4] for r in rows) / max(1, sum(r[3] for r in rows))))
    buckets = [(0.0, 0.0, "完全没有"), (0.0, 0.05, "≈0"), (0.05, 0.25, "差"),
               (0.25, 0.5, "中"), (0.5, 0.8, "好"), (0.8, 1.01, "很好")]
    for lo, hi, name in buckets:
        sel = [r for r in rows if lo < (r[2] / max(1, r[1])) <= hi]
        if lo == 0.0 and hi == 0.0:
            sel = [r for r in rows if r[2] == 0]
        print("  %-6s %3d 张" % (name, len(sel)))
    print("\n最差的 30 张（case, 有名字GT, 命中, box GT, box命中）：")
    for r in rows[:30]:
        print("   %-22s %4d %4d   box %3d %3d" % r)
    print("\n最好的 15 张：")
    for r in rows[-15:]:
        print("   %-22s %4d %4d   box %3d %3d" % r)


if __name__ == "__main__":
    main()
