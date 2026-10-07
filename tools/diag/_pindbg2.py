# -*- coding: utf-8 -*-
"""`pin_f1`（权重 0.25，现在只有 0.0474）到底卡在哪：把「键已匹配上的 box 元件」单独拎出来，
按引脚号逐对比较，看是「我们没检出这个引脚」「检出了但号错了」还是「检出了但坐标偏了」。

用法：
    python work\\tools\\diag\\_pindbg2.py --pred work\\stage3_vlm
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

import stage2_eval  # noqa: E402
from schematic_parser.data import load_json  # noqa: E402
from schematic_parser.metrics import EvalConfig, assign_components, canon_type  # noqa: E402

BOX = ("box", "boxic", "ic")


def parse_pins(doc, key):
    """-> {pinnumber: (pinname, (x, y))}。注意 pins 在**文档顶层**：`doc['pins'][元件键]`。"""
    out = {}
    pins = ((doc or {}).get("pins") or {}).get(key) or {}
    for k, v in pins.items():
        if not isinstance(v, dict):
            continue
        num = str(v.get("pinnumber") or v.get("pin_number") or "").strip()
        if not num and k.startswith("pin_"):
            num = k[4:]
        pt = v.get("point") or {}
        try:
            xy = (float(pt["x"]), float(pt["y"]))
        except Exception:  # noqa: BLE001
            xy = None
        if num:
            out[num] = (str(v.get("pinname") or ""), xy)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tol", type=float, default=5.0)
    a = ap.parse_args()
    pred = Path(a.pred)

    exact, digits = stage2_eval.build_index(stage2_eval.DEFAULT_DATA_ROOT)
    c = Counter()
    samples = {}
    for f in sorted(pred.glob("*.json")):
        if f.name.startswith("_"):
            continue
        case, _how = stage2_eval.resolve(f.stem, exact, digits)
        if case is None:
            continue
        g = load_json(case.json_path)
        p = json.loads(f.read_text(encoding="utf-8"))
        _f1, km = assign_components(g, p, EvalConfig(), {})
        for pk, gk in km.items():
            gc = g["components"].get(gk)
            if gc is None or canon_type(gc.get("type")) not in BOX:
                continue
            c["matched_box_comps"] += 1
            gp = parse_pins(g, gk)
            pp = parse_pins(p, pk)
            c["gt_pins"] += len(gp)
            c["pred_pins"] += len(pp)
            for num, (gname, gxy) in gp.items():
                if num not in pp:
                    c["缺这个引脚号"] += 1
                    if len(samples.setdefault("缺这个引脚号", [])) < 8:
                        samples["缺这个引脚号"].append(
                            (case.case_id, gk, num, gname, sorted(pp)[:12]))
                    continue
                pname, pxy = pp[num]
                c["号对上了"] += 1
                if gxy and pxy:
                    d = max(abs(gxy[0] - pxy[0]), abs(gxy[1] - pxy[1]))
                    c["号对且坐标<=tol"] += 1 if d <= a.tol else 0
                    if d > a.tol:
                        c["号对但坐标偏"] += 1
                        c["|dx|<=20" if abs(gxy[0] - pxy[0]) <= 20 else "|dx|>20"] += 1
                        c["|dy|<=20" if abs(gxy[1] - pxy[1]) <= 20 else "|dy|>20"] += 1
                        buck = int(d // 10) * 10
                        c["距离桶 %d~%d" % (buck, buck + 10)] += 1
                        if len(samples.setdefault("号对但坐标偏", [])) < 40:
                            samples["号对但坐标偏"].append(
                                (case.case_id, gk, num, gname,
                                 (round(gxy[0], 1), round(gxy[1], 1)),
                                 (round(pxy[0], 1), round(pxy[1], 1)),
                                 "dx=%.1f dy=%.1f" % (pxy[0] - gxy[0], pxy[1] - gxy[1])))
                c["号对且名字对"] += 1 if pname and pname == gname else 0
            for num in pp:
                if num not in gp:
                    c["多出来的引脚号"] += 1
    for k, v in c.most_common():
        print("  %-16s %6d" % (k, v))
    print()
    for k, s in samples.items():
        print("--- %s ---" % k)
        for it in s:
            print("  ", it)


if __name__ == "__main__":
    main()
