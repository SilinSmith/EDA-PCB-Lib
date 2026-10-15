# -*- coding: utf-8 -*-
"""**引脚轴向账本**：非 IC（c/r/l/d/…) 元件的两个引脚应该长在**符号主轴**两端
（竖着画的电容，脚在上下；横着画的电阻，脚在左右）。

本探针把「GT 的引脚轴向」与「我方 emit 的引脚轴向」逐元件对齐，回答三件事：
  1. GT 的脚是否确实沿**GT 框主轴**（验证这条规律本身）；
  2. 我方 emit 的脚是否沿同一轴向（当前实现是「哪条导线段贴着框就挂哪边」）；
  3. 差异集中在哪些类型上。

    python -X utf8 eda-pcb/tools/diag/_pinaxis.py --run Y:/workspace/EDA/work/_fast_out_dlv10
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))

from schematic_parser import metrics as M                                    # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json     # noqa: E402


def _find_pred(run: Path, cid: str):
    for p in (run / "final_results" / ("%s_Pred.json" % cid),
              run / "final_results" / ("%s.json" % cid)):
        if p.exists():
            return load_json(p)
    return None


def _pts(rec, pins_map):
    """元件记录的引脚点列表（JSON 帧）。"""
    out = []
    for _pk, pr in (pins_map or {}).items():
        pt = (pr or {}).get("point") or {}
        try:
            out.append((float(pt["x"]), float(pt["y"])))
        except Exception:
            continue
    return out


def _axis(points, box):
    """两脚相对框心的主方向：'V'（上下）/'H'（左右）。"""
    if not points or not box or len(box) < 4:
        return "?"
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    dx = max(abs(p[0] - cx) for p in points)
    dy = max(abs(p[1] - cy) for p in points)
    return "V" if dy > dx else "H"


def _major(box):
    if not box or len(box) < 4:
        return "?"
    w = abs(float(box[2]) - float(box[0]))
    h = abs(float(box[3]) - float(box[1]))
    return "V" if h > w else "H"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"Y:\workspace\EDA\work\_fast_out_dlv10")
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--show", type=int, default=0, help="打印前 N 条不一致明细")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = M.EvalConfig()
    pair = collections.Counter()          # (gt_axis, pred_axis)
    bytype = collections.Counter()
    gt_rule = collections.Counter()       # GT 脚轴向 vs GT 框主轴
    shown = 0
    for c in [x for x in discover(DEFAULT_DATA_ROOT) if x.split == a.split][:a.n]:
        raw = _find_pred(run, c.case_id)
        if raw is None:
            continue
        gt = M.normalise_doc(load_json(Path(c.json_path)))
        pred = M.normalise_doc(raw)
        det = {}
        _f, keymap = M.assign_components(gt, pred, cfg, det)
        pred2 = M.rewrite_keys(pred, keymap)
        gpin, ppin = gt.get("pins") or {}, pred2.get("pins") or {}
        for k, v in (gt.get("components") or {}).items():
            ty = M.canon_type((v or {}).get("type"))
            if ty in M.IC_TYPES or ty in ("gnd", "v", "power", ""):
                continue
            g = _pts(v, gpin.get(k))
            if len(g) < 2:
                continue
            ax_g = _axis(g, v.get("bbox"))
            gt_rule[(ax_g, _major(v.get("bbox")))] += 1
            pv = (pred2.get("components") or {}).get(k)
            p = _pts(pv, ppin.get(k)) if pv else []
            ax_p = _axis(p, (pv or {}).get("bbox")) if len(p) >= 2 else "无"
            pair[(ax_g, ax_p)] += 1
            bytype[(ty, ax_g, ax_p)] += 1
            if a.show and ax_p not in (ax_g,) and shown < a.show:
                shown += 1
                print("  case %-4s %-6s %-5s GT脚%s(框主轴%s) 我方%s  GT框%s 我方框%s"
                      % (c.case_id, k, ty, ax_g, _major(v.get("bbox")), ax_p,
                         [round(x, 1) for x in (v.get("bbox") or [])],
                         [round(x, 1) for x in ((pv or {}).get("bbox") or [])]))
    print("GT 脚轴向 × GT 框主轴：", dict(gt_rule))
    print("GT 脚轴向 × 我方脚轴向：", dict(pair))
    print("按类型（type, GT轴向, 我方轴向）-> 计数：")
    for k in sorted(bytype):
        print("   %-6s GT %-2s 我方 %-2s : %d" % (k[0], k[1], k[2], bytype[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
