# -*- coding: utf-8 -*-
"""量我方 box 引脚点相对 GT 的偏移（按元件边的方位分组），看有没有可用的系统性修正。

用法：python -X utf8 tools\diag\_pinshift.py --run <run> --split 10GT --n 10
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "diag"))

from _pinledger import _find_assigned, _find_pred, _find_stage1          # noqa: E402
from schematic_parser import metrics as M                               # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json  # noqa: E402


def _pt(d):
    if not d:
        return None
    p = d.get("point") or {}
    return (float(p.get("x", 0)), float(p.get("y", 0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)
    cfg = M.EvalConfig()
    by_side = defaultdict(list)
    allrows = []
    for c in [x for x in discover(DEFAULT_DATA_ROOT) if x.split == a.split][:a.n]:
        raw = _find_pred(run, c.case_id)
        if raw is None:
            continue
        gt = M.normalise_doc(load_json(Path(c.json_path)))
        pred = M.normalise_doc(raw)
        _c, keymap = M.assign_components(gt, pred, cfg, {})
        pred2 = M.rewrite_keys(pred, keymap)
        s1 = _find_stage1(run, c.case_id) or {}
        s1_pins = s1.get("pins") or {}
        comps1 = {x.get("id"): x for x in (s1.get("components") or [])
                  if isinstance(x, dict) and x.get("id")}
        asg = _find_assigned(run, c.case_id)
        if not asg:
            continue
        for cid, cc in (asg.get("comps") or {}).items():
            key = ((cc.get("designator") or {}).get("text") or "").strip()
            gkey = None
            for k, v in keymap.items():
                if v == key:
                    gkey = k
            gpins = {M.pin_key_to_number(pk): v for pk, v in
                     ((gt.get("pins") or {}).get(gkey) or {}).items()}
            ppins = {M.pin_key_to_number(pk): v for pk, v in
                     ((pred2.get("pins") or {}).get(key) or {}).items()}
            bb = (comps1.get(cid) or {}).get("bbox_img") or [0, 0, 0, 0]
            try:
                x0, y0, x1, y1 = [float(v) for v in bb]
            except Exception:                                               # noqa: BLE001
                continue
            for num, gp in gpins.items():
                pp = ppins.get(num)
                if not pp:
                    continue
                g = _pt(gp)
                o = _pt(pp)
                if not g or not o:
                    continue
                dx, dy = o[0] - g[0], o[1] - g[1]
                d = math.hypot(dx, dy)
                dl, dr = abs(g[0] - x0), abs(g[0] - x1)
                dt, db = abs(g[1] - y0), abs(g[1] - y1)
                side = "L" if dl == min(dl, dr, dt, db) else \
                       "R" if dr == min(dl, dr, dt, db) else \
                       "T" if dt == min(dl, dr, dt, db) else "B"
                by_side[side].append((dx, dy, d))
                allrows.append((c.case_id, key, num, side, round(dx, 1), round(dy, 1),
                                round(d, 1),
                                str((gp or {}).get("pinname") or ""),
                                "ok" if d <= cfg.pin_tol else "off"))
    print("配对上的 box 引脚 %d 根（pin_tol=%.1f）" % (len(allrows), cfg.pin_tol))

    def med(t):
        t = sorted(t)
        return t[len(t) // 2]

    def mean(t):
        return sum(t) / len(t)

    for side in ("L", "R", "T", "B"):
        v = by_side.get(side) or []
        if not v:
            continue
        n_ok = sum(1 for _dx, _dy, d in v if d <= cfg.pin_tol)
        print("  %s n=%-3d 命中 %-3d (%.2f) | dx 中位 %+6.1f 均值 %+6.1f | "
              "dy 中位 %+6.1f 均值 %+6.1f | d 中位 %5.1f"
              % (side, len(v), n_ok, 1.0 * n_ok / len(v),
                 med([x for x, _y, _d in v]), mean([x for x, _y, _d in v]),
                 med([y for _x, y, _d in v]), mean([y for _x, y, _d in v]),
                 med([d for _x, _y, d in v])))
    fix = {s: (med([x for x, _y, _d in v]), med([y for _x, y, _d in v]))
           for s, v in by_side.items()}
    n_ok0 = sum(1 for r in allrows if r[8] == "ok")
    n_ok1 = 0
    for r in allrows:
        fx, fy = fix.get(r[3], (0.0, 0.0))
        n_ok1 += 1 if math.hypot(r[4] - fx, r[5] - fy) <= cfg.pin_tol else 0
    print("  原样命中 %d/%d → 按侧中位偏移修正后 %d/%d（%s）"
          % (n_ok0, len(allrows), n_ok1, len(allrows),
             {k: (round(v[0], 1), round(v[1], 1)) for k, v in fix.items()}))
    # 只看「名字对」的子集：位置修正能不能把名对位偏的那批救成 tp
    if a.show:
        for r in allrows[:a.show]:
            print("   ", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
