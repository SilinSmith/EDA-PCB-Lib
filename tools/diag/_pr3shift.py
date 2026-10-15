"""队友产物（PR#2/PR#3 分支）的引脚点「纯几何偏移」假设检验。

问题：队友在官方 10GT 上的 `pin_f1` 只有 0.1332（我方 0.8166），其中 case 2/5/6/7/9
整张 0.0000。先看是不是**点位置约定不同**（不是读数错）：把队友的每个引脚点沿
「框中心 → 点」方向一次性内移 d 像素，再重打分，找 pin_f1 的最优 d。

用法：
  python work/_pr3shift.py --run work/_fast_out_pr3b
  python work/_pr3shift.py --run work/_fast_out_pr2 --ds 0,4,8,12,16,20,24
"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json  # noqa: E402

CASES = [str(i) for i in range(1, 11)]


def shift_pred(pred, d):
    """把每个引脚点沿「框中心→点」方向内移 d 像素（d=0 原样）。"""
    out = copy.deepcopy(pred)
    for cid_, c in (out.get("components") or {}).items():
        bb = c.get("bbox")
        pins = (out.get("pins") or {}).get(cid_)
        if not bb or not pins:
            continue
        x0, y0, x1, y1 = [float(v) for v in (bb if isinstance(bb, (list, tuple)) else
                                             [bb["x0"], bb["y0"], bb["x1"], bb["y1"]])]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        for pk, pv in pins.items():
            pt = pv.get("point") if isinstance(pv, dict) else None
            if not pt:
                continue
            px, py = float(pt["x"]), float(pt["y"])
            vx, vy = px - cx, py - cy
            n = (vx * vx + vy * vy) ** 0.5
            if n < 1e-6:
                continue
            pv["point"] = {"x": px - d * vx / n, "y": py - d * vy / n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="work/_fast_out_pr3b")
    ap.add_argument("--ds", default="0,6,10,12,14,18,24")
    a = ap.parse_args()
    ds = [float(v) for v in a.ds.split(",") if v.strip()]
    cfg = M.EvalConfig()
    cases = {}
    for cid in CASES:
        for c in discover(DEFAULT_DATA_ROOT):
            if c.case_id == cid and c.split == "10GT":
                cases[cid] = c
    print("run =", a.run)
    print("d(px) | pin_f1 均值 | 逐案 pin_f1 | 爆 0 张数")
    for d in ds:
        f1s, zeros = [], 0
        for cid, c in cases.items():
            gt = load_json(Path(c.json_path))
            pp = Path(ROOT) / a.run / "final_results" / ("%s_Pred.json" % cid)
            if not pp.exists():
                continue
            pr = shift_pred(load_json(pp), d)
            sc = M.evaluate_case(gt, pr, case_id=cid, cfg=cfg)
            f1 = sc.pin.f1
            f1s.append(f1)
            zeros += (f1 <= 1e-9)
        print("%5.1f |   %.4f    | %s | %d" % (
            d, sum(f1s) / len(f1s),
            " ".join("%.2f" % v for v in f1s), zeros))


if __name__ == "__main__":
    main()
