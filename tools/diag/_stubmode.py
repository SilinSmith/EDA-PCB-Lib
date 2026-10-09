"""`STAGE1_PIN_STUB` 各模式对 **stage1 引脚点集**的影响（按侧对齐量 P/R vs GT）。

动机：`CLUSTER_IC` 12→5 之后 case 7 丢了一整行（`sparse` 的触发条件比的是簇计数，
簇变小 ⇒ 接触点变多 ⇒ 反而不触发补桩）。新加的 `gapfill` 模式改成「补没被覆盖的长桩」。
本探针不跑整条链路，只把 `stage1.pins.find_pins` 的输出与 GT 引脚按侧对齐，量：

    tp / fp / fn（L/R 侧比 y，T/B 侧比 x，tol 默认 5px —— 即 scorer 的 `cfg.pin_tol`）

用法::

    python -X utf8 tools/diag/_stubmode.py --ci 5 --modes sparse,gap0,gap8,gap12,union
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "eda-pcb"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from stage1 import pins as P          # noqa: E402

GT_DIR = PKG / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
IC_TYPES = ("box", "bigic", "ic")


def gt_path(cid: str, gt_root: Path):
    """GT json：`<root>/<cid>/<cid>.json`，或 `<root>/<cid>*/<cid>-*_target_named.json`
    （派生集在 `200_train_cases` / `10_GTcase` 下的命名）。"""
    cands = []
    for d in [gt_root / cid] + sorted(gt_root.glob(cid + "*")):
        if d.is_dir():
            cands.append(d / ("%s.json" % cid))
            cands += sorted(d.glob("*target_named*.json"))
    for p in cands:
        if p.is_file():
            return p
    return None


def side_of(bx: float, by: float, bb) -> str:
    x0, y0, x1, y1 = bb
    d = {"L": abs(bx - x0), "R": abs(bx - x1), "T": abs(by - y0), "B": abs(by - y1)}
    return min(d, key=d.get)


def q_side(q, bb):
    x0, y0, x1, y1 = bb
    if q[0] < x0 - 1:
        return "L"
    if q[0] > x1 + 1:
        return "R"
    if q[1] < y0 - 1:
        return "T"
    if q[1] > y1 + 1:
        return "B"
    return None


def set_mode(mode: str):
    P.STUB_FILL = True
    P.STUB_UNION = (mode == "union")
    P.STUB_SPARSE = (mode == "sparse")
    P.STUB_GAP = mode.startswith("gap")
    P.STUB_MINLEN = float(mode[3:]) if mode.startswith("gap") and mode[3:] else 0.0
    if mode in ("0", "off"):
        P.STUB_FILL = False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--run", default=str(ROOT / "work" / "_fast_out_dlv10"))
    ap.add_argument("--gt-root", default=str(GT_DIR),
                    help="GT 根目录（派生集用 200_train_cases / 10_GTcase）")
    ap.add_argument("--ci", type=float, default=5.0)
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--modes", default="sparse,gap0,gap5,gap8,gap10,gap12,union")
    ap.add_argument("--free", action="store_true",
                    help="桩证据改用「抹掉文字框」的墨迹（pincol.text_free_ink）")
    ap.add_argument("--pad", type=float, default=0.0, help="text_free_ink 的外扩 px")
    ap.add_argument("--byside", action="store_true",
                    help="额外按 L/R/T/B 汇总 tp/fp/fn（看缺脚缺在哪一侧）")
    a = ap.parse_args(argv)

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    tot = {m: [0, 0, 0] for m in modes}          # mode -> [tp, fp, fn]
    tot_n = {m: 0 for m in modes}
    tot_sd = {m: {s: [0, 0, 0] for s in ("L", "R", "T", "B")} for m in modes}
    for cid in [c.strip() for c in a.cases.split(",") if c.strip()]:
        s1p = Path(a.run) / "runs" / "stage1" / ("%s.stage1.json" % cid)
        gtj = None
        for _r in [x for x in str(a.gt_root).split(";") if x.strip()]:
            gtj = gt_path(cid, Path(_r))
            if gtj is not None:
                break
        if not (s1p.exists() and gtj is not None and gtj.exists()):
            print("  skip %s" % cid)
            continue
        s1 = json.loads(s1p.read_text(encoding="utf-8"))
        img = cv2.imdecode(np.fromfile(s1["image"]["path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        H = img.shape[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink = (gray < 170).astype(np.uint8)
        if a.free:
            from stage1 import pincol as _pc
            ink = _pc.text_free_ink(ink, s1.get("texts") or [], pad=a.pad)
        js = json.loads(gtj.read_text(encoding="utf-8"))
        segs = s1["wire_detail"]["segments"]
        comps = s1["components"]
        ics = [c for c in comps if str(c.get("type", "")).lower() in IC_TYPES]

        # 我们的 IC -> GT 元件名（按左上角距离 < 40px）
        pairs = []
        for c in ics:
            bb = [float(v) for v in c["bbox_img"]]
            best, bk = None, None
            for nm, rec in (js.get("components") or {}).items():
                if str(rec.get("type", "")).lower() not in IC_TYPES:
                    continue
                gb = [float(v) for v in rec["bbox"]]
                gbi = [gb[0], H - gb[3], gb[2], H - gb[1]]
                d = abs(gbi[0] - bb[0]) + abs(gbi[1] - bb[1])
                if best is None or d < best:
                    best, bk = d, nm
            if bk is not None and best is not None and best < 40:
                pairs.append((c, bb, bk))

        line = ["case %-3s IC=%d" % (cid, len(pairs))]
        csd = {s: [0, 0, 0] for s in ("L", "R", "T", "B")} if a.byside else None
        for mode in modes:
            set_mode(mode)
            P.CLUSTER_IC = a.ci
            out = P.find_pins(comps, segs, texts=(s1.get("texts") or []), ink=ink)
            tp = fp = fn = 0
            n_pts = 0
            for c, bb, nm in pairs:
                gt = {}
                for pk, rec in (js.get("pins") or {}).get(nm, {}).items():
                    pt = rec.get("point") or {}
                    x, y = float(pt.get("x", 0.0)), H - float(pt.get("y", 0.0))
                    gt.setdefault(side_of(x, y, bb), []).append(
                        (y if side_of(x, y, bb) in ("L", "R") else x))
                got = {}
                for q in out.get(c["id"]) or []:
                    sd = q_side(q, bb)
                    if sd is None:
                        continue
                    got.setdefault(sd, []).append(q[1] if sd in ("L", "R") else q[0])
                    n_pts += 1
                for sd in ("L", "R", "T", "B"):
                    gg = sorted(gt.get(sd) or [])
                    pp = sorted(got.get(sd) or [])
                    used = [False] * len(gg)
                    s_tp = s_fp = 0
                    for v in pp:
                        hit = -1
                        for i, gv in enumerate(gg):
                            if not used[i] and abs(gv - v) <= a.tol:
                                hit = i
                                break
                        if hit >= 0:
                            used[hit] = True
                            tp += 1
                            s_tp += 1
                        else:
                            fp += 1
                            s_fp += 1
                    s_fn = sum(1 for u in used if not u)
                    fn += s_fn
                    tot_sd[mode][sd][0] += s_tp
                    tot_sd[mode][sd][1] += s_fp
                    tot_sd[mode][sd][2] += s_fn
                    if csd is not None:
                        csd[sd][0] += s_tp
                        csd[sd][1] += s_fp
                        csd[sd][2] += s_fn
            tot[mode][0] += tp
            tot[mode][1] += fp
            tot[mode][2] += fn
            tot_n[mode] += n_pts
            line.append("%s:%d/%d/%d(n=%d)" % (mode, tp, fp, fn, n_pts))
        if csd is not None:
            line.append("| " + " ".join(
                "%s %d/%d/%d" % (sd, csd[sd][0], csd[sd][1], csd[sd][2])
                for sd in ("L", "R", "T", "B") if sum(csd[sd])))
        print("  ".join(line))

    print()
    for mode in modes:
        tp, fp, fn = tot[mode]
        p = tp / float(tp + fp) if tp + fp else 0.0
        r = tp / float(tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        print("  %-8s CI=%-4g tp=%3d fp=%3d fn=%3d  P=%.4f R=%.4f F1=%.4f  点数 %d"
              % (mode, a.ci, tp, fp, fn, p, r, f, tot_n[mode]))
    if a.byside:
        print()
        print("  按侧汇总（gt 侧按离框边最近的边判定，我方点按框外所在边判定）")
        for mode in modes:
            print("  --- %s" % mode)
            for s in ("L", "R", "T", "B"):
                tp, fp, fn = tot_sd[mode][s]
                if tp + fp + fn == 0:
                    continue
                rr = tp / float(tp + fn) if tp + fn else 0.0
                pp = tp / float(tp + fp) if tp + fp else 0.0
                print("      %s  tp=%3d fp=%3d fn=%3d  P=%.3f R=%.3f" % (s, tp, fp, fn, pp, rr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
