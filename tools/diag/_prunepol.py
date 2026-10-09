"""_prunepol.py -- 判别「框内 GT 导线边到底死在哪一道过滤上」。

背景（用户 m42009 的互相等待）：
- `_chainmeas.py` 量到 28 张里 662/3786 条 GT 导线边（17.5%）有采样点落在**我们自己
  检出的元件框内**，而它们只覆盖了 17 条（R=0.0257）；框外 1076/3124（R=0.3444）。
- `_occpol.py` 已证伪「提线前把自己的框涂白」这个解释（改 `occlude=[]` 后边召回**逐位
  相同**）。所以要在**上游过滤**里找：`extract_wires` 有三道裁剪
  （`min_extent` 小分量 + `rescue_pad` 贴框豁免、`max_axis_dev`、`_is_frame` 图框）。

本探针在 28 张上逐个关掉这些闸门，并把边覆盖拆成「框内 / 框外」两栏，看哪一道闸门
真正吃掉了框内那 662 条。

臂：
  pipe       = 管线口径（`pipeline.py:215-219` 的原参数）
  off_ext    = min_extent=0, rescue_pad=0（不按连通分量删任何线）
  off_axis   = max_axis_dev=90（不按 45° 族偏离删）
  off_frame  = drop_frame=False（不删贴边图框线）
  nofilt     = 三者全关
用法：python -X utf8 work\\tools\\dev\\_prunepol.py --cases-file work\\_cases_w28.txt
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
for p in (str(EDA), str(EDA / "tools" / "eval"), str(ROOT / "work" / "tools" / "dev")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402

from stage1.wires import extract_wires  # noqa: E402
import _wireedge as WE  # noqa: E402
import schematic_parser as S  # noqa: E402
import stage1_wires_eval as SE  # noqa: E402

PIPE = dict(erase_pad=1.0, drop_frame=True, max_axis_dev=12.0,
            min_extent=80.0, rescue_pad=8.0)
ARMS = {
    "pipe": {},
    "off_ext": dict(min_extent=0.0, rescue_pad=0.0),
    "off_axis": dict(max_axis_dev=90.0),
    "off_frame": dict(drop_frame=False),
    "nofilt": dict(min_extent=0.0, rescue_pad=0.0, max_axis_dev=90.0, drop_frame=False),
}


def load_cases(f: str):
    txt = io.open(f, encoding="utf-8-sig").read()
    return [c.strip() for c in txt.replace("\r", "").replace("\n", ",").split(",") if c.strip()]


def edge_samples(GA, n=16):
    """每条 GT 边沿长度取 n 个采样点（闭区间含两端）。"""
    out = []
    for x1, y1, x2, y2 in np.asarray(GA, np.float64).reshape(-1, 4):
        ts = np.linspace(0.0, 1.0, n)
        out.append(np.stack([x1 + (x2 - x1) * ts, y1 + (y2 - y1) * ts], axis=1))
    return out


def in_boxes(pts, boxes, pad=0.0):
    if not boxes:
        return False
    b = np.asarray(boxes, np.float64)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    ok = ((x >= b[None, :, 0] - pad) & (x <= b[None, :, 2] + pad) &
          (y >= b[None, :, 1] - pad) & (y <= b[None, :, 3] + pad))
    return bool(ok.any())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-file", default=str(ROOT / "work" / "_cases_w28.txt"))
    ap.add_argument("--stage1-dir", default=str(ROOT / "work" / "_full220" / "runs" / "stage1"))
    ap.add_argument("--png-dir", default=str(ROOT / "work" / "_submit_in220"))
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--pad", type=float, default=2.0,
                    help="框内判定的外扩（与 _chainmeas 的 hop3 对齐，默认 2.0）")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cases = load_cases(a.cases_file)
    if a.limit:
        cases = cases[:a.limit]
    tot = {k: dict(cin=0, nin=0, cout=0, nout=0, cp=0, np_=0, seg=0) for k in ARMS}
    for cid in cases:
        png = Path(a.png_dir) / f"{cid}.png"
        sj = Path(a.stage1_dir) / f"{cid}.stage1.json"
        case = S.find_case(cid, S.DEFAULT_DATA_ROOT)
        if not (png.exists() and sj.exists() and case is not None):
            print(f"{cid}: skip")
            continue
        bgr = cv2.imread(str(png), cv2.IMREAD_COLOR)
        H = bgr.shape[0]
        s1 = json.load(io.open(sj, encoding="utf-8"))
        boxes = [[float(v) for v in c["bbox_img"]] for c in s1["components"]]
        txts = [[float(v) for v in t["bbox_img"]] for t in (s1.get("texts") or [])]
        GA = WE.gt_segs(case, H)
        GA = np.asarray(GA, np.float64).reshape(-1, 4) if GA else np.zeros((0, 4))
        SAMP = edge_samples(GA)
        isin = (np.array([in_boxes(p, boxes, a.pad) for p in SAMP], bool)
                if len(SAMP) else np.zeros(0, bool))
        line = "%-14s n=%4d (in %4d)" % (cid, len(GA), int(isin.sum()))
        for name, over in ARMS.items():
            kw = dict(PIPE)
            kw.update(over)
            occ = boxes
            w, _ = extract_wires(bgr, occlude=occ, erase=txts, **kw)
            PA = SE.seg_arrays(w["segments"])
            cg, ng, cp, npd = WE.score(GA, PA, a.tol)
            # 逐边覆盖（判据同 stage1_wires_eval：一条边的全部采样点都落在某段 5px 内）
            cov = np.zeros(len(GA), bool)
            if len(GA) and len(PA):
                P = np.asarray(PA, np.float64).reshape(-1, 4)
                for i, pts in enumerate(SAMP):
                    ok = True
                    for p in pts:
                        d = SE.dist_pts_to_segs(p[None, :], P)
                        if d[0] > a.tol:
                            ok = False
                            break
                    cov[i] = ok
            t = tot[name]
            t["cin"] += int((cov & isin).sum())
            t["nin"] += int(isin.sum())
            t["cout"] += int((cov & ~isin).sum())
            t["nout"] += int((~isin).sum())
            t["cp"] += cp
            t["np_"] += npd
            t["seg"] += len(w["segments"])
            line += " | %s %3d/%3d %3d/%3d" % (name, int((cov & isin).sum()), int(isin.sum()),
                                               int((cov & ~isin).sum()), int((~isin).sum()))
        print(line)
    print("\n=== 合计 ===")
    for name in ARMS:
        t = tot[name]
        Rin = t["cin"] / max(1, t["nin"])
        Rout = t["cout"] / max(1, t["nout"])
        P = t["cp"] / max(1, t["np_"])
        F1 = 2 * (t["cin"] + t["cout"]) / max(1, t["nin"] + t["nout"])
        F1 = 2 * F1 * P / max(1e-9, F1 + P)
        print("  %-9s 框内 %3d/%4d R=%.4f | 框外 %4d/%4d R=%.4f | 段 %5d/%5d P=%.4f | F1=%.4f | segs %d"
              % (name, t["cin"], t["nin"], Rin, t["cout"], t["nout"], Rout,
                 t["cp"], t["np_"], P, F1, t["seg"]))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
