# -*- coding: utf-8 -*-
"""`_wireedge.py` —— 导线掩膜的**彩色边缘证据通道** A/B（用户 m40034 的第一项）。

背景：`pipeline/pskel.py:87 ink_mask` 三档（`adaptive`/`otsu`/`bgdiff`）全部建在
**灰度**上；颜色只在 `wire_colour_mask` 里当**过滤器**（把不合色相的像素删掉），
所以「等亮度但不同色」的细走线在灰度阈值下根本进不来。本探针量的是新通道
`pskel.colour_edge_mask`（Di Zenzo 彩色梯度，`STAGE1_WIRE_EDGE=1`）能救回多少。

三条臂（都用 stage1 真实的 `extract_wires` 参数：occlude=元件框、erase=文字框）：
  base       STAGE1_WIRE_EDGE=0  STAGE1_WIRE_COLOUR=off      纯灰度
  edge       STAGE1_WIRE_EDGE=1  STAGE1_WIRE_COLOUR=off      灰度 ∪ 彩色边缘
  edge+rel   STAGE1_WIRE_EDGE=1  STAGE1_WIRE_COLOUR=rel      再加**相对**彩度门

判据与 `tools/eval/stage1_wires_eval.py` 完全一致（GT 边采样 16 点，**全部点**都在
预测段 5px 内才算这条边被覆盖；精度同理反向），保证和历史读数可比。

用法：
    python work\\tools\\dev\\_wireedge.py --cases-file work\\_cases9.txt
    python work\\tools\\dev\\_wireedge.py --cases-file work\\_cases220.txt --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
for p in (str(EDA), str(EDA / "tools" / "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402
from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402
import stage1_wires_eval as SE  # noqa: E402
from stage1 import wires as W  # noqa: E402


def imread_unicode(p: Path) -> np.ndarray:
    buf = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def gt_segs(case, H: float):
    gt = json.load(open(case.json_path, encoding="utf-8"))
    out = []
    for net in gt["nets"].values():
        for pts in (net.get("edges") or {}).values():
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                out.append([[float(pts[i]["x"]), H - float(pts[i]["y"])],
                            [float(pts[i + 1]["x"]), H - float(pts[i + 1]["y"])]])
    return out


def score(GA, PA, tol: float):
    """返回 (边覆盖, GT 边数, 预测段覆盖, 预测段数) —— 与 stage1_wires_eval 同判据。"""
    n_g = GA.shape[0]
    n_p = PA.shape[0]
    cov_g = cov_p = 0
    if n_g and n_p:
        pts, of = [], []
        for i in range(n_g):
            pts.append(SE.sample_polyline(GA[i, :2], GA[i, 2:], 16))
            of.append(np.full(16, i))
        d = SE.dist_pts_to_segs(np.concatenate(pts, 0), PA)
        worst = np.full(n_g, 0.0)
        np.maximum.at(worst, np.concatenate(of), d)
        cov_g = int((worst <= tol).sum())
        ppts, pof = [], []
        for i in range(n_p):
            ppts.append(SE.sample_polyline(PA[i, :2], PA[i, 2:], 16))
            pof.append(np.full(16, i))
        pd = SE.dist_pts_to_segs(np.concatenate(ppts, 0), GA)
        pworst = np.full(n_p, 0.0)
        np.maximum.at(pworst, np.concatenate(pof), pd)
        cov_p = int((pworst <= tol).sum())
    return cov_g, n_g, cov_p, n_p


ARMS = [
    ("base", {"STAGE1_WIRE_EDGE": "0", "STAGE1_WIRE_COLOUR": "off",
              "STAGE1_WIRE_EDGE_NEAR": "2"}),
    ("rel", {"STAGE1_WIRE_EDGE": "0", "STAGE1_WIRE_COLOUR": "rel",
             "STAGE1_WIRE_EDGE_NEAR": "2"}),
    ("edge", {"STAGE1_WIRE_EDGE": "1", "STAGE1_WIRE_COLOUR": "off",
              "STAGE1_WIRE_EDGE_NEAR": "0"}),
    ("edge_near", {"STAGE1_WIRE_EDGE": "1", "STAGE1_WIRE_COLOUR": "off",
                   "STAGE1_WIRE_EDGE_NEAR": "2"}),
    ("edge+rel", {"STAGE1_WIRE_EDGE": "1", "STAGE1_WIRE_COLOUR": "rel",
                  "STAGE1_WIRE_EDGE_NEAR": "2"}),
    # ★ 用户 m42009：「背景是白的，导线颜色不确定，只要从背景过渡到任意颜色都可能是
    #   导线」⇒ 前景判据换成与**局部背景**的 Lab ΔE（`STAGE1_WIRE_INK=bglab`）。
    ("ink_lab", {"STAGE1_WIRE_EDGE": "0", "STAGE1_WIRE_COLOUR": "off",
                 "STAGE1_WIRE_INK": "bglab"}),
    # 加「框内用局部纸色填充」（消接缝）后的 bglab。
    ("ink_lab+locfill", {"STAGE1_WIRE_EDGE": "0", "STAGE1_WIRE_COLOUR": "off",
                         "STAGE1_WIRE_INK": "bglab", "STAGE1_OCC_LOCAL": "1"}),
    # ★ 用户 m42009：「还要用其他手段交叉验证才能得出这是导线」⇒ 色相门从**一票否决**
    #   改成「未确认像素只要与已确认像素连通就留下」（`STAGE1_WIRE_UNCONF=near`）。
    ("dom_veto", {"STAGE1_WIRE_EDGE": "0", "STAGE1_WIRE_COLOUR": "dominant",
                  "STAGE1_WIRE_UNCONF": "veto"}),
    ("dom_near", {"STAGE1_WIRE_EDGE": "0", "STAGE1_WIRE_COLOUR": "dominant",
                  "STAGE1_WIRE_UNCONF": "near"}),
    ("dom_keep", {"STAGE1_WIRE_EDGE": "0", "STAGE1_WIRE_COLOUR": "dominant",
                  "STAGE1_WIRE_UNCONF": "keep"}),
    ("ink_lab+dom_near", {"STAGE1_WIRE_EDGE": "0", "STAGE1_WIRE_COLOUR": "dominant",
                          "STAGE1_WIRE_INK": "bglab", "STAGE1_WIRE_UNCONF": "near",
                          "STAGE1_OCC_LOCAL": "1"}),
    # 交付默认口径（`STAGE1_WIRE_COLOUR=hint`，问一次模型导线是什么颜色）下的否决 vs 交叉验证。
    ("hint_veto", {"STAGE1_WIRE_EDGE": "0", "STAGE1_WIRE_COLOUR": "hint",
                   "STAGE1_WIRE_UNCONF": "veto"}),
    ("hint_near", {"STAGE1_WIRE_EDGE": "0", "STAGE1_WIRE_COLOUR": "hint",
                   "STAGE1_WIRE_UNCONF": "near"}),
]

# 每条臂都要先清掉的开关（否则上一条臂的值会漏到这条）
_ENV_KEYS = ("STAGE1_WIRE_EDGE", "STAGE1_WIRE_EDGE_NEAR", "STAGE1_WIRE_COLOUR",
             "STAGE1_WIRE_INK", "STAGE1_WIRE_UNCONF", "STAGE1_WIRE_UNCONF_NEAR",
             "STAGE1_OCC_LOCAL")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-file", default=str(ROOT / "work" / "_cases9.txt"))
    ap.add_argument("--stage1-dir", default=str(ROOT / "work" / "_full220" / "runs" / "stage1"))
    ap.add_argument("--png-dir", default=str(ROOT / "work" / "_submit_in220"))
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default="base,edge,edge+rel")
    a = ap.parse_args()

    want = [s.strip() for s in Path(a.cases_file).read_text(encoding="utf-8-sig").split(",") if s.strip()]
    if a.limit:
        want = want[:a.limit]
    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    arms = [nm for nm in a.arms.split(",") if nm]
    tot = {nm: [0, 0, 0, 0] for nm in arms}
    rows = []
    for cid in want:
        s1 = Path(a.stage1_dir) / ("%s.stage1.json" % cid)
        png = Path(a.png_dir) / ("%s.png" % cid)
        if not s1.exists() or not png.exists():
            print("  跳过 %s（缺 stage1 或 png）" % cid)
            continue
        case = cases.get(cid)
        if case is None:
            print("  跳过 %s（GT 里没有这个 case）" % cid)
            continue
        d = json.load(open(s1, encoding="utf-8"))
        H = float(d["image"]["h"])
        bgr = imread_unicode(png)
        comps = [c["bbox_img"] for c in d.get("components", [])]
        texts = [t["bbox_img"] for t in d.get("texts", [])]
        GA = SE.seg_arrays(gt_segs(case, H))
        row = {"case": cid}
        for nm in arms:
            env = dict(ARMS)[nm]
            for k in _ENV_KEYS:
                os.environ.pop(k, None)
            os.environ.update(env)
            w, _t = W.extract_wires(
                bgr, occlude=comps, erase=texts, erase_pad=1.0, drop_frame=True,
                max_axis_dev=12.0, min_extent=80.0, rescue_pad=8.0)
            PA = SE.seg_arrays(w["segments"])
            cg, ng, cp, npd = score(GA, PA, a.tol)
            row[nm] = (cg, ng, cp, npd)
            tot[nm][0] += cg
            tot[nm][1] += ng
            tot[nm][2] += cp
            tot[nm][3] += npd
        rows.append(row)

    print("\n=== 逐张（边覆盖 / GT 边数 · 段覆盖 / 预测段数）===")
    for r in rows:
        parts = []
        for nm in arms:
            cg, ng, cp, npd = r[nm]
            parts.append("%s %d/%d %d/%d" % (nm, cg, ng, cp, npd))
        print("  %-14s %s" % (r["case"], " | ".join(parts)))
    print("\n=== 合计 ===")
    for nm in arms:
        cg, ng, cp, npd = tot[nm]
        R = cg / max(1, ng)
        P = cp / max(1, npd)
        F = 2 * R * P / max(1e-9, R + P)
        print("  %-9s 边覆盖 %4d/%4d R=%.4f | 段覆盖 %5d/%5d P=%.4f | F1=%.4f"
              % (nm, cg, ng, R, cp, npd, P, F))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
