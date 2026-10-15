"""_occpol.py -- 「我们自己的框该不该在提线前涂白」的口径实验（用户 m42009 的互相等待）。

背景：`stage1/wires.py:occlude_boxes` 在提线前把**我们自己检出的元件框**涂成背景色。
`_chainmeas.py` 量到：GT 导线边里 17.5% 至少有一个采样点落在这些框内，而它们的覆盖率
只有 2.6%（框外 34.4%）。但 `bridge_boxes`（框内交叉验证桥接）在 10 张里只桥出 **0** 根
——说明丢掉的那些**不是**「两截共线短桩被框切开」，而是「导线端点走进框里就没了」。

本探针直接换四种遮挡口径，量导线级边覆盖/段精度，用来判断「涂白」到底是不是元凶：
  keep  : `occlude=[]`          —— 一根都不涂（元件本体轮廓也会变成导线，精度可能崩）
  all   : `occlude=boxes`       —— 现状（全涂）
  big   : 只涂「大框」（min side >= --big）—— 只清掉真正的 IC 本体，小元件（电阻/电容
          的框里往往就是导线端点与位号）保持透明
  text  : `occlude=boxes` 但 `rescue_pad=0`（不豁免贴框短桩）—— 反向对照

用法：python -X utf8 work\\tools\\dev\\_occpol.py --cases-file work\\_cases_w28.txt
"""
from __future__ import annotations

import argparse
import io
import json
import math
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


def load_cases(f: str):
    txt = io.open(f, encoding="utf-8-sig").read()
    return [c.strip() for c in txt.replace("\r", "").replace("\n", ",").split(",") if c.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-file", default=str(ROOT / "work" / "_cases_w28.txt"))
    ap.add_argument("--stage1-dir", default=str(ROOT / "work" / "_full220" / "runs" / "stage1"))
    ap.add_argument("--png-dir", default=str(ROOT / "work" / "_submit_in220"))
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--big", type=float, default=60.0, help="大框判定：min(w,h) >= 这个值")
    ap.add_argument("--anchor", type=float, default=6.0,
                    help="sel 臂的锚定半径：keep 段端点离已检出引脚点 <= 这个值才算有交叉验证")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cases = load_cases(a.cases_file)
    if a.limit:
        cases = cases[:a.limit]
    pol = ("keep", "all", "big", "text")
    tot = {k: {"cg": 0, "ng": 0, "cp": 0, "np": 0} for k in pol + ("sel",)}
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
        big = [b for b in boxes if min(b[2] - b[0], b[3] - b[1]) >= a.big]
        GA = WE.gt_segs(case, H)
        GA = np.asarray(GA, np.float64).reshape(-1, 4) if GA else np.zeros((0, 4))
        out = {}
        keepsegs = []
        for k in pol:
            occ = {"keep": [], "all": boxes, "big": big, "text": boxes}[k]
            rp = 0.0 if k == "text" else 8.0
            w, _ = extract_wires(bgr, occlude=occ, erase=txts, erase_pad=1.0,
                                 drop_frame=True, max_axis_dev=12.0,
                                 min_extent=80.0, rescue_pad=rp)
            PA = SE.seg_arrays(w["segments"])
            if k == "keep":
                keepsegs = w["segments"]
            cg, ng, cp, npd = WE.score(GA, PA, a.tol)
            out[k] = (cg, ng, cp, npd)
            tot[k]["cg"] += cg
            tot[k]["ng"] += ng
            tot[k]["cp"] += cp
            tot[k]["np"] += npd
        # ★ 第五臂 `sel`（用户 m42009 的「交叉验证」）：不涂框（保留框内导线证据），
        #   但**只接纳有独立证据锚定的段** —— 端点落在已检出引脚点 6px 内的才要。
        #   问的是：能不能用「引脚证据」把那 +147 条 GT 边挑出来，而不吞下 +1615 条 fp 段。
        pinpts = []
        for v in (s1.get("pins") or {}).values():
            for q in (v or []):
                try:
                    pinpts.append((float(q[0]), float(q[1])))
                except (TypeError, IndexError):
                    pass
        sel = []
        for sgm in keepsegs:
            if any(math.hypot(pt[0] - q[0], pt[1] - q[1]) <= a.anchor
                   for pt in sgm for q in pinpts):
                sel.append(sgm)
        cg, ng, cp, npd = WE.score(GA, SE.seg_arrays(sel), a.tol)
        tot["sel"]["cg"] += cg
        tot["sel"]["ng"] += ng
        tot["sel"]["cp"] += cp
        tot["sel"]["np"] += npd
        out["sel"] = (cg, ng, cp, npd)
        print("sel_segs %4d/%4d" % (len(sel), len(keepsegs)), end="  ")
        print("%-14s boxes %3d (big %2d) | keep %4d/%4d %4d/%4d | all %4d/%4d %4d/%4d | "
              "big %4d/%4d %4d/%4d | text %4d/%4d %4d/%4d"
              % (cid, len(boxes), len(big),
                 out["keep"][0], out["keep"][1], out["keep"][2], out["keep"][3],
                 out["all"][0], out["all"][1], out["all"][2], out["all"][3],
                 out["big"][0], out["big"][1], out["big"][2], out["big"][3],
                 out["text"][0], out["text"][1], out["text"][2], out["text"][3]))
    print("\n=== 合计（边覆盖 R / 段精度 P / F1）===")
    for k in pol + ("sel",):
        t = tot[k]
        R = t["cg"] / max(1, t["ng"])
        P = t["cp"] / max(1, t["np"])
        F1 = 2 * R * P / max(1e-9, R + P)
        print("  %-5s 边 %4d/%4d R=%.4f | 段 %5d/%5d P=%.4f | F1=%.4f"
              % (k, t["cg"], t["ng"], R, t["cp"], t["np"], P, F1))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
