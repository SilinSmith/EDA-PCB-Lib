"""从官方 220 张 GT 里把「真实文字的图 + 真值串」挖出来（弱监督标注）。

为什么：合成语料的文字只有 4~7px 高、字体是 PIL 内置字体，和真实图纸的矢量字体有域差。
而官方 GT 里本来就有全部真值串（位号 `Name`、参数 `value`、引脚名 `pinname`、引脚号
`pin_<number>`），配上 `cvparts.textbox` 的定位框就能造出**真图真题**。

标注是弱监督的（靠几何锚点猜哪个框对应哪条真值），所以每条都记录 `n_cand`（该锚点
附近有几个候选框）用来事后卡精度。

跑法（CWD = Y:\\workspace\\EDA\\eda-pcb）：
    $env:PYTHONIOENCODING='utf-8'
    python Y:\\workspace\\EDA\\work\\ocr_real_build.py --out Y:\\workspace\\EDA\\work\\ocr_real
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

import cv2
import numpy as np

from schematic_parser import data as gtdata
from cvparts import textbox as tb

sys.stdout.reconfigure(encoding="utf-8")

R_COMP = 45.0     # 元件文字（位号/参数）候选框的搜索半径
R_PIN = 30.0      # 引脚文字候选框的搜索半径
MAXLEN = 24       # 和 textrec.config.MAX_TEXT_LEN 一致


def box_dist(r, box):
    """点-矩形距离；r 是先验点或矩形。"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if len(r) == 2:
        return ((cx - r[0]) ** 2 + (cy - r[1]) ** 2) ** 0.5
    dx = max(r[0] - cx, 0, cx - r[2])
    dy = max(r[1] - cy, 0, cy - r[3])
    return (dx * dx + dy * dy) ** 0.5


def inside(box, rect, margin=0.0):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (rect[0] - margin <= cx <= rect[2] + margin
            and rect[1] - margin <= cy <= rect[3] + margin)


def build_case(case):
    doc = gtdata.load_json(case.json_path)
    img = cv2.imread(case.png_path)
    if img is None:
        return []
    H = img.shape[0]
    boxes = tb.detect_boxes(img)
    if not boxes:
        return []
    comps = doc.get("components") or {}
    pins = doc.get("pins") or {}

    # 每个元件的图像坐标矩形
    crect = {}
    for cid, comp in comps.items():
        bx = comp["bbox"]
        crect[cid] = (bx[0], H - bx[3], bx[2], H - bx[1])
    # 框被哪个元件框包住（用来排除「别人的位号」）
    own_of_box = [None] * len(boxes)
    for i, b in enumerate(boxes):
        for cid, r in crect.items():
            if inside(b, r, margin=2.0):
                own_of_box[i] = cid
                break

    recs = []

    def rec(role, text, bi, dist, n_cand, cid=None, extra=None):
        text = (text or "").strip()
        if not text or len(text) > MAXLEN:
            return
        b = boxes[bi]
        w, h = b[2] - b[0], b[3] - b[1]
        if w < 1 or h < 3:
            return
        d = {"case": case.case_id, "split": case.split, "role": role, "text": text,
             "bbox": [float(v) for v in b], "w": float(w), "h": float(h),
             "dist": float(dist), "n_cand": int(n_cand)}
        if cid is not None:
            d["cid"] = cid
        if extra:
            d.update(extra)
        recs.append(d)

    # ---- 元件级：位号 / 参数 ----
    for cid, comp in comps.items():
        r = crect[cid]
        cand = []
        for i, b in enumerate(boxes):
            if own_of_box[i] not in (None, cid):
                continue          # 这个框属于别的元件
            d = box_dist(r, b)
            if d <= R_COMP:
                cand.append((d, i))
        cand.sort()
        want = []
        if comp.get("Name"):
            want.append(("designator", comp["Name"]))
        if comp.get("value"):
            want.append(("value", str(comp["value"])))
        for k, (role, text) in enumerate(want):
            if k >= len(cand):
                break
            rec(role, text, cand[k][1], cand[k][0], len(cand), cid,
                {"n_want": len(want), "cand_rank": k})

    # ---- 引脚级：引脚号 / 引脚名 ----
    for cid, pp in pins.items():
        r = crect.get(cid)
        for pk, pv in pp.items():
            pt = pv["point"]
            px, py = pt["x"], H - pt["y"]
            num = pk.split("_", 1)[1] if "_" in pk else pk
            # 引脚号：锚 = 引脚点，且**不在元件框内**
            out = [(box_dist((px, py), b), i) for i, b in enumerate(boxes)
                   if (r is None or not inside(b, r, margin=-1.0))]
            out.sort()
            n_out = sum(1 for d, _ in out if d <= R_PIN)
            if out and out[0][0] <= R_PIN:
                rec("pin_number", num, out[0][1], out[0][0], n_out, cid)
            # 引脚名：锚 = 在元件框内、离引脚点最近的框；非 box 类退化为最近框
            nm = pv.get("pinname")
            if nm:
                if r is not None and (r[2] - r[0]) > 12 and (r[3] - r[1]) > 12:
                    ins = [(box_dist((px, py), b), i) for i, b in enumerate(boxes)
                           if inside(b, r, margin=1.0)]
                    ins.sort()
                    n_ins = sum(1 for d, _ in ins if d <= R_PIN)
                    if ins and ins[0][0] <= R_PIN:
                        rec("pin_name", nm, ins[0][1], ins[0][0], n_ins, cid,
                            {"anchored": "inside"})
                        continue
                got = [(box_dist((px, py), b), i) for i, b in enumerate(boxes)]
                got.sort()
                n_got = sum(1 for d, _ in got if d <= R_PIN)
                if got and got[0][0] <= R_PIN:
                    rec("pin_name", nm, got[0][1], got[0][0], n_got, cid,
                        {"anchored": "nearest"})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"Y:\workspace\EDA\work\ocr_real")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cases = gtdata.discover(gtdata.DEFAULT_DATA_ROOT)
    if a.limit:
        cases = cases[:a.limit]
    cases = sorted(cases, key=lambda c: c.case_id)

    rows = []
    for k, c in enumerate(cases):
        rs = build_case(c)
        split = "val" if (k % a.val_every == 0) else "train"
        for r in rs:
            r["fold"] = split
        rows.extend(rs)
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(cases)} cases, {len(rows)} rows", flush=True)

    p = os.path.join(a.out, "manifest.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n写出 {p}：{len(rows)} 条")
    print("  role × fold:", dict(Counter((r["role"], r["fold"]) for r in rows)))
    print("  n_cand==1 占比:", {k: round(v, 3) for k, v in
          ((role, sum(1 for r in rows if r["role"] == role and r["n_cand"] == 1)
            / max(1, sum(1 for r in rows if r["role"] == role)))
           for role in sorted({r["role"] for r in rows}))})
    print("  字符集大小:", len({ch for r in rows for ch in r["text"]}))
    print("  字符集:", "".join(sorted({ch for r in rows for ch in r["text"]})))


if __name__ == "__main__":
    main()
