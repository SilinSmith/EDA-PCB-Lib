# -*- coding: utf-8 -*-
"""从新补充训练集（页面级 GT）**收字**给 CRNN 训练：不靠 GT 框，靠我们自己的检测框 + 近似字符串匹配。

为什么不是「弱配对」（`tools/dev/import_new_dataset.py`）
--------------------------------------------------------
那份清单按「文字框中心到元件框 / 引脚点」的几何距离配对，抽检只有 ~56% 正确：符号体框
里套住的是图形（`+ ◁ ▷`）、或者抢到邻居的文字。而这批 GT **没有文字框**，几何本身就不可靠。

本脚本换一条路：检测框用**我们自己的** `detect_texts`（这正好是推理时会看到的那套框），
配对用**字符串**而不是几何——
  * `exact`：该框在目标角色上的读数与 GT 串完全相同（干净，但只是自洽，教不会新东西）；
  * `near` ：读数与 GT 串的编辑距离 ≤ `min(2, len//2)`（**这是价值所在**：说明框找对了、
             但模型读错了；标签取 GT 串 ⇒ 直接把「纠正样例」喂进训练）。
几何距离只用来在多个候选间挑一个（并做边缘化：最优与次优离得太近就整条丢弃）。

两条纯度闸（都是实测踩出来的）：
  * 短串（`51`/`12`）的 edit 1 毫无意义 ⇒ 长度 < 3 只收 exact；
  * 引脚红点、绿色导线会被读成数字 ⇒ 框内「红/绿强饱和像素」占比 > `--marker` 的整条丢弃
    （注意不能用 `colored_frac`：位号/值是**深蓝**字，也会被它判成"有色"）。

坐标只影响挑哪个框，不进入产物：产物是 `{image, bbox, text, role}`，与坐标系无关。

用法
----
    cd Y:\\workspace\\EDA
    python -X utf8 eda-pcb\\tools\\dev\\harvest_labels.py --limit 3 --dbg
    python -X utf8 eda-pcb\\tools\\dev\\harvest_labels.py            # 48 组全量
产物：`work/_boxes_ocr.jsonl`（逐条候选）、`work/_man_new_ocr.jsonl`（训练清单）、
      `work/_harvest_stats.json`（每角色 exact/near 计数与几何中位）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import zlib
from pathlib import Path
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EDA = Path(__file__).resolve().parents[2]
REPO = EDA.parent
STAGE = EDA / "work" / "_newdata_stage"
CHARSET = json.loads((EDA / "textrec" / "data" / "charset.json").read_text(encoding="utf-8"))["chars"]
CSET = set(CHARSET)
ROLES_ALL = ("designator", "value", "pin_number", "pin_name")
ROLE_ID = {"designator": 0, "value": 1, "pin_name": 2, "pin_number": 3,
           "net_label": 4, "note": 5}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"\s+", "", s)
    return s.upper().replace("Μ", "U").replace("μ", "U")


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 3:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def ok_text(t: str) -> bool:
    t = t or ""
    if not t or len(t) > 13:
        return False
    return all(c in CSET for c in t)


def marker_frac(bgr, box) -> float:
    """框内「**红/绿**强饱和像素占比」——引脚红点、绿色导线都会落到这，蓝/黑文字不会。"""
    import numpy as np
    h, w = bgr.shape[:2]
    x0 = max(0, int(box[0])); y0 = max(0, int(box[1]))
    x1 = min(w, int(np.ceil(box[2]))); y1 = min(h, int(np.ceil(box[3])))
    if x1 <= x0 or y1 <= y0:
        return 1.0
    c = bgr[y0:y1, x0:x1].astype(np.int16)
    b, g, r = c[:, :, 0], c[:, :, 1], c[:, :, 2]
    red = (r > 90) & (r - np.maximum(g, b) > 50)
    grn = (g > 90) & (g - np.maximum(r, b) > 50)
    return float((red | grn).mean())


def anchor_dist(box, anchor, is_point: bool) -> float:
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    if is_point:
        return ((cx - anchor[0]) ** 2 + (cy - anchor[1]) ** 2) ** 0.5
    ax0, ay0, ax1, ay1 = anchor
    dx = max(ax0 - cx, 0.0, cx - ax1)
    dy = max(ay0 - cy, 0.0, cy - ay1)
    return (dx * dx + dy * dy) ** 0.5


def gt_targets(doc: dict) -> List[Dict]:
    out = []
    for key, comp in (doc.get("components") or {}).items():
        if not key or key.startswith("∅") or re.fullmatch(r"p\d+", key):
            continue
        bb = [float(v) for v in (comp.get("bbox") or [])]
        if len(bb) != 4:
            continue
        out.append(dict(text=key, role="designator", anchor=bb, point=False))
        val = (comp.get("value") or "").strip()
        if val and val != key:
            out.append(dict(text=val, role="value", anchor=bb, point=False))
    for _ck, pins in (doc.get("pins") or {}).items():
        for pkey, pin in (pins or {}).items():
            num = pkey[4:] if pkey.startswith("pin_") else pkey
            pt = pin.get("point") or {}
            if pt.get("x") is None:
                continue
            out.append(dict(text=num, role="pin_number", anchor=(float(pt["x"]), float(pt["y"])),
                            point=True))
            nm = (pin.get("pinname") or "").strip()
            if nm:
                out.append(dict(text=nm, role="pin_name", anchor=(float(pt["x"]), float(pt["y"])),
                                point=True))
    return [t for t in out if ok_text(t["text"])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 组（冒烟）")
    ap.add_argument("--root", default="", help="案例根目录（默认 work/_newdata_stage）")
    ap.add_argument("--out", default="work/_man_new_ocr.jsonl")
    ap.add_argument("--boxes", default="work/_boxes_ocr.jsonl")
    ap.add_argument("--stats", default="work/_harvest_stats.json")
    ap.add_argument("--near", type=int, default=2, help="near 层的编辑距离上限")
    ap.add_argument("--r-frac", type=float, default=0.05, help="几何半径 = max(60, frac*W)")
    ap.add_argument("--marker", type=float, default=0.25, help="红/绿像素占比上限")
    ap.add_argument("--dbg", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(EDA))
    from stage2.text import TextEngine, detect_texts, imread_unicode  # noqa: E402

    engine = TextEngine()
    root = Path(a.root) if a.root else STAGE
    cases = sorted(p for p in root.iterdir() if p.is_dir())
    if a.limit:
        cases = cases[:a.limit]
    man, boxrows, stats = [], [], {}
    for k, cdir in enumerate(cases, 1):
        png = next(iter(sorted(cdir.glob("*.png"))), None)
        js = next(iter(sorted(cdir.glob("*.json"))), None)
        if png is None or js is None:
            continue
        bgr = imread_unicode(png)
        h, w = bgr.shape[:2]
        rmax = max(60.0, a.r_frac * w)
        boxes = detect_texts(bgr)
        reads = engine.read(bgr, boxes, ROLES_ALL)
        doc = json.loads(Path(js).read_text(encoding="utf-8"))
        targets = gt_targets(doc)

        pairs = []                                  # (edit, dist, ti, bi)
        mfrac: Dict[int, float] = {}
        for ti, t in enumerate(targets):
            nt = norm(t["text"])
            near_cap = max(1, min(a.near, len(nt) // 2))
            if len(nt) < 3:
                near_cap = 0                       # 短串的 edit 1 没有意义
            for bi, rec in enumerate(reads):
                got = rec["reads"].get(t["role"])
                if not got or not got[0]:
                    continue
                nr = norm(got[0])
                if not nr:
                    continue
                d = lev(nr, nt)
                if d > near_cap:
                    continue
                if bi not in mfrac:
                    mfrac[bi] = marker_frac(bgr, rec["bbox"])
                if mfrac[bi] > a.marker:
                    continue                       # 引脚红点 / 绿色导线被当成字
                dist = anchor_dist([float(v) for v in rec["bbox"]], t["anchor"], t["point"])
                if dist > rmax:
                    continue
                pairs.append((d, dist, ti, bi))
        pairs.sort()
        used_t, used_b = set(), set()
        for d, dist, ti, bi in pairs:
            if ti in used_t or bi in used_b:
                continue
            t = targets[ti]
            used_t.add(ti)
            used_b.add(bi)
            tier = "exact" if d == 0 else "near"
            bb = [round(float(v), 1) for v in reads[bi]["bbox"]]
            row = dict(image=str(png), case=cdir.name, src="real", role=t["role"],
                       role_id=ROLE_ID[t["role"]], text=t["text"], bbox=bb,
                       tier=tier, edit=d, dist=round(dist, 1),
                       split=("val" if (zlib.crc32(cdir.name.encode("utf-8")) % 10) == 0
                              else "train"))
            man.append(row)
            boxrows.append(dict(case=cdir.name, image=str(png), bbox=bb, text=t["text"],
                                role=t["role"], tier=tier, edit=d, dist=round(dist, 1)))
            s = stats.setdefault(t["role"], {"exact": 0, "near": 0, "dist": []})
            s[tier] += 1
            s["dist"].append(round(dist, 1))
        if a.dbg:
            cov = {r: (sum(1 for t in targets if t["role"] == r),
                       sum(1 for m in man if m["case"] == cdir.name and m["role"] == r))
                   for r in ROLES_ALL}
            print("  %-10s boxes %4d targets %4d 覆盖 %s" % (cdir.name, len(boxes), len(targets), cov))
        if k % 10 == 0:
            print("  ... %d/%d 组，累计 %d 条" % (k, len(cases), len(man)))

    (REPO / a.out).write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in man),
                              encoding="utf-8")
    (REPO / a.boxes).write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in boxrows),
                                encoding="utf-8")
    tot = {r: {k2: v2 for k2, v2 in s.items() if k2 != "dist"} for r, s in stats.items()}
    (REPO / a.stats).write_text(json.dumps(tot, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n记录 %d（train %d / val %d）" % (
        len(man), sum(1 for m in man if m["split"] == "train"),
        sum(1 for m in man if m["split"] == "val")))
    for r, s in stats.items():
        ds = sorted(s["dist"])
        print("  %-11s exact %4d near %4d 中位几何距离 %.1f" % (
            r, s["exact"], s["near"], ds[len(ds) // 2] if ds else -1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
