# -*- coding: utf-8 -*-
"""GT 可信度审计：val77 上把每个 FN / FP 重新归类，回答「是模型错还是标注错」。

与 `work/tools/eval/fn_fp.py` **完全相同**的推理参数：
    weights = eda-pcb/yolo_part/runs/detect/pcb43_v2_from_best/weights/best.pt
    conf 0.05, iou 0.7, imgsz 1024, device 0
    配对判据 CRIT="corner", CORNER_TOL=20px（官方口径，见 schematic_parser/stamp.py:12）

分类
----
FN（漏检的 GT）:
  TRUEMISS : 该 GT 周围 40px 内**没有任何类别**的预测
  CLASSCONF: 同位置有**别的类**的预测（corner<=20 或 IoU>=0.3）—— 报对方类名+conf
  DEADGT   : GT 框内几乎没墨（框内暗像素 < 2%，或外环 6px 墨迹中位 < 0.02）= 框住空白
FP（对不上 GT 的预测）:
  NEARMISS : 最近 GT 的 corner 差在 (20,40] / (40,60] —— 位置对、框偏
  CLASSCONF: 同位置（corner<=20 或 IoU>=0.3）有**别的类**的 GT
  REALFP   : 60px 内没有任何 GT

用法:
  $env:PYTHONPATH='Y:\workspace\EDA\eda-pcb'
  K:\\python\\311\\python.exe work\\tools\\diag\\_gtfaith.py --out work\\tools\\diag\\_gtfaith_report.txt
"""
import argparse
import json
import re
import statistics as stx
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

DEF_W = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\runs\detect"
             r"\pcb43_v2_from_best\weights\best.pt")
# v2 的 junction 已失效（Y:\workspace\EDA\trainset_v2 不存在），实际数据在 delete\ 下
V2 = Path(r"Y:\workspace\EDA\delete\训练集_v2")
V3 = Path(r"Y:\workspace\EDA\trainset_v3")
CORNER_TOL = 20.0
NEAR1, NEAR2 = 40.0, 60.0
AROUND = 40.0          # TRUEMISS 的「附近」半径
INK_T = 128            # 暗像素阈值（图上墨迹是暗的、底是白，mean~246）
INK_BOX_MIN = 0.02     # 框内墨迹占比低于此 => 框空白
INK_RING_MED = 0.02    # 外环墨迹中位低于此 => 周围也空白


# ------------------------------------------------------------------ geometry
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def corner_dev(a, b):
    return max(abs(a[i] - b[i]) for i in range(4))


def hit(a, b):
    return corner_dev(a, b) <= CORNER_TOL


def center_dist(a, b):
    return (((a[0] + a[2]) - (b[0] + b[2])) / 2) ** 2 + \
           (((a[1] + a[3]) - (b[1] + b[3])) / 2) ** 2


def center_dist_px(a, b):
    return center_dist(a, b) ** 0.5


def box_dist(a, b):
    """两矩形间距（0 = 相交/接触）。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def overlap_loose(p, g):
    return corner_dev(p, g) <= CORNER_TOL or iou(p, g) >= 0.3


# 官方 GT 是「按方向固定尺寸的图章」，见 eda-pcb/schematic_parser/stamp.py
# 这里列出需要在报告里做「图章尺寸」判定的几个类的全部方向尺寸。
STAMP = {
    "r": [(8, 20), (20, 8)],
    "c": [(16, 20), (20, 16)],
    "gnd": [(20, 19), (19, 20)],
    "v": [(10, 10)],
    "testpoint": [(20, 20)],
    "pin": [(20, 20)],
    "net_input": [(30, 10), (10, 30), (44, 11), (11, 44)],
    "net_output": [(30, 10), (10, 30)],
    "net_bidirection": [(30, 10), (10, 30)],
    "l": [(34, 34)],
    "led": [(20, 20)],
    "d": [(20, 20)],
    "fuse": [(22, 22)],
    "block": [(20, 20)],
}
# 父 agent 实测：train 侧存在 `synth_<id>` 同源文件 => 这些源图的 val 样本被污染
LEAK_IDS = {"0002", "0011", "0026", "0027", "0029", "0050", "0151"}


def source_id(stem: str) -> str:
    """`200_train_cases-0050__t00_01` -> `0050`; `10_GTcase-0037-Datasheet__full` -> `0037`."""
    base = stem.split("__")[0]
    runs = re.findall(r"\d+", base)
    return runs[-1] if runs else base


def stamp_rank(cname, box):
    """图章尺寸命中的排名（1 = 该类最常见尺寸），未命中返回 None。"""
    lst = STAMP.get(cname)
    if not lst:
        return None
    w, h = round(box[2] - box[0]), round(box[3] - box[1])
    for i, (sw, sh) in enumerate(lst, 1):
        if abs(w - sw) <= 1 and abs(h - sh) <= 1:
            return i
    return None


# ------------------------------------------------------------------ data
def load_gt(lbl: Path, w: int, h: int):
    out = []
    if not lbl.exists():
        return out
    for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def ring_pixels(x1, y1, x2, y2, w, h, d=6):
    """框外环 d px（不含框本身）的像素坐标。"""
    ax1, ay1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    ax2, ay2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if ax2 <= ax1 or ay2 <= ay1:
        return [], []
    ox1, oy1 = ax1 - d, ay1 - d
    ox2, oy2 = ax2 + d, ay2 + d
    rx, ry = [], []
    for yy in range(oy1, oy2):
        if 0 <= yy < h:
            for xx in range(ox1, ox2):
                if not (0 <= xx < w):
                    continue
                if ax1 <= xx < ax2 and ay1 <= yy < ay2:
                    continue          # 框内
                rx.append(xx); ry.append(yy)
    return rx, ry


def shape_sig(gray, box):
    """判定框内是「矩形电阻体」还是「锯齿/电感笔画」。

    返回 (列向暗游程数中位, 游程数的变异系数CV, 框填充率)。
    矩形体: 每一列都是同一条实心带 => 游程数恒为 1，CV≈0，填充率高。
    锯齿笔画: 斜线在每一列只占 1~2 个像素，且随列上下移动 => 填充率极低。
    """
    h, w = gray.shape
    ax1, ay1 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    ax2, ay2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    if ax2 <= ax1 or ay2 <= ay1:
        return 0.0, 0.0, 0.0
    sub = (gray[ay1:ay2, ax1:ax2] < INK_T)
    bw, bh = sub.shape[1], sub.shape[0]
    if bw < 2 or bh < 2:
        return 0.0, 0.0, float(sub.mean())
    vertical = bw >= bh          # 长轴水平 => 按列统计
    if not vertical:
        sub = sub.T
    ns = []
    for c in range(sub.shape[1]):
        col = sub[:, c]
        n, prev = 0, False
        for v in col:
            if v and not prev:
                n += 1
            prev = v
        ns.append(n)
    ns = np.asarray(ns, dtype=np.float64)
    med = float(np.median(ns))
    cv = float(ns.std() / ns.mean()) if ns.mean() > 0 else 0.0
    return med, cv, float(sub.mean())


def ink_stats(gray, box):
    """返回 (框内墨迹占比, 外环6px墨迹中位, 外环墨迹均值, 框内标准差)。"""
    x1, y1, x2, y2 = box
    h, w = gray.shape
    ax1, ay1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    ax2, ay2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if ax2 <= ax1 or ay2 <= ay1:
        return 0.0, 0.0, 0.0
    sub = gray[ay1:ay2, ax1:ax2]
    ink_in = float((sub < INK_T).mean())
    sd = float(sub.std())
    rx, ry = ring_pixels(x1, y1, x2, y2, w, h, 6)
    if rx:
        ring = gray[np.asarray(ry), np.asarray(rx)]
        # 「外环墨迹中位」= 环上像素的暗/亮中位数（0/1 采样）
        s = (ring < INK_T).astype(np.float32)
        med = float(np.median(s))
        meanv = float(s.mean())
    else:
        med, meanv = 0.0, 0.0
    return ink_in, med, meanv, sd


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEF_W))
    ap.add_argument("--root", default=str(V3))
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=str(Path(__file__).with_name("_gtfaith_report.txt")))
    ap.add_argument("--classes", default="gnd,v,r,c,testpoint,net_input,box,l")
    ap.add_argument("--dump", default="")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    names = model.names
    if not isinstance(names, dict):
        names = {i: n for i, n in enumerate(names)}

    root = Path(a.root)
    idir = root / "images" / a.split
    ldir = root / "labels" / a.split
    imgs = sorted(p for p in idir.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg"))

    want = [c.strip() for c in a.classes.split(",") if c.strip()]
    ids = {n: i for i, n in names.items() if n in want}

    FN = defaultdict(list)        # cls -> [{img,box,rcls,rs,conf,..}]
    FP = defaultdict(list)
    ALLGT = defaultdict(list)     # cls -> [ink_in, ring_med, ring_mean, bw, bh]
    GTCNT = Counter()             # (grp, cls) -> GT 数
    dup_pool = defaultdict(list)  # cls -> [{box,n}] 同框重复预测
    tot_gt, tot_fn, tot_pred, tot_fp = Counter(), Counter(), Counter(), Counter()
    per_img = {}
    cache = {}

    def gray_of(p: Path):
        if p not in cache:
            cache.clear()
            cache[p] = np.asarray(Image.open(str(p)).convert("L"), dtype=np.uint8)
        return cache[p]

    for i, p in enumerate(imgs):
        with Image.open(str(p)) as im:
            W, H = im.size
        gt = load_gt(ldir / (p.stem + ".txt"), W, H)
        for c, *_ in gt:
            tot_gt[c] += 1
        r = model.predict(str(p), conf=a.conf, iou=0.7, imgsz=a.imgsz,
                          verbose=False, device=a.device)[0]
        preds = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            for b, c, s in zip(xyxy, cl, cf):
                preds.append({"box": tuple(float(v) for v in b),
                              "cls": int(c), "score": float(s)})
        for q in preds:
            tot_pred[q["cls"]] += 1

        # ---- 与 fn_fp.py 完全一致的一对一配对
        used = set()
        pairs = []
        for q in sorted(preds, key=lambda x: -x["score"]):
            best, bi = None, -1
            for m, g in enumerate(gt):
                if m in used or g[0] != q["cls"]:
                    continue
                if not hit(q["box"], g[1:]):
                    continue
                v = -corner_dev(q["box"], g[1:])
                if best is None or v > best:
                    best, bi = v, m
            if bi >= 0:
                used.add(bi)
                pairs.append((bi, q))
        matched = {id(q) for _, q in pairs}

        g = gray_of(p)
        # ---- 每个 GT 的墨迹统计（用于「框住空白/框在文字上」的签名检验）
        for gg in gt:
            if gg[0] not in ids.values():
                continue
            GTCNT[("LEAK" if source_id(p.stem) in LEAK_IDS else "CLEAN", gg[0])] += 1
            bx = gg[1:]
            if bx[2] - bx[0] < 4 or bx[3] - bx[1] < 4:
                continue
            ai, am, an, _ = ink_stats(g, bx)
            rmed, rcv, fill = shape_sig(g, bx)
            ALLGT[gg[0]].append((ai, am, an, bx[2] - bx[0], bx[3] - bx[1],
                                 rmed, rcv, fill))

        # ---- FN
        for m, gg in enumerate(gt):
            if m in used:
                continue
            tot_fn[gg[0]] += 1
            box = gg[1:]
            near = [q for q in preds
                    if center_dist_px(q["box"], box) <= AROUND
                    or box_dist(q["box"], box) <= AROUND]
            near.sort(key=lambda q: -q["score"])
            same = [q for q in preds if overlap_loose(q["box"], box)]
            same.sort(key=lambda q: -q["score"])
            # 只把**别的类**的预测算作「类混淆」；同类的算「重复预测」，不是漏检
            conf_pred = [q for q in same if q["cls"] != gg[0]]
            dup_pred = [q for q in same if q["cls"] == gg[0]]
            ink_in, ink_ring, ink_ring_mean, sd = ink_stats(g, box)
            rmed, rcv, fill = shape_sig(g, box)
            dead = ink_in < INK_BOX_MIN or ink_ring < INK_RING_MED
            # 「框住空白 / 框在文字或导线上」：框内墨迹很局部（文字/导线）而框外环
            # 基本空白；或框内本身就几乎没墨。这类 FN 其实是**标注错**，不该算漏检。
            sr = stamp_rank(names[gg[0]], box)
            # 「框住空白」= 框内没有任何暗像素
            blank = ink_in < 1e-6
            # 「框在文字上」= 框内是实心墨、但框外环完全空白（图章扣在字/导线上）
            text_sig = (ink_ring_mean < 0.005 and ink_in >= 0.02 and sr is not None)
            # 「框在锯齿/电感笔画上，但标注写成 r」：
            # 实测 (8,20) 框的填充率中位=0.315（实心矩形体），(20,8) 框中位=0.024
            # （横放的锯齿/细笔画）。故阈值按框的长边取向分开取，避免误判竖放电阻体。
            bw_ = box[2] - box[0]
            bh_ = box[3] - box[1]
            thin_lim = 0.02 if bw_ > bh_ else 0.15
            zigzag = (names[gg[0]] == "r" and sr is not None
                      and 4 <= min(bw_, bh_) <= 12
                      and fill < thin_lim)
            no_count = bool(blank or text_sig or zigzag or dup_pred)
            if conf_pred:
                kind = "CLASSCONF"
            elif dup_pred:
                kind = "DUP"
            elif near:
                kind = "NEAROTHER"
            elif blank or text_sig or zigzag:
                kind = "DEADGT"
            else:
                kind = "TRUEMISS"
            if dup_pred:
                dup_pool[gg[0]].append(dict(box=box, n=len(dup_pred)))
            grp = "LEAK" if source_id(p.stem) in LEAK_IDS else "CLEAN"
            FN[gg[0]].append(dict(
                img=p, box=box, kind=kind, sr=sr, nc=no_count, grp=grp,
                ringmean=ink_ring_mean, rmed=rmed, rcv=rcv, fill=fill,
                blank=blank, text_sig=text_sig, zigzag=zigzag,
                dups=dup_pred,
                rcls=(names[conf_pred[0]["cls"]] if conf_pred else
                      (names[dup_pred[0]["cls"]] if dup_pred else
                       (names[near[0]["cls"]] if near else None))),
                rconf=(conf_pred[0]["score"] if conf_pred else
                       (dup_pred[0]["score"] if dup_pred else
                        (near[0]["score"] if near else None))),
                n_pred_same=len(same), n_pred_near=len(near),
                ink=ink_in, ink_ring=ink_ring, ink_ring_mean=ink_ring_mean, sd=sd,
                bw=box[2] - box[0], bh=box[3] - box[1]))

        # ---- FP
        for q in preds:
            if id(q) in matched:
                continue
            tot_fp[q["cls"]] += 1
            box = q["box"]
            # nearest GT of the SAME class, and nearest GT of ANY class
            cs = [(corner_dev(box, gg[1:]), names[gg[0]], gg[1:], gg[0])
                  for gg in gt]
            cs.sort(key=lambda t: t[0])
            same_gts = [t for t in cs if t[3] == q["cls"]]
            loose = [gg for gg in gt if overlap_loose(box, gg[1:])]
            loose = sorted(loose, key=lambda gg: corner_dev(box, gg[1:]))
            if loose:
                kind = "CLASSCONF"
                near_cls = names[loose[0][0]]
                near_dev = corner_dev(box, loose[0][1:])
            elif cs and cs[0][0] <= NEAR2:
                kind = "NEARMISS"
                near_cls = cs[0][1]
                near_dev = cs[0][0]
            else:
                kind = "REALFP"
                near_cls = cs[0][1] if cs else None
                near_dev = cs[0][0] if cs else None
            FP[q["cls"]].append(dict(
                img=p, box=box, score=q["score"], kind=kind, grp=grp,
                near_cls=near_cls, near_dev=near_dev,
                same_cls=near_cls if same_gts and same_gts[0][0] == near_dev else (
                    same_gts[0][1] if same_gts else None),
                same_dev=same_gts[0][0] if same_gts else None,
                same_cls_only=(names[same_gts[0][3]] if same_gts else None),
                same_dev_only=(same_gts[0][0] if same_gts else None),
                bw=box[2] - box[0], bh=box[3] - box[1]))
        per_img[p.stem] = dict(gt=len(gt), pred=len(preds),
                               fn=sum(1 for m, _ in enumerate(gt) if m not in used),
                               fp=sum(1 for q in preds if id(q) not in matched))
        if (i + 1) % 20 == 0:
            print("  ...%d/%d" % (i + 1, len(imgs)), flush=True)

    cache.clear()
    lines = []
    P = lines.append

    P("=" * 96)
    P("_gtfaith.py  root=%s  split=%s  conf=%g imgsz=%d 判据 corner<=%gpx"
      % (root, a.split, a.conf, a.imgsz, CORNER_TOL))
    P("images=%d   类别索引 %s" % (len(imgs), {k: v for k, v in sorted(ids.items(),
                                                                    key=lambda kv: kv[1])}))
    P("=" * 96)

    P("\n### 总账")
    P("%-12s%7s%7s%9s%8s%8s%9s" % ("cls", "GT", "FN", "FN率", "pred", "FP", "精确率"))
    for nm in want:
        cid = ids.get(nm)
        if cid is None:
            continue
        gg, f = tot_gt[cid], tot_fn[cid]
        pr, fp = tot_pred[cid], tot_fp[cid]
        P("%-12s%7d%7d%9s%8d%8d%9s" % (
            nm, gg, f, ("%.1f%%" % (100.0 * f / gg) if gg else "-"),
            pr, fp, ("%.3f" % ((pr - fp) / pr) if pr else "-")))

    P("\n### A. FN 重新归类")
    P("%-12s%7s%9s%9s%9s%6s%9s" % ("cls", "FN", "TRUEMISS", "CLASSCONF",
                                     "NEAROTHER", "DUP", "DEADGT"))
    for nm in want:
        cid = ids.get(nm)
        if cid is None:
            continue
        k = Counter(s["kind"] for s in FN[cid])
        if not tot_fn[cid]:
            continue
        P("%-12s%7d%9d%9d%9d%6d%9d" % (nm, tot_fn[cid], k["TRUEMISS"],
                                        k["CLASSCONF"], k["NEAROTHER"], k["DUP"],
                                        k["DEADGT"]))
    P("\n### A0. 「同框重复预测」假 FN（模型在同一 GT 位置输出了 >1 个同类框，贪心配对只领走一个）")
    P("%-12s%10s%10s%12s" % ("cls", "重复GT数", "其中FN", "占该类GT"))
    for nm in want:
        cid = ids.get(nm)
        if cid is None:
            continue
        ndup = len(dup_pool[cid])
        nfn = sum(1 for s in FN[cid] if s["kind"] == "DUP")
        if ndup or nfn:
            P("%-12s%10d%10d%12s" % (nm, ndup, nfn,
                                      "%.1f%%" % (100.0 * nfn / max(1, tot_gt[cid]))))
    P("\n### A0b. 合并 FN 假阳性账（不只 DUP，还含 DEADGT / 图章扣文字）")
    P("%-12s%7s%10s%10s%12s" % ("cls", "GT", "原始FN", "不应算FN", "修正召回率"))
    for nm in want:
        cid = ids.get(nm)
        if cid is None or not tot_gt[cid]:
            continue
        f = FN[cid]
        nc = sum(1 for s in f if s["nc"])
        P("%-12s%7d%10d%10d%12s" % (nm, tot_gt[cid], len(f), nc,
                                     "%.3f" % ((tot_gt[cid] - len(f) + nc) / tot_gt[cid])))

    P("\n### A2. CLASSCONF 汇总：该类的 FN 被模型预测成了哪些**别的**类")
    for nm in want:
        cid = ids.get(nm)
        if cid is None:
            continue
        cc = Counter(s["rcls"] for s in FN[cid] if s["kind"] == "CLASSCONF")
        if cc:
            P("  FN %-11s %d 个类混淆 -> %s" % (nm, sum(cc.values()),
                                                dict(cc.most_common(12))))
    P("\n### A2b. 类混淆矩阵（行=被漏的 GT 类，列=模型在该位置预测的类，含 conf）")
    for nm in want:
        cid = ids.get(nm)
        if cid is None:
            continue
        rows = [s for s in FN[cid] if s["kind"] == "CLASSCONF"]
        if not rows:
            continue
        cc = Counter((s["rcls"], round(s["rconf"], 2)) for s in rows)
        P("  %-11s <- %s" % (nm, ", ".join("%s@%.2f x%d" % (k[0], k[1], v)
                                           for k, v in cc.most_common(12))))
    P("\n### A2c. r → l 专项（owner 判断 1 的核心）")
    if rid0 := ids.get("r"):
        rows = [s for s in FN[rid0] if s["kind"] == "CLASSCONF"]
        P("  r 的 FN 共 %d，其中 CLASSCONF=%d；预测为 l 的=%d"
          % (len(FN[rid0]), len(rows),
             sum(1 for s in FN[rid0] if s["rcls"] == "l")))
        P("  r 被预测成的类分布: %s"
          % dict(Counter(s["rcls"] for s in rows).most_common()))
    if lid0 := ids.get("l"):
        P("  l 侧: GT=%d FN=%d pred=%d FP=%d；l 的 FP 错误来源=%s"
          % (tot_gt[lid0], tot_fn[lid0], tot_pred[lid0], tot_fp[lid0],
             dict(Counter(s["near_cls"] for s in FP[lid0]).most_common())))
        P("  l 的 GT 的框尺寸: %s"
          % dict(Counter((round(x[3] / 4) * 4, round(x[4] / 4) * 4)
                         for x in ALLGT[lid0]).most_common(8)))
        P("  r 的 GT 的框尺寸: %s"
          % dict(Counter((round(x[3] / 4) * 4, round(x[4] / 4) * 4)
                         for x in ALLGT[rid0]).most_common(8)))
    P("\n### A3. NEAROTHER（附近有别类预测但既不重合）明细")
    for nm in want:
        cid = ids.get(nm)
        if cid is None:
            continue
        cc = Counter(s["rcls"] for s in FN[cid] if s["kind"] == "NEAROTHER")
        if cc:
            P("  FN %-11s 附近预测类: %s" % (nm, dict(cc.most_common(10))))
    P("\n### A4. DEADGT（框内墨迹<2% 或 外环6px中位<0.02）明细")
    for nm in want:
        cid = ids.get(nm)
        if cid is None:
            continue
        d = [s for s in FN[cid] if s["kind"] == "DEADGT"]
        if d:
            P("  FN %-11s DEADGT=%d ; 前 8 个: %s" % (
                nm, len(d),
                ", ".join("%s[%.0fx%.0f ink=%.3f ringmed=%.2f ringmean=%.3f]"
                          % (s["img"].stem[:34], s["bw"], s["bh"], s["ink"],
                             s["ink_ring"], s["ink_ring_mean"])
                          for s in d[:8])))

    P("\n### A1b. 「不该算漏检」的 FN 明细判定（blank=框内零墨 / text=图章扣文字 / zigzag=r 图章扣锯齿笔画 / DUP=同框重复）")
    P("%-12s%6s%8s%6s%6s%7s%6s%8s" % ("cls", "FN", "不应算",
                                       "blank", "text", "zigzag", "DUP", "TRUEMISS"))
    for nm in want:
        cid = ids.get(nm)
        if cid is None or not tot_fn[cid]:
            continue
        f = FN[cid]
        P("%-12s%6d%8d%6d%6d%7d%6d%8d" % (
            nm, len(f), sum(1 for s in f if s["nc"]),
            sum(1 for s in f if s["blank"]),
            sum(1 for s in f if s["text_sig"]),
            sum(1 for s in f if s["zigzag"]),
            sum(1 for s in f if s["kind"] == "DUP"),
            sum(1 for s in f if s["kind"] == "TRUEMISS")))
    P("\n### A1d. r 的 FN 逐框判定（owner 判断 1 的直接证据）")
    if rid_n := ids.get("r"):
        P("  %-36s%12s%8s%7s%7s%7s%7s%9s" % ("img", "box", "kind", "ink",
                                               "fill", "游程中位", "游程CV", "预测"))
        for s in sorted(FN[rid_n], key=lambda s: s["img"].stem):
            P("  %-36s%12s%8s%7.3f%7.3f%7.1f%7.2f%9s"
              % (s["img"].stem[:36],
                 "%.0f,%.0f,%.0f,%.0f" % s["box"], s["kind"], s["ink"],
                 s["fill"], s["rmed"], s["rcv"], s["rcls"] or "-"))
    totfn_g = Counter((s["grp"], nm) for nm in want if ids.get(nm) is not None
                      for s in FN[ids[nm]])
    P("\n### A1c. 按组统计（LEAK = 源图 id 在 train 侧有同源 `synth_` 文件；CLEAN = 真 held-out）")
    for grp in ("LEAK", "CLEAN"):
        P("  -- %s 组" % grp)
        P("  %-12s%8s%8s%8s%10s%8s%8s" % ("cls", "GT", "FN", "FN率", "不应算",
                                           "CLASSC", "DEADGT"))
        for nm in want:
            cid = ids.get(nm)
            if cid is None:
                continue
            f = [s for s in FN[cid] if s["grp"] == grp]
            gc = GTCNT.get((grp, cid), 0)
            if not gc:
                continue
            k = Counter(s["kind"] for s in f)
            P("  %-12s%8d%8d%8s%10d%8d%8d" % (
                nm, gc, len(f), ("%.1f%%" % (100.0 * len(f) / gc)) if gc else "-",
                sum(1 for s in f if s["nc"]), k["CLASSCONF"], k["DEADGT"]))
        P("  %-12s%8s%8s%8s%8s" % ("类(pred)", "pred", "FP", "REALFP", "近GT<=20"))
        for nm in want:
            cid = ids.get(nm)
            if cid is None:
                continue
            q = [s for s in FP[cid] if s["grp"] == grp]
            if not q:
                continue
            P("  %-12s%8d%8d%8d%8d" % (
                nm, len(q), len(q),
                sum(1 for s in q if s["kind"] == "REALFP"),
                sum(1 for s in q if s["near_dev"] is not None
                    and s["near_dev"] <= CORNER_TOL)))


    P("\n### A5. 每个类的 GT 框「墨迹/形状签名」全量统计")
    P("  %-12s%6s%9s%9s%11s%11s%11s%11s" % ("cls", "GT", "框内中位", "框内<2%",
                                             "外环均值中位", "填充率中位",
                                             "游程数中位", "游程CV中位"))
    for nm in want:
        cid = ids.get(nm)
        if cid is None or not ALLGT[cid]:
            continue
        v = ALLGT[cid]
        P("  %-12s%6d%9.3f%9d%11.3f%11.3f%11.1f%11.2f" % (
            nm, len(v),
            stx.median(x[0] for x in v),
            sum(1 for x in v if x[0] < INK_BOX_MIN),
            stx.median(x[2] for x in v),
            stx.median(x[7] for x in v),
            stx.median(x[5] for x in v),
            stx.median(x[6] for x in v)))
    P("\n### A5b. r 的 GT 框形状签名分布（填充率 = 框内暗像素占比）")
    if rid_n := ids.get("r"):
        v = ALLGT[rid_n]
        P("  填充率分位: 最小=%.3f 10%%=%.3f 25%%=%.3f 中位=%.3f 75%%=%.3f 90%%=%.3f 最大=%.3f"
          % (min(x[7] for x in v),
             sorted(x[7] for x in v)[len(v) // 10],
             sorted(x[7] for x in v)[len(v) // 4],
             stx.median(x[7] for x in v),
             sorted(x[7] for x in v)[3 * len(v) // 4],
             sorted(x[7] for x in v)[9 * len(v) // 10],
             max(x[7] for x in v)))
        for t in (0.10, 0.20, 0.30):
            zig = [x for x in v if x[7] < t]
            P("  阈值 %.2f: 细笔画(锯齿)=%d  实心(矩形体)=%d   细笔画尺寸=%s"
              % (t, len(zig), len(v) - len(zig),
                 dict(Counter((round(x[3] / 2) * 2, round(x[4] / 2) * 2)
                              for x in zig).most_common(6))))
        P("  【交叉表】按尺寸 x 填充率阈值 0.20")
        for wh in ((8, 20), (20, 8)):
            sub = [x for x in v if (round(x[3]), round(x[4])) == wh]
            if not sub:
                continue
            z = sum(1 for x in sub if x[7] < 0.20)
            P("    %-8s n=%3d  填充率<0.20(细笔画/锯齿)=%3d  填充率>=0.20(实心块)=%3d  "
              "填充率中位=%.3f" % ("%dx%d" % wh, len(sub), z, len(sub) - z,
                                   stx.median(x[7] for x in sub)))
    P("\n### A5c. l 与 r 在【同样的 (8,20)/(20,8) 框尺寸】下的填充率对比（判别锯齿 vs 实心体）")
    for nm in ("l", "r"):
        cid = ids.get(nm)
        if cid is None or not ALLGT[cid]:
            continue
        v = [x for x in ALLGT[cid]
             if (round(x[3]), round(x[4])) in ((8, 20), (20, 8), (32, 8), (8, 32),
                                               (36, 8), (8, 36))]
        if v:
            P("  %-3s n=%d  填充率: 中位=%.3f 最小=%.3f 最大=%.3f  <0.20的=%d"
              % (nm, len(v), stx.median(x[7] for x in v),
                 min(x[7] for x in v), max(x[7] for x in v),
                 sum(1 for x in v if x[7] < 0.20)))

    P("\n### B. FP 重新归类")

    P("%-12s%8s%10s%10s%9s" % ("cls", "FP", "NEARMISS", "CLASSCONF", "REALFP"))
    for nm in want:
        cid = ids.get(nm)
        if cid is None or not tot_fp[cid]:
            continue
        k = Counter(s["kind"] for s in FP[cid])
        P("%-12s%8d%10d%10d%9d" % (nm, tot_fp[cid], k["NEARMISS"],
                                   k["CLASSCONF"], k["REALFP"]))

    P("\n### B2. NEARMISS 距离带宽（最近 GT 的 corner 差）")
    for nm in want:
        cid = ids.get(nm)
        if cid is None:
            continue
        d = [s["near_dev"] for s in FP[cid] if s["kind"] == "NEARMISS"]
        if d:
            P("  FP %-11s n=%d  20~40px=%d  40~60px=%d  中位=%.1f  最大=%.1f"
              % (nm, len(d), sum(1 for v in d if v <= NEAR1),
                 sum(1 for v in d if NEAR1 < v <= NEAR2), stx.median(d), max(d)))
            cc = Counter(s["near_cls"] for s in FP[cid] if s["kind"] == "NEARMISS")
            P("       最近 GT 的类: %s" % dict(cc.most_common(10)))
    P("\n### B2b. 每个 FP 的「同类最近 GT」距离（与 NEARMISS 的最近任意类 GT 对比）")
    for nm in want:
        cid = ids.get(nm)
        if cid is None or not tot_fp[cid]:
            continue
        d = [s["same_dev_only"] for s in FP[cid] if s["same_dev_only"] is not None]
        if d:
            P("  FP %-11s 有同类GT的=%d/%d  同类距<=20:%d  20~40:%d  40~60:%d  >60:%d"
              % (nm, len(d), tot_fp[cid],
                 sum(1 for v in d if v <= CORNER_TOL),
                 sum(1 for v in d if CORNER_TOL < v <= NEAR1),
                 sum(1 for v in d if NEAR1 < v <= NEAR2),
                 sum(1 for v in d if v > NEAR2)))

    P("\n### B3. REALFP 明细（60px 内没有任何 GT）")
    for nm in want:
        cid = ids.get(nm)
        if cid is None:
            continue
        d = sorted([s for s in FP[cid] if s["kind"] == "REALFP"],
                   key=lambda s: -s["score"])
        if d:
            P("  FP %-11s REALFP=%d  最高 conf: %s" % (
                nm, len(d),
                ", ".join("%s conf=%.2f %.0fx%.0f" % (s["img"].stem[:30], s["score"],
                                                      s["bw"], s["bh"]) for s in d[:10])))

    # ---------------- B1: r 漏检处预测成了什么 ----------------
    rid = ids.get("r")
    P("\n" + "=" * 96)
    P("### B1. r 的漏检位置上，模型预测了什么类")
    if rid is not None:
        fr = FN[rid]
        P("r: GT=%d  FN=%d  预测总数=%d  FP=%d" % (tot_gt[rid], len(fr),
                                                  tot_pred[rid], tot_fp[rid]))
        lis = ids.get("l")
        P("  漏检位置的预测构成: %s"
          % dict(Counter(("DUP(同类重复预测)" if s["kind"] == "DUP" else
                          (s["rcls"] or "(完全无预测)")) for s in fr).most_common(15)))
        P("  其中 DUP=%d（同框重复预测，**不应算漏检**）, CLASSCONF(别的类)=%d, "
          "TRUEMISS=%d, DEADGT=%d, NEAROTHER=%d"
          % (sum(1 for s in fr if s["kind"] == "DUP"),
             sum(1 for s in fr if s["kind"] == "CLASSCONF"),
             sum(1 for s in fr if s["kind"] == "TRUEMISS"),
             sum(1 for s in fr if s["kind"] == "DEADGT"),
             sum(1 for s in fr if s["kind"] == "NEAROTHER")))
        n_l = sum(1 for s in fr if s["rcls"] == "l")
        P("  其中预测为 l 的: %d / %d = %.1f%%（owner 判断 1 的检验点）"
          % (n_l, len(fr), 100.0 * n_l / max(1, len(fr))))
        P("  修正后 r 的真实漏检 = %d - %d(不应算) = %d，修正召回率 = %.3f"
          % (len(fr), sum(1 for s in fr if s["nc"]), len(fr) - sum(1 for s in fr if s["nc"]),
             (tot_gt[rid] - len(fr) + sum(1 for s in fr if s["nc"])) / tot_gt[rid]))
        # 按 conf 排序的明细
        det = sorted([s for s in fr if s["rcls"]], key=lambda s: -s["rconf"])
        P("  有预测的 %d 个，conf 降序前 25:" % len(det))
        for s in det[:25]:
            P("     conf=%.3f  预测=%-12s kind=%-9s 同框同类重复=%d  GT框=%.0fx%.0f  img=%s"
              % (s["rconf"], s["rcls"], s["kind"], len(s["dups"]),
                 s["bw"], s["bh"], s["img"].stem))
        # r 与 l 的全局判别：模型在 r 的 GT 上回什么、在 l 的 GT 上回什么
        P("  [对照] r 的 GT 被正确配对 vs 漏: 见总账；l 的 GT=%d FN=%d pred=%d FP=%d"
          % (tot_gt[lis] if lis is not None else -1,
             tot_fn[lis] if lis is not None else -1,
             tot_pred[lis] if lis is not None else -1,
             tot_fp[lis] if lis is not None else -1))
        # 几何：漏掉的 r 的尺寸
        if fr:
            bwv = sorted(s["bw"] for s in fr)
            bhv = sorted(s["bh"] for s in fr)
            P("  漏掉的 r 尺寸: bw 中位=%.0f [%.0f..%.0f]  bh 中位=%.0f [%.0f..%.0f]"
              % (stx.median(bwv), bwv[0], bwv[-1], stx.median(bhv), bhv[0], bhv[-1]))
            imc = Counter(s["img"].stem for s in fr)
            P("  分布在 %d 张图; top8: %s" % (len(imc),
                                             dict(imc.most_common(8))))
            inkv = sorted(s["ink"] for s in fr)
            P("  框内墨迹占比: 中位=%.3f 最小=%.3f 最大=%.3f  低于2%%的=%d"
              % (stx.median(inkv), inkv[0], inkv[-1],
                 sum(1 for v in inkv if v < INK_BOX_MIN)))

    # ---------------- B2: fp_r 的最近 GT ----------------
    P("\n" + "=" * 96)
    P("### B2. fp_r 每个预测的最近 GT 类与 corner 距离")
    if rid is not None and FP[rid]:
        fp_r = FP[rid]
        P("fp_r 共 %d 个" % len(fp_r))
        k = Counter(s["kind"] for s in fp_r)
        P("  归类: NEARMISS=%d CLASSCONF=%d REALFP=%d"
          % (k["NEARMISS"], k["CLASSCONF"], k["REALFP"]))
        cc = Counter(s["near_cls"] for s in fp_r)
        P("  最近 GT 的类分布: %s" % dict(cc.most_common(10)))
        dev = [s["near_dev"] for s in fp_r if s["near_dev"] is not None]
        if dev:
            P("  最近 GT corner 距离: <=20px(有重合)=%d  20~40=%d  40~60=%d  >60=%d"
              % (sum(1 for v in dev if v <= CORNER_TOL),
                 sum(1 for v in dev if CORNER_TOL < v <= NEAR1),
                 sum(1 for v in dev if NEAR1 < v <= NEAR2),
                 sum(1 for v in dev if v > NEAR2)))
        sv = [s["same_dev_only"] for s in fp_r if s["same_dev_only"] is not None]
        if sv:
            P("  最近【同类 r】GT 距离: <=20px=%d  20~40=%d  40~60=%d  >60=%d  无同类GT=%d"
              % (sum(1 for v in sv if v <= CORNER_TOL),
                 sum(1 for v in sv if CORNER_TOL < v <= NEAR1),
                 sum(1 for v in sv if NEAR1 < v <= NEAR2),
                 sum(1 for v in sv if v > NEAR2), len(fp_r) - len(sv)))
        P("  明细（conf 降序，前 30）: img | conf | 预测框 | 最近GT类 | corner | 最近同类r | corner")
        for s in sorted(fp_r, key=lambda s: -s["score"])[:30]:
            P("     %-36s %.3f  %.0fx%.0f  近GT=%-12s %6.1f  近r=%-5s %s"
              % (s["img"].stem[:36], s["score"], s["bw"], s["bh"],
                 s["near_cls"], s["near_dev"] if s["near_dev"] is not None else -1,
                 s["same_cls_only"],
                 ("%.1f" % s["same_dev_only"]) if s["same_dev_only"] is not None else "-"))

    # ---------------- B3: fn_net_input ----------------
    nid = ids.get("net_input")
    P("\n" + "=" * 96)
    P("### B3. fn_net_input 的 GT 框")
    if nid is not None:
        fnn = FN[nid]
        P("net_input: GT=%d  FN=%d  pred=%d  FP=%d"
          % (tot_gt[nid], len(fnn), tot_pred[nid], tot_fp[nid]))
        if fnn:
            bwv = sorted(s["bw"] for s in fnn)
            bhv = sorted(s["bh"] for s in fnn)
            P("  尺寸: bw 中位=%.0f [%.0f..%.0f]   bh 中位=%.0f [%.0f..%.0f]"
              % (stx.median(bwv), bwv[0], bwv[-1], stx.median(bhv), bhv[0], bhv[-1]))
            sc = Counter((round(s["bw"] / 2) * 2, round(s["bh"] / 2) * 2) for s in fnn)
            P("  尺寸直方(2px归并, 前12): %s" % dict(sc.most_common(12)))
            inkv = sorted(s["ink"] for s in fnn)
            rngv = sorted(s["ink_ring"] for s in fnn)
            P("  框内墨迹占比: 中位=%.3f 最小=%.3f 最大=%.3f  <2%%的=%d(%s)"
              % (stx.median(inkv), inkv[0], inkv[-1],
                 sum(1 for v in inkv if v < INK_BOX_MIN),
                 "%.1f%%" % (100.0 * sum(1 for v in inkv if v < INK_BOX_MIN) / len(inkv))))
            rngv = sorted(s["ink_ring_mean"] for s in fnn)
            rm = sorted(s["ink_ring"] for s in fnn)
            P("  外环6px墨迹: 均值中位=%.3f 最小=%.3f 最大=%.3f ; 中位判据=0 的个数=%d/%d"
              % (stx.median(rngv), rngv[0], rngv[-1],
                 sum(1 for v in rm if v < 1e-9), len(rm)))
            P("  归类: %s" % dict(Counter(s["kind"] for s in fnn)))
            imc = Counter(s["img"].stem for s in fnn)
            P("  所在图 (%d 张): %s" % (len(imc),
                                        ", ".join("%s x%d" % (k2, v) for k2, v in
                                                  imc.most_common())))
            P("  逐框明细:")
            for s in sorted(fnn, key=lambda s: (s["img"].stem, s["box"][1])):
                P("     %-36s box=(%.0f,%.0f,%.0f,%.0f) %.0fx%.0f  ink=%.3f ring=%.3f  "
                  "预测=%s%s"
                  % (s["img"].stem[:36], s["box"][0], s["box"][1], s["box"][2],
                     s["box"][3], s["bw"], s["bh"], s["ink"], s["ink_ring"],
                     s["rcls"] or "无",
                     (" conf=%.3f" % s["rconf"]) if s["rconf"] else ""))

    P("\n### 每图 FN/FP 计数（FN+FP>0）")
    for k2, v in sorted(per_img.items()):
        if v["fn"] or v["fp"]:
            P("  %-38s gt=%-4d pred=%-4d fn=%-3d fp=%d"
              % (k2[:38], v["gt"], v["pred"], v["fn"], v["fp"]))

    if a.dump:
        json.dump(dict(
            fn={names[c]: [dict(img=s["img"].name,
                                box=[round(x, 2) for x in s["box"]],
                                kind=s["kind"], rcls=s["rcls"],
                                rconf=s["rconf"], ink=round(s["ink"], 4),
                                ring=round(s["ink_ring"], 4)) for s in FN[c]]
                for c in FN},
            fp={names[c]: [dict(img=s["img"].name,
                                box=[round(x, 2) for x in s["box"]],
                                score=s["score"], kind=s["kind"],
                                near_cls=s["near_cls"],
                                near_dev=(round(s["near_dev"], 2)
                                          if s["near_dev"] is not None else None),
                                same_dev=(round(s["same_dev_only"], 2)
                                          if s["same_dev_only"] is not None else None))
                for s in FP[c]] for c in FP},
            per_img=per_img,
        ), open(a.dump, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        P("\n[dump] %s" % a.dump)

    txt = "\n".join(lines)
    print(txt)
    Path(a.out).write_text(txt, encoding="utf-8")
    print("\n[written] %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
