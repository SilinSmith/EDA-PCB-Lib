# -*- coding: utf-8 -*-
"""训练集标签体检：回答用户 m07249 的七条质疑。

    & 'K:\python\311\python.exe' Y:\workspace\EDA\work\tools\diag\_label_audit.py

`work\\tools\\eval\\fn_fp.py` 判 FN / FP 的口径是「**同类** 且 IoU>=0.5」。这条口径有
一个必然副作用：**一个位置对、大小/位置略偏的框会同时记一个 FN 和一个 FP**。用户看到
的「误检其实都是对的，就是歪了」就是这么来的。

本脚本对 val 集逐张跑 YOLO，然后分四件事：
A. 训练集构造的**泄漏**（同一张图 `__full` 与 `__t00_xx` 切片是否同时进了 train 和 val）
B. **FP 归因**：每个虚警去找「同类最近的 GT」和「任意类最近的 GT」，看它到底是
   「框歪」(same-IoU 高) 还是「类错」(any-IoU 高、类不同) 还是「真虚警」
C. **FN 归因**：每个漏检去找「任意类最近的预测」，看它到底是「类错」还是「真漏」
D. **类间混淆矩阵**（按 IoU>=0.5 的错配统计）
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOTDATA = Path(r"Y:\workspace\EDA\trainset_v2")
DEF_W = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\runs\detect"
             r"\pcb43_v2_from_best\weights\best.pt")

# `200_train_cases-0002__t00_01` / `10_GTcase-0037-Datasheet__full` -> 原图 key
TILE_RE = re.compile(r"^(?P<key>.+?)__(?:full|t\d{2}_\d{2})$")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(txt: Path, w: int, h: int):
    out = []
    if not txt.exists():
        return out
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def key_of(stem: str) -> str:
    m = TILE_RE.match(stem)
    return m.group("key") if m else stem


def leakage(root: Path) -> None:
    print("=" * 78)
    print("A. 训练集构造的泄漏 / 切片重复")
    tr = {p.stem for p in (root / "images" / "train").glob("*.png")}
    va = {p.stem for p in (root / "images" / "val").glob("*.png")}
    te = {p.stem for p in (root / "images" / "test").glob("*.png")}
    print("  train %d 张 / val %d 张 / test %d 张" % (len(tr), len(va), len(te)))
    ktr = Counter(key_of(s) for s in tr)
    kva = Counter(key_of(s) for s in va)
    kte = Counter(key_of(s) for s in te)
    both = sorted(set(ktr) & set(kva))
    print("  **同一个原图同时出现在 train 和 val 的：%d 个** " % len(both))
    n_img = sum(ktr[k] for k in both) + sum(kva[k] for k in both)
    print("     涉及 %d 张切片图（train %d + val %d），占 val 的 %.1f%%"
          % (n_img, sum(ktr[k] for k in both), sum(kva[k] for k in both),
             100.0 * sum(kva[k] for k in both) / max(1, len(va))))
    for k in both[:12]:
        print("     %-46s train %d 片 / val %d 片" % (k[:46], ktr[k], kva[k]))
    b2 = sorted(set(ktr) & set(kte))
    print("  同一个原图同时出现在 train 和 test 的：%d 个 %s"
          % (len(b2), b2[:4]))
    print("  val 里每个原图平均切成 %.2f 片；train 里平均 %.2f 片"
          % (sum(kva.values()) / max(1, len(kva)), sum(ktr.values()) / max(1, len(ktr))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOTDATA))
    ap.add_argument("--weights", default=str(DEF_W))
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.root)
    leakage(root)

    from ultralytics import YOLO
    model = YOLO(a.weights)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    names = {int(k): str(v) for k, v in names.items()}
    imgs = sorted((root / "images" / a.split).glob("*.png"))
    if a.limit:
        imgs = imgs[:a.limit]
    lbldir = root / "labels" / a.split

    fp_cat = Counter()
    fp_by_cls = defaultdict(Counter)
    fn_cat = Counter()
    fn_by_cls = defaultdict(Counter)
    conf_mat = Counter()          # (GT 类, 预测类) —— 位置对上了但类不同
    iou_same_fp = []              # 虚警与「同类最近 GT」的 IoU
    iou_any_fp = []
    iou_any_fn = []
    dup_pair = 0                  # 两个预测框 IoU>=0.7 且同类（自家 NMS 没滤掉的重复）
    dup_n = 0
    size_by_cls = defaultdict(list)

    for i, p in enumerate(imgs):
        from PIL import Image
        with Image.open(str(p)) as im:
            W, H = im.size
        gt = load_gt(lbldir / (p.stem + ".txt"), W, H)
        r = model.predict(str(p), conf=a.conf, iou=0.7, imgsz=a.imgsz,
                          verbose=False, device=0)[0]
        preds = []
        if r.boxes is not None and len(r.boxes):
            for b, c, s in zip(r.boxes.xyxy.cpu().numpy(),
                               r.boxes.cls.cpu().numpy().astype(int),
                               r.boxes.conf.cpu().numpy()):
                preds.append({"box": tuple(float(v) for v in b), "cls": int(c),
                              "score": float(s)})
        # 同类贪心一对一
        used = set()
        for q in sorted(preds, key=lambda x: -x["score"]):
            best, bi = 0.0, -1
            for m, g in enumerate(gt):
                if m in used or g[0] != q["cls"]:
                    continue
                v = iou(q["box"], g[1:])
                if v > best:
                    best, bi = v, m
            if best >= 0.5:
                used.add(bi)
                q["hit"] = True
        # --- GT 侧：漏检归因 ---
        for m, g in enumerate(gt):
            if m in used:
                continue
            gb = g[1:]
            size_by_cls[names[g[0]]].append((gb[2] - gb[0]) * (gb[3] - gb[1]))
            best_any, bc = 0.0, -1
            for q in preds:
                v = iou(q["box"], gb)
                if v > best_any:
                    best_any, bc = v, q["cls"]
            iou_any_fn.append(best_any)
            if best_any >= 0.5:
                cat = "类错(位置对上了)"
                conf_mat[(names[g[0]], names[bc])] += 1
            elif best_any >= 0.3:
                cat = "框歪"
            else:
                cat = "真漏检"
            fn_cat[cat] += 1
            fn_by_cls[names[g[0]]][cat] += 1
        # --- 预测侧：虚警归因 ---
        for q in preds:
            if q.get("hit"):
                continue
            s_same, s_any, ac = 0.0, 0.0, -1
            for g in gt:
                v = iou(q["box"], g[1:])
                if v > s_any:
                    s_any, ac = v, g[0]
                if g[0] == q["cls"] and v > s_same:
                    s_same = v
            iou_same_fp.append(s_same)
            iou_any_fp.append(s_any)
            if s_same >= 0.3:
                cat = "框歪(同类就在旁边)"
            elif s_any >= 0.5:
                cat = "类错(其实是 %s)" % names[ac]
                conf_mat[(names[ac], names[q["cls"]])] += 1
            elif s_any >= 0.3:
                cat = "框歪/跨类"
            else:
                cat = "真虚警"
            fp_cat[cat] += 1
            fp_by_cls[names[q["cls"]]][cat] += 1
        # 自家重复框
        for x in range(len(preds)):
            for y in range(x + 1, len(preds)):
                if preds[x]["cls"] == preds[y]["cls"] and \
                        iou(preds[x]["box"], preds[y]["box"]) >= 0.7:
                    dup_pair += 1
        dup_n += len(preds)
        if (i + 1) % 20 == 0:
            print("  ...%d/%d" % (i + 1, len(imgs)), flush=True)

    def pct(d):
        t = sum(d.values()) or 1
        return "  ".join("%s %d(%.0f%%)" % (k, v, 100.0 * v / t)
                         for k, v in d.most_common())

    print()
    print("=" * 78)
    print("B. 虚警 FP 归因（共 %d 个）" % sum(fp_cat.values()))
    print("  " + pct(fp_cat))
    same_hi = sum(1 for v in iou_same_fp if v >= 0.3)
    print("  与「同类最近 GT」IoU>=0.3 的：%d 个 (%.1f%%) —— 这些位置本来是对的"
          % (same_hi, 100.0 * same_hi / max(1, len(iou_same_fp))))
    print()
    print("C. 漏检 FN 归因（共 %d 个）" % sum(fn_cat.values()))
    print("  " + pct(fn_cat))
    any_hi = sum(1 for v in iou_any_fn if v >= 0.5)
    print("  与「任意类最近预测」IoU>=0.5 的：%d 个 (%.1f%%) —— 这些其实检出来了，是类判错"
          % (any_hi, 100.0 * any_hi / max(1, len(iou_any_fn))))
    print()
    print("D. 预测框自身重复（同类 IoU>=0.7，ultralytics 的 iou=0.7 没滤掉）")
    print("  共 %d 对 / 预测框 %d 个" % (dup_pair, dup_n))
    print()
    print("E. 类间混淆 top20 —— (GT 类 -> 预测类), 位置对上了只是类不同")
    for (g, q), n in conf_mat.most_common(20):
        print("     %-16s -> %-16s %4d" % (g, q, n))
    print()
    print("F. 逐类 FP / FN 结构（只列被质疑的类）")
    for nm in ("box", "c", "r", "l", "net_input", "testpoint", "gnd", "v"):
        fpb = fp_by_cls.get(nm) or Counter()
        fnb = fn_by_cls.get(nm) or Counter()
        if not fpb and not fnb:
            continue
        sz = size_by_cls.get(nm) or []
        sz = sorted(sz)[len(sz) // 2] if sz else 0
        print("  --- %s（漏检框面积中位数 %.0f px²）" % (nm, sz))
        print("      FP: " + (pct(fpb) if fpb else "无"))
        print("      FN: " + (pct(fnb) if fnb else "无"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
