# -*- coding: utf-8 -*-
"""在真实 val 上算「真类 → 预测类」的混淆统计（IoU 贪心匹配）。

用法:
    python confusion.py <weights.pt> [--data ...] [--split val] [--imgsz 1024] [--conf 0.001] [--iou 0.5]

输出:
    - 每个真类的 召回/被误判成谁/漏检
    - 每个预测类的 精度/它吃掉了哪些真类（虚警来源）
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DATA_ROOT = r"Y:\workspace\EDA\训练集_v2"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(path: Path, w: int, h: int):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append((c, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.7)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    import numpy as np
    from PIL import Image
    from ultralytics import YOLO

    root = Path(args.root)
    classes = (root / "classes.txt").read_text(encoding="utf-8").splitlines()
    # 注意: --cache disk 会在图片旁边留下同名 .npy（ultralytics 的磁盘缓存），必须按扩展名过滤
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    imgs = sorted(p for p in (root / "images" / args.split).glob("*") if p.suffix.lower() in exts)
    model = YOLO(args.weights)

    pair = Counter()          # (true_id, pred_id)
    missed = Counter()        # true_id -> n
    fp = Counter()            # pred_id -> n  (background false positives)
    inst = Counter()          # true_id -> n
    kind_inst = Counter()     # (kind, true_id) -> n
    kind_miss = Counter()     # (kind, true_id) -> n
    BKG = -1

    def imkind(name: str) -> str:
        return "full" if "__full" in name else "tile"

    for i, ip in enumerate(imgs):
        im = Image.open(ip)
        w, h = im.size
        gts = load_gt(root / "labels" / args.split / (ip.stem + ".txt"), w, h)
        kind = imkind(ip.name)
        for c, _ in gts:
            inst[c] += 1
            kind_inst[(kind, c)] += 1
        r = model.predict(str(ip), imgsz=args.imgsz, conf=args.conf, iou=args.nms,
                          verbose=False, device="0")[0]
        preds = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            preds = [(int(cl[k]), xyxy[k].tolist(), float(cf[k])) for k in range(len(cl))]
        # 贪心：按 IoU 从高到低配对
        cands = []
        for gi, (gc, gb) in enumerate(gts):
            for pi, (pc, pb, _) in enumerate(preds):
                v = iou(gb, pb)
                if v >= args.iou:
                    cands.append((v, gi, pi))
        cands.sort(reverse=True)
        used_g, used_p = set(), set()
        for v, gi, pi in cands:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            pair[(gts[gi][0], preds[pi][0])] += 1
        for gi, (gc, _) in enumerate(gts):
            if gi not in used_g:
                missed[gc] += 1
                kind_miss[(kind, gc)] += 1
                pair[(gc, BKG)] += 1
        for pi, (pc, _, _) in enumerate(preds):
            if pi not in used_p:
                fp[pc] += 1
                pair[(BKG, pc)] += 1
        if (i + 1) % 20 == 0:
            print("  ...%d/%d" % (i + 1, len(imgs)), flush=True)

    def nm(c):
        return "background" if c == BKG else (classes[c] if 0 <= c < len(classes) else str(c))

    tot = sum(inst.values())
    print("\nweights:", args.weights, " split:", args.split, " IoU>=", args.iou)
    print("GT instances:", tot, " missed:", sum(missed.values()),
          "(%.1f%%)" % (100.0 * sum(missed.values()) / max(1, tot)))

    print("\n=== full sheet vs tile：弱类漏检到底出在哪类图上 ===")
    print("%-18s %18s %18s" % ("true", "full (inst/missed/R)", "tile (inst/missed/R)"))
    rows = []
    for c in inst:
        fi, fm = kind_inst[("full", c)], kind_miss[("full", c)]
        ti, tm = kind_inst[("tile", c)], kind_miss[("tile", c)]
        rows.append((missed[c], c, fi, fm, ti, tm))
    for _, c, fi, fm, ti, tm in sorted(rows, reverse=True)[:14]:
        fR = "%.3f" % (1.0 - fm / fi) if fi else "  -  "
        tR = "%.3f" % (1.0 - tm / ti) if ti else "  -  "
        print("%-18s %6d/%-6d %-8s %6d/%-6d %-8s" % (nm(c), fi, fm, fR, ti, tm, tR))
    for kind in ("full", "tile"):
        I = sum(v for (k, _), v in kind_inst.items() if k == kind)
        M = sum(v for (k, _), v in kind_miss.items() if k == kind)
        print("  %-5s TOTAL inst=%d missed=%d recall=%.3f"
              % (kind, I, M, 1.0 - M / max(1, I)))

    print("\n=== 每个真类：漏检 / 被误判成谁（按漏检数排序） ===")
    print("%-18s %6s %7s %7s   %s" % ("true", "inst", "missed", "recall", "top confusions"))
    for c in sorted(inst, key=lambda c: -missed[c]):
        conf = sorted(((pair[(c, p)], p) for p in set(p for (t, p) in pair if t == c and p != c)),
                      reverse=True)[:4]
        s = ", ".join("%s x%d" % (nm(p), n) for n, p in conf if n) or "-"
        print("%-18s %6d %7d %7.3f   %s"
              % (nm(c), inst[c], missed[c], 1.0 - missed[c] / max(1, inst[c]), s))

    print("\n=== 每个预测类的虚警来源（background 行 = 凭空多出来的框） ===")
    print("%-18s %7s   %s" % ("pred", "fp", "top true classes it stole"))
    for p in sorted(fp, key=lambda p: -fp[p]):
        src = sorted(((pair[(t, p)], t) for t in set(t for (t, _) in pair if t != BKG)),
                     reverse=True)[:4]
        s = ", ".join("%s x%d" % (nm(t), n) for n, t in src if n) or "-"
        print("%-18s %7d   %s" % (nm(p), fp[p], s))

    print("\n=== 最大的前 %d 个混淆对（不含对角） ===" % args.top)
    off = [(n, t, p) for (t, p), n in pair.items() if t != p]
    for n, t, p in sorted(off, reverse=True)[:args.top]:
        print("  %-16s -> %-16s %5d" % (nm(t), nm(p), n))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
