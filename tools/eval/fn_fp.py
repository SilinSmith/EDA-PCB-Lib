"""把漏检(FN)与误检(FP)样本抠出来，做成可肉眼审阅的接触表。

判据：conf 很低（默认 0.05）时模型仍然漏掉的 GT = 真漏检；
      conf 很高却对不上任何 GT 的预测 = 真虚警。

每格 = 一个以目标为中心、外扩 2.2 倍的上下文裁剪（放大 2 倍），
红线 = 漏掉的 GT 框，蓝线 = 对不上的预测框。

用法:
  python fn_fp.py [--weights ...] [--conf 0.05] [--n 8] [--classes gnd,v,r,c,testpoint,net_input]
"""
import sys, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:\workspace\EDA\trainset_v2")
DEF_W = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\runs\detect\pcb43_v2_from_best\weights\best.pt")
OUT = Path(r"Y:\workspace\EDA\work\preview")
OUT.mkdir(parents=True, exist_ok=True)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ---- 判据 ------------------------------------------------------------------
# 官方判据是「四角差 <= 20px」（schematic_parser/stamp.py:12 的 bbox_tol=20），
# 不是 IoU。在 r 的 8x20 图章框上，IoU>=0.5 只容忍中心偏 ±1.3px（x 方向偏
# 2.7px 就掉到 0.5 以下），会把大量「位置对、框偏」的检出同时记成一笔 FN 和
# 一笔 FP。默认必须用官方口径，否则我们一直在追假目标。
CRIT = "corner"          # corner | iou
CORNER_TOL = 20.0
IOU_MIN = 0.5


def corner_dev(a, b) -> float:
    """四角最大绝对差（官方 bbox_ok 用的量）。越小越好。"""
    return max(abs(a[i] - b[i]) for i in range(4))


def hit(a, b) -> bool:
    """按当前判据判断预测 a 与 GT b 是否算同一颗。"""
    if CRIT == "iou":
        return iou(a, b) >= IOU_MIN
    return corner_dev(a, b) <= CORNER_TOL


def load_gt(lbl: Path, w: int, h: int):
    out = []
    if not lbl.exists():
        return out
    for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0])); cx, cy, bw, bh = (float(v) for v in p[1:5])
        x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
        out.append((c, x1, y1, x2, y2))
    return out


def render(samples, title, path, cls_name, kind):
    """samples: list of dict(img=Path, box=(x1,y1,x2,y2), w,h, score)"""
    if not samples:
        return None
    CW = CH = 150; PAD = 6; COLS = 4
    rows = (len(samples) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * (CW + PAD) + PAD, rows * (CH + 30 + PAD) + 30 + PAD),
                      (14, 17, 23))
    d = ImageDraw.Draw(sheet)
    d.text((PAD + 2, 8), title, fill=(230, 238, 246))
    for i, s in enumerate(samples):
        cx = PAD + (i % COLS) * (CW + PAD)
        cy = 30 + PAD + (i // COLS) * (CH + 30 + PAD)
        im = Image.open(str(s["img"])).convert("L")
        W, H = im.size
        x1, y1, x2, y2 = s["box"]
        bw, bh = x2 - x1, y2 - y1
        k = 2.2
        mx, my = max(bw, 12) * k / 2, max(bh, 12) * k / 2
        ccx, ccy = (x1 + x2) / 2, (y1 + y2) / 2
        nx1 = max(0, int(ccx - mx)); ny1 = max(0, int(ccy - my))
        nx2 = min(W, int(ccx + mx)); ny2 = min(H, int(ccy + my))
        if nx2 - nx1 < 4 or ny2 - ny1 < 4:
            continue
        crop = im.crop((nx1, ny1, nx2, ny2)).convert("RGB")
        sx, sy = CW / crop.width, CH / crop.height
        crop = crop.resize((CW, CH), Image.LANCZOS)
        sheet.paste(crop, (cx, cy))
        # 画框（换算到裁剪坐标系）
        col = (250, 70, 70) if kind == "FN" else (80, 160, 255)
        rx1 = (x1 - nx1) * sx; ry1 = (y1 - ny1) * sy
        rx2 = (x2 - nx1) * sx; ry2 = (y2 - ny1) * sy
        d.rectangle([cx + rx1, cy + ry1, cx + rx2, cy + ry2], outline=col, width=2)
        d.rectangle([cx, cy, cx + CW, cy + CH], outline=(60, 70, 86))
        d.text((cx + 2, cy + CH + 3), f"{cls_name} {int(bw)}x{int(bh)}px", fill=(215, 225, 235))
        tail = f"{s['img'].stem[:20]}"
        if s.get("score") is not None:
            tail = f"conf {s['score']:.2f} | " + tail
        d.text((cx + 2, cy + CH + 15), tail, fill=(125, 137, 152))
    sheet.save(path)
    return sheet.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEF_W))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--classes", default="gnd,v,r,c,testpoint,net_input,box")
    ap.add_argument("--crit", choices=("corner", "iou"), default="corner",
                    help="配对判据；corner=官方四角差<=20px（默认），iou=IoU>=0.5（旧的、过严）")
    ap.add_argument("--corner-tol", type=float, default=20.0, dest="corner_tol")
    a = ap.parse_args()

    global CRIT, CORNER_TOL
    CRIT, CORNER_TOL = a.crit, a.corner_tol

    from ultralytics import YOLO
    model = YOLO(a.weights)
    names = model.names

    imgdir = Path(a.root) / "images" / a.split
    lbldir = Path(a.root) / "labels" / a.split
    imgs = sorted(p for p in imgdir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))

    FN = defaultdict(list); FP = defaultdict(list)
    tot_gt = defaultdict(int); tot_fn = defaultdict(int)
    tot_pred = defaultdict(int); tot_fp = defaultdict(int)

    for i, p in enumerate(imgs):
        with Image.open(str(p)) as im:
            W, H = im.size
        gt = load_gt(lbldir / (p.stem + ".txt"), W, H)
        for c, *_ in gt:
            tot_gt[c] += 1
        r = model.predict(str(p), conf=a.conf, iou=0.7, imgsz=a.imgsz,
                          verbose=False, device=0)[0]
        preds = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            for b, c, s in zip(xyxy, cl, cf):
                preds.append({"box": tuple(float(v) for v in b), "cls": int(c), "score": float(s)})
        for q in preds:
            tot_pred[q["cls"]] += 1

        # 贪心一对一匹配（按 conf 降序）；配对判据见文件顶部 CRIT
        used = set()
        pairs = []
        for j, q in enumerate(sorted(preds, key=lambda x: -x["score"])):
            best, bi = None, -1
            for m, g in enumerate(gt):
                if m in used or g[0] != q["cls"]:
                    continue
                if not hit(q["box"], g[1:]):
                    continue
                # 同判据下取最优（corner 越小越好 / iou 越大越好）
                v = iou(q["box"], g[1:]) if CRIT == "iou" else -corner_dev(q["box"], g[1:])
                if best is None or v > best:
                    best, bi = v, m
            if bi >= 0:
                used.add(bi); pairs.append((bi, q))
        for m, g in enumerate(gt):
            if m in used:
                continue
            tot_fn[g[0]] += 1
            FN[g[0]].append(dict(img=p, box=g[1:]))
        matched = {id(q) for _, q in pairs}
        for q in preds:
            if id(q) in matched:
                continue
            tot_fp[q["cls"]] += 1
            FP[q["cls"]].append(dict(img=p, box=q["box"], score=q["score"]))
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(imgs)}", flush=True)

    print(f"\nimages {len(imgs)}  conf {a.conf}  imgsz {a.imgsz}  "
          f"判据 {CRIT}" + (f"(四角差<={CORNER_TOL:g}px)" if CRIT == "corner" else f"(IoU>={IOU_MIN})"))
    print(f"\n{'cls':<14}{'GT':>7}{'FN':>7}{'FN率':>9}{'pred':>8}{'FP':>8}{'精确率':>9}")
    want = [c.strip() for c in a.classes.split(",") if c.strip()]
    ids = {}
    for cid, nm in (names.items() if isinstance(names, dict) else enumerate(names)):
        if nm in want:
            ids[nm] = int(cid)
    for nm in want:
        cid = ids.get(nm)
        if cid is None:
            continue
        g, f = tot_gt[cid], tot_fn[cid]
        pr, fp = tot_pred[cid], tot_fp[cid]
        prec = (pr - fp) / pr if pr else 0.0
        print(f"{nm:<14}{g:>7}{f:>7}{(f/g if g else 0):>9.1%}{pr:>8}{fp:>8}{prec:>9.3f}")

    print()
    # FN 到底是「整类都难」还是「某几张图难」？
    from collections import Counter
    for nm in want:
        cid = ids.get(nm)
        if cid is None:
            continue
        cnt = Counter(s["img"].stem for s in FN[cid])
        tot = sum(cnt.values())
        if not tot:
            continue
        top = cnt.most_common(5)
        share = sum(v for _, v in top) / tot
        t = ", ".join(f"{k[:26]}:{v}" for k, v in top)
        print(f"  FN集中度 {nm:<11} 共{tot:>4} 个，分布在 {len(cnt):>3} 张图；"
              f"top5 图占 {share:>4.0%}  ->  {t}")
    print()
    for nm in want:
        cid = ids.get(nm)
        if cid is None:
            continue
        # FN：优先展示「大目标也漏」——按 GT 框面积降序
        f = sorted(FN[cid], key=lambda s: -((s["box"][2] - s["box"][0]) * (s["box"][3] - s["box"][1])))
        f = f[:a.n]
        if f:
            sz = render(f, f"漏检 FN — {nm}  (conf {a.conf} 下仍未检出；红框=GT)",
                        OUT / f"fn_{nm}.png", nm, "FN")
            print(f"  fn_{nm}.png  n={len(f)}  {sz}")
        # FP：优先展示「高置信虚警」
        q = sorted(FP[cid], key=lambda s: -s.get("score", 0))
        q = q[:a.n]
        if q:
            sz = render(q, f"误检 FP — {nm}  (对不上任何 GT；蓝框=预测，数字=置信度)",
                        OUT / f"fp_{nm}.png", nm, "FP")
            print(f"  fp_{nm}.png  n={len(q)}  {sz}")


if __name__ == "__main__":
    main()
