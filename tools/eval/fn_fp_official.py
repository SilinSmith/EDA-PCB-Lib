"""按【官方口径】重做 FN/FP 接触表。

和 `fn_fp.py` 的区别（`fn_fp.py` 用的是 YOLO mAP 口径，会误导）：
  判据   : max(|pred[i]-gt[i]|) <= 20px（官方 problem_statement.txt:175-184），不是 IoU>=0.5
  框     : 预测框先贴成规范图章（schematic_parser/stamp.py），GT 本身就是图章

每格画两个框：GT 用红色，预测用蓝色（FN 格画与 GT 同类、最近的预测；FP 格画最近的同类 GT），
这样一眼能看出「没找到」还是「找到了但差一点」。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser.stamp import snap_box, STAMP_SIDE  # noqa: E402

OUT = Path(r"Y:\workspace\EDA\work\preview\official")
OUT.mkdir(parents=True, exist_ok=True)
TOL = 20.0


def load_gt(lbl: Path, w: int, h: int):
    out = []
    if not lbl.exists():
        return out
    for line in lbl.read_text().splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        c = int(t[0]); cx, cy, bw, bh = map(float, t[1:5])
        out.append((c, np.array([(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h])))
    return out


def corner_diff(a, b):
    return max(abs(a[i] - b[i]) for i in range(4))


def match_official(preds, gts):
    """preds/gts: list[(cls, box)]. 官方口径一对一贪心匹配（同类 + 角点差 <= TOL）。"""
    cand = []
    for i, p in enumerate(preds):
        pc, pb = p[0], p[1]
        for j, (gc, gb) in enumerate(gts):
            if pc != gc:
                continue
            d = corner_diff(pb, gb)
            if d <= TOL:
                cand.append((d, i, j))
    cand.sort(key=lambda t: t[0])
    up, ug, pairs = set(), set(), []
    for d, i, j in cand:
        if i in up or j in ug:
            continue
        up.add(i); ug.add(j); pairs.append((i, j))
    return pairs


def nearest(preds, cls, box):
    best, bd = None, 1e18
    for p in preds:
        if p[0] != cls:
            continue
        d = corner_diff(p[1], box)
        if d < bd:
            bd, best = d, p
    return best, bd


def render(pairs, title, path, cls_name, kind):
    """pairs: list of dict(img, gt_box, other_box, other_col, sub)"""
    if not pairs:
        return None
    CW = CH = 150; PAD = 6; COLS = 4
    rows = (len(pairs) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS*(CW+PAD)+PAD, rows*(CH+32+PAD)+32+PAD), (14, 17, 23))
    d = ImageDraw.Draw(sheet)
    d.text((PAD+2, 9), title, fill=(230, 238, 246))
    for i, s in enumerate(pairs):
        cx = PAD + (i % COLS) * (CW + PAD)
        cy = 32 + PAD + (i // COLS) * (CH + 32 + PAD)
        try:
            im = Image.open(str(s["img"])).convert("L")
        except Exception:
            continue
        W, H = im.size
        g = s["gt_box"]
        boxes = [(g, (250, 70, 70))]
        if s.get("other_box") is not None:
            boxes.append((s["other_box"], s.get("other_col", (80, 160, 255))))
        ax1 = min(b[0][0] for b in boxes); ay1 = min(b[0][1] for b in boxes)
        ax2 = max(b[0][2] for b in boxes); ay2 = max(b[0][3] for b in boxes)
        k = 2.2
        mx, my = max(ax2-ax1, 14)*k/2, max(ay2-ay1, 14)*k/2
        ccx, ccy = (ax1+ax2)/2, (ay1+ay2)/2
        nx1, ny1 = max(0, int(ccx-mx)), max(0, int(ccy-my))
        nx2, ny2 = min(W, int(ccx+mx)), min(H, int(ccy+my))
        if nx2-nx1 < 4 or ny2-ny1 < 4:
            continue
        crop = im.crop((nx1, ny1, nx2, ny2)).convert("RGB").resize((CW, CH), Image.LANCZOS)
        sheet.paste(crop, (cx, cy))
        sx, sy = CW/crop.width, CH/crop.height
        for (bx, col) in boxes:
            d.rectangle([cx+(bx[0]-nx1)*sx, cy+(bx[1]-ny1)*sy,
                         cx+(bx[2]-nx1)*sx, cy+(bx[3]-ny1)*sy], outline=col, width=2)
        d.rectangle([cx, cy, cx+CW, cy+CH], outline=(60, 70, 86))
        d.text((cx+2, cy+CH+4), f"{cls_name} gt {int(g[2]-g[0])}x{int(g[3]-g[1])}px",
               fill=(215, 225, 235))
        d.text((cx+2, cy+CH+17), s["sub"], fill=(125, 137, 152))
    sheet.save(path)
    return sheet.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default=r"Y:\workspace\EDA\eda-pcb\yolo_part\runs\detect\pcb43_v2_from_best\weights\best.pt")
    ap.add_argument("--root", default=r"Y:\workspace\EDA\trainset_v2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--classes", default="")
    a = ap.parse_args()

    import re
    txt = (Path(a.root) / "data.yaml").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"names:\s*\[(.*?)\]", txt, re.S)
    names = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()] if m else \
        (Path(a.root) / "classes.txt").read_text().split()
    want = [c.strip() for c in a.classes.split(",") if c.strip()] if a.classes else None

    from ultralytics import YOLO
    model = YOLO(a.weights)
    img_dir = Path(a.root) / "images" / a.split
    lab_dir = Path(a.root) / "labels" / a.split

    fn_pool = defaultdict(list)
    fp_pool = defaultdict(list)
    st = defaultdict(Counter)
    for lp in sorted(lab_dir.glob("*.txt")):
        ip = img_dir / (lp.stem + ".png")
        if not ip.exists():
            ip = img_dir / (lp.stem + ".jpg")
        if not ip.exists():
            continue
        W, H = Image.open(ip).size
        gts = load_gt(lp, W, H)
        r = model.predict(str(ip), imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        raw = [(int(c), b.astype(float), float(s)) for b, c, s in
               zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        preds = []
        for c, b, s in raw:
            nm = names[c]
            preds.append((c, np.array(snap_box(b, nm)), s))
        pairs = match_official(preds, gts)
        pset = {i for i, _ in pairs}
        gset = {j for _, j in pairs}
        for j, (c, gb) in enumerate(gts):
            if j not in gset:
                st[names[c]].update(fn=1)
                pr, d = nearest(preds, c, gb)
                fn_pool[names[c]].append({
                    "img": ip, "gt_box": gb,
                    "other_box": None if pr is None else pr[1],
                    "other_col": (80, 160, 255),
                    "sub": (f"conf {pr[2]:.2f} | 差 {d:.0f}px | {lp.stem[:22]}" if pr
                            else f"同类无预测 | {lp.stem[:22]}"),
                    "d": d,
                })
        for i, (c, pb, sc) in enumerate(preds):
            if i not in pset:
                st[names[c]].update(fp=1)
                ng, d = nearest([(gc, gb) for gc, gb in gts], c, pb)
                fp_pool[names[c]].append({
                    "img": ip, "gt_box": pb,
                    "other_box": None if ng is None else ng[1],
                    "other_col": (250, 70, 70),
                    "sub": f"conf {sc:.2f} | 最近GT差 {d:.0f}px | {lp.stem[:22]}",
                    "d": d,
                })

    print(f"{'class':<16}{'gt':>6}{'fn':>6}{'fp':>6}   FN率")
    rows = []
    for nm, c in st.items():
        gt = c["fn"] + (c["tp"] if "tp" in c else 0)
        rows.append((c["fn"], nm, c["fn"], c["fp"]))
    # 重新算 GT 总数
    total = Counter()
    for lp in sorted(lab_dir.glob("*.txt")):
        for line in lp.read_text().splitlines():
            t = line.split()
            if len(t) >= 5:
                total[names[int(t[0])]] += 1
    for nm in sorted(st, key=lambda x: -st[x]["fn"]):
        c = st[nm]
        gt = total[nm]
        print(f"{nm:<16}{gt:>6}{c['fn']:>6}{c['fp']:>6}   {c['fn']/gt*100 if gt else 0:.1f}%")

    for nm in sorted(st):
        if want and nm not in want:
            continue
        fnl = sorted(fn_pool[nm], key=lambda s: -s["d"])[:a.n]
        fpl = sorted(fp_pool[nm], key=lambda s: -s["d"])[:a.n]
        render(fnl, f"FN {nm}  conf {a.conf}  官方口径(角点差<=20px) 红=GT 蓝=最近预测",
               OUT / f"fn_{nm}.png", nm, "FN")
        render(fpl, f"FP {nm}  conf {a.conf}  官方口径(角点差<=20px) 蓝=预测 红=最近GT",
               OUT / f"fp_{nm}.png", nm, "FP")
    meta = {nm: {"gt": total[nm], "fn": st[nm]["fn"], "fp": st[nm]["fp"]} for nm in st}
    (OUT / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(f"\n图写到 {OUT}")


if __name__ == "__main__":
    main()
