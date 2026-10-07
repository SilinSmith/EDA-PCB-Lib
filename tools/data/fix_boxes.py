"""把 GT 的固定尺寸图章扩成符号真实画出来的范围。

思路：图章只是一个锚点标记。符号本体是「粗墨迹」的连通域，导线是细线。
      1) 以图章为中心开一个窗口
      2) 二值化取墨迹
      3) 形态学开运算去掉细导线，留下符号本体
      4) 取与图章重叠最多的那个连通域，它的外接框就是符号真实范围

用法:
  python fix_boxes.py --class r --cases 60 --sheet
  python fix_boxes.py --class v --cases 60 --k 3
"""
import argparse, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser import discover, load_json          # noqa: E402

ROOT = r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)"
SPLITS = ("10GT", "10_GTcase", "200_train_cases")
OUT = Path(r"Y:\workspace\EDA\work\preview")
INK = 170


def grow(gray, box, k=3, winf=5.0, maxarea=14.0):
    """返回 (grown_box, ok, why)。box 是图像坐标 (x1,y1,x2,y2)。"""
    H, W = gray.shape
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = max(bw * winf / 2, bw / 2 + 8), max(bh * winf / 2, bh / 2 + 8)
    X1, Y1 = int(max(0, cx - hw)), int(max(0, cy - hh))
    X2, Y2 = int(min(W, cx + hw)), int(min(H, cy + hh))
    if X2 - X1 < 4 or Y2 - Y1 < 4:
        return box, False, "window_too_small"
    sub = gray[Y1:Y2, X1:X2]
    ink = (sub < INK).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, ker)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(thick, 8)
    if n <= 1:
        return box, False, "no_ink"
    # 图章在窗口内的坐标
    sx1, sy1 = int(max(0, x1 - X1)), int(max(0, y1 - Y1))
    sx2, sy2 = int(min(sub.shape[1], x2 - X1)), int(min(sub.shape[0], y2 - Y1))
    stamp = np.zeros_like(thick)
    if sx2 > sx1 and sy2 > sy1:
        stamp[sy1:sy2, sx1:sx2] = 1
    best, bestov = 0, 0
    for j in range(1, n):
        m = lab == j
        ov = int((m & (stamp > 0)).sum())
        if ov > bestov:
            best, bestov = j, ov
    if best == 0:
        # 图章没压到任何粗墨迹 -> 退回：整个窗口里最大的连通域
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return box, False, "no_cc"
        best = 1 + int(areas.argmax())
    bx, by, bwd, bht = stats[best, cv2.CC_STAT_LEFT], stats[best, cv2.CC_STAT_TOP], \
        stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT]
    gx1, gy1, gx2, gy2 = X1 + bx, Y1 + by, X1 + bx + bwd, Y1 + by + bht
    if bwd * bht > maxarea * bw * bh:
        return box, False, "merged_with_wire"
    # 长出来的框必须至少在一个方向显著超过图章，否则等于没长
    if bwd <= bw * 1.05 and bht <= bh * 1.05:
        return box, False, "stamp_only"
    if bwd < bw * 0.8 and bht < bh * 0.8:
        return box, False, "shrank"
    return (float(gx1), float(gy1), float(gx2), float(gy2)), True, "grown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--cases", type=int, default=60)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--winf", type=float, default=5.0)
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--per", type=int, default=12)
    a = ap.parse_args()

    cases = [c for c in discover(ROOT, SPLITS)][:a.cases]
    rows = []
    why = defaultdict(int)
    for c in cases:
        try:
            doc = load_json(c.json_path)
            gray = np.asarray(Image.open(str(c.png_path)).convert("L"), dtype=np.uint8)
        except Exception:
            continue
        H = gray.shape[0]
        for key, v in (doc.get("components") or {}).items():
            if v.get("type") != a.cls:
                continue
            x1, y1, x2, y2 = v["bbox"]
            box = (x1, H - y2, x2, H - y1)
            if box[2] - box[0] < 3 or box[3] - box[1] < 3:
                continue
            g, ok, r = grow(gray, box, a.k, a.winf)
            why[r] += 1
            rows.append((c.case_id, key, box, g, ok, c.png_path, gray.shape[::-1]))

    print(f"class={a.cls}  cases={len(cases)}  instances={len(rows)}  k={a.k}  winf={a.winf}")
    print("结果:", dict(why))
    good = [r for r in rows if r[4]]
    if good:
        rw = [ (r[3][2]-r[3][0])/(r[2][2]-r[2][0]) for r in good ]
        rh = [ (r[3][3]-r[3][1])/(r[2][3]-r[2][1]) for r in good ]
        ra = [ x*y for x,y in zip(rw,rh) ]
        import statistics as st
        for nm, arr in (("宽比 grown/stamp", rw), ("高比", rh), ("面积比", ra)):
            q = sorted(arr)
            p = lambda t: q[min(len(q)-1, int(t*len(q)))]
            print(f"  {nm:<18} p10 {p(.1):.2f}  p50 {p(.5):.2f}  p90 {p(.9):.2f}  mean {st.mean(arr):.2f}")
        # 扩完之后还剩多少与图章相同
        same = sum(1 for r in good if abs(r[3][2]-r[3][0]-(r[2][2]-r[2][0]))<1 and abs(r[3][3]-r[3][1]-(r[2][3]-r[2][1]))<1)
        print(f"  与图章尺寸相同: {same}/{len(good)}")

    if a.sheet and rows:
        per = a.per
        CW = 108; PAD = 4
        take = [r for r in rows if r[4]][:per*3] or rows[:per*3]
        rowsn = (len(take)+per-1)//per
        sheet = Image.new("RGB", (per*(CW+PAD)+PAD, rowsn*(CW+20+PAD)+PAD), (14,17,23))
        d = ImageDraw.Draw(sheet)
        for i,(cid,key,box,g,ok,path,wh) in enumerate(take):
            im = Image.open(str(path)).convert("RGB")
            r_, c_ = divmod(i, per)
            x = PAD + c_*(CW+PAD); y = PAD + r_*(CW+20+PAD)
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            span = max(box[2]-box[0], box[3]-box[1])*3.4
            hw = max(span/2, 30); hh = max((box[3]-box[1])*2.6/2, 30)
            X1,Y1 = int(max(0,cx-hw)), int(max(0,cy-hh))
            X2,Y2 = int(min(im.width,cx+hw)), int(min(im.height,cy+hh))
            if X2-X1<3 or Y2-Y1<3: continue
            crop = im.crop((X1,Y1,X2,Y2))
            sc = min(CW/crop.width, CW/crop.height)
            crop = crop.resize((max(1,int(crop.width*sc)), max(1,int(crop.height*sc))), Image.LANCZOS)
            ox, oy = x+(CW-crop.width)//2, y+14+(CW-crop.height)//2
            sheet.paste(crop,(ox,oy))
            dd = ImageDraw.Draw(sheet)
            dd.rectangle([ox+(box[0]-X1)*sc, oy+(box[1]-Y1)*sc, ox+(box[2]-X1)*sc, oy+(box[3]-Y1)*sc], outline=(255,60,60), width=1)
            if ok:
                dd.rectangle([ox+(g[0]-X1)*sc, oy+(g[1]-Y1)*sc, ox+(g[2]-X1)*sc, oy+(g[3]-Y1)*sc], outline=(60,255,120), width=2)
            d.text((x, y+2), f"{key} {round(box[2]-box[0])}x{round(box[3]-box[1])} -> {round(g[2]-g[0])}x{round(g[3]-g[1])}", fill=(200,210,225))
            d.text((x, y+CW+15), f"{cid} {'OK' if ok else 'FAIL'}", fill=(150,160,180))
        p = OUT / f"fixbox_{a.cls}.png"
        sheet.save(p)
        print(f"  wrote {p} ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
