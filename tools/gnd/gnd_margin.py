"""GT bbox 是不是把符号切掉了？—— 对 gnd / v / r / c / testpoint 量「框外还有多少墨」。

判据：把 GT 框外扩 R 像素得到外环，数外环里的墨迹占比。
- 外环墨迹占比高  => 框是紧贴一个局部、周围还有实质内容（文字、导线、符号其余部分）
- 同时看「是否有大连通域触到框边」，这是「被切断」的硬证据。

输出：
  1. 逐类统计表（占比分位数、被判为 clipped 的比例）
  2. work/preview/margin_<cls>.png —— 前后对比接触表：左=原框，右=外扩 1.9x 后的完整上下文
"""
import sys, math, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageDraw
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser import discover, load_json          # noqa: E402

DATA_ROOT = r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)"
SPLITS = ("10GT", "10_GTcase", "200_train_cases")
CLASSES = ["gnd", "v", "r", "c", "testpoint", "box", "net_input"]
INK = 170
RING = 6               # 外环宽度（像素）
PER_SHEET = 12         # 每类抽多少张对比图
OUTDIR = Path(r"Y:\workspace\EDA\work\preview")
OUTDIR.mkdir(parents=True, exist_ok=True)


def source_of(stem: str) -> str:
    s = stem.lower()
    for k, v in (("kicad", "KiCad"), ("altium", "Altium"), ("datasheet", "Datasheet"),
                 ("jlc", "jlc"), ("easyeda", "jlc")):
        if k in s:
            return v
    return "unknown"


def load_gray(p: Path) -> np.ndarray:
    im = Image.open(str(p)).convert("L")
    return np.asarray(im, dtype=np.uint8)


def biggest_cc_touch(mask: np.ndarray) -> bool:
    """框内墨迹的最大连通域是否触到框边（= 符号被切断的硬证据）。用 4 邻域 BFS。"""
    h, w = mask.shape
    if h == 0 or w == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    best = 0; best_touch = False
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]; seen[y0, x0] = True
        n = 0; touch = False
        while stack:
            y, x = stack.pop(); n += 1
            if y == 0 or x == 0 or y == h - 1 or x == w - 1:
                touch = True
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True; stack.append((ny, nx))
        if n > best:
            best, best_touch = n, touch
    return best_touch


def main():
    rows = defaultdict(list)          # cls -> list of dict
    picks = defaultdict(list)         # cls -> samples for the contact sheet
    ncase = 0
    for case in discover(DATA_ROOT, SPLITS):
        try:
            doc = load_json(case.json_path)
            gray = load_gray(Path(str(case.png_path)))
        except Exception:
            continue
        H, W = gray.shape
        ink = gray < INK
        src = source_of(Path(str(case.png_path)).stem)
        got = Counter()
        for key, comp in (doc.get("components") or {}).items():
            ct = comp.get("type")
            if ct not in CLASSES:
                continue
            x1, y1, x2, y2 = comp["bbox"]
            iy1, iy2 = int(round(H - y2)), int(round(H - y1))
            ix1, ix2 = int(round(x1)), int(round(x2))
            if ix2 - ix1 < 3 or iy2 - iy1 < 3:
                continue
            ix1 = max(0, ix1); iy1 = max(0, iy1); ix2 = min(W, ix2); iy2 = min(H, iy2)
            inner = ink[iy1:iy2, ix1:ix2]
            if inner.size == 0:
                continue
            # 外环：外扩 RING 后减掉原框
            ex1, ey1 = max(0, ix1 - RING), max(0, iy1 - RING)
            ex2, ey2 = min(W, ix2 + RING), min(H, iy2 + RING)
            outer = ink[ey1:ey2, ex1:ex2]
            ring_area = outer.size - inner.size
            ring_ink = int(outer.sum()) - int(inner.sum())
            ring_frac = ring_ink / ring_area if ring_area > 0 else 0.0
            inn_frac = float(inner.mean())
            rec = dict(case=case.case_id, src=src, key=key,
                       w=ix2 - ix1, h=iy2 - iy1, area=(ix2 - ix1) * (iy2 - iy1),
                       ring_frac=ring_frac, inn_frac=inn_frac,
                       cut=biggest_cc_touch(inner),
                       box=(ix1, iy1, ix2, iy2))
            rows[ct].append(rec)
            if ct in ("gnd",) and len(picks[ct]) < 400:
                picks[ct].append((rec, gray))
        ncase += 1
        if ncase % 40 == 0:
            print(f"  ...{ncase} cases", flush=True)

    print(f"\ncases {ncase}")
    print(f"\n{'cls':<11}{'n':>6}{'ring_frac p50':>15}{'p90':>8}{'p99':>8}"
          f"{'ring>0.06':>11}{'框被切断':>10}{'框内墨占比 p50':>16}{'中位框 w x h':>16}")
    for ct in CLASSES:
        v = rows.get(ct) or []
        if not v:
            continue
        rf = np.array([r["ring_frac"] for r in v])
        inf = np.array([r["inn_frac"] for r in v])
        w = np.array([r["w"] for r in v]); h = np.array([r["h"] for r in v])
        cut = np.mean([r["cut"] for r in v])
        print(f"{ct:<11}{len(v):>6}{np.percentile(rf,50):>15.4f}{np.percentile(rf,90):>8.4f}"
              f"{np.percentile(rf,99):>8.4f}{np.mean(rf>0.06):>11.1%}{cut:>10.1%}"
              f"{np.percentile(inf,50):>16.4f}{int(np.median(w)):>9} x {int(np.median(h)):<5}")

    # 按来源看 gnd
    print("\ngnd by source:")
    by = defaultdict(list)
    for r in rows.get("gnd", []):
        by[r["src"]].append(r["ring_frac"])
    for s, vals in sorted(by.items(), key=lambda kv: -len(kv[1])):
        a = np.array(vals)
        print(f"  {s:<17} n={len(a):<5} ring_frac p50 {np.percentile(a,50):.4f} "
              f"p90 {np.percentile(a,90):.4f}  ring>0.06 {np.mean(a>0.06):.1%}")

    # 接触表：挑外环墨迹最多的一批，左原框 / 右 1.9x 上下文
    for ct in ("gnd", "v", "r", "c"):
        v = rows.get(ct) or []
        if not v:
            continue
        v.sort(key=lambda r: -r["ring_frac"])
        sel = v[:PER_SHEET]
        # 需要原图，按 case 再取一次
        cache = {}
        CELLW, CELLH = 96, 96
        pad = 4
        sheet = Image.new("RGB", ((CELLW * 2 + pad * 3) * 6, (CELLH + 26 + pad) * 2),
                          (16, 18, 24))
        d = ImageDraw.Draw(sheet)
        x = pad; y = pad; col = 0
        for rec in sel:
            if rec["case"] not in cache:
                case = [c for c in discover(DATA_ROOT, SPLITS) if c.case_id == rec["case"]]
                if not case:
                    continue
                cache[rec["case"]] = (load_gray(Path(str(case[0].png_path))),
                                      Path(str(case[0].png_path)).stem)
            gray, stem = cache[rec["case"]]
            bx1, by1, bx2, by2 = rec["box"]
            # 左：原框
            a = Image.fromarray(gray[by1:by2, bx1:bx2])
            # 右：外扩到 1.9 倍宽高、以框中心为中心
            cw, ch = bx2 - bx1, by2 - by1
            k = 1.9
            nx1 = max(0, int((bx1 + bx2) / 2 - cw * k / 2))
            ny1 = max(0, int((by1 + by2) / 2 - ch * k / 2))
            nx2 = min(gray.shape[1], int((bx1 + bx2) / 2 + cw * k / 2))
            ny2 = min(gray.shape[0], int((by1 + by2) / 2 + ch * k / 2))
            b = Image.fromarray(gray[ny1:ny2, nx1:nx2])
            for im, ox in ((a, 0), (b, CELLW + pad)):
                im = im.convert("RGB")
                im.thumbnail((CELLW, CELLH), Image.LANCZOS)
                sheet.paste(im, (x + ox, y))
                d.rectangle([x + ox, y, x + ox + im.width, y + im.height], outline=(70, 80, 96))
            cap = f"{ct} {rec['src'][:6]} {rec['w']}x{rec['h']} ring{rec['ring_frac']:.2f}"
            d.text((x, y + CELLH + 3), cap, fill=(200, 210, 220))
            d.text((x, y + CELLH + 14), f"{stem[:16]}", fill=(120, 130, 145))
            col += 1
            x += CELLW * 2 + pad * 3
            if col % 6 == 0:
                x = pad; y += CELLH + 26 + pad
        p = OUTDIR / f"margin_{ct}.png"
        sheet.save(p)
        print(f"  wrote {p}  ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
