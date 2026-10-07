"""查一个 case 的 GT：某个类的每个框长什么样、框里到底有没有东西。

用法:
  python gt_look.py 0050 --type r
  python gt_look.py 0027 --type net_input --sheet
  python gt_look.py 0050 --all
"""
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser import discover, load_json          # noqa: E402

DATA_ROOT = r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)"
SPLITS = ("10GT", "10_GTcase", "200_train_cases")
OUT = Path(r"Y:\workspace\EDA\work\preview")
INK = 170


def find(cid):
    for c in discover(DATA_ROOT, SPLITS):
        if c.case_id.lstrip("0") == cid.lstrip("0") or c.case_id == cid:
            return c
    return None


def to_img(box, H):
    x1, y1, x2, y2 = box
    return (x1, H - y2, x2, H - y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--type", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--per", type=int, default=12)
    a = ap.parse_args()

    c = find(a.case)
    if c is None:
        print("case not found:", a.case); return
    doc = load_json(c.json_path)
    gray = np.asarray(Image.open(str(c.png_path)).convert("L"), dtype=np.uint8)
    H, W = gray.shape
    comps = doc.get("components") or {}
    print(f"case {c.case_id}  split={c.split}  png={Path(c.png_path).name}  {W}x{H}")
    print(f"components {len(comps)}  pins_groups {len(doc.get('pins') or {})}  nets {len(doc.get('nets') or {})}")

    from collections import Counter
    hist = Counter(v.get("type") for v in comps.values())
    print("type histogram:", dict(hist.most_common()))

    items = []
    for k, v in comps.items():
        if a.type and v.get("type") != a.type:
            continue
        box = to_img(v["bbox"], H)
        x1, y1, x2, y2 = [int(round(t)) for t in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        patch = gray[y1:y2, x1:x2] if w and h else np.zeros((1, 1), np.uint8)
        ink = float((patch < INK).mean()) if patch.size else 0.0
        items.append((k, v.get("type"), v.get("value"), (x1, y1, x2, y2), w, h, ink, patch))

    if not a.all and not a.type:
        return
    items.sort(key=lambda t: -t[5])
    print(f"\n{'key':<12}{'type':<12}{'value':<10}{'w x h':<12}{'ink':<8}img-box")
    for k, ty, val, box, w, h, ink, _ in items[:80]:
        print(f"{k:<12}{str(ty):<12}{str(val):<10}{f'{w} x {h}':<12}{ink:<8.4f}{box}")
    if len(items) > 80:
        print(f"... 共 {len(items)} 个")

    empt = [t for t in items if t[6] < 0.02]
    print(f"\n几乎空的框(ink<0.02): {len(empt)}/{len(items)}")
    print(f"框宽 <=3px: {sum(1 for t in items if t[4] <= 3)}   框高 <=3px: {sum(1 for t in items if t[5] <= 3)}")

    if a.sheet and items:
        per = a.per
        CW = CH = 96; PAD = 4
        rows = (min(len(items), per * 4) + per - 1) // per
        sheet = Image.new("RGB", (per * (CW + PAD) + PAD, rows * (CH + 20 + PAD) + PAD), (14, 17, 23))
        d = ImageDraw.Draw(sheet)
        for i, (k, ty, val, box, w, h, ink, patch) in enumerate(items[:per * 4]):
            r, col = divmod(i, per)
            x = PAD + col * (CW + PAD); y = PAD + r * (CH + 20 + PAD)
            # 左：原框；右：外扩 2.2 倍上下文
            for j, (f, tag) in enumerate(((1.0, "box"), (2.2, "ctx"))):
                cw, ch = box[2] - box[0], box[3] - box[1]
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                hw, hh = cw * f / 2, ch * f / 2
                X1, Y1 = int(max(0, cx - hw)), int(max(0, cy - hh))
                X2, Y2 = int(min(W, cx + hw)), int(min(H, cy + hh))
                if X2 - X1 < 3 or Y2 - Y1 < 3:
                    continue
                im = Image.fromarray(gray[Y1:Y2, X1:X2]).convert("RGB").resize((CW // 2 - 1, CH), Image.LANCZOS)
                if j == 0:
                    dd = ImageDraw.Draw(im); dd.rectangle([0, 0, im.width - 1, im.height - 1], outline=(255, 60, 60))
                sheet.paste(im, (x + j * (CW // 2), y + 14))
            d.text((x, y + 2), f"{ty} {w}x{h} ink{ink:.2f}", fill=(200, 210, 225))
            d.text((x, y + CH + 15), f"{k} {c.case_id}", fill=(150, 160, 180))
        p = OUT / f"gt_{c.case_id}_{a.type or 'all'}.png"
        sheet.save(p)
        print(f"wrote {p} ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
