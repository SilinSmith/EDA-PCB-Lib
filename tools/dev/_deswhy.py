# -*- coding: utf-8 -*-
"""**位号为什么错**：把「认不出来（字形/字体）」和「挂到别的元件上」分开量。

对每个 case：
  1. GT 元件（`components` 的键就是位号）与 stage1 元件按 **JSON 坐标 IoU** 配对；
  2. 取 `_assigned` 里那个元件被赋的 `designator` 槽（`text`/`box`/`conf`/`_by`）；
  3. 若 `text != GT 键` ⇒ 记一条「错」，并**按那个 box 从原图裁片**：
     * 用同一个 CRNN **重读这个 box** —— 若重读 == GT，说明**框里有正确文字、是我们没读到/没用它**；
       若重读 == 我们的错值，说明**这个框在模型眼里就是那个错字**（字形/字体问题）。
  4. 顺带统计 `_by` 来源（`solve:ocr:designator` / `solve:vlm:enrich` / …）。

裁片拼成一张对照图（4× 放大），标签写 `GT=… ours=… reread=… conf=… by=…`。

    python -X utf8 work\tools\dev\_deswhy.py --limit 30 --out work\_deswhy
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

from PIL import Image, ImageDraw, ImageFont          # noqa: E402
from textrec.infer import Recognizer                # noqa: E402

WORK = Path(r"Y:\workspace\EDA\work")
EDA = Path(r"Y:\workspace\EDA\eda-pcb")
DATA = EDA / "yolo_part" / "Dataset(waitforpreprocess)"
SPLITS = ("200_train_cases", "10_GTcase", "10GT")


def find_png(case: str) -> Path | None:
    for root in (WORK / "full220" / "images", WORK / "full220"):
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.png")):
            if p.stem == case:
                return p
    return None


def find_gt(case: str) -> dict | None:
    for sp in SPLITS:
        root = DATA / sp
        if not root.is_dir():
            continue
        for pat in ("%s.json" % case, "%s-*.json" % case, "%s_*.json" % case):
            for p in sorted(root.rglob(pat)):
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(d, dict) and d.get("components"):
                    return d
    return None


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def canon(t: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", (t or "")).upper()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(WORK / "_f220_A" / "_assigned"), help="`_assigned` 目录")
    ap.add_argument("--stage1", default=str(WORK / "_full220" / "runs" / "stage1"))
    ap.add_argument("--out", default=str(WORK / "_deswhy"))
    ap.add_argument("--cases-file", default=str(WORK / "_cases220.txt"))
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--sheet-n", type=int, default=40)
    ap.add_argument("--ckpt", default=str(WORK / "ocr_mix" / "runs" / "best.pt"))
    ap.add_argument("--no-reread", action="store_true")
    a = ap.parse_args()

    raw = Path(a.cases_file).read_text(encoding="utf-8")
    seen, cases = set(), []
    for tok in re.split(r"[,\s]+", raw):
        tok = tok.strip()
        if not tok or "-" in tok or not re.match(r"^[0-9A-Za-z_]+$", tok) or tok in seen:
            continue
        seen.add(tok)
        cases.append(tok)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = None
    if not a.no_reread:
        rec = Recognizer.load(a.ckpt, device="cpu")
        print("[ckpt] %s  classes=%d" % (Path(a.ckpt).name, rec.n_class if hasattr(rec, "n_class") else -1))

    n_pair = n_wrong = 0
    by_cnt, verdict = Counter(), Counter()
    rows = []
    for case in cases[: a.limit]:
        gp, sp, ap_ = find_gt(case), Path(a.stage1) / ("%s.stage1.json" % case), Path(a.dir) / ("%s.json" % case)
        if not (gp and sp.is_file() and ap_.is_file()):
            continue
        s1 = json.loads(sp.read_text(encoding="utf-8"))
        asg = json.loads(ap_.read_text(encoding="utf-8"))
        comps = {c["id"]: c for c in s1.get("components", [])}
        used = set()
        for gk, gv in (gp.get("components") or {}).items():
            gb = gv.get("bbox")
            if not gb:
                continue
            best, bi = 0.0, None
            for cid, c in comps.items():
                if cid in used:
                    continue
                v = iou(gb, c.get("bbox_json") or [0, 0, 0, 0])
                if v > best:
                    best, bi = v, cid
            if bi is None or best < a.min_iou:
                continue
            used.add(bi)
            n_pair += 1
            slot = (asg.get("comps", {}).get(bi) or {}).get("designator") or {}
            ours = slot.get("text") or ""
            by_cnt[str(slot.get("_by"))] += 1
            if canon(ours) == canon(gk):
                verdict["pair_ok"] += 1
                continue
            n_wrong += 1
            box = slot.get("box")
            if not box:
                verdict["no_box"] += 1
                continue
            img = None
            png = find_png(case)
            if png is not None:
                img = Image.open(png).convert("RGB")
                crop = img.crop((int(box[0]) - 2, int(box[1]) - 2, int(box[2]) + 2, int(box[3]) + 2))
            else:
                crop = Image.new("RGB", (60, 18), "white")
            rr = ""
            if rec is not None and img is not None:
                try:
                    rr = (rec.read_many(img.convert("L"), [(list(box), "designator")]) or [("", 0)])[0][0]
                except Exception as e:                                   # noqa: BLE001
                    rr = "ERR:%s" % e
            if canon(rr) == canon(gk):
                verdict["reread==GT（框里读得出，管线却没用）"] += 1
            elif canon(rr) == canon(ours):
                verdict["reread==ours（模型在该框就读错）"] += 1
            else:
                verdict["reread=其它"] += 1
            rows.append((case, gk, ours, rr, slot.get("conf"), slot.get("_by"), crop))

    for (case, gk, ours, rr, conf, by, crop) in rows[: a.sheet_n]:
        pass
    if rows:
        scale, cap_h, wmax = 4, 14, 0
        tiles = []
        for (case, gk, ours, rr, conf, by, crop) in rows[: a.sheet_n]:
            c2 = crop.resize((crop.width * scale, crop.height * scale), Image.NEAREST)
            cap = "%s GT=%-6s ours=%-8s reread=%-8s conf=%.2f by=%s" % (
                case, gk, ours or "-", rr or "-", conf or 0.0, (by or "-").replace("solve:", ""))
            tiles.append((c2, cap))
            wmax = max(wmax, c2.width + 620)
        f = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 13)
        H = sum(max(t[0].height, cap_h) + 6 for t in tiles)
        sheet = Image.new("RGB", (min(1600, wmax + 20), H + 10), "white")
        d = ImageDraw.Draw(sheet)
        y = 5
        for t, cap in tiles:
            d.text((5, y + 2), cap, fill=(0, 0, 0), font=f)
            sheet.paste(t, (620, y))
            y += max(t.height, cap_h) + 6
        p = out / "sheet.png"
        sheet.save(p)
        print("[sheet] %s  (%d 行)" % (p, len(tiles)))

    print("配对 %d，其中位号错 %d（%.1f%%）" % (n_pair, n_wrong, 100.0 * n_wrong / max(1, n_pair)))
    print("判定: " + "  ".join("%s=%d" % (k, v) for k, v in verdict.most_common()))
    print("_by : " + "  ".join("%s=%d" % (k, v) for k, v in by_cnt.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
