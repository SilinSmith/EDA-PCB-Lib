# -*- coding: utf-8 -*-
"""官方渲染用的是哪一款字体？—— 拿真实裁片（GT 文本已知）去套候选字体的字形。

做法：对 manifest 里 dist==0 的行裁出真实墨迹 → 二值化 → 墨迹紧框 → **按高度归一**
（保留长宽比，宽度信息也参与比较，因为等宽/比例字体差别正在这里）→ 灰度块；
候选字体用 PIL 渲染同一文本（4× 超采样再缩下来，模仿抗锯齿），同样归一；
两个 32×128 的灰度块做「允许 ±4px 平移」的最大相关系数。按字体汇总均值排名。

**注意**：官方 220 张里有多个渲染器变体（`jlc` 3355 行、`Datasheet` 836、`Altium Designer` 313、
`KiCad`/`kicad` 83、`other` 89、10GT 的 `1.png`…`10.png` 307），字体可能一个变体一款 —— 用
`--variant jlc` 之类的过滤分开套。

用法：
  python -X utf8 work\tools\dev\_fontfit.py --role value --variant jlc --n 100
  python -X utf8 work\tools\dev\_fontfit.py --role designator --variant Datasheet --n 80
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORK = Path(r"Y:\workspace\EDA\work")
MPL = Path(r"Y:\workspace\EDA\eda-pcb\eda_yolo\Lib\site-packages\matplotlib\mpl-data\fonts\ttf")

CANDS = [
    ("cour.ttf", Path(r"C:\Windows\Fonts\cour.ttf")),
    ("courbd.ttf", Path(r"C:\Windows\Fonts\courbd.ttf")),
    ("courbi.ttf", Path(r"C:\Windows\Fonts\courbi.ttf")),
    ("lucon.ttf", Path(r"C:\Windows\Fonts\lucon.ttf")),
    ("consola.ttf", Path(r"C:\Windows\Fonts\consola.ttf")),
    ("consolab.ttf", Path(r"C:\Windows\Fonts\consolab.ttf")),
    ("CascadiaMono.ttf", Path(r"C:\Windows\Fonts\CascadiaMono.ttf")),
    ("CascadiaCode.ttf", Path(r"C:\Windows\Fonts\CascadiaCode.ttf")),
    ("DejaVuSansMono.ttf", MPL / "DejaVuSansMono.ttf"),
    ("DejaVuSansMono-Bold.ttf", MPL / "DejaVuSansMono-Bold.ttf"),
    ("cmtt10.ttf", MPL / "cmtt10.ttf"),
    ("cmr10.ttf", MPL / "cmr10.ttf"),
    ("DejaVuSerif.ttf", MPL / "DejaVuSerif.ttf"),
    ("DejaVuSerif-Bold.ttf", MPL / "DejaVuSerif-Bold.ttf"),
    ("times.ttf", Path(r"C:\Windows\Fonts\times.ttf")),
    ("arial.ttf", Path(r"C:\Windows\Fonts\arial.ttf")),
    ("arialbd.ttf", Path(r"C:\Windows\Fonts\arialbd.ttf")),
    ("calibri.ttf", Path(r"C:\Windows\Fonts\calibri.ttf")),
    ("segoeui.ttf", Path(r"C:\Windows\Fonts\segoeui.ttf")),
]
SS = 4
CH, CW = 32, 128          # 归一化画布（高×宽）
MAX_SHIFT = 4


def _norm(a: np.ndarray) -> np.ndarray | None:
    if a.size == 0:
        return None
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 8:
        return None
    ink = a < lo + 0.55 * (hi - lo)
    ys, xs = np.where(ink)
    if len(xs) < 4:
        return None
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    h, w = y2 - y1, x2 - x1
    if h < 2 or w < 2:
        return None
    sub = 255.0 - a[y1:y2, x1:x2].astype(np.float32)
    sub -= sub.min()
    if sub.max() > 0:
        sub /= sub.max()
    scale = (CH - 8) / float(h)
    if w * scale > CW - 2:
        scale = (CW - 2) / float(w)
    nw, nh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
    img = Image.fromarray((sub * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    canvas = np.zeros((CH, CW), dtype=np.float32)
    canvas[4:4 + nh, 1:1 + nw] = np.asarray(img, dtype=np.float32) / 255.0
    return canvas


def _best_corr(a: np.ndarray, b: np.ndarray) -> float:
    best = -1.0
    for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
            bb = np.roll(np.roll(b, dy, axis=0), dx, axis=1)
            x = a.ravel() - a.mean()
            y = bb.ravel() - bb.mean()
            d = float(np.linalg.norm(x) * np.linalg.norm(y))
            if d > 0:
                best = max(best, float(x @ y / d))
    return best


def render(text: str, font_path: Path, size: int) -> np.ndarray | None:
    try:
        f = ImageFont.truetype(str(font_path), size * SS)
    except Exception:
        return None
    pad = 6 * SS
    img = Image.new("L", (max(60, len(text) * size * SS + 2 * pad), 4 * size * SS), 255)
    d = ImageDraw.Draw(img)
    try:
        d.text((pad, pad), text, font=f, fill=0)
    except Exception:
        return None
    return np.asarray(img.resize((img.width // SS, img.height // SS), Image.LANCZOS),
                      dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(WORK / "official_text" / "manifest.jsonl"))
    ap.add_argument("--role", default="value")
    ap.add_argument("--variant", default="")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    rows, cache = [], {}
    for line in Path(a.manifest).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("role") != a.role or r.get("dist", 0) != 0:
            continue
        if len(r.get("text") or "") < 3:
            continue
        if a.variant and a.variant.lower() not in Path(r["image"]).name.lower():
            continue
        rows.append(r)
        if len(rows) >= a.n:
            break
    print("样本 %d 条（role=%s, dist==0, variant=%r）" % (len(rows), a.role, a.variant))

    scores = defaultdict(list)
    for r in rows:
        p = Path(r["image"])
        if not p.is_file():
            continue
        if p not in cache:
            cache[p] = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        g = cache[p]
        x1, y1, x2, y2 = [int(round(float(v))) for v in r["bbox"]]
        real = _norm(g[y1:y2, x1:x2])
        if real is None:
            continue
        for name, path in CANDS:
            if not path.is_file():
                continue
            best = None
            for size in range(7, 16):
                rr = render(r["text"], path, size)
                if rr is None:
                    continue
                cand = _norm(rr)
                if cand is None:
                    continue
                c = _best_corr(real, cand)
                if best is None or c > best:
                    best = c
            if best is not None:
                scores[name].append(best)

    rank = sorted(((sum(v) / len(v), k, len(v)) for k, v in scores.items() if v), reverse=True)
    print("排名（平均相关系数）：")
    for s, name, k in rank[:a.top]:
        print("  %.4f  %-26s n=%d" % (s, name, k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
