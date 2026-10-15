# -*- coding: utf-8 -*-
"""**官方字体合成语料**：用与官方渲染器同款/同类的字体直接合成文字裁片，补上合成语料的字体域缺口。

动机（本次实测）：
* 官方 220 张里按渲染器变体分：`jlc` 3355 行（67%）、`Datasheet` 836、`Altium Designer` 313、
  `KiCad`/`kicad` 83、10GT 的 `1.png`…`10.png` 307、`other` 89。
* 拿真实裁片（GT 文本已知）套候选字体的字形（`work\tools\dev\_fontfit.py`，按高度归一 +
  ±4px 最大相关）：**`jlc` → arial 0.875**、**`Datasheet` → arial 0.888**、
  **`Altium Designer` → times 0.866 / arial 0.840**；而 10GT 那批目视是**衬线等宽**（≈ courbd）。
* 但项目自己的合成语料（`eda-pcb\synth\symbols.py`）把位号/值/引脚号都画成 **`mono` = consola.ttf**
  （值/器件名偶用 `sans`）⇒ **字体类别就不对**（等宽 vs 比例无衬线）。

本工具因此**不画整张原理图**，只直接合成紧贴文字的裁片（比走整图快几个数量级），
字号从官方裁片的**真实墨迹高度分布**里采样（按 role），字体从上面的实测排名里采样。

产物（textrec 训练用 manifest 格式）：
  ``{"image":…, "case":…, "src":"font", "split":"train|val", "role":…, "role_id":…, "text":…, "bbox":[…]}``
裁片以**多行拼进大图**（sheet）的方式落盘，省 inode 也省 IO。

用法：
  python -X utf8 work\tools\dev\mk_fonttext.py --out work\fonttext --n 120000 --sheets 4
  python -X utf8 work\tools\dev\mk_fonttext.py --out work\fonttext_probe --n 400 --sheet-side 900
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

WORK = Path(r"Y:\workspace\EDA\work")
MPL = Path(r"Y:\workspace\EDA\eda-pcb\eda_yolo\Lib\site-packages\matplotlib\mpl-data\fonts\ttf")

ROLE_ID = {"designator": 0, "value": 1, "pin_name": 2, "pin_number": 3, "net_label": 4, "note": 5}

# 按实测排名分组：官方主档（jlc/Datasheet）是比例无衬线，Altium 偏衬线，10GT 是衬线等宽。
FONT_GROUPS = {
    "sans": [("arial", Path(r"C:\Windows\Fonts\arial.ttf"), 5),
             ("arialbd", Path(r"C:\Windows\Fonts\arialbd.ttf"), 2),
             ("segoeui", Path(r"C:\Windows\Fonts\segoeui.ttf"), 2),
             ("calibri", Path(r"C:\Windows\Fonts\calibri.ttf"), 2)],
    "serif": [("times", Path(r"C:\Windows\Fonts\times.ttf"), 3),
              ("DejaVuSerif", MPL / "DejaVuSerif.ttf", 1),
              ("cmr10", MPL / "cmr10.ttf", 1)],
    "mono": [("consola", Path(r"C:\Windows\Fonts\consola.ttf"), 2),
             ("consolab", Path(r"C:\Windows\Fonts\consolab.ttf"), 1),
             ("courbd", Path(r"C:\Windows\Fonts\courbd.ttf"), 3),
             ("cour", Path(r"C:\Windows\Fonts\cour.ttf"), 2),
             ("lucon", Path(r"C:\Windows\Fonts\lucon.ttf"), 1),
             ("DejaVuSansMono", MPL / "DejaVuSansMono.ttf", 1),
             ("CascadiaMono", Path(r"C:\Windows\Fonts\CascadiaMono.ttf"), 1)],
}
# 变体 → 字体组权重（对齐实测：jlc/Datasheet 是 sans，Altium 偏 serif，KiCad/10GT 偏 mono）
VARIANT_MIX = {
    "jlc": {"sans": 0.80, "serif": 0.08, "mono": 0.12},
    "Datasheet": {"sans": 0.75, "serif": 0.10, "mono": 0.15},
    "Altium": {"sans": 0.35, "serif": 0.45, "mono": 0.20},
    "KiCad": {"sans": 0.35, "serif": 0.10, "mono": 0.55},
    "10GT": {"sans": 0.15, "serif": 0.10, "mono": 0.75},
}
# 官方 220 张语料里的实测变体频次（4974 行）：jlc 3355 / Datasheet 836 / Altium 313 / 10GT 307 / 其它 ~172
VARIANT_WEIGHT = {"jlc": 3355, "Datasheet": 836, "Altium": 313, "KiCad": 120, "10GT": 307}

UNITS = ["", "R", "k", "kΩ", "M", "MΩ", "Ω", "mΩ", "pF", "nF", "uF", "F", "H", "uH", "nH",
         "V", "mV", "A", "mA", "uA", "W", "kW", "Hz", "kHz", "MHz", "%", "s", "ms", "us"]
PREFIX = ["R", "C", "L", "U", "Q", "D", "J", "P", "TP", "X", "Y", "FB", "SW", "K", "LED",
          "RN", "CN", "TVS", "F", "BT", "MK", "ANT", "TP", "MTG"]
PINROOT = ["GPIO", "VDD", "VSS", "VBAT", "VBUS", "VREG", "XTAL", "VDDBAT", "VDDA", "GND",
           "AGND", "PGND", "DGND", "EN", "RESET", "NRST", "SWDIO", "SWCLK", "TX", "RX",
           "SDA", "SCL", "MISO", "MOSI", "SCK", "CS", "CLK", "TAB", "TOG", "DN1", "DP1",
           "USB", "P_", "P", "IO", "A", "D"]
OKCHARS = set(" %+-./0-9A-Z_a-zΩ∅")


def pick(weighted):
    tot = sum(w for _, _, w in weighted)
    r = random.random() * tot
    acc = 0.0
    for name, path, w in weighted:
        acc += w
        if r <= acc:
            return name, path
    return weighted[-1][0], weighted[-1][1]


def gen_text(role: str, vocab: dict, rng: random.Random, synth_frac: float) -> str:
    pool = vocab.get(role) or []
    if pool and rng.random() > synth_frac:
        return rng.choice(pool)
    if role == "designator":
        return "%s%d%s" % (rng.choice(PREFIX), rng.randint(1, 260),
                           rng.choice(["", "", "", "A", "B"]))
    if role == "pin_number":
        style = rng.random()
        if style < 0.6:
            return str(rng.randint(1, 256))
        if style < 0.8:
            return "%s%d" % (rng.choice("ABP"), rng.randint(1, 64))
        return "%d" % rng.randint(1, 99)
    if role == "value":
        if rng.random() < 0.55:
            num = rng.choice(["%.0f" % rng.randint(1, 999), "%.1f" % (rng.random() * 100),
                              "%.2f" % (rng.random() * 10), "%d.%d" % (rng.randint(1, 99), rng.randint(0, 9))])
            unit = rng.choice(UNITS)
            sep = " " if rng.random() < 0.08 else ""
            return num + sep + unit
        return rng.choice(PINROOT) + str(rng.randint(0, 99))
    if role == "pin_name":
        if rng.random() < 0.5 and pool:
            base = rng.choice(pool)
        else:
            base = rng.choice(PINROOT) + rng.choice(["", "%d" % rng.randint(0, 48),
                                                     "_%d" % rng.randint(0, 12)])
            if rng.random() < 0.25:
                base += rng.choice(["+", "-", "_1", "2"])
        return base[:24]
    return rng.choice(pool) if pool else "NET_%d" % rng.randint(1, 99)


def sanitize(t: str) -> str:
    return "".join(c for c in t if c in OKCHARS)


def load_stats(manifest: Path, max_rows: int = 0):
    vocab = defaultdict(list)
    heights = defaultdict(list)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = sanitize(str(r.get("text") or ""))
        if not t:
            continue
        role = r.get("role")
        vocab[role].append(t)
        bb = r.get("bbox") or []
        if len(bb) == 4:
            h = float(bb[3]) - float(bb[1])
            if 3 <= h <= 40:
                heights[role].append(h)
        if max_rows and sum(len(v) for v in vocab.values()) > max_rows:
            break
    return vocab, heights


def render_text(text: str, font_path: Path, target_h: float) -> Image.Image | None:
    """按目标墨迹高度渲染并做轻增广，返回 RGB 小图。"""
    ss = 4
    size = max(4, int(round(target_h)))
    try:
        f = ImageFont.truetype(str(font_path), size * ss)
    except Exception:
        return None
    pad = 4 * ss
    img = Image.new("L", (max(24, int(len(text) * size * ss * 0.9) + 2 * pad), 4 * size * ss), 255)
    d = ImageDraw.Draw(img)
    try:
        d.text((pad, pad), text, font=f, fill=random.randint(0, 60))
    except Exception:
        return None
    a = np.asarray(img)
    ink = a < 128
    ys, xs = np.where(ink)
    if len(xs) < 3:
        return None
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    cur_h = max(1, y2 - y1)
    scale = target_h / cur_h
    sub = img.crop((max(0, x1 - 1), max(0, y1 - 1), x2 + 1, y2 + 1))
    nw, nh = max(2, int(round(sub.width * scale))), max(2, int(round(sub.height * scale)))
    sub = sub.resize((nw, nh), Image.LANCZOS)
    if random.random() < 0.35:
        sub = sub.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.7)))
    bg = random.choice([255, 255, 250, 245, 240])
    rgb = Image.new("RGB", (nw + 6, nh + 6), (bg, bg, bg))
    rgb.paste(Image.merge("RGB", (sub, sub, sub)), (3, 3))
    if random.random() < 0.25:                      # 淡彩底 / 彩字（官方有彩色值文字）
        tint = random.choice([(230, 230, 250), (250, 240, 230), (235, 245, 235)])
        rgb = Image.blend(rgb, Image.new("RGB", rgb.size, tint), 0.35)
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WORK / "fonttext"))
    ap.add_argument("--manifest", default=str(WORK / "official_text" / "manifest.jsonl"))
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--sheets", type=int, default=4, help="把裁片拼进几张 sheet（0 = 一图一裁片）")
    ap.add_argument("--sheet-side", type=int, default=2048)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--synth-frac", type=float, default=0.4, help="用文法生成（而非照抄真实词汇）的比例")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab, heights = load_stats(Path(a.manifest))
    print("[vocab] " + "  ".join("%s=%d" % (k, len(v)) for k, v in sorted(vocab.items())))
    print("[ink-h] " + "  ".join("%s=%.1f±%.1f" % (k, float(np.mean(v)), float(np.std(v)))
                                 for k, v in sorted(heights.items())))
    roles = [r for r in ROLE_ID if vocab.get(r)]
    variants = list(VARIANT_MIX)

    rows = []
    made = 0
    sheet_i, sheet_img = -1, None
    sx, sy = 6, 6
    row_h = 40                      # 固定行高：裁片最高 ~30px + 边距，保证永不重叠

    def new_sheet():
        nonlocal sheet_i, sheet_img, sx, sy
        if sheet_img is not None:
            sheet_img.save(out / ("sheet_%02d.png" % sheet_i))
        sheet_i += 1
        sheet_img = Image.new("RGB", (a.sheet_side, a.sheet_side), "white")
        sx, sy = 6, 6

    if a.sheets:
        new_sheet()

    while made < a.n:
        role = rng.choice(roles)
        text = gen_text(role, vocab, rng, a.synth_frac)
        if not text or len(text) > 28:
            continue
        variant = rng.choices(list(VARIANT_WEIGHT), weights=[VARIANT_WEIGHT[v] for v in VARIANT_WEIGHT])[0]
        mix = VARIANT_MIX[variant]
        group = rng.choices(list(mix), weights=[mix[g] for g in mix])[0]
        fname, fpath = pick(FONT_GROUPS[group])
        hs = heights.get(role) or [9.0]
        target_h = max(5.0, min(30.0, float(np.random.choice(hs)) * rng.uniform(0.85, 1.15)))
        crop = render_text(text, fpath, target_h)
        if crop is None:
            continue
        if not a.sheets:
            p = out / ("c%06d.png" % made)
            crop.save(p)
            rows.append({"image": str(p), "case": "font", "src": "font",
                         "split": "val" if rng.random() < a.val_frac else "train",
                         "role": role, "role_id": ROLE_ID[role], "text": text,
                         "bbox": [3, 3, crop.width - 3, crop.height - 3]})
            made += 1
            continue
        if sx + crop.width + 4 > a.sheet_side:
            sx, sy = 6, sy + row_h
        if sy + row_h + 4 > a.sheet_side:
            new_sheet()
        py = sy + (row_h - crop.height) // 2
        sheet_img.paste(crop, (sx, py))
        rows.append({"image": str(out / ("sheet_%02d.png" % sheet_i)), "case": "font", "src": "font",
                     "split": "val" if rng.random() < a.val_frac else "train",
                     "role": role, "role_id": ROLE_ID[role], "text": text,
                     "bbox": [sx + 3, py + 3, sx + crop.width - 3, py + crop.height - 3]})
        sx += crop.width + 4
        made += 1
        if made % 20000 == 0:
            print("  ... %d" % made)

    if a.sheets and sheet_img is not None:
        sheet_img.save(out / ("sheet_%02d.png" % sheet_i))
    mf = out / "manifest.jsonl"
    with mf.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    cnt = Counter(r["role"] for r in rows)
    print("写好 %s  n=%d  %s" % (mf, len(rows), dict(cnt)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
