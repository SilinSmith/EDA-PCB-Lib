# -*- coding: utf-8 -*-
"""对照图：同一个框、同一个模型，在**两条预处理路径**下的读数。

左 = 训练侧口径（框原样喂给 `crop_gray`）
右 = 线上口径（`stage2/text.py:pad_box` 先按 0.30h+2px 外扩，再交给 `crop_gray`）

渲染的是**模型真正吃到的那张 32px 输入**（放大 6 倍），所以能直接看出字形被压小了多少。
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EDA = Path(__file__).resolve().parents[3] / "eda-pcb"
sys.path.insert(0, str(EDA))

import cv2                                                     # noqa: E402
from stage2.text import TextEngine, pad_box, imread_unicode    # noqa: E402
from textrec.dataset import crop_gray                          # noqa: E402


def render(img_pil, bbox, up=6):
    c = crop_gray(img_pil, bbox)
    a = np.asarray(c, dtype=np.uint8)
    h, w = a.shape
    big = cv2.resize(a, (max(1, w * up), h * up), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((big.shape[0] + 24, big.shape[1] + 8, 3), 255, np.uint8)
    canvas[24:24 + big.shape[0], 4:4 + big.shape[1]] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    return canvas


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    man = [json.loads(l) for l in
           Path(r"Y:\workspace\EDA\work\ocr_real\manifest.jsonl").read_text(encoding="utf-8").splitlines()
           if l.strip()]
    val = [m for m in man if m.get("split") == "val"]
    eng = TextEngine()
    rec = eng.recognizer()
    by = collections.defaultdict(list)
    for m in val:
        by[m["image"]].append(m)

    pairs = []
    for img_path, ms in by.items():
        img = imread_unicode(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        pil = Image.open(img_path).convert("L")
        online = eng.read(img, [m["bbox"] for m in ms], [m["role"] for m in ms])
        raw = rec.read_many(gray, [(tuple(m["bbox"]), m["role"]) for m in ms])
        for m, o, r in zip(ms, online, raw):
            oo = (o.get("reads") or {}).get(m["role"])
            oo = oo[0] if oo else ""
            if oo != m["text"] and r[0] == m["text"]:
                pairs.append((m, pil, oo, r[0]))

    # 每个角色挑几条
    pick = []
    seen = collections.Counter()
    for m, pil, oo, rr in pairs:
        if seen[m["role"]] >= limit // 2:
            continue
        seen[m["role"]] += 1
        pick.append((m, pil, oo, rr))
        if len(pick) >= limit:
            break

    tiles = []
    for m, pil, oo, rr in pick:
        left = render(pil, m["bbox"])
        right = render(pil, pad_box(tuple(m["bbox"])))
        h = max(left.shape[0], right.shape[0])
        w = left.shape[1] + right.shape[1] + 14
        cv = np.full((h, w, 3), 255, np.uint8)
        cv[:left.shape[0], :left.shape[1]] = left
        cv[:right.shape[0], left.shape[1] + 14:left.shape[1] + 14 + right.shape[1]] = right
        head = np.full((20, w, 3), 255, np.uint8)
        cv2.putText(head, "GT=%s | train=%s | online=%s"
                    % (m["text"][:12], rr[:12], (oo or "-")[:12]),
                    (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1, cv2.LINE_AA)
        tiles.append(np.vstack([head, cv]))

    tw = max(t.shape[1] for t in tiles)
    out = np.full((sum(t.shape[0] + 6 for t in tiles), tw, 3), 245, np.uint8)
    y = 0
    for t in tiles:
        out[y:y + t.shape[0], :t.shape[1]] = t
        y += t.shape[0] + 6
    dst = Path(r"Y:\workspace\EDA\work\preview\_padab.png")
    cv2.imwrite(str(dst), out)
    print("候选 %d 条，渲染 %d 条 -> %s" % (len(pairs), len(tiles), dst))
    for m, pil, oo, rr in pick:
        print("   %-12s GT=%-12s train=%-12s online=%s" % (m["role"], m["text"], rr, oo or "-"))


if __name__ == "__main__":
    main()
