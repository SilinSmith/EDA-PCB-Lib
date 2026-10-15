# -*- coding: utf-8 -*-
"""量 `stage1.wires._hint_wire_hue` 的真实代价：整图 vs 缩到 1024，以及有多少张图会触发它。

背景：`_hint_wire_hue` 在 **stage1** 里被无条件调用（`extract_wires` 开头），
判据是「彩度像素 >= 200 个」。原来把整图发给 VLM，14 MPx 的图单次 **303 s**。

用法：
    python work\\tools\\diag\\_huetime.py 0003 0005 0077
    python work\\tools\\diag\\_huetime.py --count          # 只数触发张数（不调 VLM）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

import cv2                                                              # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover           # noqa: E402
from stage2_eval import build_index, resolve                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--reps", type=int, default=1, help="每个组合重复几次（分辨冷启动/热）")
    a = ap.parse_args()

    from pipeline import pskel
    from vlm import client

    if a.count:
        n_hit = n_gray = 0
        rows = []
        for c in discover(DEFAULT_DATA_ROOT):
            bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            try:
                npx = int((pskel.chroma(bgr) >= 22.0).sum())
            except Exception:
                npx = -1
            if npx >= 200:
                n_hit += 1
                rows.append((w * h, c.case_id, w, h, npx))
            else:
                n_gray += 1
        rows.sort(reverse=True)
        print("会触发色相询问的图 %d 张 / 灰度跳过 %d 张（共 %d）" % (
            n_hit, n_gray, n_hit + n_gray))
        print("其中最大的 12 张：")
        for a_, cid, w, h, npx in rows[:12]:
            print("  %-8s %5dx%-5d %.1f MPx  彩度像素 %d" % (cid, w, h, a_ / 1e6, npx))
        return 0

    exact, digits = build_index(DEFAULT_DATA_ROOT)
    from stage1 import wires as WR
    for cid in (a.cases or ["0003"]):
        case, _ = resolve(cid, exact, digits)
        if case is None:
            print("找不到", cid)
            continue
        bgr = cv2.imread(str(case.png_path), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        small = WR._hue_small(bgr)
        print("== %s %dx%d (%.1f MPx) -> 缩放后 %dx%d" % (
            cid, w, h, w * h / 1e6, small.shape[1], small.shape[0]))
        for i in range(a.reps):
            for name, img in (("原图", bgr), ("缩到1024", small)):
                t0 = time.time()
                txt = client.ask(img, WR.WIRE_HUE_PROMPT, max_tokens=24)
                print("   [%d] %-10s %6.2fs  -> %r" % (i, name, time.time() - t0, txt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
