# -*- coding: utf-8 -*-
"""VLM 网格批量读文字：把 N 个裁片排成带编号的网格，一次调用读回全部。

动机：单裁片一次调用 ~6-8 s（qwen3.5-9b 带思维链），一张图 ~500 个文字框根本付不起。
若网格能把**单裁片**成本压到 <1 s，VLM 才有资格进管线。

    cd Y:\\workspace\\EDA
    python work\\tools\\diag\\_vlmgrid.py --limit 12 --per 32 --grid 8
    python work\\tools\\diag\\_vlmgrid.py --limit 12 --per 32 --grid 16 --which det
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "work" / "tools" / "diag"))

from _vlmread import MODEL, ask_vlm, collect  # noqa: E402

GRID_PROMPT = (
    "图里是一个 %d 行 %d 列的表格。每个格子的**左侧**有一个黑底白字的编号，"
    "格子内容是取自电路原理图的一小段文字。\n"
    "请逐格读出格子内容里的文字（不要读编号），只输出一个 JSON 数组，形如\n"
    "[{\"id\": 编号, \"text\": \"该格文字\"}, ...]\n"
    "看不清的格子 text 写 \"?\"。除了这个 JSON 数组不要输出任何其它内容。")


def make_grid(crops, cols, cell_h=64, label_w=54, pad=8, max_w=760):
    n = len(crops)
    rows = (n + cols - 1) // cols
    cells = []
    for c in crops:
        h, w = c.shape[:2]
        if h <= 0 or w <= 0:
            cells.append(np.full((cell_h, 40, 3), 255, np.uint8))
            continue
        sc = cell_h / float(h)
        nw = int(round(w * sc))
        nw = max(6, min(nw, max_w))
        cells.append(cv2.resize(c, (nw, cell_h), interpolation=cv2.INTER_CUBIC))
    cell_w = max(x.shape[1] for x in cells) + label_w + pad * 2
    canvas = np.full((rows * (cell_h + pad * 2) + pad,
                      cols * cell_w + pad, 3), 255, np.uint8)
    for k, x in enumerate(cells):
        r, cc = divmod(k, cols)
        y0 = pad + r * (cell_h + pad * 2)
        x0 = pad + cc * cell_w
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 2, y0 + cell_h + pad),
                      (210, 210, 210), 1)
        cv2.rectangle(canvas, (x0 + 2, y0), (x0 + label_w - 4, y0 + cell_h),
                      (0, 0, 0), -1)
        txt = str(k + 1)
        fs = 1.5 if len(txt) == 1 else 1.15
        cv2.putText(canvas, txt, (x0 + 8, y0 + cell_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
        canvas[y0:y0 + cell_h, x0 + label_w:x0 + label_w + x.shape[1]] = x
    return canvas, rows, cols


def parse_grid(txt, n):
    t = (txt or "").strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", t).strip()
    out = {}
    obj = None
    for pat in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pat, t, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                break
            except Exception:  # noqa: BLE001
                obj = None
    if obj is None:
        return out
    if isinstance(obj, dict):
        obj = obj.get("items") or obj.get("results") or list(obj.values())
    if not isinstance(obj, list):
        return out
    for it in obj:
        if not isinstance(it, dict):
            continue
        k = it.get("id", it.get("index", it.get("编号")))
        v = it.get("text", it.get("content", it.get("文字", "")))
        try:
            k = int(k)
        except Exception:  # noqa: BLE001
            continue
        if 1 <= k <= n:
            out[k] = str(v).strip()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--per", type=int, default=32)
    ap.add_argument("--pad", type=float, default=10.0)
    ap.add_argument("--grid", type=int, default=8, help="每次调用装几个裁片")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--which", default="tight", choices=("tight", "det"))
    ap.add_argument("--model", default="")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cell", type=int, default=64)
    ap.add_argument("--save", default="")
    ap.add_argument("--show", type=int, default=0, help="打印前 N 次调用的原始回答")
    a = ap.parse_args(argv)

    rows = collect(a)
    if a.which == "det":
        rows = [r for r in rows if r.get("det") is not None]
    n = len(rows)
    print("样本 %d 个（%d 张图），网格 %d/次，which=%s"
          % (n, len({r["case"] for r in rows}), a.grid, a.which))
    if not n:
        return 1

    batches = [rows[i:i + a.grid] for i in range(0, n, a.grid)]
    made = [make_grid([r[a.which] for r in b], a.cols, a.cell) for b in batches]
    grids = [m[0] for m in made]
    if a.save:
        cv2.imwrite(a.save, grids[0])
        print("saved:", a.save, grids[0].shape)

    prompt = GRID_PROMPT % (((len(batches[0]) + a.cols - 1) // a.cols), a.cols)

    def job(pair):
        b, g = pair
        t, dt = ask_vlm(g, prompt=prompt, model=a.model or MODEL)
        return b, t, dt, parse_grid(t, len(b))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(job, zip(batches, grids)))
    wall = time.time() - t0

    got = {}
    ncall_ok = 0
    for k, (b, t, dt, m) in enumerate(res):
        if m:
            ncall_ok += 1
        if a.show and k < a.show:
            print("\n--- 调用 %d 原始回答（%.1f s, 解析出 %d 项）---\n%s"
                  % (k, dt, len(m), (t or "")[:1200]))
        for r in b:
            got[r["i"]] = m.get(r["i"] - b[0]["i"] + 1, "")
    ok = sum(1 for r in rows if got.get(r["i"], "") == r["txt"])
    print("\n调用 %d 次，解析出内容 %d 次，墙钟 %.1f s（%.2f s/裁片，%.1f s/次）"
          % (len(res), ncall_ok, wall, wall / n, wall / len(res)))
    print("=== 网格批量 exact ===")
    print("  vlm(grid %d, %s)  %5d/%d = %.4f"
          % (a.grid, a.which, ok, n, ok / max(1, n)))

    print("\n前 20 条：")
    for r in rows[:20]:
        print("  %-4d GT=%-14s got=%-14s %s"
              % (r["i"], r["txt"][:14], str(got.get(r["i"], ""))[:14],
                 "OK" if got.get(r["i"], "") == r["txt"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
