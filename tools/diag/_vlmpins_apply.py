# -*- coding: utf-8 -*-
"""用 VLM 给 stage2 的预测补上两件从没读对过的东西：**IC 位号** 与 **IC 引脚号/名**。

为什么必须一起做（`_vlmdesig.py` / `_pincover.py` / `_pingrid.py` 实测）：

* `pin_f1`（权重 0.25）的 `flat()` 里有 `if comp not in box_comps: continue`，
  而 `box_comps` 取自 **GT** 的 box 元件键。我们 box 键命中率只有 **0.309**
  ⇒ 引脚读得再准，键不对就一分不得（第一次只补引脚：tp 69 → 54）。
* VLM 读位号（裁片 = 预测 bbox 外扩 60，坐标翻正后）**0.65 与 GT 键一致**，
  读出非空 0.75，0.16 s/个。管线现在靠 OCR + `canon_designator`，只有 0.309。
* VLM 逐引脚读（每个已检出引脚点裁一个不对称窗口、打红圈、8 个拼网格）
  **编号 0.9697 / 编号+名称 0.8636**，0.81 s/元件。管线现在 `pin` 召回 0.0074。

坐标坑（踩过一次）：预测的 `bbox` / `point` 是**左下角 y 向上**，与 GT 一致；
拿去索引图像（y 向下）前必须 `y_img = H - y`，否则裁片上下镜像、读出来全是垃圾。

    cd Y:\\workspace\\EDA
    python work\\tools\\diag\\_vlmpins_apply.py --in work\\stage2_v17 --out work\\_vp --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "diag"))

from schematic_parser.data import discover  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402
from _vlmread import ask_vlm  # noqa: E402
from _pingrid import (COLORS, PROMPT, adaptive_window, mark, parse_grid,  # noqa: E402
                      tile)

IC = ("box", "ic", "boxic")
DESIG_PROMPT = (
    "这是电路原理图里的**一个元器件**（IC / 模块）。\n"
    "在该元器件外框**之外**（通常紧贴右上角或左上角）印着它的**位号**，"
    "形如「字母前缀 + 数字」，例如 U1、U5、IC2、J3、SW1。\n"
    "只回那个位号本身，不要解释、不要换行、不要任何其它字符。\n"
    "如果图里印的是 `U?`、`J?` 这种带问号的未标注形式，就原样回 `U?`。\n"
    "完全找不到就回 NONE。")
DESIG_RE = re.compile(r"^[A-Za-z]{1,3}\d{1,4}$")
FKEY_RE = re.compile(r"^p\d{3,4}$")


def _crop(bgr, pt, center, outer, inner, cross):
    h, w = bgr.shape[:2]
    px, py = float(pt[0]), float(pt[1])
    dx, dy = float(center[0]) - px, float(center[1]) - py
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-3:
        dx, dy, n = 1.0, 0.0, 1.0
    ux, uy = dx / n, dy / n
    ax0, ay0 = px - ux * outer, py - uy * outer
    ax1, ay1 = px + ux * inner, py + uy * inner
    x0 = max(0, int(min(ax0, ax1) - cross)); x1 = min(w, int(max(ax0, ax1) + cross))
    y0 = max(0, int(min(ay0, ay1) - cross)); y1 = min(h, int(max(ay0, ay1) + cross))
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr[y0:y1, x0:x1].copy(), (px - x0, py - y0)


def _crop_box(bgr, ib, margin):
    h, w = bgr.shape[:2]
    x0 = max(0, int(ib[0] - margin)); y0 = max(0, int(ib[1] - margin))
    x1 = min(w, int(ib[2] + margin)); y1 = min(h, int(ib[3] + margin))
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr[y0:y1, x0:x1]


def read_designator(bgr, ib, args, stat):
    im = _crop_box(bgr, ib, args.desig_margin)
    if im is None:
        return None
    txt, _dt = ask_vlm(im, prompt=DESIG_PROMPT, model=args.model)
    stat[0] += 1
    t = re.sub(r"[^0-9A-Za-z?]", "", (txt or "").strip().splitlines()[0] if (txt or "").strip() else "")
    return t or None


def read_pins(bgr, points, ctr, outer, inner, cross, args, stat):
    crops, rels = [], []
    for q in points:
        c = _crop(bgr, q, ctr, outer, inner, cross)
        if c is None:
            crops.append(None); rels.append(None)
            continue
        im, rel = c
        mark(im, rel, args.mr, args.mt, COLORS[args.color])
        crops.append(im); rels.append(rel)
    ans = {}
    for s in range(0, len(crops), args.per):
        sub, subidx = [], []
        for i in range(s, min(s + args.per, len(crops))):
            if crops[i] is not None:
                sub.append(crops[i]); subidx.append(i)
        if not sub:
            continue
        g = tile(sub, cols=args.cols)
        txt, _dt = ask_vlm(g, prompt=PROMPT, model=args.model)
        stat[1] += 1
        for i, (num, name) in parse_grid(txt).items():
            if 0 <= i < len(sub):
                ans[subidx[i]] = (num, name)
    return ans


def enrich_case(pred_path, outdir, png_path, args, log):
    pred = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    cid = pred.get("case") or Path(pred_path).stem
    if png_path is None:
        shutil.copyfile(pred_path, Path(outdir) / Path(pred_path).name)
        return "no_png"
    bgr = imread_unicode(png_path)
    H = float(bgr.shape[0])
    comps = pred.get("components") or {}
    pins = pred.get("pins") or {}
    stat = [0, 0]
    n_ren = n_pin = 0
    renames = {}

    for key, cv_ in list(comps.items()):
        if str(cv_.get("type") or "").lower() not in IC:
            continue
        bb = cv_.get("bbox")
        if not bb:
            continue
        ibl = (bb[0], H - bb[3], bb[2], H - bb[1])
        if args.designator:
            got = read_designator(bgr, ibl, args, stat)
            if got and DESIG_RE.match(got) and got != key and got not in comps:
                renames[key] = got
                n_ren += 1
    # 改名：components / pins / nets.hyperGraph
    if renames:
        comps = {(renames.get(k, k)): v for k, v in comps.items()}
        pins = {(renames.get(k, k)): v for k, v in pins.items()}
        for net in (pred.get("nets") or {}).values():
            hg = net.get("hyperGraph")
            if isinstance(hg, str):
                for old, new in renames.items():
                    hg = re.sub(r"(?<![0-9A-Za-z_])%s\." % re.escape(old), "%s." % new, hg)
                net["hyperGraph"] = hg
    # 引脚：对改名后的每个 IC 元件，逐点问 VLM
    changed = False
    for key, cv_ in comps.items():
        if str(cv_.get("type") or "").lower() not in IC:
            continue
        cur = pins.get(key) or {}
        if len(cur) < args.min_pins:
            continue
        bb = cv_.get("bbox")
        if not bb:
            continue
        points, keep = [], []
        for pk, info in cur.items():
            pt = (info or {}).get("point") or {}
            try:
                points.append((float(pt["x"]), H - float(pt["y"])))
                keep.append((pk, info))
            except Exception:  # noqa: BLE001
                pass
        if not points:
            continue
        outer, inner, cross = adaptive_window(bb, args.inner, args.cross, args.outer, args.ref)
        ctr = ((bb[0] + bb[2]) / 2.0, H - (bb[1] + bb[3]) / 2.0)
        ans = read_pins(bgr, points, ctr, outer, inner, cross, args, stat)
        # 合并：VLM 读到的号优先；读不到就**原样保留**（绝不丢引脚——丢一个就多一个
        # `pin.missing`，实测会让 `pin.missing` 302→308、总分倒退）。
        newpins = {}
        used = set()
        for i, (pk, info) in enumerate(keep):
            num = name = ""
            if i in ans:
                num, name = (ans[i][0] or "").strip(), (ans[i][1] or "").strip()
            old_name = str((info or {}).get("pinname") or "")
            if num and num in used:
                num = ""
            if num:
                used.add(num)
                newpins["pin_%s" % num] = {"pinname": name or old_name, "point": info["point"]}
                n_pin += 1
                changed = True
            else:
                newpins[pk] = info
        if changed and newpins:
            pins[key] = newpins
    pred["components"] = comps
    pred["pins"] = pins
    (Path(outdir) / ("%s.json" % cid)).write_text(
        json.dumps(pred, ensure_ascii=False), encoding="utf-8")
    log.append((cid, n_ren, stat[0], stat[1], n_pin))
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-desig", dest="designator", action="store_false", default=True)
    ap.add_argument("--desig-margin", type=float, default=60.0)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--min-pins", type=int, default=2)
    ap.add_argument("--outer", type=float, default=34.0)
    ap.add_argument("--inner", type=float, default=78.0)
    ap.add_argument("--cross", type=float, default=58.0)
    ap.add_argument("--ref", type=float, default=762.0)
    ap.add_argument("--mr", type=int, default=4)
    ap.add_argument("--mt", type=int, default=2)
    ap.add_argument("--color", default="red", choices=tuple(COLORS))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3.5-9b"))
    a = ap.parse_args(argv)

    indir = Path(a.indir); outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pngs = {}
    for c in discover(splits=None) if False else discover():
        pngs[c.case_id] = c.png_path
    wanted = {s.strip() for s in a.cases.split(",") if s.strip()}
    jobs = []
    for f in sorted(indir.glob("*.json")):
        if wanted and f.stem not in wanted:
            continue
        jobs.append((str(f), pngs.get(f.stem)))
        if a.limit and len(jobs) >= a.limit:
            break
    for f in sorted(indir.glob("*.json")):
        outdir.joinpath("%s.json" % f.stem)
    print("待处理 %d 张" % len(jobs))
    log = []
    t0 = time.time()

    def work(job):
        try:
            return enrich_case(job[0], str(outdir), job[1], a, log)
        except Exception as e:  # noqa: BLE001
            print("  !! %s: %r" % (Path(job[0]).stem, e))
            shutil.copyfile(job[0], outdir / Path(job[0]).name)
            return repr(e)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, jobs))
    wall = time.time() - t0
    print("\n完成 %d 张，墙钟 %.1f s（%.2f s/张）" % (len(log), wall, wall / max(1, len(log))))
    print("  改名元件 %d 个｜位号调用 %d 次｜网格调用 %d 次｜改写引脚 %d 个"
          % (sum(r[1] for r in log), sum(r[2] for r in log),
             sum(r[3] for r in log), sum(r[4] for r in log)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
