# -*- coding: utf-8 -*-
"""只量**键已经匹配上**的 r/c 元件（也就是真正会被 `component_text_ok` 比较的那批）
上，管线与 VLM 的 value 逐字命中率。

`_vlmval.py` 是按 bbox IoU 配对的，和「键匹配」不是同一批元件；而
`component_text_ok` 只在键匹配（或坐标匹配）的配对上跑。所以要看值覆盖能不能赚分，
必须在这批元件上比。

    python work\\tools\\diag\\_vlmval3.py --pred work\\stage2_v17 --margin 80 --pv v2 --limit 0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
import numpy as np  # noqa: E402
from schematic_parser.data import discover, load_json  # noqa: E402
from schematic_parser.metrics import EvalConfig, assign_components, canon_type  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402
from vlm import client  # noqa: E402

PROMPT2 = (
    "这是 PCB 原理图的一个局部截图，画面中心是一个分立元件（电阻/电容/电感等）。\n"
    "请**先找到它的位号**（如 C5、R12），再读出它的**参数值**。只回一行，格式：\n"
    "位号=值\n"
    "四条硬约束：\n"
    "1. 值的字符**原样照抄**，一个字符都不许改、不许规范化 —— 印着 `100nF` 就写 `100nF`，"
    "不许写成 `0.1uF`、不许写成 `10nF`；印着 `10kΩ` 就写 `10kΩ`。\n"
    "2. 绝对不要把容差/耐压那一行（`10%`、`16V`、`50V` 这样的）当成值。\n"
    "3. 只报**画面正中心**那个元件自己的值，不要报旁边元件的。\n"
    "4. 实在找不到值就写 位号=NONE。"
)
cfg = EvalConfig()


def norm(s):
    return re.sub(r"\s+", "", (s or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "stage2_v17"))
    ap.add_argument("--margin", type=float, default=80.0)
    ap.add_argument("--scale", type=int, default=client.SCALE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--mdump", default=str(ROOT / "work" / "_vlmval3.json"))
    a = ap.parse_args()

    import cv2
    pred_dir = Path(a.pred)
    jobs = []
    for c in discover():
        if not c.json_path or not c.png_path:
            continue
        f = pred_dir / ("%s.json" % c.case_id)
        if not f.exists():
            continue
        gt = load_json(c.json_path)
        pr = json.loads(f.read_text(encoding="utf-8"))
        _f1, keymap = assign_components(gt, pr, cfg, {})
        for gk, pk in keymap.items():
            g = (gt.get("components") or {}).get(gk) or {}
            p = (pr.get("components") or {}).get(pk) or {}
            if canon_type(g.get("type")) not in ("r", "c"):
                continue
            if not g.get("value") or not p.get("bbox"):
                continue
            jobs.append((c, gk, p["bbox"], g["value"], p.get("value")))
    jobs.sort(key=lambda x: (x[0].case_id, x[1]))
    rng = np.random.RandomState(a.seed)
    if a.limit and len(jobs) > a.limit:
        idx = rng.permutation(len(jobs))[:a.limit]
        jobs = [jobs[i] for i in sorted(idx)]
    print("键匹配上的 r/c 且 GT 值非空：%d 个（margin %.0f）" % (len(jobs), a.margin))

    def work(j):
        c, gk, bb, gv, pv = j
        try:
            bgr = imread_unicode(c.png_path)
            H, W = bgr.shape[:2]
            ib = (bb[0], H - bb[3], bb[2], H - bb[1])      # y 向上 -> 图像坐标
            m = a.margin
            x0 = max(0, int(ib[0] - m)); y0 = max(0, int(ib[1] - m))
            x1 = min(W, int(ib[2] + m) + 1); y1 = min(H, int(ib[3] + m) + 1)
            im = bgr[y0:y1, x0:x1]
            if im.size == 0:
                return None
            if a.scale > 1:
                im = cv2.resize(im, None, fx=a.scale, fy=a.scale,
                                interpolation=cv2.INTER_CUBIC)
            txt, _dt = client.ask(im, PROMPT2, max_tokens=1024)
            first = (txt or "").strip().splitlines()
            first = first[0].strip() if first else ""
            val = first.split("=", 1)[1].strip() if "=" in first else first
            return dict(case=c.case_id, key=gk, gt=gv, pipe=pv, raw=first, vlm=val)
        except Exception as ex:  # noqa: BLE001
            return dict(case=c.case_id, key=gk, gt=gv, pipe=pv, raw="ERR:%s" % ex, vlm="")

    t0 = time.time()
    res = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(work, jobs)):
            if r:
                res.append(r)
            if (i + 1) % 100 == 0:
                print("  ... %d/%d  %.0fs" % (i + 1, len(jobs), time.time() - t0))
    dt = time.time() - t0
    n = len(res)
    p_ok = sum(1 for r in res if norm(r["pipe"]) == norm(r["gt"]))
    v_ok = sum(1 for r in res if norm(r["vlm"]) == norm(r["gt"]))
    both = sum(1 for r in res if norm(r["vlm"]) == norm(r["gt"]) == norm(r["pipe"]))
    onlyv = sum(1 for r in res if norm(r["vlm"]) == norm(r["gt"]) != norm(r["pipe"]))
    onlyp = sum(1 for r in res if norm(r["pipe"]) == norm(r["gt"]) != norm(r["vlm"]))
    print()
    print("| 键匹配上的 r/c（GT 值非空） | %d |" % n)
    print("| 管线对 | %d = %.4f |" % (p_ok, p_ok / max(1, n)))
    print("| **VLM 对** | %d = %.4f |" % (v_ok, v_ok / max(1, n)))
    print("| 都对 %d | 只 VLM 对 %d | 只管线对 %d | 都错 %d |"
          % (both, onlyv, onlyp, n - both - onlyv - onlyp))
    print("| 墙钟 %.1fs | %.2f s/个 |" % (dt, dt / max(1, n)))
    print()
    print("--- 只管线对（覆盖会改坏）最多 15 ---")
    k = 0
    for r in res:
        if norm(r["pipe"]) == norm(r["gt"]) != norm(r["vlm"]):
            print("  %-14s %-5s GT=%-12s 管线=%-12s VLM=%s" % (r["case"], r["key"], r["gt"], r["pipe"], r["vlm"]))
            k += 1
            if k >= 15:
                break
    print()
    print("--- 只 VLM 对（覆盖会改好）最多 15 ---")
    k = 0
    for r in res:
        if norm(r["vlm"]) == norm(r["gt"]) != norm(r["pipe"]):
            print("  %-14s %-5s GT=%-12s 管线=%-12s VLM=%s" % (r["case"], r["key"], r["gt"], r["pipe"], r["vlm"]))
            k += 1
            if k >= 15:
                break
    print()
    print("--- VLM 错得最离谱（都错）最多 12 ---")
    k = 0
    for r in res:
        if norm(r["vlm"]) != norm(r["gt"]) and norm(r["pipe"]) != norm(r["gt"]):
            print("  %-14s %-5s GT=%-12s 管线=%-12s VLM=%s" % (r["case"], r["key"], r["gt"], r["pipe"], r["vlm"]))
            k += 1
            if k >= 12:
                break
    if a.mdump:
        Path(a.mdump).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("dump ->", a.mdump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
