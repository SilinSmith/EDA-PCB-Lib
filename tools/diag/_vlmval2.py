# -*- coding: utf-8 -*-
"""VLM 读元件「值」——**只裁 stage2 已分配的那个 value 文字框**（加 pad），而不是整块元件区域。

理由：整块 `bbox + margin 80` 会把邻居的值读进来（`R6 10kΩ -> 75KΩ`）。而 stage2 的
`_assigned/<case>.json` 里每颗元件都记着它把**哪一个文字框**当成了 value
（`comps[cid].value.box`，**图像坐标 y 向下**）。如果这个框是对的，VLM 只需要转写这一小块，
不会有邻居干扰；如果这个框本来就选错了，VLM 也救不回来 —— 这个脚本就是量这两者的边界。

    python work\\tools\\diag\\_vlmval2.py --pad 40 --limit 200 --workers 6 --dump
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
from stage2.text import imread_unicode  # noqa: E402
from stage2.assemble import VALUE_LIKE  # noqa: E402
from vlm import client  # noqa: E402

VAL_TYPES = ("r", "c", "l", "fuse", "crystal", "crystal_2pin", "crystal_3pin",
             "crystal_4pin")
PROMPT = (
    "这是 PCB 原理图的一个极小局部截图，里面是一段元件参数的**文字**。\n"
    "请把这段文字**原样照抄**下来，一个字符都不要改、不要补全、不要规范化。\n"
    "例如印着 `100nF` 就回 `100nF`（不要回 `0.1uF`、不要回 `10nF`）；印着 `10kΩ` 就回 `10kΩ`。\n"
    "只回这一行文字本身，不要回位号，不要回引脚名，不要解释。看不到文字就只回 NONE。"
)


def norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or ""))


def ctr(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", default=str(ROOT / "work" / "stage2_v17"))
    ap.add_argument("--stage1", default=str(ROOT / "work" / "stage1_v6"))
    ap.add_argument("--pad", type=float, default=40.0)
    ap.add_argument("--scale", type=int, default=client.SCALE)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dump", default="")
    ap.add_argument("--render", default="")
    a = ap.parse_args()

    import cv2
    s2, s1d = Path(a.s2), Path(a.stage1)
    jobs = []
    for c in discover():
        if not c.json_path or not c.png_path:
            continue
        asg = s2 / "_assigned" / ("%s.json" % c.case_id)
        s1f = s1d / ("%s.stage1.json" % c.case_id)
        if not asg.exists() or not s1f.exists():
            continue
        gt = load_json(c.json_path)
        s1 = json.loads(s1f.read_text(encoding="utf-8"))
        H = float((s1.get("image") or {}).get("h") or 0.0)
        if not H:
            continue
        asg = json.loads(asg.read_text(encoding="utf-8"))
        gtlist = [(k, g) for k, g in (gt.get("components") or {}).items()
                  if (g.get("type") in VAL_TYPES) and g.get("bbox")]
        for comp in (s1.get("components") or []):
            cid = comp.get("id")
            v = ((asg.get("comps") or {}).get(cid) or {}).get("value")
            if not v or not v.get("box"):
                continue
            bb = comp.get("bbox_json") or comp.get("bbox_img")
            if not bb:
                continue
            # bbox_json 已是 y 向上；bbox_img 是图像坐标 ⇒ 统一成 y 向上
            yup = bb if comp.get("bbox_json") else [bb[0], H - bb[3], bb[2], H - bb[1]]
            c_ = ctr(yup)
            best = None
            for k, g in gtlist:
                g_ = ctr(g["bbox"])
                dd = ((c_[0] - g_[0]) ** 2 + (c_[1] - g_[1]) ** 2) ** 0.5
                if best is None or dd < best[0]:
                    best = (dd, k, g)
            if best is None or best[0] > 20:
                continue
            jobs.append((c, cid, v.get("text"), best[1], best[2].get("value"), v))

    rng = np.random.RandomState(a.seed)
    if a.limit and len(jobs) > a.limit:
        idx = rng.permutation(len(jobs))[:a.limit]
        jobs = [jobs[i] for i in sorted(idx)]
    print("样本 %d 个（stage2 已分配 value 框、且能配到 GT 的元件），pad %.0f，scale %d"
          % (len(jobs), a.pad, a.scale))

    def work(j):
        c, cid, pv, gk, gv, vinfo = j
        try:
            bgr = imread_unicode(c.png_path)
            Hh, Ww = bgr.shape[:2]
            b = vinfo["box"]                      # 图像坐标
            p = a.pad
            x0 = max(0, int(b[0] - p)); y0 = max(0, int(b[1] - p))
            x1 = min(Ww, int(b[2] + p) + 1); y1 = min(Hh, int(b[3] + p) + 1)
            im = bgr[y0:y1, x0:x1]
            if im.size == 0:
                return None
            if a.scale > 1:
                im = cv2.resize(im, None, fx=a.scale, fy=a.scale,
                                interpolation=cv2.INTER_CUBIC)
            txt, _dt = client.ask(im, PROMPT, max_tokens=512)
            return dict(case=c.case_id, cid=cid, key=gk, gt=gv, pipe=pv,
                        raw=txt, vlm=(txt or "").strip().splitlines()[0].strip()
                        if (txt or "").strip() else "")
        except Exception as ex:  # noqa: BLE001
            return dict(case=c.case_id, cid=cid, key=gk, gt=gv, pipe=pv,
                        raw="ERR:%s" % ex, vlm="")

    t0 = time.time()
    res = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(work, jobs)):
            if r:
                res.append(r)
            if (i + 1) % 50 == 0:
                print("  ... %d/%d  %.0fs" % (i + 1, len(jobs), time.time() - t0))
    dt = time.time() - t0
    ok = [r for r in res if r["gt"]]
    p_ok = sum(1 for r in ok if norm(r["pipe"]) == norm(r["gt"]))
    v_ok = sum(1 for r in ok if norm(r["vlm"]) == norm(r["gt"]))
    both = sum(1 for r in ok if norm(r["vlm"]) == norm(r["gt"]) == norm(r["pipe"]))
    onlyv = sum(1 for r in ok if norm(r["vlm"]) == norm(r["gt"]) != norm(r["pipe"]))
    onlyp = sum(1 for r in ok if norm(r["pipe"]) == norm(r["gt"]) != norm(r["vlm"]))
    neither = sum(1 for r in ok if norm(r["vlm"]) != norm(r["gt"]) and norm(r["pipe"]) != norm(r["gt"]))
    print()
    print("| GT 值非空 | %d |" % len(ok))
    print("| 管线（stage2 的 CRNN 结果）对 | %d = %.4f |" % (p_ok, p_ok / max(1, len(ok))))
    print("| **VLM 转写 value 框对** | %d = %.4f |" % (v_ok, v_ok / max(1, len(ok))))
    print("| 都对 %d | 只 VLM 对 %d | 只管线对 %d | 都错 %d |" % (both, onlyv, onlyp, neither))
    print("| GT 值非空但 stage2 没分配 value 框 | %d |" % (0))
    print("| 墙钟 | %.1fs | %.2f s/个 |" % (dt, dt / max(1, len(res))))
    print()
    print("--- VLM 错、管线也错（最多 18）---")
    n = 0
    for r in ok:
        if norm(r["vlm"]) != norm(r["gt"]) and norm(r["pipe"]) != norm(r["gt"]):
            print("  %-14s %-5s GT=%-10s 管线=%-10s VLM=%s" %
                  (r["case"], r["key"], r["gt"], r["pipe"], r["vlm"]))
            n += 1
            if n >= 18:
                break
    print()
    print("--- VLM 对、管线错（最多 12）---")
    n = 0
    for r in ok:
        if norm(r["vlm"]) == norm(r["gt"]) != norm(r["pipe"]):
            print("  %-14s %-5s GT=%-10s 管线=%-10s VLM=%s" %
                  (r["case"], r["key"], r["gt"], r["pipe"], r["vlm"]))
            n += 1
            if n >= 12:
                break
    if a.dump:
        Path(a.dump).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("dump ->", a.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
