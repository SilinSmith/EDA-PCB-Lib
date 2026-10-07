# -*- coding: utf-8 -*-
"""VLM 读 IC 的**器件型号**（GT 的 `Name` 字段）—— 用来中和 box 那 276 个 text_mismatch。

GT 里 `box` 元件的 `Name` 就是芯片型号（`MSP432E401YTPDT`/`AMS1117-3.3`/`78M12`），
而 `component_text_ok` 对 IC_TYPES 比的就是这个字段。我们现在的 `Name` 是别的文字串
（`VCC1`、`PA14/200Ω`），996 个 box 只有 2 个 tp。

    python work\\tools\\diag\\_vlmname.py --limit 0 --margin 8 --workers 6 --dump
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

PROMPT = (
    "这是 PCB 原理图里一个集成电路(IC)的局部截图。\n"
    "请读出这颗芯片上印的**器件型号 / part number**（例如 MSP432E401YTPDT、AMS1117-3.3、78M12、STM32F103C8T6）。\n"
    "要求：\n"
    "1. 只回型号本身，不要回位号（如 U1/R3/C5），不要回引脚名（如 VDD/PA14），不要回参数（如 100nF）。\n"
    "2. 型号里的字母数字原样照抄，不要补全、不要规范化、不要猜。\n"
    "3. 如果有多个候选，每行一个，最可能的放第一行。\n"
    "4. 图上确实没有印型号就只回 NONE。"
)
BOX = ("box", "ic", "boxic")


def norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).upper()


def parse(txt: str):
    out = []
    for ln in (txt or "").splitlines():
        s = ln.strip().strip("`*#- ").strip()
        s = re.sub(r"^\d+[.、)]\s*", "", s)
        if not s or s.upper() == "NONE":
            continue
        if VALUE_LIKE.search(s):
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "stage2_v17"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--margin", type=float, default=8.0)
    ap.add_argument("--scale", type=int, default=client.SCALE)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()

    import cv2
    rng = np.random.RandomState(a.seed)
    jobs = []
    for c in discover():
        if not c.json_path or not c.png_path:
            continue
        gt = load_json(c.json_path)
        for k, g in (gt.get("components") or {}).items():
            if (g.get("type") or "") not in BOX or not g.get("bbox"):
                continue
            nm = g.get("Name") or g.get("name")
            jobs.append((c, k, g["bbox"], nm))
    jobs.sort(key=lambda x: (x[0].case_id, x[1]))
    if a.limit:
        idx = rng.permutation(len(jobs))[:a.limit]
        jobs = [jobs[i] for i in sorted(idx)]
    print("待测 GT box 元件 %d 个，margin %.0f，scale %d" % (len(jobs), a.margin, a.scale))

    def work(j):
        c, k, b, nm = j
        try:
            bgr = imread_unicode(c.png_path)
            H = float(bgr.shape[0])
            ib = (b[0], H - b[3], b[2], H - b[1])          # GT 是左下 y 向上
            m = a.margin
            x0 = max(0, int(ib[0] - m)); y0 = max(0, int(ib[1] - m))
            x1 = min(bgr.shape[1], int(ib[2] + m) + 1); y1 = min(bgr.shape[0], int(ib[3] + m) + 1)
            if x1 - x0 < 6 or y1 - y0 < 6:
                return None
            im = bgr[y0:y1, x0:x1]
            if a.scale > 1:
                im = cv2.resize(im, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_CUBIC)
            txt, _dt = client.ask(im, PROMPT, max_tokens=512)
            return dict(case=c.case_id, key=k, gt=nm, raw=txt, cands=parse(txt))
        except Exception as ex:  # noqa: BLE001
            return dict(case=c.case_id, key=k, gt=nm, raw="ERR:%s" % ex, cands=[])

    t0 = time.time()
    res = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(work, jobs)):
            if r:
                res.append(r)
            if (i + 1) % 50 == 0:
                print("  ... %d/%d  %.0fs" % (i + 1, len(jobs), time.time() - t0))
    dt = time.time() - t0
    def variants(r):
        """模型可能把 `MAX14571 MAX14572 MAX14573` 拆成多行 ⇒ 也算拼接形式。"""
        v = [norm(x) for x in r["cands"]]
        if len(v) > 1:
            v.append(norm(" ".join(r["cands"])))
            v.append(norm("/".join(r["cands"])))
            v.append(norm("".join(r["cands"])))
        return v

    ok = [r for r in res if r["gt"]]
    hit = sum(1 for r in ok if norm(r["gt"]) in variants(r))
    hit1 = sum(1 for r in ok if r["cands"] and norm(r["cands"][0]) == norm(r["gt"]))
    ctrl = sum(1 for r in res if not r["gt"] and r["cands"])
    print()
    print("| GT Name 非空 | %d | 首行命中 | %d = %.4f | 任一候选命中 | %d = %.4f |"
          % (len(ok), hit1, hit1 / max(1, len(ok)), hit, hit / max(1, len(ok))))
    print("| GT Name 为空 | %d | 我们仍读出东西 | %d |" % (len(res) - len(ok), ctrl))
    print("| 墙钟 | %.1fs |  %.2f s/个 |" % (dt, dt / max(1, len(res))))
    print()
    print("--- GT Name 非空的错误样例（最多 20）---")
    n = 0
    for r in ok:
        if norm(r["gt"]) not in variants(r):
            print("  %-16s %-4s GT=%-22s 读=%s" % (r["case"], r["key"], r["gt"], r["cands"][:3]))
            n += 1
            if n >= 20:
                break
    if a.dump:
        Path(a.dump).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("dump ->", a.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
