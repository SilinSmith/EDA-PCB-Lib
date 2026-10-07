# -*- coding: utf-8 -*-
"""单颗 IC 的 VLM 引脚读取调试：看裁片尺寸、思维链、finish_reason。

    cd Y:\\workspace\\EDA
    python work\\tools\\diag\\_vlmpin_dbg.py --case 1 --key U1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "diag"))

from schematic_parser.data import discover, load_json  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402
from _vlmread import _chat_once  # noqa: E402
from _vlmpin import PIN_PROMPT, TERSE_PROMPT  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="1")
    ap.add_argument("--key", default="U1")
    ap.add_argument("--margin", type=float, default=26.0)
    ap.add_argument("--max-side", type=int, default=0, help=">0 时把长边缩到该值")
    ap.add_argument("--style", default="terse", choices=("json", "terse"))
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3.5-9b"))
    a = ap.parse_args(argv)

    for c in discover():
        if c.case_id != a.case or not c.json_path:
            continue
        gt = load_json(c.json_path)
        v = (gt.get("components") or {}).get(a.key)
        if not v:
            print("没有这个元件"); return 1
        bgr = imread_unicode(c.png_path)
        H = bgr.shape[0]
        b = v["bbox"]
        ib = (b[0], H - b[3], b[2], H - b[1])
        x0 = max(0, int(ib[0] - a.margin)); y0 = max(0, int(ib[1] - a.margin))
        x1 = min(bgr.shape[1], int(ib[2] + a.margin))
        y1 = min(bgr.shape[0], int(ib[3] + a.margin))
        im = bgr[y0:y1, x0:x1]
        print("图 %s  %dx%d" % (c.png_path.split("\\")[-1], bgr.shape[1], bgr.shape[0]))
        print("元件 %s Name=%s  bbox=%s" % (a.key, v.get("Name"), b))
        print("裁片 %dx%d" % (im.shape[1], im.shape[0]))
        if a.max_side and max(im.shape[:2]) > a.max_side:
            s = a.max_side / float(max(im.shape[:2]))
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            print("缩到 %dx%d" % (im.shape[1], im.shape[0]))
        out = os.path.join(str(ROOT), "work", "preview", "_pindbg_%s_%s.png" % (a.case, a.key))
        cv2.imwrite(out, im)
        print("saved:", out)
        prompt = PIN_PROMPT if a.style == "json" else TERSE_PROMPT
        txt, fin, usage = _chat_once(im, prompt, a.model, a.budget, 900)
        print("\nfinish_reason =", fin)
        print("usage =", usage)
        print("content(len=%d) = %r" % (len(txt or ""), (txt or "")[:600]))
        return 0
    print("没找到 case", a.case)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
