# -*- coding: utf-8 -*-
"""给单张图的 stage1 做 cProfile，找出 total_s 之外的「黑洞」。

背景：`work\\_full220` 全量跑里，大图（4491x3173、5100x3296…）单张 `total_s`
高达 315s，而 `timing` 里 yolo/cv/wires/rays/text/bigic/pins 加起来只有 4.3s。

用法：
    python work\\tools\\diag\\_prof1.py 0003
    python work\\tools\\diag\\_prof1.py 0003 --top 25
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover          # noqa: E402
from stage2_eval import build_index, resolve                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "work" / "_prof1"))
    a = ap.parse_args()

    exact, digits = build_index(DEFAULT_DATA_ROOT)
    case, _ = resolve(a.case, exact, digits)
    if case is None:
        print("找不到 case", a.case)
        return 1
    out = Path(a.out) / case.case_id
    out.mkdir(parents=True, exist_ok=True)

    from stage1 import pipeline as P
    from stage1.detect import DEFAULT_WEIGHTS
    kw = dict(weights=DEFAULT_WEIGHTS, imgsz=1600, conf=0.25, device="0")
    pr = cProfile.Profile()
    pr.enable()
    r = P.run_case(case, **kw)
    pr.disable()
    print("timing:", r.get("timing"))
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(a.top)
    txt = s.getvalue()
    # 只留最有用的一段
    lines = txt.splitlines()
    print("\n".join(lines[:12]))
    print("\n".join(lines[-a.top - 1:]))
    (out / "_profile.txt").write_text(txt, encoding="utf-8")
    print("完整 profile 写到", out / "_profile.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
