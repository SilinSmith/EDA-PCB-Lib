# -*- coding: utf-8 -*-
"""引脚为什么丢：逐 IC 把 GT 引脚号 / 我们发出的引脚号 / OCR 读到的号摆在一起。

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_pinmiss.py [--pred 目录] [--show 0001,0002] [--all]
默认 pred 目录 work\\_fusion_out（融合产物），同时可给 --base 目录对照。

它回答三个问题：
  ① GT 的 box 元件引脚，我们发出了几个（缺几个）；
  ② 缺的那些，OCR 到底读没读到（`_assigned` 里的 `pins[k].number`）；
  ③ 读到了却被丢掉的，是撞号、越界，还是被 `BOX_PIN_DROP_UNREAD` 闸掉了。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M                     # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover  # noqa: E402


def gt_of(case: str):
    for c in discover(DEFAULT_DATA_ROOT, SPLITS):
        if c.case_id == case:
            return c
    return None


def load_gt(case: str):
    c = gt_of(case)
    if c is None:
        return None
    import eda_pcb_stub  # noqa: F401  (占位，永远不会走到)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fusion_out"))
    ap.add_argument("--assigned",
                    default=str(ROOT / "work" / "_submit_out220r36" / "runs"
                                / "stage2" / "_assigned"))
    ap.add_argument("--show", default="")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    print("== 待补：先确认 GT 引脚结构 ==")
    cases = [x.case_id for x in discover(DEFAULT_DATA_ROOT, SPLITS)]
    print("case 数 %d，样例 %s" % (len(cases), cases[:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
