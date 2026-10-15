# -*- coding: utf-8 -*-
"""`STAGE2_BOX_PIN_NEED_NAME` 的**离线单变量 A/B**（纯 CPU，不需要 GPU / OCR）。

为什么能离线：那个闸门在 `stage2/assemble.py:759` 的 `build_doc` 里，
而 `build_doc(stage1, assigned, case)` 的两个输入**都已经落盘**
（`runs/stage1/<case>.stage1.json` 与 `runs/stage2/_assigned/<case>.json`）。
`BOX_PIN_NEED_NAME` 是模块级全局、在函数里按调用时读 ⇒ 直接改
`assemble.BOX_PIN_NEED_NAME` 就能切开关，不必重跑 OCR。

它量的是「**号已读对、名字没读到**的引脚桩被整条丢掉」这件事值多少分
（第十批：官方 9 张缺的 45 根 IC 引脚里有 23 根属于这一类）。

用法：
    python work\\tools\\diag\\_gateab.py --run work\\_fast_out_reg9 --cases 1,2,3,4,5,7,8,9,10
    python work\\tools\\diag\\_gateab.py --run work\\_full220 --all
然后分别打分：
    python work\\tools\\eval\\stage2_eval.py work\\_gate_ab\\on  --json work\\_gate_on.json
    python work\\tools\\eval\\stage2_eval.py work\\_gate_ab\\off --json work\\_gate_off.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EDA = ROOT.parent / "eda-pcb"
sys.path.insert(0, str(EDA))

from stage2 import assemble                                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "_fast_out_reg9"))
    ap.add_argument("--cases", default="1,2,3,4,5,7,8,9,10")
    ap.add_argument("--all", action="store_true", help="用 _assigned 目录里全部 case")
    ap.add_argument("--out", default=str(ROOT / "_gate_ab"))
    a = ap.parse_args()

    run = Path(a.run)
    s1 = run / "runs" / "stage1"
    asg = run / "runs" / "stage2" / "_assigned"
    if not asg.is_dir():
        raise SystemExit("找不到 %s" % asg)
    cases = (sorted(p.stem for p in asg.glob("*.json")) if a.all
             else [c.strip() for c in a.cases.split(",") if c.strip()])
    print("run=%s  cases=%d" % (run, len(cases)))

    # 载入一次，两个臂复用（build_doc 不改输入）
    docs = {}
    miss = []
    for c in cases:
        p1 = s1 / ("%s.stage1.json" % c)
        p2 = asg / ("%s.json" % c)
        if not (p1.is_file() and p2.is_file()):
            miss.append(c)
            continue
        docs[c] = (json.loads(p1.read_text(encoding="utf-8")),
                   json.loads(p2.read_text(encoding="utf-8")))
    if miss:
        print("⚠ 缺产物 %d 张：%s" % (len(miss), miss[:20]))

    for arm, val in (("on", True), ("off", False)):
        d = Path(a.out) / arm
        d.mkdir(parents=True, exist_ok=True)
        assemble.BOX_PIN_NEED_NAME = val
        n_pin = 0
        for c, (doc, asgdoc) in docs.items():
            pred = assemble.build_doc(doc, asgdoc, c)
            pred.pop("__pin_index", None)
            n_pin += len(pred.get("pins") or {})
            (d / ("%s.json" % c)).write_text(
                json.dumps(pred, ensure_ascii=False), encoding="utf-8")
        print("  %-4s BOX_PIN_NEED_NAME=%-5s 元件键 %d  -> %s"
              % (arm, val, n_pin, d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
