# -*- coding: utf-8 -*-
"""在指定的一批 case 上跑 VLM 补强，产出干净的 `<case>.json`（供融合层消费）。

为什么要单独做这一步：`_submit_out220r32` 的 `stage3_vlm` 是用**那一次运行的
stage2** 补强的，直接拿来配 `_submit_out220r36` 的 stage2 会串味。融合层要比较的
必须是同一条链：`stage1 → stage2 → enrich → fusion`。

用法：python _fzenrich.py [case ...]
"""
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
PCB = ROOT / "eda-pcb"
sys.path.insert(0, str(PCB))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover  # noqa: E402
from vlm import enrich as E                                           # noqa: E402

RUN = ROOT / "work" / "_submit_out220r36" / "runs"
OUT = Path(os.environ.get("FZ_ENRICH_OUT", str(ROOT / "work" / "_fusion_vlm")))

png_of = {}
for c in discover(DEFAULT_DATA_ROOT, SPLITS):
    png_of.setdefault(c.case_id, c.png_path)
    d = "".join(ch for ch in c.case_id if ch.isdigit())
    if d:
        png_of.setdefault(d, c.png_path)


def _norm(c):
    """数据集里 220 张里同时存在 `1`..`10`（裸数字名）和 `0001`..（补零名），
    它们是**不同的图**（`1` 有 22 个元件、`0001` 有 83 个）。所以先按原样认，
    认不出再补零 —— 无脑 `zfill(4)` 会把裸名悄悄改成另一张图。"""
    if c in png_of:
        return c
    if c.isdigit() and len(c) < 4:
        return c.zfill(4)
    return c


cases = [_norm(c) for c in sys.argv[1:]]
if not cases:
    cases = ["%04d" % i for i in (1, 2, 3, 4, 5, 7, 8, 9, 10)]

missing = [c for c in cases if c not in png_of]
print("要补强 %d 张，找不到 PNG 的：%s" % (len(cases), missing or "无"))
# ⚠ `enrich.DEFAULTS` 里 `name=False`，而**生产路径是开的**
# （`submission\run.py` 的 `-vlm-name default="on"` ⇒ `"name": True`）。
# 不显式打开，`read_name` 一次都不会跑（实测 `name_calls=0`），
# 而 `Name` 字段恰好是 IC 那批分的主要来源。
st = E.enrich_dir(RUN / "stage2", OUT, png_of, workers=4, cases=cases,
                  opts={"name": True})
print(json.dumps({k: v for k, v in st.items() if k != "failed"},
                 ensure_ascii=False, indent=1))
if st.get("failed"):
    print("失败样例：", st["failed"][:5])
