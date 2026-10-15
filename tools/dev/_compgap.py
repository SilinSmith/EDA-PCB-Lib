# -*- coding: utf-8 -*-
"""component 失分溯源：GT 里哪些元件一个都没配上（连坐标都没配上），按「伪元件 ∅NNN / 普通位号」分开。

动机（`work\_ro220b_metrics.json` 的 train200 汇总）：`component.missing 2289`、`component.extra 2158`
远大于 `component.text_mismatch 583`，所以派生档 component F1（0.6066）主要不是文字问题。
要判定这 2289 是多少是 `∅NNN`（电源/地/网络伪元件）、多少是真的漏检，以及坐标是否在同一个坐标系里。

用法：
  python -X utf8 work\tools\dev\_compgap.py --dir work\_ro220b --cases-file work\_cases220.txt --limit 40
  python -X utf8 work\tools\dev\_compgap.py --dir work\_ro220b --cases-file work\_cases220.txt --show 6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

WORK = Path(r"Y:\workspace\EDA\work")
sys.path.insert(0, str(WORK / "tools" / "dev"))
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

import mk_official_textset as M          # noqa: E402
import stage2_eval as SE                 # noqa: E402
from schematic_parser.metrics import EvalConfig, assign_components   # noqa: E402


def suite(n: str) -> str:
    if re.match(r"^\d{1,2}$", n):
        return "10GT"
    if re.match(r"^\d{4}$", n):
        return "train200"
    return "10_GTcase"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="预测 JSON 目录（融合层直出：<case>.json 平铺）")
    ap.add_argument("--cases-file", default=str(WORK / "_cases220.txt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=0, help="打印几个未配上的元件（含 GT bbox 与最近预测）")
    a = ap.parse_args()

    names = [s.strip() for s in Path(a.cases_file).read_text(encoding="utf-8")
             .strip().split(",") if s.strip()]
    if a.limit:
        names = names[:a.limit]
    idx, digits = M.build_index()
    cfg = EvalConfig()

    st = Counter()
    for i, name in enumerate(names):
        p = Path(a.dir) / ("%s.json" % name)
        if not p.is_file():
            st["缺预测"] += 1
            continue
        case, _how = SE.resolve(name, idx, digits)
        if case is None:
            st["缺 GT"] += 1
            continue
        gt = json.loads(Path(case.json_path).read_text(encoding="utf-8"))
        pred = json.loads(p.read_text(encoding="utf-8"))
        gc = gt.get("components") or {}
        f1, keymap = assign_components(gt, pred, cfg)
        covered = set(keymap.values())
        s = suite(name)
        for k, g in gc.items():
            pseudo = M.junk_key(k)
            got = k in covered
            tag = ("伪元件" if pseudo else "普通") + ("配上" if got else "没配上")
            st["%s/%s" % (s, tag)] += 1
        pc = pred.get("components") or {}
        if isinstance(pc, dict):
            npred = len(pc)
        else:
            npred = len(pc)
        st["%s/预测元件数" % s] += npred
        if a.show and not pseudo and i < a.show:
            miss = [k for k in gc if k not in covered][:6]
            print("[%s] %s 未配上示例 %s" % (s, name, miss))

    keys = sorted(st)
    suites = ("10GT", "10_GTcase", "train200")
    for s in suites:
        row = {k.split("/", 1)[1]: v for k, v in st.items() if k.startswith(s + "/")}
        if not row:
            continue
        print("[%s] %s" % (s, json.dumps(row, ensure_ascii=False)))
        tot = row.get("普通配上", 0) + row.get("普通没配上", 0)
        pse = row.get("伪元件配上", 0) + row.get("伪元件没配上", 0)
        print("        普通 %d（没配上 %d, %.1f%%）  伪元件 %d（没配上 %d, %.1f%%）  预测 %d"
              % (tot, row.get("普通没配上", 0),
                 100.0 * row.get("普通没配上", 0) / max(1, tot),
                 pse, row.get("伪元件没配上", 0),
                 100.0 * row.get("伪元件没配上", 0) / max(1, pse),
                 row.get("预测元件数", 0)))
    other = {k: v for k, v in st.items() if not k.split("/")[0] in suites}
    if other:
        print("其它:", other)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
