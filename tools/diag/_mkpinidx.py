# -*- coding: utf-8 -*-
"""为已有的 stage2 产物补出 <case>.pinindex.json（纯 CPU，不跑任何模型）。

融合层要改交付文档的 `pin_<N>` 键，必须先知道「第 k 个候选 -> 哪个 pin 键」。
`build_doc` 内部有这张表（`pin_pk`），但以前不导出。现在它导出 `__pin_index`，
而 `build_doc` 只吃 (stage1 文档, _assigned, case) 三样 JSON —— 所以旧产物不必
重跑 stage1/stage2，直接在这里补出来即可。

同时逐张核对「重装配出来的交付文档」与磁盘上那份是否逐字节相同：
不同就说明 `_assigned` 不足以复现，脚本会跳过该张不写 sidecar。

用法（workdir = Y:\\workspace\\EDA\\eda-pcb）：
    python ..\\work\\tools\\diag\\_mkpinidx.py [--run ..\\work\\_submit_out220r36\\runs] [--dry]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")
from stage2.assemble import build_doc            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=r"..\work\_submit_out220r36\runs")
    ap.add_argument("--dry", action="store_true", help="只核对，不写 sidecar")
    a = ap.parse_args()

    s1d = os.path.join(a.run, "stage1")
    s2d = os.path.join(a.run, "stage2")
    asd = os.path.join(s2d, "_assigned")

    cases = sorted(f[:-5] for f in os.listdir(asd) if f.endswith(".json"))
    same = diff = skip = written = 0
    tot_comp = tot_pin = 0
    errs = []
    for c in cases:
        try:
            s1 = json.load(open(os.path.join(s1d, c + ".stage1.json"), encoding="utf-8"))
            aa = json.load(open(os.path.join(asd, c + ".json"), encoding="utf-8"))
            ref = json.load(open(os.path.join(s2d, c + ".json"), encoding="utf-8"))
        except Exception as e:                       # noqa: BLE001
            errs.append("%s 读失败 %s: %s" % (c, type(e).__name__, e))
            skip += 1
            continue
        d = build_doc(s1, aa, c)
        pidx = d.pop("__pin_index", None)
        eq = (json.dumps(d, ensure_ascii=False, sort_keys=True)
              == json.dumps(ref, ensure_ascii=False, sort_keys=True))
        if not eq:
            diff += 1
            errs.append("%s 重装配与磁盘文档不等价 -> 跳过" % c)
            continue
        same += 1
        tot_comp += len(pidx or {})
        tot_pin += sum(len(v) for v in (pidx or {}).values())
        if not a.dry:
            with open(os.path.join(s2d, c + ".pinindex.json"), "w",
                      encoding="utf-8") as f:
                json.dump(pidx or {}, f, ensure_ascii=False)
            written += 1

    print("张数 %d｜逐字节等价 %d｜不等价 %d｜读失败 %d" % (len(cases), same, diff, skip))
    print("pin_index 元件数合计 %d｜引脚数合计 %d" % (tot_comp, tot_pin))
    print("写出 sidecar %d 个%s" % (written, "（--dry，没写）" if a.dry else ""))
    for e in errs[:20]:
        print("  !!", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
