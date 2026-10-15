"""GT 引脚键的**约定**普查：键是「物理引脚号」还是「1..N 计数器」？

为什么要量这个：`schematic_parser/metrics.py:426 pin_f1()` 按 **(元件, 键里的数字)**
配对，再比 `pinname` 与点距（`pin_tol`）。所以「键里的数字」是物理号还是顺序号，
**决定 VLM 引脚通道（把号改成读到的物理号）是加分还是掉分**。

三个 split：
  * `10GT`           —— 官方公开 golden 10 张（`stage2_eval` 的官方口径来源）
  * `10_GTcase`      —— 官方另一批（`0025-Altium Designer` 这类）
  * `200_train_cases`—— 自建派生集（`*_target_named.json`，JLC 导出后自动命名）

判据：对每个元件的引脚键集合 `ns`
  * `ns == [1..len(ns)]` ⇒ 计数器式（counter）
  * 否则                 ⇒ 物理号式（physical），并记下最大键与是否含 >N 的号
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "..", "eda-pcb", "yolo_part",
                    "Dataset(waitforpreprocess)")
ROOT = os.path.abspath(ROOT)


def keynum(pk):
    m = re.search(r"(\d+)", str(pk))
    return int(m.group(1)) if m else None


def gt_of(case_dir):
    fs = [f for f in glob.glob(os.path.join(case_dir, "*.json"))
          if "leak" not in os.path.basename(f)]
    # 派生集用 `*_target_named.json`，官方 `10GT` 用 `<case>.json`
    named = [f for f in fs if f.endswith("_target_named.json")]
    plain = [f for f in fs if os.path.basename(f)[:-5] ==
             os.path.basename(case_dir)]
    for f in (named or plain or fs):
        try:
            return json.load(open(f, encoding="utf-8")), os.path.basename(f)
        except Exception:                                      # noqa: BLE001
            continue
    return None, None


def main(argv):
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", "..", "..", "eda-pcb"))
        from schematic_parser.data import IC_TYPES          # noqa: E402
        from schematic_parser.metrics import canon_type      # noqa: E402
    except Exception as exc:                                # noqa: BLE001
        print("!! 引不到 schematic_parser：", exc)
        IC_TYPES, canon_type = frozenset(), (lambda t: t)
    for split in ("10GT", "10_GTcase", "200_train_cases"):
        d = os.path.join(ROOT, split)
        if not os.path.isdir(d):
            print("!! 没有目录", d)
            continue
        cnt = collections.Counter()
        ic = collections.Counter()
        biggest = 0
        pn_numeric = 0
        pn_total = 0
        examples = []
        for case in sorted(os.listdir(d)):
            cd = os.path.join(d, case)
            if not os.path.isdir(cd):
                continue
            j, fn = gt_of(cd)
            if not j:
                continue
            for comp, g in (j.get("pins") or {}).items():
                if not isinstance(g, dict) or not g:
                    continue
                ns = sorted(x for x in (keynum(k) for k in g)
                            if x is not None)
                if not ns:
                    continue
                biggest = max(biggest, ns[-1])
                ok = ns == list(range(1, len(ns) + 1))
                cnt["counter" if ok else "physical"] += 1
                cnt["comp"] += 1
                if not ok and len(examples) < 4:
                    examples.append((case, comp, ns[:12]))
                for info in g.values():
                    pn_total += 1
                    if re.fullmatch(r"\d+", str((info or {}).get("pinname") or "")):
                        pn_numeric += 1
                # ★ 只看 **IC（box）且 >=5 脚**：2 脚的 R/C 上「1..2」两种约定
                #   本来就重合，混进来会把结论冲淡。
                comps = j.get("components") or {}
                if (canon_type((comps.get(comp) or {}).get("type"))
                        in IC_TYPES and len(ns) >= 5):
                    ic["comp"] += 1
                    ic["counter" if ok else "physical"] += 1
        tot = max(cnt["comp"], 1)
        print("== %-16s 元件 %4d | 计数器式 %4d (%.1f%%) | 物理号式 %4d "
              "| 最大键 %3d | pinname 纯数字 %.1f%%"
              % (split, cnt["comp"], cnt["counter"],
                 100.0 * cnt["counter"] / tot, cnt["physical"], biggest,
                 100.0 * pn_numeric / max(pn_total, 1)))
        print("     ├ IC(box) 且 >=5 脚：%d 个，其中计数器式 %d (%.1f%%)"
              % (ic["comp"], ic["counter"],
                 100.0 * ic["counter"] / max(ic["comp"], 1)))
        for e in examples:
            print("     物理号例：%s %s -> %s" % e)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
