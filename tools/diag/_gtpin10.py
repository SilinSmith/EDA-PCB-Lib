# -*- coding: utf-8 -*-
"""官方 10GT 的 **IC(box) 引脚键**到底是什么形状 —— 决定「引脚号对不上」能不能赢。

背景（`docs/RESULTS.md` §2.3）：GT 引脚在评分里按 `(元件, pin_key_to_number(pk))` 配对，
`_pinoracle.py` 显示 64.9% 的 GT IC 引脚在我们的预测里**连那个键都没有**。
在**派生档**（`200_train_cases`）上，GT 的 `pin_N` 后缀常是内部计数器（`pin_445`），
那部分数学上不可赢。官方档是不是也这样，必须单独量。

用法：
    python work/tools/diag/_gtpin10.py            # 10GT 全量
    python work/tools/diag/_gtpin10.py --show 5   # 每类打印几个例子
"""
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
EDA = os.path.abspath(os.path.join(HERE, "..", "..", "..", "eda-pcb"))
ROOT = os.path.join(EDA, "yolo_part", "Dataset(waitforpreprocess)", "10GT")

IC_TYPES = {"box", "ic", "chip", "module"}


def _canon(t):
    return str(t or "").strip().lower()


def keynum(pk):
    s = str(pk)
    if s.lower().startswith("pin_"):
        s = s[4:]
    elif s.lower().startswith("pin"):
        s = s[3:]
    return s


def main(argv):
    show = 0
    if "--show" in argv:
        show = int(argv[argv.index("--show") + 1])
    cases = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    tot = Counter()
    per_case = []
    samples = {}
    for c in cases:
        jp = None
        for dirpath, _dirs, files in os.walk(os.path.join(ROOT, c)):
            for f in files:
                if f.endswith(".json"):
                    jp = os.path.join(dirpath, f)
                    break
            if jp:
                break
        if not jp:
            print("!! %s 没有 json" % c)
            continue
        gt = json.load(open(jp, encoding="utf-8"))
        comps = gt.get("components") or {}
        pins = gt.get("pins") or {}
        boxc = {k for k, v in comps.items() if _canon((v or {}).get("type")) in IC_TYPES}
        n_all = n_box = 0
        for comp, group in pins.items():
            if not isinstance(group, dict):
                continue
            for pk, info in group.items():
                n_all += 1
                if comp not in boxc:
                    continue
                n_box += 1
                num = keynum(pk)
                if re.fullmatch(r"\d{1,4}", num):
                    k = "纯数字"
                elif re.fullmatch(r"[A-Za-z]{1,2}\d{1,3}", num):
                    k = "字母+数字(A1/B12)"
                elif re.fullmatch(r"[A-Za-z]+\d*", num):
                    k = "纯字母"
                else:
                    k = "其它(含斜杠等)"
                tot[k] += 1
                samples.setdefault(k, []).append((c, comp, pk, (info or {}).get("pinname")))
        per_case.append((c, n_box, n_all))
    print("10GT 目录：%s" % ROOT)
    print("case 数：%d" % len(per_case))
    print()
    print("| case | GT IC(box) 元件 | IC 引脚 | 全部引脚 |")
    print("|---|---|---|---|")
    for c, nb, na in per_case:
        print("| %s | | %d | %d |" % (c, nb, na))
    print()
    s = sum(tot.values())
    print("GT IC 引脚键形状（共 %d 根）：" % s)
    for k, v in tot.most_common():
        print("  %-18s %5d  %5.1f%%" % (k, v, 100.0 * v / max(1, s)))
    print()
    print("平均每 case IC 引脚 %.1f 根" % (s / max(1, len(per_case))))
    if show:
        for k, v in tot.most_common():
            print()
            print("== %s 例（%d 根）==" % (k, v))
            for c, comp, pk, nm in samples[k][:show]:
                print("  %s %s %s pinname=%r" % (c, comp, pk, nm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
