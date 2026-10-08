# -*- coding: utf-8 -*-
"""量化「官方元器件库 cpntLibrary 能给我方当前产物补多少分」的机会。

只做统计，不改管线。对每个 case：
  1) 用官方口径的 `assign_components` 把 pred 元件对到 GT 元件（顺带拿到 keymap）；
  2) 对**双方都匹配上、且 GT 是 box/IC** 的元件，逐引脚比 pinname；
  3) 把差异分成三桶：
       ok        —— 名字已经对
        lib_fix   —— 名字错，但 GT 名在库里该型号的 name 集合里（⇒ 库先验可修）
        no_lib    —— 名字错，库也救不了
       missing   —— pred 压根没有这个 pin
  4) 元件名（Name）单独统计：GT 名能命中库、而我方 Name 不同的情况，数「库能不能把 pred 名拉到 GT」。

用法：
    python -X utf8 work\\tools\\diag\\_libopp.py --pred work\\_fnew --cases 1,2,3,4,5,7,8,9,10
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.metrics import (  # noqa: E402
    EvalConfig, assign_components, canon_type, rewrite_keys)
from schematic_parser.data import IC_TYPES, pin_key_to_number  # noqa: E402

GT9 = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
LIB = ROOT / "work" / "data" / "cpntLibrary.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "work" / "tools" / "diag"))
from _libcov import Lib, load, norm  # noqa: E402


def pinname_of(pin_rec) -> str:
    return (pin_rec or {}).get("pinname") or ""


def _dist1(a: str, b: str) -> bool:
    """a 与 b 的编辑距离是否恰好为 1（插入/删除/替换一位）。"""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(l)):
        if l[:i] + l[i + 1:] == s:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="含 <case>.json 的目录（如 work\\_fnew）")
    ap.add_argument("--cases", default="1,2,3,4,5,7,8,9,10")
    ap.add_argument("--gt", default=str(GT9))
    ap.add_argument("--lib", default=str(LIB))
    ap.add_argument("--fuzzy", type=float, default=0.85)
    ap.add_argument("--verbose", type=int, default=0)
    a = ap.parse_args()

    lib = Lib(load(Path(a.lib)))
    gtroot, predroot = Path(a.gt), Path(a.pred)
    cfg = EvalConfig()

    pin = Counter()
    name = Counter()
    num = Counter()
    per_case = []
    libkeys_used: Counter = Counter()

    for c in [x.strip() for x in a.cases.split(",") if x.strip()]:
        gtf = gtroot / c / ("%s.json" % c)
        pf = predroot / ("%s.json" % c)
        if not (gtf.exists() and pf.exists()):
            print("  ! 跳过 case %s（缺 %s 或 %s）" % (c, gtf, pf))
            continue
        gt = json.loads(gtf.read_text(encoding="utf-8"))
        pred = json.loads(pf.read_text(encoding="utf-8"))
        f1, keymap = assign_components(gt, pred, cfg)
        pred2 = rewrite_keys(pred, keymap)

        gc, gpin, ppin = gt.get("components", {}), gt.get("pins", {}), pred2.get("pins", {})
        n_fix = n_nolib = n_missing = n_ok = 0
        n_ic = n_ic_lib = 0
        for k, rec in gc.items():
            if canon_type(rec.get("type")) not in IC_TYPES:
                continue
            if k not in pred2.get("components", {}):      # 元件没对上
                continue
            n_ic += 1
            gtname = rec.get("Name") or ""
            cand = lib.exact(gtname) or lib.normalized(gtname) or \
                [x[0] for x in lib.fuzzy(gtname, a.fuzzy)]
            libkey = cand[0] if cand else None
            if libkey:
                n_ic_lib += 1
                libkeys_used[libkey] += 1
            libnames = lib.pin_names(libkey) if libkey else Counter()
            # 元件名本身
            pname = (pred2["components"].get(k) or {}).get("Name") or ""
            if pname != gtname:
                # 库能把预测名拉到 GT 吗？（预测名模糊命中某个库键，且该键 == GT 名归一化）
                pcand = lib.exact(pname) or lib.normalized(pname) or \
                    [x[0] for x in lib.fuzzy(pname, a.fuzzy)]
                if libkey and pcand and norm(pcand[0]) == norm(libkey):
                    name["lib_fix"] += 1
                elif libkey and gtname and norm(pname) != norm(gtname):
                    name["mismatch_other"] += 1
            else:
                name["ok"] += 1
            # 引脚
            for pk, pv in (gpin.get(k) or {}).items():
                gname = pinname_of(pv)
                if k not in ppin or pk not in ppin[k]:
                    pin["missing"] += 1
                    n_missing += 1
                    continue
                pname_pin = pinname_of(ppin[k][pk])
                if pname_pin == gname:
                    pin["ok"] += 1
                    n_ok += 1
                elif gname and gname in libnames:
                    pin["lib_fix"] += 1
                    n_fix += 1
                    if a.verbose:
                        print("    [pin-fix] %s %s.%s  %r -> %r" % (c, k, pk, pname_pin, gname))
                else:
                    pin["no_lib"] += 1
                    n_nolib += 1
                    if a.verbose:
                        print("    [pin-no]  %s %s.%s  %r -> %r" % (c, k, pk, pname_pin, gname))
            # ---- 引脚号通道：库能不能当「号码词表」用 --------------------
            gnums = [pin_key_to_number(pk) for pk in (gpin.get(k) or {})]
            pnums = [pin_key_to_number(pk) for pk in (ppin.get(k) or {})]
            gset, pset = set(gnums), set(pnums)
            lnums = set(str(x) for x in lib.pin_numbers(libkey)) if libkey else set()
            for n in gset:
                num["ok" if n in pset else "missing"] += 1
            for n in pset - gset:
                num["extra"] += 1
                if not lnums:
                    continue
                if n not in lnums:
                    num["extra_not_in_lib"] += 1
                # 库词表能不能把它拉回某个我们缺的 GT 号？（字符串距离 ≤1 或前缀关系）
                cand = [g for g in (gset - pset)
                        if g in lnums and (g.startswith(n) or n.startswith(g)
                                           or _dist1(n, g))]
                if cand:
                    num["lib_snap"] += 1
                    if a.verbose:
                        print("    [num-snap] %s %s  %r -> %s" % (c, k, n, cand[:3]))
                else:
                    num["lib_nosnap"] += 1
        per_case.append((c, n_ic, n_ic_lib, n_ok, n_fix, n_nolib, n_missing))
        print("  case %-3s IC %2d（库命中 %2d） | pinname 对 %3d / 库可修 %3d / 库救不了 %3d / 缺 %3d"
              % (c, n_ic, n_ic_lib, n_ok, n_fix, n_nolib, n_missing))

    tot = sum(pin.values())
    print("\n===== pinname 汇总（%d 个 GT 引脚，仅统计双方都匹配上的 box/IC）=====" % tot)
    for k in ("ok", "lib_fix", "no_lib", "missing"):
        if tot:
            print("  %-8s %4d  %5.1f%%" % (k, pin[k], 100.0 * pin[k] / tot))
    print("  ⇒ 库先验的**上界增分**只可能来自 lib_fix 这 %d 个" % pin["lib_fix"])
    nt = sum(name.values())
    print("===== 元件名（Name）汇总（%d 个匹配上的 IC）=====" % nt)
    for k in ("ok", "lib_fix", "mismatch_other"):
        if nt:
            print("  %-14s %4d  %5.1f%%" % (k, name[k], 100.0 * name[k] / nt))
    print("  用到的库键 top:", ", ".join("%s×%d" % (k, v) for k, v in libkeys_used.most_common(8)))
    nt2 = sum(num.values())
    print("===== 引脚号通道（%d 个 GT 号 + 我方多余号）=====" % nt2)
    for k in ("ok", "missing", "extra", "extra_not_in_lib", "lib_snap", "lib_nosnap"):
        if nt2:
            print("  %-16s %4d" % (k, num[k]))
    print("  ⇒ 用库当号码词表的上界：把 %d 个错号拉回 GT 号（另有 %d 个连库都没有 ⇒ 可直接判废）"
          % (num["lib_snap"], num["extra_not_in_lib"]))


if __name__ == "__main__":
    main()
