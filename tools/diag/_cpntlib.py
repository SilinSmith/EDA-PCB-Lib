"""官方元件库 `赛题资料/cpntLibrary` 到底能救多少 **box(IC) 引脚名**。

口径来源（`schematic_parser/metrics.py`）：
  · `pin_f1` 占 0.25，**只统计 GT 里 type 为 box 的元件**（`EvalConfig.pin_box_only=True`）；
  · 一条引脚算 tp 的条件：`(comp, pin_key_to_number(pk))` 键相同 + `pinname` **字符串全等**
    + `point` 距离 <= 5px（`metrics.py:458-473`）。
  · 于是库的用法是：**已知 IC 的 `Name`（= 元件型号）+ 引脚号 ⇒ 直接查表得到引脚名**。

本探针只回答「上限有多少」，不改管线：
  A. GT 里 box 元件的 `Name` 能在库里查到多少（四种归一化策略）；
  B. 查到之后，按 **引脚号** 查出来的名字集合里有没有 GT 的 `pinname`（区分大小写全等/不区分）；
  C. 名字集合的**歧义度**（同一 number 对多个不同 name 时能不能唯一定位）；
  D. 增量：对照我们 `--pred` 的预测，现在名字错/缺的引脚里，有多少能被库直接修对
     （前提是**我们那条引脚的号码本身是对的**，否则查表查不到）。

用法：
  python -X utf8 eda-pcb\\tools\\diag\\_cpntlib.py --lib 赛题资料\\cpntLibrary \\
      --pred work\\_s2_220mix --cases-file work\\_cases_w28.txt --show 12
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "diag"))
from schematic_parser import data as D                                # noqa: E402
from schematic_parser.metrics import canon_type                       # noqa: E402
import adjmeas as AM                                                  # noqa: E402

SUF_C = re.compile(r"_C\d+$")
SUF_N = re.compile(r"_\d+$")
SUF_J = re.compile(r"_JX$")


def norm_aggr(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", (s or "").lower())


def build_index(lib: dict):
    idx = {"exact": {}, "low": {}, "suf": {}, "aggr": {}}
    for k in lib:
        idx["exact"].setdefault(k, k)
        idx["low"].setdefault(k.lower(), k)
        base = k
        for rx in (SUF_C, SUF_N, SUF_J):
            base = rx.sub("", base)
        idx["suf"].setdefault(base.lower(), k)
        idx["aggr"].setdefault(norm_aggr(base), k)
        idx["aggr"].setdefault(norm_aggr(k), k)
    return idx


def lookup(idx, name: str):
    """返回 (策略, 库键) 或 (None, None)。顺序＝从最保守到最宽松。"""
    if not name:
        return None, None
    for st, key in (("exact", name), ("low", name.lower())):
        if key in idx[st]:
            return st, idx[st][key]
    base = name
    for rx in (SUF_C, SUF_N, SUF_J):
        base = rx.sub("", base)
    if base.lower() in idx["suf"]:
        return "suf", idx["suf"][base.lower()]
    a = norm_aggr(base)
    if a and a in idx["aggr"]:
        return "aggr", idx["aggr"][a]
    return None, None


def read_cases(path) -> list:
    txt = Path(path).read_text(encoding="utf-8-sig", errors="ignore").replace(",", "\n")
    return [t.strip() for t in txt.splitlines() if t.strip() and not t.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=str(ROOT.parent / "赛题资料" / "cpntLibrary"))
    ap.add_argument("--pred", default="")
    ap.add_argument("--cases-file", default=str(ROOT.parent / "work" / "_cases_w28.txt"))
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                                 # noqa: BLE001
        pass
    lib = json.loads(Path(a.lib).read_text(encoding="utf-8"))
    idx = build_index(lib)
    print("库条目 %d（归一化索引 exact/low/suf/aggr = %d/%d/%d/%d）"
          % (len(lib), len(idx["exact"]), len(idx["low"]), len(idx["suf"]), len(idx["aggr"])))

    T = collections.Counter()
    strat = collections.Counter()
    amb = collections.Counter()
    fixable = collections.Counter()
    rows = []
    for nm in read_cases(a.cases_file):
        _, gt = AM.find_gt(nm)
        if gt is None:
            T["无GT"] += 1
            continue
        pred = None
        if a.pred:
            p = Path(a.pred) / ("%s.json" % nm)
            if p.exists():
                pred = json.loads(p.read_text(encoding="utf-8"))
        gc = gt.get("components") or {}
        gp = gt.get("pins") or {}
        for cid, rec in gc.items():
            if canon_type(rec.get("type")) not in D.IC_TYPES:
                continue                      # pin_f1 只算 box
            T["gt_box"] += 1
            pins = gp.get(cid) or {}
            T["gt_box_pins"] += len(pins)
            if not pins:
                continue
            st, lk = lookup(idx, str(rec.get("Name") or ""))
            if st is None:
                T["型号查不到"] += 1
                T["型号查不到_引脚"] += len(pins)
                if len(rows) < a.show:
                    rows.append((nm, cid, rec.get("Name"), "-", len(pins), 0, 0))
                continue
            strat[st] += 1
            libp = lib[lk].get("pins") or []
            bynum = collections.defaultdict(set)
            for pr in libp:
                n = str(pr.get("number") or "").strip()
                if n:
                    bynum[n].add(str(pr.get("name") or "").strip())
            got = ex = amb_ok = 0
            our = (pred or {}).get("pins", {}).get(cid) or {}
            for pk, info in pins.items():
                bad = True
                num = D.pin_key_to_number(pk)
                names = bynum.get(str(num))
                if names:
                    amb[len(names)] += 1
                    gname = str((info or {}).get("pinname") or "")
                    if gname in names:
                        got += 1
                        bad = False
                        if len(names) == 1:
                            ex += 1
                        else:
                            amb_ok += 1
                if a.pred and bad:
                    # 我们现在这条引脚是什么状态？库能不能修对？
                    mine = our.get(pk)
                    if mine is None:
                        cur = "missing"
                    else:
                        cur = ("ok" if str(mine.get("pinname")) == str((info or {}).get("pinname"))
                               else "name_mismatch")
                    fixable[cur] += 1
                    if names and str((info or {}).get("pinname") or "") in names and len(names) == 1:
                        fixable[cur + "_库可唯一修对"] += 1
                    elif names and str((info or {}).get("pinname") or "") in names:
                        fixable[cur + "_库可修对但歧义"] += 1
            T["库命中_引脚"] += len(pins)
            T["库命中_名字对"] += got
            T["库命中_唯一"] += ex
            T["库命中_歧义但对"] += amb_ok
            if len(rows) < a.show:
                rows.append((nm, cid, rec.get("Name"), "%s:%s" % (st, lk), len(pins), got, ex))

    print("-" * 78)
    for k in ("gt_box", "gt_box_pins", "型号查不到", "型号查不到_引脚",
              "库命中_引脚", "库命中_名字对", "库命中_唯一", "库命中_歧义但对"):
        print("  %-22s %6d" % (k, T[k]))
    n = max(1, T["gt_box_pins"])
    print("  → 全量引脚里，库按 pin number 查得到的名字包含 GT 名的比例  %.1f%%"
          % (100.0 * T["库命中_名字对"] / n))
    print("  → 其中能**唯一**定位的                                     %.1f%%"
          % (100.0 * T["库命中_唯一"] / n))
    print("  归一化策略命中分布: %s" % dict(strat))
    print("  同一 number 对应不同 name 的个数分布: %s"
          % dict(sorted(amb.items())[:8]))
    if a.pred:
        print("-" * 78)
        print("  我们当前这些引脚的状态（库命中元件内）与前缀计数：")
        for k, v in sorted(fixable.items()):
            print("     %-34s %5d" % (k, v))
    if rows:
        print("-" * 78)
        print("  %-14s %-6s %-18s %-24s %6s %6s %6s" %
              ("case", "cid", "GT Name", "库键(策略)", "引脚", "名对", "唯一"))
        for r in rows:
            print("  %-14s %-6s %-18s %-24s %6d %6d %6d" % r)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
