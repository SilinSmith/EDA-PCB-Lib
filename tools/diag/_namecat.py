"""引脚名错配的**成因分类**：截断 / 扩读 / 字符混淆 / 完全无关。

用法：
    python work\\tools\\diag\\_namecat.py --pred work\\_gt10_along\\runs\\stage2 --split 10GT [--top 40]

动机：`_namefix.py` 已经量出 10GT 上「名」的逐字相等率（59.8%），但没回答
「剩下的 40% 里有多少是**文字框截断**造成的」—— 如果那一类占多数，那么
「把朝外方向（左右边上就是背向引脚的那一侧）的框扩出去重读」就是下一个杠杆；
如果多数是无关联的错读，扩框就没用。

判据（对我们读到的 `ours` 与 GT 的 `theirs`，都大写去非字母数字后比）：
  * `trunc`：`ours` 是 `theirs` 的**真前缀/真子串**（我们少读了字符）
  * `over` ：`theirs` 是 `ours` 的真子串（我们多读了字符或粘了邻居）
  * `near`：编辑距离 1（且长度 >= 4）—— 纯字符混淆
  * `same_len_diff`：等长但有多处不同
  * `other`：其余
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "eda-pcb")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from schematic_parser import metrics as M                       # noqa: E402
from schematic_parser.data import discover, load_json           # noqa: E402

cfg = M.EvalConfig()
_ALNUM = re.compile(r"[^A-Za-z0-9]")


def _n(s: str) -> str:
    return _ALNUM.sub("", (s or "").upper())


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    cases = [c for c in discover() if c.split == a.split]
    cats: collections.Counter = collections.Counter()
    bynum: collections.Counter = collections.Counter()
    rows = []
    for c in cases:
        pp = os.path.join(a.pred, "%s.json" % c.case_id)
        if not os.path.exists(pp):
            continue
        gt = load_json(c.json_path)
        pr = load_json(pp)
        try:
            _f1, km = M.assign_components(gt, pr, cfg)
        except Exception as e:                                   # noqa: BLE001
            print("  ! %s: %s" % (c.case_id, e))
            continue
        if not km:
            continue
        back = {g: k for k, g in km.items()}
        gtp = gt.get("pins") or {}
        prp = pr.get("pins") or {}
        for gk, pk in back.items():
            gcomp = (gt.get("components") or {}).get(gk) or {}
            if M.canon_type(gcomp.get("type")) not in M.IC_TYPES:
                continue
            ga = gtp.get(gk) or {}
            pa = prp.get(pk) or {}
            if not ga or not pa:
                continue
            # ---- 通道 A：**按引脚号**配对（= 官方口径的 `pin.name_mismatch` 那一桶）----
            for gk2, gv in ga.items():
                if not isinstance(gv, dict):
                    continue
                num = M.pin_key_to_number(gk2)
                ov = pa.get("pin_%s" % num)
                if not isinstance(ov, dict):
                    continue
                on = str(ov.get("pinname") or "")
                gn = str(gv.get("pinname") or "")
                if on == gn:
                    bynum["number_matched_name_ok"] += 1
                    continue
                o, t = _n(on), _n(gn)
                if not o:
                    cat = "empty"
                elif not t:
                    cat = "gt_empty"
                elif o in t:
                    cat = "trunc"
                elif t in o:
                    cat = "over"
                elif o.isdigit():
                    cat = "our_name_is_a_number"
                elif _lev(o, t) <= 1 and min(len(o), len(t)) >= 3:
                    cat = "near(edit<=1)"
                else:
                    cat = "other"
                bynum["number_matched_name_BAD"] += 1
                bynum["bad:" + cat] += 1
                rows.append((cat, c.case_id, gk, "pin_%s" % num, num, on, gn,
                             str(ov.get("box") or "")))
            # ---- 通道 B：按**最近点**配对（看点对齐之后名字本身的质量）----
            gpts = [((float(v.get("point", {}).get("x", 0)),
                      float(v.get("point", {}).get("y", 0))), k2, v)
                    for k2, v in ga.items() if isinstance(v, dict)]
            pts = [((float(v.get("point", {}).get("x", 0)),
                     float(v.get("point", {}).get("y", 0))), k2, v)
                   for k2, v in pa.items() if isinstance(v, dict)]
            used = set()
            for gp, gk2, gv in gpts:
                best = None
                for i, (qp, pk2, pv) in enumerate(pts):
                    if i in used:
                        continue
                    d = math.hypot(gp[0] - qp[0], gp[1] - qp[1])
                    if best is None or d < best[0]:
                        best = (d, i, pk2, pv)
                if best is None or best[0] > a.tol:
                    continue
                used.add(best[1])
                ours = str(best[3].get("pinname") or "")
                theirs = str(gv.get("pinname") or "")
                if ours == theirs:
                    cats["exact"] += 1
                    continue
                o, t = _n(ours), _n(theirs)
                if not o:
                    cat = "empty"
                elif not t:
                    cat = "gt_empty"
                elif o != t and o in t:
                    cat = "trunc"
                elif o != t and t in o:
                    cat = "over"
                elif _lev(o, t) <= 1 and min(len(o), len(t)) >= 4:
                    cat = "near(edit<=1)"
                elif len(o) == len(t):
                    cat = "same_len_diff"
                else:
                    cat = "other"
                cats[cat] += 1

    tot = sum(cats.values())
    print("\n%s：引脚名错配成因分类" % a.split)
    print("\n[A] **按引脚号**配对（= 官方 `pin.name_mismatch` 那桶，%d 对）"
          % (bynum["number_matched_name_ok"] + bynum["number_matched_name_BAD"]))
    print("   number_matched_name_ok  %5d" % bynum["number_matched_name_ok"])
    print("   number_matched_name_BAD %5d" % bynum["number_matched_name_BAD"])
    for k, n in sorted(bynum.items()):
        if k.startswith("bad:"):
            print("      %-24s %5d" % (k[4:], n))
    print("\n[B] **按最近点**配对（<=%.1fpx，%d 对）" % (a.tol, tot))
    for k, n in cats.most_common():
        print("   %-16s %5d  %6.2f%%" % (k, n, 100.0 * n / max(tot, 1)))
    ex = cats.get("exact", 0)
    if tot - ex:
        print("\n   （非精确对里，可被「扩框重读」救的 = trunc + over = %d / %d = %.1f%%）"
              % (cats.get("trunc", 0) + cats.get("over", 0), tot - ex,
                 100.0 * (cats.get("trunc", 0) + cats.get("over", 0)) / (tot - ex)))
    print("\ntop %d 「号对上了但名不对」：" % a.top)
    for r in rows[:a.top]:
        print("   %-24s case=%-3s %-6s %-7s ours=%-14r gt=%-14r"
              % (r[0], r[1], r[2], r[4], r[5], r[6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
