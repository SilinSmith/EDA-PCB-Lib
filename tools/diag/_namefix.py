"""引脚名（`pinname`）的字符级混淆普查 —— `pin_f1` 的 tp 要求 pinname **逐字**相等。

用法：
    python work\\tools\\diag\\_namefix.py --pred work\\_submit_out220r21\\runs\\stage2 [--tol 5] [--top 30]

做法：走官方 `M.assign_components` 拿 keymap，把 GT 的每个 box 引脚与**我们同一颗元件
上最近的那个引脚点**配对（`--tol` 内），然后统计：
  * 逐字相等率
  * 施加若干「归一化」之后的相等率（`I/1`、`O/0`、`rn/m`、去非字母数字、去下划线/斜杠/空格）
  * 最常见的 (ours, gt) 差异对
目的：判断「我们的 OCR 是不是系统性地把某个字符读成了另一个」——如果是，
一个字符级后处理就能把一大批 `pin.name_mismatch` 变成 tp。
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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "eda-pcb") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "eda-pcb"))

from schematic_parser import metrics as M                    # noqa: E402
from schematic_parser.data import load_json                  # noqa: E402

cfg = M.EvalConfig()


def norm_variants(s: str) -> dict:
    """返回若干种归一化下的字符串。"""
    out = {}
    out["raw"] = s
    t = s.replace("1", "I")
    out["1->I"] = t
    out["I->1"] = s.replace("I", "1")
    t0 = s.replace("0", "O")
    out["0->O"] = t0
    out["O->0"] = s.replace("O", "0")
    t2 = s.replace("1", "I").replace("0", "O")
    out["1->I,0->O"] = t2
    t3 = s.replace("I", "1").replace("O", "0")
    out["I->1,O->0"] = t3
    t4 = s.replace("1", "I").replace("0", "O").replace("5", "S").replace("8", "B")
    out["+5->S,8->B"] = t4
    t5 = s.replace("I", "").replace("1", "")
    out["drop I/1"] = t5
    out["upper"] = s.upper()
    t6 = "".join(ch for ch in s.upper() if ch.isalnum())
    out["alnum"] = t6
    t7 = s.upper().replace("1", "I")
    out["alnum+1->I"] = "".join(ch for ch in t7 if ch.isalnum())
    t8 = s.upper().replace("0", "O").replace("1", "I")
    out["alnum+0->O,1->I"] = "".join(ch for ch in t8 if ch.isalnum())
    return out


VARIANT_KEYS = None
NUMSTAT: collections.Counter = collections.Counter()
INTKEY = re.compile(r"^\d+$")


def _sane_key(pk: str, nkeys: int) -> bool:
    """GT 的 `pin_<...>` 后缀是不是「像真实封装引脚号」：纯整数、1..256。"""
    suf = pk[4:] if pk.startswith("pin_") else pk
    return bool(INTKEY.match(suf)) and 1 <= int(suf) <= 256


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="200_train_cases")
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()

    from schematic_parser.data import discover
    cases = [c for c in discover() if c.split == a.split]
    globals()["VARIANT_KEYS"] = None
    hit = collections.Counter()
    pairs = collections.Counter()
    base = 0
    eq_any = collections.Counter()
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
            ga = gtp.get(gk) or {}
            pa = prp.get(pk) or {}
            if not ga or not pa:
                continue
            # ⚠️ 必须只留 GT 里 type 是 IC 的元件：非 IC 的 pinname 两边都是
            # `1`/`2` 这类位置号，会把相等率抬到 90% 以上（假信号）。
            gcomp = (gt.get("components") or {}).get(gk) or {}
            if M.canon_type(gcomp.get("type")) not in M.IC_TYPES:
                continue
            gpts = []
            for gk2, v in ga.items():
                if not isinstance(v, dict):
                    continue
                v = dict(v)
                v["_k"] = gk2
                gpts.append(((float(v.get("point", {}).get("x", 0)),
                              float(v.get("point", {}).get("y", 0))), v))
            pts = []
            for pk2, v in pa.items():
                if not isinstance(v, dict):
                    continue
                v = dict(v)
                v["_k"] = pk2
                pts.append(((float(v.get("point", {}).get("x", 0)),
                             float(v.get("point", {}).get("y", 0))), v))
            used = set()
            for gp, gv in gpts:
                best = None
                for i, (pp2, pv) in enumerate(pts):
                    if i in used:
                        continue
                    d = math.hypot(gp[0] - pp2[0], gp[1] - pp2[1])
                    if best is None or d < best[0]:
                        best = (d, i, pv)
                if best is None or best[0] > a.tol:
                    continue
                used.add(best[1])
                ours = str(best[2].get("pinname") or "")
                theirs = str(gv.get("pinname") or "")
                base += 1
                pairs[(ours, theirs)] += 1
                # ---- 号的通道：点已对齐的前提下，我们的 key 后缀与 GT 的比 ----
                gsuf = str(gv.get("_k") or "")
                ours_nm = str(best[2].get("_k") or "")
                sane = _sane_key(gsuf, 0)
                bucket = "sane" if sane else "insane"
                NUMSTAT[bucket + "_total"] += 1
                if gsuf == ours_nm:
                    NUMSTAT[bucket + "_numok"] += 1
                if ours == theirs:
                    NUMSTAT["nameok_" + bucket] += 1
                if ours == theirs:
                    eq_any["raw"] += 1
                vo = norm_variants(ours)
                vt = norm_variants(theirs)
                if VARIANT_KEYS is None:
                    globals()["VARIANT_KEYS"] = list(vo)
                for k in vo:
                    if vo[k] and vo[k] == vt[k]:
                        eq_any[k] += 1
    print("aligned pin pairs (nearest within %.1fpx, same component): %d" % (a.tol, base))
    if not base:
        return 0
    print()
    print("%-20s %8s %8s" % ("normalization", "equal", "rate"))
    for k in (globals()["VARIANT_KEYS"] or ["raw"]):
        print("%-20s %8d %7.2f%%" % (k, eq_any[k], 100.0 * eq_any[k] / base))
    print()
    print("top %d (ours, gt) mismatching pairs:" % a.top)
    shown = 0
    for (o, g), n in pairs.most_common():
        if o == g:
            continue
        print("   %4d  ours=%-22r gt=%-22r" % (n, o, g))
        shown += 1
        if shown >= a.top:
            break
    # ---- 引脚「号」的可读性分层：决定 num 通道还有多少空间 ----
    print()
    print("number channel, split by whether the GT key is a *sane package pin*")
    stat = collections.Counter()
    for key, n in NUMSTAT.items():
        stat[key] += n
    for k in ("sane_total", "sane_numok", "insane_total", "insane_numok",
              "nameok_sane", "nameok_insane"):
        print("   %-16s %6d" % (k, stat[k]))
    st, sn = stat["sane_total"], stat["sane_numok"]
    it, iN = stat["insane_total"], stat["insane_numok"]
    if st:
        print("   sane   num_ok rate = %.1f%%  (n=%d)" % (100.0 * sn / st, st))
    if it:
        print("   insane num_ok rate = %.1f%%  (n=%d)" % (100.0 * iN / it, it))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
