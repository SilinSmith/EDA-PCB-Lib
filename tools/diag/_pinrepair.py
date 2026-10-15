#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""引脚文本的「受约束修复」探针（§5.133 第 0 步）。只测量，不改提取逻辑。

核心 GT-blind 机制：**用引脚名集合反查库条目**（而不是靠 `partname` 的字符串相似）。
`stage2/libpins.py` 的 `key_for(partname)` 用的是元件名模糊匹配，OCR 一旦把元件名读歪
（`MP223CGU2`/`15ND`/`79WM12`/`SY1813BADC`）就查不到库；但**我们已读出的引脚名**与库里
该器件引脚名集合的重合度很高，用 Dice 反查就能定位到唯一库条目，再用库的名字↔号映射
补全/纠正引脚名与号。全部信号都不看 GT。

GT 只用于**评估**：`--dump` 打逐槽表，`--apply` 给出真实分数效果。

用法：
  python -X utf8 tools/diag/_pinrepair.py [--run-dir ...] [--cases 1,2,...] [--dump] [--apply]
"""
import argparse
import importlib.util
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EDA = os.path.dirname(os.path.dirname(HERE))
if EDA not in sys.path:
    sys.path.insert(0, EDA)

_spec = importlib.util.spec_from_file_location("_pinrot2", os.path.join(HERE, "_pinrot2.py"))
R2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R2)

from schematic_parser.data import DEFAULT_DATA_ROOT, discover          # noqa: E402
from schematic_parser.metrics import (                                  # noqa: E402
    EvalConfig, assign_components, rewrite_keys, pin_f1, net_hypergraph_f1,
)

CASES10 = [str(i) for i in range(1, 11)]
DEFAULT_RUN = "Y:/workspace/EDA/work/_fast_out_delivery_default"
LIB = os.path.join(EDA, "data", "BoxICLibrary_200cases.json")

CONFUSION = [("G", "C"), ("O", "0"), ("O", "D"), ("I", "1"), ("I", "L"), ("S", "5"),
             ("B", "8"), ("Z", "2"), ("A", "4"), ("U", "V"), ("Q", "O"), ("G", "6"),
             ("N", "M"), ("R", "P"), ("E", "F"), ("T", "7"), ("C", "6"), ("D", "0"),
             ("L", "1"), ("M", "H"), ("W", "V"), ("N", "H"), ("V", "Y"), ("U", "J"),
             ("C", "G"), ("H", "N")]
CONF = set()
for _a, _b in CONFUSION:
    CONF.add((_a, _b))
    CONF.add((_b, _a))
SUB_C = 0.35


def norm(s):
    return re.sub(r"[^0-9A-Za-z/+()\-]", "", str(s or "")).upper()


def _sub(a, b):
    return SUB_C if (a, b) in CONF else 1.0


def wdist(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 1.0
    la, lb = len(a), len(b)
    prev = [float(j) for j in range(lb + 1)]
    for i in range(1, la + 1):
        cur = [float(i)] + [0.0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1.0, cur[j - 1] + 1.0,
                         prev[j - 1] + (0.0 if a[i - 1] == b[j - 1] else _sub(a[i - 1], b[j - 1])))
        prev = cur
    return prev[lb] / float(max(la, lb))


def load_lib():
    raw = json.load(io.open(LIB, encoding="utf-8"))
    keys = list(raw.keys())
    names, name_num, all_names = {}, {}, set()
    for k in keys:
        ns, n2n = set(), {}
        for p in (raw[k].get("pins") or []):
            nm, nu = str(p.get("name") or "").strip(), str(p.get("number") or "").strip()
            if nm:
                ns.add(norm(nm))
                all_names.add(norm(nm))
                if nu:
                    n2n.setdefault(norm(nm), set()).add(str(nu))
        names[k] = ns
        name_num[k] = {a: sorted(b) for a, b in n2n.items()}
    return dict(raw=raw, keys=keys, names=names, name_num=name_num, all_names=all_names)


def ident_lib(ns, L):
    """用我方读出的引脚名集合反查库条目。返回 (dice, key, hits)。"""
    best = (0.0, None, 0)
    for lk in L["keys"]:
        ln = L["names"][lk]
        if not ln:
            continue
        hits = sum(1 for a in ns if min((wdist(a, b) for b in ln), default=9.0) <= 0.25)
        dice = 2.0 * hits / float(len(ns) + len(ln))
        if dice > best[0]:
            best = (dice, lk, hits)
    return best


def plan(slots, L, a):
    """GT-blind 规划：返回 (lib_key, dice, hits, [(slot_dict, name, number, how), ...])。"""
    elig = [s for s in slots if s["trust"] == "trusted"]
    if a.strict_reason:
        elig = [s for s in elig if str(s.get("reason") or "") == "frame_attached_normal_stroke"]
    if len(elig) < 2:
        return None, 0.0, 0, []
    ns = set(norm(s["nam"]) for s in elig if s["nam"] and str(s["nam"]).strip())
    ns |= set(norm(s["nam"]) for s in slots if s["trust"] == "trusted" and s["nam"])
    dice, lk, hits = ident_lib(ns, L)
    if lk is None or dice < a.dice or hits < a.hits:
        return lk, dice, hits, []
    ln = L["names"][lk]
    n2n = L["name_num"][lk]
    # 1) 直接匹配
    used = set()
    out = []
    scored = []
    for s in elig:
        rd = norm(s["nam"])
        if not rd:
            continue
        bd, bn = 9.0, None
        for c in ln:
            if c in used:
                continue
            d = wdist(rd, c)
            if d < bd:
                bd, bn = d, c
        scored.append((bd, s, bn))
    scored.sort(key=lambda t: t[0])
    taken = set()
    for bd, s, bn in scored:
        if bn is None or bn in used or bd > a.thr:
            continue
        used.add(bn)
        taken.add(id(s))
        out.append((s, bn, num_of(n2n, bn), "match"))
    # 2) 消元：未被占用的库名 / 未匹配的槽
    free = [s for s in elig if id(s) not in taken]
    left = [c for c in ln if c not in used]
    if free and left and a.elim > 0.0:
        pairs = []
        for s in free:
            rd = norm(s["nam"])
            for c in left:
                pairs.append((wdist(rd, c) if rd else a.elim, s, c))
        pairs.sort(key=lambda t: t[0])
        us, uc = set(), set()
        for d, s, c in pairs:
            if id(s) in us or c in uc or d > a.elim:
                continue
            us.add(id(s))
            uc.add(c)
            out.append((s, c, num_of(n2n, c), "elim"))
    return lk, dice, hits, out


def num_of(n2n, name):
    v = n2n.get(name)
    if not v:
        return None
    v = [x for x in v if str(x).isdigit()]
    return v[0] if len(v) == 1 else None


def run(a):
    L = load_lib()
    index = {str(c.case_id): c for c in discover(a.data_root)}
    cfg = EvalConfig()
    base_pin = base_net = ap_pin = ap_net = 0.0
    ncase = 0
    stat = dict(fix=0, broke=0, phr=0, miss=0, invent=0, stillbad=0)
    rows = []
    for cid in [x.strip() for x in a.cases.split(",") if x.strip()]:
        case = index.get(cid)
        if case is None:
            continue
        r = R2.analyse_case(cid, case, a.run_dir, cfg, a.show)
        if r is None:
            continue
        base_pin += r["base_pin"]
        base_net += r["base_net"]
        ncase += 1
        bykey = {}
        for s in r["slots"]:
            bykey.setdefault(s["key"], []).append(s)
        pending = []
        for key, ss in bykey.items():
            lk, dice, hits, plan_ = plan(ss, L, a)
            g = ss[0]["gnear"]
            rows.append(("HDR", cid, key, "", "lib=%s dice=%.2f hits=%d n_plan=%d" % (lk, dice, hits, len(plan_))))
            for s in ss:
                gn = s["gnear"]
                gname = norm(gn[1][1]) if gn else ""
                cur = norm(s["nam"])
                got = next((p for p in plan_ if p[0] is s), None)
                newname = got[1] if got else cur
                if got and got[3] == "match" and newname == cur:
                    got = None
                    newname = cur
                # 评估（before = 槽当前读到的名，after = 修复后的名）
                before_ok = bool(gname) and cur == gname
                after = newname if got else cur
                after_ok = bool(gname) and after == gname
                if got:
                    if after_ok and not before_ok:
                        stat["fix"] += 1
                    elif before_ok and not after_ok:
                        stat["broke"] += 1
                    elif after_ok:
                        stat["phr"] += 1
                    elif not gname:
                        stat["invent"] += 1
                    else:
                        stat["miss"] += 1
                else:
                    if before_ok:
                        stat["phr"] += 1
                    elif gname:
                        stat["stillbad"] += 1
                if got:
                    gnum = got[2] or (str(s["num"]) if s["num"] and str(s["num"]).isdigit() else None)
                    if gnum:
                        pending.append((key, gnum, newname, s["doc_pt"], s["edge"]))
                rows.append(("ROW", cid, key, s["slot"], "%s %-8r %-8r -> %s/%s %s | GT %s"
                             % (s["edge"], s["num"], s["nam"],
                                (got[2] if got else "-"), (newname if got else "-"),
                                (got[3] if got else "keep"),
                                ("none" if not gn else "%s/%s" % (gn[1][0], gn[1][1])))))
        if a.apply:
            doc = R2._apply_oracle(r["base_doc"], pending)
            cp = pin_f1(r["gt"], doc, cfg, {}).f1
            cn = net_hypergraph_f1(r["gt"], doc, cfg, {})[0].f1
            ap_pin += cp
            ap_net += cn
            print("  case %-3s pin %.4f -> %.4f (%+.4f) | netH %.4f -> %.4f (%+.4f) | 计划 %d"
                  % (cid, r["base_pin"], cp, cp - r["base_pin"], r["base_net"], cn, cn - r["base_net"],
                     len(pending)))
        else:
            ap_pin += r["base_pin"]
            ap_net += r["base_net"]
    if a.dump:
        print("case comp slot detail")
        for t in rows:
            print("  " + "  ".join(str(x) for x in t[1:]))
    n = max(1, ncase)
    print()
    print("== 库条目「引脚名集合反查」修复（dice>=%.2f hits>=%d thr=%.2f elim=%.2f）=="
          % (a.dice, a.hits, a.thr, a.elim))
    print("命中 GT 脚的槽：修好 %d / 改错 %d / 仍错 %d / 修到非 GT 名 %d / 本已对且未动 %d / 幻影槽被发明 %d"
          % (stat["fix"], stat["broke"], stat["stillbad"], stat["miss"], stat["phr"], stat["invent"]))
    print("分数：pin %.4f -> %.4f (%+.4f = %+.2f 分) | netH %.4f -> %.4f (%+.4f = %+.2f 分)"
          % (base_pin / n, ap_pin / n, (ap_pin - base_pin) / n, 25.0 * (ap_pin - base_pin) / n,
             base_net / n, ap_net / n, (ap_net - base_net) / n, 35.0 * (ap_net - base_net) / n))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--cases", default=",".join(CASES10))
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--dice", type=float, default=0.55)
    ap.add_argument("--hits", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.34)
    ap.add_argument("--elim", type=float, default=0.80)
    ap.add_argument("--strict-reason", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
