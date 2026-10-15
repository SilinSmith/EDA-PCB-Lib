# -*- coding: utf-8 -*-
"""并网失分归因：GT 网成员里我方缺的那些，到底是「缺脚」还是「脚号对不上」。

用法：python -X utf8 work\\_netgap.py [arm=defconfirm] [cases=work/_cases10gt.txt]
分类：comp_missing（该元件我方压根没发出）/ no_pin_at_all / pin_pos_ok_num_diff（有脚、
      脚点对得上、只是号不同）/ pin_pos_far / gt_no_point
"""
import collections
import io
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDA = os.path.join(ROOT, "eda-pcb")
if EDA not in sys.path:
    sys.path.insert(0, EDA)
sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser import metrics as M  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "defconfirm"
CASES = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "work", "_cases10gt.txt")
PRED = os.path.join(ROOT, "work", "_fast_out_" + ARM, "final_results")

IDX = {}
for _c in discover(DEFAULT_DATA_ROOT):
    if _c.split == "200_train_cases":
        if _c.json_path.endswith("target_named.json"):
            IDX[_c.case_id] = _c
    elif _c.split == "10GT":
        IDX[_c.case_id] = _c
    elif _c.split == "10_GTcase" and _c.json_path.endswith("_target.json"):
        IDX[_c.case_id] = _c


def members(hg):
    if not hg:
        return []
    t = str(hg).strip()
    if t.startswith("("):
        t = t[1:]
    if t.endswith(")"):
        t = t[:-1]
    if not t or t == "\u2205":
        return []
    return [p.strip() for p in t.split(",") if p.strip()]


def split_member(m):
    if "." not in m:
        return m, None
    c, p = m.rsplit(".", 1)
    return c, p


def _xy(p):
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    return float(p[0]), float(p[1])


tally = collections.Counter()
bytype = collections.Counter()
examples = []
cfg = M.EvalConfig()
tot_mem = tot_tp = 0
for cid in [x.strip() for x in io.open(CASES, encoding="utf-8").read().strip().split(",") if x.strip()]:
    case = IDX.get(cid)
    pp = os.path.join(PRED, cid + "_Pred.json")
    if case is None or not os.path.exists(pp):
        continue
    gt, pr0 = load_json(case.json_path), load_json(pp)
    _f, km = M.assign_components(gt, pr0, cfg, {})
    pr = M.rewrite_keys(pr0, km)
    gtc, prc = gt.get("components") or {}, pr.get("components") or {}
    gtp, prp = gt.get("pins") or {}, pr.get("pins") or {}
    allour = set()
    for nk, nv in (pr.get("nets") or {}).items():
        allour |= set(members((nv or {}).get("hyperGraph")))
    for nk, nv in (gt.get("nets") or {}).items():
        gm = members((nv or {}).get("hyperGraph"))
        tot_mem += len(gm)
        for m in gm:
            c, num = split_member(m)
            if m in allour:
                tot_tp += 1
                continue
            t = M.canon_type((gtc.get(c) or {}).get("type"))
            if c not in prc:
                tag = "comp_missing"
            else:
                gpv = None
                for pk, pv in (gtp.get(c) or {}).items():
                    if M.pin_key_to_number(pk) == num:
                        gpv = pv
                        break
                if gpv is None or not gpv.get("point"):
                    tag = "gt_no_point"
                else:
                    gx, gy = _xy(gpv["point"])
                    best = None
                    for pk, pv in (prp.get(c) or {}).items():
                        if not pv.get("point"):
                            continue
                        x, y = _xy(pv["point"])
                        d = math.hypot(x - gx, y - gy)
                        if best is None or d < best:
                            best = d
                    if best is None:
                        tag = "no_pin_at_all"
                    elif best <= 5.0:
                        tag = "pin_pos_ok_num_diff"
                    else:
                        tag = "pin_pos_far"
            tally[tag] += 1
            bytype[(t, tag)] += 1
            if len(examples) < 14:
                examples.append((cid, nk, m, t, tag))

print("== 并网成员归因 %s cases=%s ==" % (ARM, CASES))
print("GT 网成员 %d｜我方命中 %d｜缺 %d（我方多出的 fp 不在本表）" % (tot_mem, tot_tp, sum(tally.values())))
for k, v in tally.most_common():
    print("   %-22s %4d" % (k, v))
print("按类型（类型, 归因）:")
for (t, tag), v in sorted(bytype.items(), key=lambda kv: -kv[1]):
    print("   %-10s %-22s %4d" % (t, tag, v))
print("例子:")
for e in examples:
    print("   ", e)
