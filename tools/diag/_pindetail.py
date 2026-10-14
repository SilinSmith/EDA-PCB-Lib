# -*- coding: utf-8 -*-
"""§5.111 取证：某个 case 的引脚通道（protocol2 的 `pin_f1`）为什么低。

protocol2 的 pin 通道只看 **GT 里 type 为 box 的元件**的 (元件, 脚号) 键：
键在 GT 有、我方没有 ⇒ `pin.missing`（fn）；我方多出来 ⇒ `pin.extra`（fp）；
同键但 `pinname` 不同 / `point` 距离 > `pin_tol` ⇒ 各记一次 fp+fn。
本探针把 `evaluate_case` 的 `diagnostic` 分桶打出来，再把**逐脚对照**列清楚
（缺的、多的、名字不符的、位置偏的），用来判断是「整个元件没配上」还是「脚级错」。

用法（从 `Y:\\workspace\\EDA` 根跑）：
    python -X utf8 work\\_pindetail.py --case 8 [--run gsd9] [--max 60]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import (DEFAULT_DATA_ROOT, IC_TYPES, discover,   # noqa: E402
                                   normalise_doc, pin_key_to_number)
from schematic_parser.metrics import (EvalConfig, assign_components,       # noqa: E402
                                      canon_type, evaluate_case, rewrite_keys)


def _pt(raw):
    if isinstance(raw, dict):
        raw = (raw.get('x'), raw.get('y'))
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None


ap = argparse.ArgumentParser()
ap.add_argument("--case", default="8")
ap.add_argument("--run", default="gsd9")
ap.add_argument("--max", type=int, default=60)
args = ap.parse_args()

cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c
if args.case not in cases:
    sys.exit("没有这个 case：%s（有：%s）" % (args.case, ",".join(sorted(cases))))
cas = cases[args.case]
pred_p = ROOT / "work" / ("_fast_out_%s" % args.run) / "final_results" / ("%s_Pred.json" % args.case)
if not pred_p.exists():
    sys.exit("没有预测：%s" % pred_p)
gt_doc = json.load(open(str(cas.json_path), encoding="utf-8"))
pred_doc = json.load(open(str(pred_p), encoding="utf-8"))

cfg = EvalConfig()
sc = evaluate_case(gt_doc, pred_doc, args.case, cfg)
print("case %s  图 %s" % (args.case, cas.png_path))
print("预测 %s" % pred_p)
print("总分 %.2f | 元件 %.4f 引脚 %.4f 网络 %.4f 导线 %.4f"
      % (sc.score, sc.component.f1, sc.pin.f1, sc.net_hypergraph.f1, sc.net_line.f1))
print("引脚通道 tp/fp/fn = %d / %d / %d  (precision %.4f recall %.4f)"
      % (sc.pin.tp, sc.pin.fp, sc.pin.fn, sc.pin.precision, sc.pin.recall))
print("--- 分桶 ---")
for k in sorted(sc.diagnostic):
    print("    %-28s %s" % (k, sc.diagnostic[k]))

gt, pred = normalise_doc(gt_doc), normalise_doc(pred_doc)
comp, keymap = assign_components(gt, pred, cfg, {})
pred2 = rewrite_keys(pred, keymap)
box = {k for k, v in gt['components'].items()
       if canon_type((v or {}).get('type')) in IC_TYPES}
print("--- GT 里的 box 元件 %d 个：%s" % (len(box), ",".join(sorted(box))))
print("--- keymap（预测键 -> GT 键）：%s" % (keymap or "（空）"))


def flat(section, only=None):
    out = {}
    for c, g in section.items():
        if only is not None and c not in only:
            continue
        if not isinstance(g, dict):
            continue
        for pk, info in g.items():
            out[(c, pin_key_to_number(pk))] = info
    return out


gf, pf = flat(gt['pins'], box), flat(pred2['pins'], box)
rows = []
for key in sorted(set(gf) | set(pf)):
    g, p = gf.get(key), pf.get(key)
    if g is None:
        tag = "多余 fp"
    elif p is None:
        tag = "缺失 fn"
    elif not isinstance(p, dict):
        tag = "坏形状"
    elif p.get('pinname') != g.get('pinname'):
        tag = "名字不符"
    else:
        a, b = _pt(g.get('point')), _pt(p.get('point'))
        if a is None or b is None:
            tag = "无 point"
        elif ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 > cfg.pin_tol:
            tag = "位置偏 %.1fpx" % ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        else:
            tag = "ok"
    rows.append((tag, key[0], str(key[1]),
                 (g or {}).get('pinname'), (p or {}).get('pinname'),
                 _pt((g or {}).get('point')), _pt((p or {}).get('point'))))
print("--- 逐脚（GT box 元件的键 ∪ 我方同键）= %d 行 ---" % len(rows))
for tag, c, num, gn, pn, gp, pp in rows[:args.max]:
    print("    %-12s %-8s pin %-4s gt_name=%-10s pred_name=%-10s gt_pt=%s pred_pt=%s"
          % (tag, c, num, gn, pn,
             ("(%.0f,%.0f)" % gp) if gp else None, ("(%.0f,%.0f)" % pp) if pp else None))
print("--- 我方 box 元件（GT box 键 + 预测里其它有脚的元件） ---")
for c in sorted(pred2['pins']):
    g = pred2['pins'][c]
    if isinstance(g, dict) and g:
        print("    %-10s 脚数 %d  %s" % (c, len(g), "（在 GT box 列表里）" if c in box else "（GT 没有/不是 box）"))
