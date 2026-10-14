# -*- coding: utf-8 -*-
"""引脚通道的**判分漏斗**（逐键复刻 `metrics.pin_f1` 的四道关，把损失分类）。

判分器（`schematic_parser/metrics.py:446 pin_f1`）对 GT 里 `canon_type ∈ IC_TYPES`
的元件，按 `(元件, 脚号)` 建键，逐键要求：①预测端有同键 ②`pinname` 逐字符相等
③点距 ≤ `cfg.pin_tol`（5.0px）。本探针把每个键的失败原因归类，并额外给出
「去掉某一维」的对照，用来回答：**少脚 / 脚号错 / 名字错 / 几何偏** 各占多少。

用法（从 Y:\\workspace\\EDA 根跑）：
    python -X utf8 work\\_pr3funnel.py --runs pr3v,pr3b,gsd9
    python -X utf8 work\\_pr3funnel.py --runs pr3v --cases 1,2,3 --keys
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import (DEFAULT_DATA_ROOT, IC_TYPES, discover,   # noqa: E402
                                   normalise_doc, pin_key_to_number)
from schematic_parser.metrics import (EvalConfig, assign_components,       # noqa: E402
                                      canon_type, rewrite_keys)

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="pr3v,pr3b,gsd9")
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--keys", action="store_true", help="逐键打印失败原因")
a = ap.parse_args()

CFG = EvalConfig()


def _pt(raw):
    if isinstance(raw, dict):
        raw = (raw.get("x"), raw.get("y"))
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None


def flat(section, only):
    out = {}
    for comp, group in (section or {}).items():
        if only is not None and comp not in only:
            continue
        if not isinstance(group, dict):
            continue
        for pk, info in group.items():
            out[(comp, pin_key_to_number(pk))] = info if isinstance(info, dict) else {}
    return out


cases = {}
for c in discover(DEFAULT_DATA_ROOT):
    cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
    cases[cid] = c

for run in a.runs.split(","):
    print("=" * 100)
    print("run = %s" % run)
    hdr = ("%-4s %5s %5s %5s | %5s %5s %5s %5s | %5s %6s %6s | %6s %6s" % (
        "case", "gtN", "tp", "F1", "miss", "name", "pos", "extra",
        "5px", "med", "p90", "同件5", "同件20"))
    print(hdr)
    print("-" * len(hdr))
    tot = dict(gt=0, tp=0, miss=0, name=0, pos=0, extra=0, near5=0, same5=0, same20=0)
    dists = []
    same_d = []
    for cid in a.cases.split(","):
        if cid not in cases:
            continue
        gt = normalise_doc(json.load(open(str(cases[cid].json_path), encoding="utf-8")))
        box = {k for k, v in gt["components"].items()
               if canon_type((v or {}).get("type")) in IC_TYPES}
        gf = flat(gt["pins"], box)
        pp = (ROOT / "work" / ("_fast_out_%s" % run) / "final_results"
              / ("%s_Pred.json" % cid))
        if not pp.exists():
            print("%-4s   （没有预测 %s）" % (cid, pp.name))
            continue
        pred = normalise_doc(json.load(open(str(pp), encoding="utf-8")))
        _, keymap = assign_components(gt, pred, CFG, {})
        pred = rewrite_keys(pred, keymap)
        pf = flat(pred["pins"], box)

        n_tp = n_miss = n_name = n_pos = n_extra = 0
        near5 = 0
        d_list = []
        by_comp = {}
        for (c, p), info in pf.items():
            by_comp.setdefault(c, []).append(_pt(info.get("point")))
        same_hits = 0
        same20 = 0
        for key, g in gf.items():
            comp = key[0]
            p = pf.get(key)
            gp = _pt(g.get("point"))
            if p is None:
                n_miss += 1
                if gp:
                    cands = [q for q in by_comp.get(comp, []) if q]
                    if cands:
                        d = min(math.hypot(gp[0] - q[0], gp[1] - q[1]) for q in cands)
                        same_d.append(d)
                        if d <= 5.0:
                            same_hits += 1
                        if d <= 20.0:
                            same20 += 1
                continue
            pp_ = _pt(p.get("point"))
            if (p.get("pinname") or "") != (g.get("pinname") or ""):
                n_name += 1
            elif gp is None or pp_ is None:
                n_pos += 1
            elif math.hypot(gp[0] - pp_[0], gp[1] - pp_[1]) > CFG.pin_tol:
                n_pos += 1
                d_list.append(math.hypot(gp[0] - pp_[0], gp[1] - pp_[1]))
            else:
                n_tp += 1
        n_extra = len([k for k in pf if k not in gf])
        f1 = (2.0 * n_tp / (2.0 * n_tp + n_extra + n_miss + n_name + n_pos)) if gf else 0.0
        med = statistics.median(d_list) if d_list else float("nan")
        p90 = (sorted(d_list)[int(0.9 * (len(d_list) - 1))] if d_list else float("nan"))
        print("%-4s %5d %5d %5.3f | %5d %5d %5d %5d | %5s %6s %6s | %6d %6d" % (
            cid, len(gf), n_tp, f1, n_miss, n_name, n_pos, n_extra,
            "%.0f%%" % (100.0 * near5 / max(1, len(gf))),
            "%.1f" % med if d_list else "-", "%.1f" % p90 if d_list else "-",
            same_hits, same20))
        tot["gt"] += len(gf); tot["tp"] += n_tp; tot["miss"] += n_miss
        tot["name"] += n_name; tot["pos"] += n_pos; tot["extra"] += n_extra
        tot["near5"] += near5
        tot["same5"] += same_hits; tot["same20"] += same20
        dists.extend(d_list)
        if a.keys:
            for key, g in sorted(gf.items()):
                p = pf.get(key)
                gp = _pt(g.get("point"))
                if p is None:
                    same = sorted(k[1] for k in pf if k[0] == key[0])
                    gkeys = sorted(k[1] for k in gf if k[0] == key[0])
                    print("    MISS case%s %s.%s gtpin=%s gt点=%s | 我方该元件键=%s | GT键=%s" % (
                        cid, key[0], key[1], g.get("pinname"),
                        ("(%.0f,%.0f)" % gp) if gp else "-", same, gkeys))
                    continue
                pp_ = _pt(p.get("point"))
                if (p.get("pinname") or "") != (g.get("pinname") or ""):
                    print("    NAME %s.%s gt=%r pred=%r" % (key[0], key[1],
                                                            g.get("pinname"), p.get("pinname")))
                elif gp and pp_ and math.hypot(gp[0] - pp_[0], gp[1] - pp_[1]) > CFG.pin_tol:
                    print("    POS  %s.%s gt=%s pred=%s d=%.1f" % (
                        key[0], key[1], gp, pp_, math.hypot(gp[0] - pp_[0], gp[1] - pp_[1])))
    f1 = (2.0 * tot["tp"] / (2.0 * tot["tp"] + tot["extra"] + tot["miss"] + tot["name"] + tot["pos"])
          if tot["gt"] else 0.0)
    print("-" * len(hdr))
    print("合计 gt=%d tp=%d F1=%.4f | 缺键 %d · 名字错 %d · 位置超差 %d · 多余 %d" % (
        tot["gt"], tot["tp"], f1, tot["miss"], tot["name"], tot["pos"], tot["extra"]))
    print("     位置超差者点距 中位 %s 均值 %s n=%d | 缺键里「同元件任意预测点在 5px 内」%d / 20px 内 %d"
          % ("%.1f" % statistics.median(dists) if dists else "-",
             "%.1f" % statistics.mean(dists) if dists else "-", len(dists),
             tot["same5"], tot["same20"]))
    if same_d:
        print("     缺键 GT 点 → 同元件最近预测点：中位 %.1f 均值 %.1f n=%d（<=5px 的 %d）"
              % (statistics.median(same_d), statistics.mean(same_d), len(same_d),
                 sum(1 for d in same_d if d <= 5.0)))
