# -*- coding: utf-8 -*-
"""`pin_f1` 的失配按**元件**归因：extra / missing / name_mismatch / position_mismatch 各是谁贡献的。

用官方口径 `metrics.pin_f1` 的同一套 `assign_components` 结果，把每条失配摊回元件键上。

    python work\\tools\\diag\\_pinextra.py --pred work\\_gt10_vlmh
    python work\\tools\\diag\\_pinextra.py --pred work\\_gt10_s2h --top 25
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M          # noqa: E402
from schematic_parser.data import discover          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="预测目录（每张 <case>.json）")
    ap.add_argument("--cases", default="", help="逗号分隔；默认全部")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    pred_dir = Path(a.pred)
    if not pred_dir.is_absolute():
        pred_dir = ROOT / pred_dir
    cases = {c.case_id: c for c in discover()}
    want = [x.strip() for x in a.cases.split(",") if x.strip()] or None

    cfg = M.EvalConfig()
    IC = set(M.IC_TYPES)
    buckets = collections.Counter()          # 失配种类 -> 次数
    by_comp = collections.defaultdict(int)   # (种类, 元件键) -> 次数
    by_type = collections.defaultdict(int)   # (种类, canon_type) -> 次数

    for cid, c in sorted(cases.items()):
        if want and cid not in want:
            continue
        pj = pred_dir / ("%s.json" % cid)
        if not (pj.is_file() and c.json_path):
            continue
        gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
        pr = json.loads(pj.read_text(encoding="utf-8"))
        f1, keymap = M.assign_components(gt, pr, cfg)
        g2p = {gk: pk for pk, gk in keymap.items()}
        # GT 侧：只算 IC 类型（pin_box_only）
        gcomps = {k: v for k, v in (gt.get("components") or {}).items()
                  if str(v.get("type") or "").lower() in IC}
        gpins = gt.get("pins") or {}
        ppins = pr.get("pins") or {}
        for gk, gv in gcomps.items():
            gsec = gpins.get(gk) or {}
            pk = g2p.get(gk)
            psec = (ppins.get(pk) or {}) if pk else {}
            gflat = {}
            for pkey, info in gsec.items():
                num = M.pin_key_to_number(pkey)
                gflat[num] = info
            pflat = {}
            for pkey, info in psec.items():
                num = M.pin_key_to_number(pkey)
                pflat[num] = info
            for num, g in gflat.items():
                p = pflat.get(num)
                if p is None:
                    buckets["extra"] += 0     # 占位：extra 在下面按 pred 侧算
            for num, p in pflat.items():
                g = gflat.get(num)
                if g is None:
                    buckets["extra"] += 1
                    by_comp["extra", pk or "(未配上GT)"] += 1
                    by_type["extra", str(gv.get("type"))] += 1
                else:
                    if p.get("pinname") != g.get("pinname"):
                        buckets["name_mismatch"] += 1
                        by_comp["name_mismatch", pk] += 1
                        by_type["name_mismatch", str(gv.get("type"))] += 1
                    pa, ga = p.get("point") or {}, g.get("point") or {}
                    try:
                        d = ((float(pa["x"]) - float(ga["x"])) ** 2
                             + (float(pa["y"]) - float(ga["y"])) ** 2) ** 0.5
                    except Exception:  # noqa: BLE001
                        d = 0.0
                    if d > cfg.pin_tol:
                        buckets["position_mismatch"] += 1
                        by_comp["position_mismatch", pk] += 1
                        by_type["position_mismatch", str(gv.get("type"))] += 1
            for num, g in gflat.items():
                if num not in pflat:
                    buckets["missing"] += 1
                    by_comp["missing", pk or "(未配上GT)"] += 1
                    by_type["missing", str(gv.get("type"))] += 1
        # 没有 GT IC 但 pred 有 IC 的（组件键整个错）另算
    print("失配总量: %s" % dict(buckets))
    print()
    print("== 按 (种类, 元件键) 排 %d 名 ==" % a.top)
    for (kind, key), n in sorted(by_comp.items(), key=lambda kv: -kv[1])[:a.top]:
        print("   %-18s %-16s %d" % (kind, key, n))
    print()
    print("== 按 (种类, 元件类型) ==")
    agg = collections.defaultdict(collections.Counter)
    for (kind, ct), n in by_type.items():
        agg[kind][ct] += n
    for kind in ("extra", "missing", "name_mismatch", "position_mismatch"):
        print("   %-18s %s" % (kind, dict(agg[kind].most_common(8))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
