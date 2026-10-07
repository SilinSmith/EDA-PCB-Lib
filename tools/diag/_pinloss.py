"""引脚失分分解：把 220 张的 `pin.missing`（8335）拆成
「box 元件的键根本没对上」与「键对上了、但引脚没枚举出来」。

官方口径（`schematic_parser/metrics.py:426-464 pin_f1`）：
  1. 只算 GT 里 `canon_type(type) in IC_TYPES = {box, ic, boxic}` 的元件；
  2. 配对键 = `(元件键, pin_key_to_number(引脚号))`，**先按身份配、不按几何配**；
  3. 逐对要 `pinname` 全等、点距 <= `pin_tol=5.0`。
`flat(pred)` 也用 GT 的 `box_comps` 过滤 ⇒ 我们的引脚如果挂在**非 GT box 键**下，
既不记 tp、也不记 fp，只在 GT 侧记成 fn。

用法：
  python work\\tools\\diag\\_pinloss.py                    # 默认 work\\_fusion_out220no4
  python work\\tools\\diag\\_pinloss.py --dir work\\_fusion_out220no4 --show 6
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import data as D                                    # noqa: E402
from schematic_parser.metrics import (                                    # noqa: E402
    EvalConfig, assign_components, canon_type, normalise_doc,
    pin_key_to_number, rewrite_keys, _point,
)


def load_cases(want=None):
    out = {}
    for c in D.discover(D.DEFAULT_DATA_ROOT):
        if not c.json_path:
            continue
        if want and c.case_id not in want:
            continue
        out[c.case_id] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "work" / "_fusion_out220no4"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--show", type=int, default=4, help="打印几个样例的引脚号对照")
    ap.add_argument("--by-split", action="store_true",
                    help="按 case.split（KiCad/Datasheet/Altium…）分组，看失分是结构性的还是我们的")
    ap.add_argument("--by-source", action="store_true",
                    help="按文件名后缀（-KiCad / -Datasheet / -Altium Designer）分组，"
                         "即**画图工具**；这才是「引脚号画没画在图上」的真正分界")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    want = {x.strip() for x in a.cases.split(",") if x.strip()} or None
    cases = load_cases(want)
    pdir = Path(a.dir)
    cfg = EvalConfig()

    agg = Counter()
    per_split: dict = {}
    box_by_type = Counter()          # box 元件按 GT type
    percase = []
    samples = []

    for cid, case in sorted(cases.items()):
        # 三种目录布局都要认：`<dir>/<case>.json`（`_fusion_out*`）、
        # `<dir>/final_results/<case>_Pred.json`（`submission\run.py` 的产物）、
        # `<dir>/runs/fusion/<case>.json`。以前只认第一种，于是拿
        # `work\_fast_out` 当 `--dir` 时会**静默统计出 0 个元件**（看着像
        # 「GT 里没有 box」，其实是根本没读到预测）。
        f = pdir / f"{cid}.json"
        if not f.exists():
            f = pdir / "final_results" / f"{cid}_Pred.json"
        if not f.exists():
            f = pdir / "runs" / "fusion" / f"{cid}.json"
        if not f.exists():
            continue
        gt = D.load_json(case.json_path)
        pred = json.loads(f.read_text(encoding="utf-8"))
        gt_n, pred_n = normalise_doc(gt), normalise_doc(pred)
        _comp, keymap = assign_components(gt_n, pred_n, cfg, {})
        pred2 = rewrite_keys(pred_n, keymap)

        box_comps = {k for k, v in gt_n["components"].items()
                     if canon_type((v or {}).get("type")) in D.IC_TYPES}
        mapped = set(keymap.values())
        matched = box_comps & mapped
        unmatched = box_comps - mapped

        def flat(sec, keys):
            out = {}
            for comp, group in sec.items():
                if comp not in keys or not isinstance(group, dict):
                    continue
                for pk, info in group.items():
                    out[(comp, pin_key_to_number(pk))] = (pk, info)
            return out

        gf = flat(gt_n["pins"], box_comps)
        pf = flat(pred2["pins"], box_comps)

        st = Counter()
        st["box_gt"] = len(box_comps)
        st["box_matched"] = len(matched)
        for k in box_comps:
            box_by_type[canon_type((gt_n["components"].get(k) or {}).get("type"))] += 1
        for (comp, _n) in gf:
            st["gt_pins"] += 1
            if comp in matched:
                st["gt_pins_matched_comp"] += 1
            else:
                st["gt_pins_unmatched_comp"] += 1
        # 我们到底在这些键下发了几个引脚
        pred_pins_per_comp = Counter(comp for (comp, _n) in pf)
        zero = sum(1 for k in matched if pred_pins_per_comp.get(k, 0) == 0)
        st["matched_box_no_pred_pin"] = zero

        for key in set(gf) | set(pf):
            comp = key[0]
            g = gf.get(key)
            p = pf.get(key)
            tag = "matched" if comp in matched else "unmatched"
            if g is None:
                st[f"fp_{tag}"] += 1
                continue
            if p is None:
                st[f"miss_{tag}"] += 1
                continue
            if p[1].get("pinname") != g[1].get("pinname"):
                st[f"name_{tag}"] += 1
                continue
            ax, bx = _point(g[1].get("point")), _point(p[1].get("point"))
            if ax is None or bx is None:
                st[f"nopoint_{tag}"] += 1
                continue
            if math.hypot(ax[0] - bx[0], ax[1] - bx[1]) > cfg.pin_tol:
                st[f"pos_{tag}"] += 1
                continue
            st[f"tp_{tag}"] += 1

        agg.update(st)
        _stem = Path(str(case.json_path)).stem
        _suf = _stem.split("-", 1)[1].rsplit("_target", 1)[0] if "-" in _stem else "?"
        per_split.setdefault(_suf if a.by_source else case.split, Counter()).update(st)
        percase.append((cid, st["gt_pins"], st["tp_matched"] + st["tp_unmatched"],
                        st["box_matched"], st["box_gt"]))

        if len(samples) < a.show:
            for k in sorted(matched, key=lambda x: -len(gt_n["pins"].get(x) or {}))[:1]:
                gk = sorted((pin_key_to_number(p) for p in (gt_n["pins"].get(k) or {})),
                            key=lambda s: (len(s), s))
                pk = sorted((pin_key_to_number(p) for p in (pred2["pins"].get(k) or {})),
                            key=lambda s: (len(s), s))
                samples.append((cid, k, canon_type((gt_n["components"].get(k) or {}).get("type")),
                                gk[:26], pk[:26]))

    print("=== 220 张合计（口径与官方 evaluate_case 一致）===")
    g = agg["gt_pins"]
    print(f"GT box 元件 {agg['box_gt']}，其中**按键匹配上** {agg['box_matched']} "
          f"({agg['box_matched']/max(1,agg['box_gt']):.1%})")
    print(f"GT box 引脚 {g}：匹配上的元件占 {agg['gt_pins_matched_comp']}"
          f"（{agg['gt_pins_matched_comp']/max(1,g):.1%}）"
          f"，**没匹配上的元件**占 {agg['gt_pins_unmatched_comp']}"
          f"（{agg['gt_pins_unmatched_comp']/max(1,g):.1%}）")
    print()
    for tag, lab in (("matched", "元件键已对上"), ("unmatched", "元件键没对上")):
        tp, nm = agg[f"tp_{tag}"], agg[f"name_{tag}"]
        ps, ms = agg[f"pos_{tag}"], agg[f"miss_{tag}"]
        fp, np_ = agg[f"fp_{tag}"], agg[f"nopoint_{tag}"]
        tot = tp + nm + ps + ms + np_
        print(f"[{lab}] GT 引脚 {tot}")
        print(f"    tp {tp}  名字错 {nm}  位置错 {ps}  没这个号 {ms}  无点 {np_}   "
              f"多出来的号(fp) {fp}")
    # 分子被官方计成事件的形式
    ev_tp = agg["tp_matched"] + agg["tp_unmatched"]
    ev_fn = agg["miss_matched"] + agg["miss_unmatched"] + agg["name_matched"] + agg["name_unmatched"] \
        + agg["pos_matched"] + agg["pos_unmatched"] + agg["nopoint_matched"] + agg["nopoint_unmatched"]
    ev_fp = agg["fp_matched"] + agg["fp_unmatched"] + agg["name_matched"] + agg["name_unmatched"] \
        + agg["pos_matched"] + agg["pos_unmatched"] + agg["nopoint_matched"] + agg["nopoint_unmatched"]
    P = ev_tp / max(1, ev_tp + ev_fp)
    R = ev_tp / max(1, ev_tp + ev_fn)
    print(f"\n按官方事件口径: tp {ev_tp} fp {ev_fp} fn {ev_fn}  P {P:.4f} R {R:.4f} "
          f"F1 {2*P*R/max(1e-9,P+R):.4f}")
    print(f"匹配上的 box 里，**一个引脚都没发**的有 {agg['matched_box_no_pred_pin']} 个 "
          f"/ {agg['box_matched']}")
    print("\nGT box 元件按类型:", dict(box_by_type.most_common(12)))
    print("\n逐 case（gt_pins, tp, box_matched/box_gt）前 12 个最差:",
          [(c, gp, tp, f"{bm}/{bg}") for c, gp, tp, bm, bg in
           sorted(percase, key=lambda x: x[2])[:12]])
    print("\n=== 引脚号对照样例（GT vs 我们）===")
    for cid, k, t, gk, pk in samples:
        print(f"  {cid} {k} ({t})\n     GT  : {gk}\n     我们: {pk}")

    if a.by_split or a.by_source:
        print("\n=== 按 case.split 分组（看失分是结构性的还是我们的）===")
        for sp, st in sorted(per_split.items(), key=lambda kv: -kv[1]["gt_pins"]):
            g = st["gt_pins"]
            tp = st["tp_matched"] + st["tp_unmatched"]
            fn = (st["miss_matched"] + st["miss_unmatched"] + st["name_matched"]
                  + st["name_unmatched"] + st["pos_matched"] + st["pos_unmatched"])
            fp = (st["fp_matched"] + st["fp_unmatched"] + st["name_matched"]
                  + st["name_unmatched"] + st["pos_matched"] + st["pos_unmatched"])
            P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
            print(f"  {sp:20} box {st['box_matched']:4}/{st['box_gt']:4} "
                  f"({st['box_matched']/max(1,st['box_gt']):4.1%})  GT引脚 {g:5}  "
                  f"键没对上丢 {st['gt_pins_unmatched_comp']:5} "
                  f"({st['gt_pins_unmatched_comp']/max(1,g):4.1%})  "
                  f"号不对 {st['miss_matched']:5}  名错 {st['name_matched']:4}  "
                  f"tp {tp:4}  F1 {2*P*R/max(1e-9,P+R):.4f}")


if __name__ == "__main__":
    main()
