# -*- coding: utf-8 -*-
"""_pinreach.py —— 官方 10 张上 `pin_f1`（权重 0.25）的**六类亏损分解 + 可达上界**（§5.129）。

形状与 §5.128 的 `_nethloss.py` 完全一致：先用官方判分器把亏空做成**可达性上界**，
不达标就当场止损；**不跑管线**，全部在已交付产物上离线重算。

与已有两个工具的分工（**不重复、不覆盖**）：
  * `tools/diag/_pinloss.py`（旧，220 张）：只分「元件键对上 / 没对上」两桶；
  * `tools/diag/_pinoracle.py`（旧，40 张）：用「同元件下离 GT 点最近的那根」做**绑定上界**；
  * **本工具（新，官方 10 张）**：六类计数 + **每类单独的神谕 Δpin_f1** +
    「与坐标口径**无关**的类全修好」的分档上界（新 GT 要换坐标口径，这一档才是真可达的）。

官方口径（`schematic_parser/metrics.py:446-485`，逐行核对）：
  * `pin_box_only=True` ⇒ 只统计 **GT 里 `canon_type(type) in IC_TYPES`**（`{'box','ic','boxic'}`，
    `data.py:33`）的元件；`flat(pred)` 也用**同一套 GT 键**过滤 ⇒ **预测里键不在 GT box 键
    集合下的引脚被整组丢弃**（既不算 tp 也不算 fp，只在 GT 侧记 fn）。
  * 键 = `(comp_key, pin_key_to_number(pk))`（`data.py:143`：`'pin_A5'->'A5'`，否则原样）。
  * **一条引脚算 tp 的充要条件**（四条同时成立）：
      1) 元件键相同（预测先经 `rewrite_keys` 用坐标匹配改写键）；
      2) 脚号相同（`pin_key_to_number` 归一后）；
      3) **`pinname` 字符串完全相等** —— 这里**没有读 `cfg.compare_text`**，
         `pinname` 是**无条件**精确比较 ⇒ `compare_text` **不参与** pin 通道；
      4) 两端 `point` 可解析（`_point`）且欧氏距离 `<= pin_tol = 5.0`。
  * 计错方式（重要，决定各类的「每一根值多少」）：
      键在 GT 不在预测          ⇒ 只 `fn += 1`   （`pin.missing`）
      键在预测不在 GT          ⇒ 只 `fp += 1`   （`pin.extra`）
      键都在但名字/坐标不过      ⇒ **`fp += 1` 且 `fn += 1`**（`pin.name_mismatch` /
                                  `pin.position_mismatch`，**双向计错**）
    ⇒ 「名字对但坐标偏 6px」对 F1 的伤害**和「压根没检出」一样**；
      而「未检出」只损 fn 一侧 ⇒ **未检出反而比锚点偏更便宜**。

六类（父 agent m01553 指定）：
  (1) 未检出    元件键在，脚号缺，且同元件下**没有同名引脚**（脚号压根没读出来）
  (2) 脚号错    同元件下**有一个 `pinname` 相同的引脚**（脚读到了，号写错）
  (3) 锚点错    键/名字都对，距离 > 5px（5~10 / 10~20 / >20 三桶）
  (4) 元件键错  GT box 元件改写后在我方**整个不存在** ⇒ 其全部引脚记 fn
  (5) 多检出    我方凭空脚（键在预测不在 GT）
  (6) 文本错    键在但 `pinname` 不等

★ 坐标口径相关性（父 agent m01553 的要求，逐类标注）：
  **无关（新 GT 换了也照样可达）**：(1) 未检出、(2) 脚号错、(4) 元件键错、(5) 多检出、(6) 文本错
  **强相关（新 GT 会把锚点改成红绿导线交点 ⇒ 会重算，先别投）**：(3) 锚点错
  ⇒ 判据用的「可达上界」= `O_detect_only`（前五类 + ⑥ 全修好，**坐标仍用我方的**）。

用法：
  python -X utf8 tools/diag/_pinreach.py --run-dir ..\\work\\_fast_out_delivery_default
  python -X utf8 tools/diag/_pinreach.py --run-dir ..\\work\\_fast_out_deterministic_default --show
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from schematic_parser.data import (IC_TYPES, DEFAULT_DATA_ROOT, discover,      # noqa: E402
                                   number_to_pin_key, pin_key_to_number)
from schematic_parser.metrics import (EvalConfig, _point, assign_components,   # noqa: E402
                                      canon_type, normalise_doc, pin_f1,
                                      rewrite_keys)

DEFAULT_RUN = "Y:/workspace/EDA/work/_fast_out_delivery_default"
CASES10 = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
POS_BUCKETS = ((5.0, 10.0), (10.0, 20.0), (20.0, 1e9))


def flat_of(section, box_comps):
    """与 `metrics.pin_f1` 内部的 `flat` 同构：键 `(comp, 脚号)` -> info。"""
    out = {}
    for comp, group in (section or {}).items():
        if box_comps is not None and comp not in box_comps:
            continue
        if not isinstance(group, dict):
            continue
        for pk, info in group.items():
            out[(comp, pin_key_to_number(pk))] = info
    return out


def copy_pins(pins):
    return {c: dict(g) if isinstance(g, dict) else g for c, g in (pins or {}).items()}


def put(pins, comp, num, info):
    pins.setdefault(comp, {})
    if not isinstance(pins[comp], dict):
        pins[comp] = {}
    pins[comp][number_to_pin_key(num)] = info


def drop(pins, comp, num):
    g = pins.get(comp)
    if not isinstance(g, dict):
        return
    for pk in list(g):
        if pin_key_to_number(pk) == num:
            del g[pk]
    if not g:
        pins.pop(comp, None)


def f1_of(gt, pins, cfg):
    return pin_f1(gt, {"components": {}, "pins": pins, "nets": {}}, cfg, {})


def analyse_case(gt_doc, pred_doc, cfg):
    gt = normalise_doc(gt_doc)
    pred = normalise_doc(pred_doc)
    _comp, keymap = assign_components(gt, pred, cfg, {})
    pred2 = rewrite_keys(pred, keymap)

    box = {k for k, v in gt["components"].items()
           if canon_type((v or {}).get("type")) in IC_TYPES}
    gf = flat_of(gt["pins"], box)
    pf = flat_of(pred2["pins"], box)
    base = pin_f1(gt, pred2, cfg, {})

    gnums, pnums = {}, {}
    for c, n in gf:
        gnums.setdefault(c, set()).add(n)
    for c, n in pf:
        pnums.setdefault(c, set()).add(n)

    # ---- (4) 元件键错：GT box 元件在**改写后**的预测 components 里不存在
    pcomps = pred2.get("components") or {}
    key_bad = [c for c in box if c not in pcomps]
    key_matched_empty = [c for c in box if c in pcomps and not pnums.get(c)]
    key_bad_pins = sum(len(gnums.get(c, ())) for c in key_bad)

    # ---- (1)(2) 元件在、脚号缺
    c1_undet, c2_wrongnum = [], []
    for c in sorted(box):
        if c not in pcomps:
            continue                                   # 归 (4)
        miss = gnums.get(c, set()) - pnums.get(c, set())
        names_ours = set()
        for n in pnums.get(c, set()):
            info = pf.get((c, n))
            if isinstance(info, dict) and info.get("pinname"):
                names_ours.add(info["pinname"])
        for num in sorted(miss):
            gname = (gf[(c, num)] or {}).get("pinname")
            if gname and gname in names_ours:
                c2_wrongnum.append((c, num, gname))
            else:
                c1_undet.append((c, num))

    # ---- (3)(6)(5)
    both = set(gf) & set(pf)
    c6_text, c3_pos, c3_buckets, bad_shape = [], [], [0, 0, 0], []
    for k in sorted(both):
        g, p = gf[k], pf[k]
        if not isinstance(g, dict) or not isinstance(p, dict):
            bad_shape.append(k)
            continue
        if p.get("pinname") != g.get("pinname"):
            c6_text.append(k)
            continue
        a, b = _point(g.get("point")), _point(p.get("point"))
        if a is None or b is None:
            bad_shape.append(k)
            continue
        d = math.hypot(a[0] - b[0], a[1] - b[1])
        if d > cfg.pin_tol:
            c3_pos.append((k, d))
            for i, (lo, hi) in enumerate(POS_BUCKETS):
                if lo < d <= hi:
                    c3_buckets[i] += 1
                    break
    c5_extra = sorted(set(pf) - set(gf))

    # ---- 单类神谕（都从 pred2 出发，只动本类）
    def _fill(pins, keys):
        for k in keys:
            g = gf.get(k)
            if isinstance(g, dict):
                put(pins, k[0], k[1], json.loads(json.dumps(g)))

    o = {}
    p_add = copy_pins(pred2["pins"]); _fill(p_add, c1_undet)
    o["O_fill"] = f1_of(gt, p_add, cfg).f1

    p_wn = copy_pins(pred2["pins"])
    for c, num, gname in c2_wrongnum:
        for n2 in sorted(pnums.get(c, ())):
            info = pf.get((c, n2))
            if isinstance(info, dict) and info.get("pinname") == gname:
                drop(p_wn, c, n2)
                break
    _fill(p_wn, [(c, n) for c, n, _g in c2_wrongnum])
    o["O_fixnum"] = f1_of(gt, p_wn, cfg).f1

    p_tx = copy_pins(pred2["pins"])
    for k in c6_text:
        cur = p_tx.get(k[0], {}).get(number_to_pin_key(k[1]))
        if isinstance(cur, dict):
            put(p_tx, k[0], k[1], dict(cur, pinname=gf[k].get("pinname")))
    o["O_fixname"] = f1_of(gt, p_tx, cfg).f1

    p_ps = copy_pins(pred2["pins"])
    for k, _d in c3_pos:
        cur = p_ps.get(k[0], {}).get(number_to_pin_key(k[1]))
        if isinstance(cur, dict) and isinstance(gf[k], dict) and gf[k].get("point"):
            put(p_ps, k[0], k[1], dict(cur, point=gf[k]["point"]))
    o["O_fixpos"] = f1_of(gt, p_ps, cfg).f1

    p_ex = copy_pins(pred2["pins"])
    for k in c5_extra:
        drop(p_ex, k[0], k[1])
    o["O_dropextra"] = f1_of(gt, p_ex, cfg).f1

    # (4) 元件键错的神谕（只补键错元件的脚）
    p_kb = copy_pins(pred2["pins"])
    _fill(p_kb, [k for k in gf if k[0] in set(key_bad)])
    o["O_fixkey"] = f1_of(gt, p_kb, cfg).f1

    # ★ 坐标口径**无关**的类全修好（④ 键 + ① 未检出 + ② 脚号 + ⑥ 文本 + ⑤ 多检），
    #    **锚点 ③ 保持我方原样**
    def _blind(hit_frac):
        """hit_frac = 新补出来的引脚里，锚点落在 GT 5px 内的比例。

        这是 `O_detect_only` 唯一无法从 GT-blind 信号里直接拿到的东西——「我们**新检出**
        一根引脚」时必须同时给出坐标。`hit_frac=1.0` 是上界（坐标照抄 GT），
        `0.0` 是下界（新补的引脚坐标必然 >5px ⇒ 变成 fp+fn，比不补还差）。
        真值介于两者之间，由「新检出引脚的锚点精度」决定 ⇒ 顺带给出**盈亏平衡命中率**。
        """
        pins = copy_pins(pred2["pins"])
        _fill(pins, [k for k in gf if k[0] in set(key_bad)])          # ④
        m = len(c1_undet) + len(c2_wrongnum)
        nh = int(round(hit_frac * m))
        idx = 0
        for k in c1_undet:                                           # ①
            g = gf.get(k)
            if not isinstance(g, dict):
                continue
            g = json.loads(json.dumps(g))
            if idx >= nh and isinstance(g.get("point"), dict):
                g["point"] = {"x": g["point"].get("x", 0.0) + 4000.0,
                              "y": g["point"].get("y", 0.0) + 4000.0}
            idx += 1
            put(pins, k[0], k[1], g)
        for c, num, gname in c2_wrongnum:                            # ②
            for n2 in sorted(pnums.get(c, ())):
                info = pf.get((c, n2))
                if isinstance(info, dict) and info.get("pinname") == gname:
                    drop(pins, c, n2)
                    break
            g = gf.get((c, num))
            if not isinstance(g, dict):
                continue
            g = json.loads(json.dumps(g))
            if idx >= nh and isinstance(g.get("point"), dict):
                g["point"] = {"x": g["point"].get("x", 0.0) + 4000.0,
                              "y": g["point"].get("y", 0.0) + 4000.0}
            idx += 1
            put(pins, c, num, g)
        for k in c6_text:                                            # ⑥
            cur = pins.get(k[0], {}).get(number_to_pin_key(k[1]))
            if isinstance(cur, dict):
                put(pins, k[0], k[1], dict(cur, pinname=gf[k].get("pinname")))
        for k in c5_extra:                                           # ⑤
            drop(pins, k[0], k[1])
        return f1_of(gt, pins, cfg).f1

    o["O_detect_only"] = _blind(1.0)
    o["O_anchor"] = o["O_fixpos"]
    curve = {}
    for _p in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
        curve["%.2f" % _p] = _blind(_p)

    # 全上界：预测引脚 = GT 引脚（照抄，含坐标与名字）
    gp = {}
    for c in box:
        g = (gt["pins"] or {}).get(c)
        if isinstance(g, dict):
            gp[c] = json.loads(json.dumps(g))
    o["O_full"] = f1_of(gt, gp, cfg).f1 if gp else 0.0

    return {
        "case": "", "base_f1": base.f1, "tp": base.tp, "fp": base.fp, "fn": base.fn,
        "n_box": len(box), "n_gpins": len(gf), "n_ppins": len(pf),
        "c1_undet": len(c1_undet), "c2_wrongnum": len(c2_wrongnum),
        "c3_pos": len(c3_pos), "c3_b0": c3_buckets[0], "c3_b1": c3_buckets[1],
        "c3_b2": c3_buckets[2],
        "c4_key_bad": len(key_bad), "c4_key_bad_pins": key_bad_pins,
        "c4_matched_empty": len(key_matched_empty),
        "c5_extra": len(c5_extra), "c6_text": len(c6_text), "bad_shape": len(bad_shape),
        "orc": o, "curve": curve,
        "detail": {
            "key_bad": [(c, len(gnums.get(c, ())),
                         canon_type((gt["components"].get(c) or {}).get("type")))
                        for c in key_bad],
            "c1": ["%s.%s" % (c, n) for c, n in c1_undet],
            "c2": ["%s.%s(%s)" % (c, n, nm) for c, n, nm in c2_wrongnum],
            "c3": ["%s.%s d=%.1f" % (k[0], k[1], d) for k, d in c3_pos],
            "c5": ["%s.%s" % (c, n) for c, n in c5_extra],
            "c6": ["%s.%s" % (k[0], k[1]) for k in c6_text],
            "recall": [[c, len(gnums.get(c, ())), len(pnums.get(c, ()))]
                       for c in sorted(box)],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--cases", default=",".join(CASES10))
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--out", default="")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    index = {}
    for c in discover(a.data_root):
        index.setdefault(str(c.case_id), c)

    cfg = EvalConfig()
    rows = []
    for cid in [s.strip() for s in a.cases.split(",") if s.strip()]:
        case = index.get(str(cid))
        if case is None:
            print("!! case %s not found" % cid)
            continue
        pp = os.path.join(a.run_dir, "final_results", "%s_Pred.json" % cid)
        if not os.path.exists(pp):
            print("!! missing %s" % pp)
            continue
        gt_doc = json.load(io.open(case.json_path, encoding="utf-8"))
        pred_doc = json.load(io.open(pp, encoding="utf-8"))
        r = analyse_case(gt_doc, pred_doc, cfg)
        r["case"] = cid
        rows.append(r)
    if not rows:
        print("no rows")
        return 1
    n = len(rows)

    def avg(k):
        return sum(r[k] for r in rows) / float(n)

    print("run-dir = %s   cases = %d" % (a.run_dir, n))
    print()
    print("== 基线（官方口径：pin_box_only / pin_tol=5.0 / pinname 无条件精确） ==")
    print("pin_f1 = %.4f   tp=%.1f fp=%.1f fn=%.1f   box元件=%.1f  GT脚=%.1f  我方脚=%.1f"
          % (avg("base_f1"), avg("tp"), avg("fp"), avg("fn"),
             avg("n_box"), avg("n_gpins"), avg("n_ppins")))
    print()
    print("== 六类计数（逐案） ==")
    print("case  pin_f1    tp  fp  fn | c1未检出 c2脚号错 | c3锚点(5-10/10-20/>20) |"
          " c4键错元件(其脚) c4键对空 | c5多检 c6文本 形状坏")
    for r in rows:
        print("%-5s %6.4f %4d %3d %3d | %7d %8d | %5d (%2d/%2d/%2d) | %10d(%4d) %8d |"
              " %6d %6d %6d"
              % (r["case"], r["base_f1"], r["tp"], r["fp"], r["fn"],
                 r["c1_undet"], r["c2_wrongnum"], r["c3_pos"],
                 r["c3_b0"], r["c3_b1"], r["c3_b2"],
                 r["c4_key_bad"], r["c4_key_bad_pins"],
                 r["c4_matched_empty"], r["c5_extra"], r["c6_text"], r["bad_shape"]))
    print("均值   %6.4f %4.1f %3.1f %3.1f | %7.1f %8.1f | %5.1f (%2.1f/%2.1f/%2.1f) |"
          " %10.1f(%4.1f) %8.1f | %6.1f %6.1f %6.1f"
          % (avg("base_f1"), avg("tp"), avg("fp"), avg("fn"),
             avg("c1_undet"), avg("c2_wrongnum"), avg("c3_pos"),
             avg("c3_b0"), avg("c3_b1"), avg("c3_b2"),
             avg("c4_key_bad"), avg("c4_key_bad_pins"),
             avg("c4_matched_empty"), avg("c5_extra"), avg("c6_text"), avg("bad_shape")))
    print()

    keys = ["O_full", "O_detect_only", "O_anchor", "O_fixkey", "O_fill", "O_fixnum",
            "O_fixname", "O_dropextra"]
    print("== 神谕（逐案 Δ = O - base） ==")
    print("case  base    " + "".join("%16s" % k for k in keys))
    for r in rows:
        print("%-5s %6.4f  " % (r["case"], r["base_f1"])
              + "".join("%+16.4f" % (r["orc"][k] - r["base_f1"]) for k in keys))
    print("%-5s %6.4f  " % ("MEAN", avg("base_f1"))
          + "".join("%+16.4f" % (sum(x["orc"][k] for x in rows) / n - avg("base_f1"))
                    for k in keys))
    print()
    for k in keys:
        print("  %-16s %.4f" % (k, sum(x["orc"][k] for x in rows) / n))
    print()
    d_det = sum(x["orc"]["O_detect_only"] for x in rows) / n - avg("base_f1")
    d_anc = sum(x["orc"]["O_anchor"] for x in rows) / n - avg("base_f1")
    d_full = sum(x["orc"]["O_full"] for x in rows) / n - avg("base_f1")
    print("★ 可达性判据（父 agent 写死：可达 pin_f1 上界 < +0.02 = +0.5 分 ⇒ 止损）")
    print("  O_detect_only（坐标口径**无关**的类全修好）Δ = %+.4f = %+.2f 分"
          % (d_det, d_det * 25))
    print("  O_anchor     （只修锚点，**与坐标口径强相关**）Δ = %+.4f = %+.2f 分"
          % (d_anc, d_anc * 25))
    print("  O_full       （全上界）Δ = %+.4f = %+.2f 分" % (d_full, d_full * 25))
    print()
    # 逐类对账：fn / fp 是否能被六类完全解释
    print("== 对账（六类是否完全解释 tp/fp/fn） ==")
    for r in rows:
        efn = r["c1_undet"] + r["c2_wrongnum"] + r["c3_pos"] + r["c6_text"]
        efp = r["c3_pos"] + r["c5_extra"] + r["c6_text"]
        flag = "" if (efn == r["fn"] and efp == r["fp"]) else "  <<< 对不上"
        print("  case %-3s fn %2d = c1 %d + c2 %d + c3 %d + c6 %d ✓  fp %2d = c3 %d + c5 %d + c6 %d%s"
              % (r["case"], r["fn"], r["c1_undet"], r["c2_wrongnum"], r["c3_pos"],
                 r["c6_text"], r["fp"], r["c3_pos"], r["c5_extra"], r["c6_text"], flag))
    efn = sum(r["c1_undet"] + r["c2_wrongnum"] + r["c3_pos"] + r["c6_text"] for r in rows)
    efp = sum(r["c3_pos"] + r["c5_extra"] + r["c6_text"] for r in rows)
    print("  合计 fn %d（实际 %d）  fp %d（实际 %d）  ⇒ 锚点占全部错项 %.1f%%"
          % (efn, sum(r["fn"] for r in rows), efp, sum(r["fp"] for r in rows),
             100.0 * 2 * sum(r["c3_pos"] for r in rows) / max(1, efn + efp)))
    print()
    print("== ★ O_detect_only 对「新补引脚的锚点命中率」敏感性 ==")
    print("  （hit=1.0 坐标照抄 GT 是上界；hit=0.0 新补引脚坐标必错 ⇒ 比不补更差，是下界）")
    cs = sorted({k for r in rows for k in r["curve"]}, key=float)
    print("  hit   " + "".join("%9s" % k for k in cs))
    print("  Δf1   " + "".join("%+9.4f" % (sum(x["curve"][k] for x in rows) / n - avg("base_f1"))
                             for k in cs))
    print("  Δ分   " + "".join("%+9.2f" % ((sum(x["curve"][k] for x in rows) / n - avg("base_f1")) * 25)
                             for k in cs))
    base0 = avg("base_f1")
    prev = None
    be = None
    for k in cs:
        d = sum(x["curve"][k] for x in rows) / n - base0
        if prev is not None and prev[1] >= 0.02 > d:
            be = (prev[0], k)
        prev = (k, d)
    if be:
        print("  ⇒ 盈亏平衡：命中率 ~%.2f 时 Δ 落到 +0.02（+0.5 分）门槛；实测我方已检出引脚的"
              % ((float(be[0]) + float(be[1])) / 2.0))
        hit = sum(r["tp"] for r in rows) / float(max(1, sum(r["n_ppins"] for r in rows)))
        print("     锚点 ≤5px 命中率 = tp/我方脚 = %.3f" % hit)
    print()
    print("== 召回结构（每元件 GT 脚数 vs 我方脚数） ==")
    tot_g = sum(r["n_gpins"] for r in rows)
    tot_p = sum(r["n_ppins"] for r in rows)
    print("GT 脚合计 %d，我方脚合计 %d，整体召回 %.3f" % (tot_g, tot_p, tot_p / max(tot_g, 1)))
    ratios = []
    for r in rows:
        for c, ng, np_ in r["detail"]["recall"]:
            ratios.append((r["case"], c, ng, np_))
    full = sum(1 for _a, _c, g, p in ratios if g and p == g)
    zero = sum(1 for _a, _c, g, p in ratios if g and p == 0)
    part = sum(1 for _a, _c, g, p in ratios if g and 0 < p < g)
    print("元件数 %d：脚数**全对** %d，**一个脚都没有** %d，部分 %d"
          % (len(ratios), full, zero, part))
    rs = sorted((p / float(g)) for _a, _c, g, p in ratios if g)
    if rs:
        def q(f):
            return rs[min(len(rs) - 1, int(f * (len(rs) - 1)))]
        print("逐元件召回率分位：p10=%.2f p25=%.2f p50=%.2f p75=%.2f p90=%.2f"
              % (q(.10), q(.25), q(.50), q(.75), q(.90)))
    big = [(a_, c, g, p) for a_, c, g, p in ratios if g >= 8]
    if big:
        bg = sum(g for _a, _c, g, _p in big)
        bp = sum(p for _a, _c, _g, p in big)
        print("引脚数 >=8 的元件 %d 个：GT 脚合计 %d，我方 %d，召回 %.3f"
              % (len(big), bg, bp, bp / float(bg)))
        for a_, c, g, p in sorted(big, key=lambda t: (t[3] / float(t[2])))[:14]:
            print("    case %-3s %-9s GT=%2d 我方=%2d  (%.2f)" % (a_, c, g, p, p / float(g)))
    # 小而全丢的元件
    small_zero = [(a_, c, g) for a_, c, g, p in ratios if g and p == 0 and g <= 3]
    print("引脚 1~3 且**全丢**的元件 %d 个" % len(small_zero))
    if small_zero[:12]:
        print("    " + " ".join("%s/%s(GT%d)" % (a_, c, g) for a_, c, g in small_zero[:12]))

    if a.out:
        io.open(a.out, "w", encoding="utf-8").write(
            json.dumps(rows, ensure_ascii=False, indent=1))
        print("\njson -> %s" % a.out)

    if a.show:
        print("\n== 非零项明细 ==")
        for r in rows:
            d = r["detail"]
            if not any(d[k] for k in ("key_bad", "c1", "c2", "c3", "c5", "c6")):
                continue
            print("\n--- case %s pin_f1 %.4f (tp %d fp %d fn %d) ---"
                  % (r["case"], r["base_f1"], r["tp"], r["fp"], r["fn"]))
            for c, ng, t in d["key_bad"]:
                print("  [4 键错] %s (type=%s) GT脚=%d 整丢" % (c, t, ng))
            if d["c2"]:
                print("  [2 号错] %s" % " ".join(d["c2"]))
            if d["c1"]:
                print("  [1 未检出 %d] %s" % (len(d["c1"]), " ".join(d["c1"][:48])))
            if d["c3"]:
                print("  [3 锚点 %d] %s" % (len(d["c3"]), " ".join(d["c3"][:48])))
            if d["c5"]:
                print("  [5 多检 %d] %s" % (len(d["c5"]), " ".join(d["c5"][:48])))
            if d["c6"]:
                print("  [6 文本 %d] %s" % (len(d["c6"]), " ".join(d["c6"][:48])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
