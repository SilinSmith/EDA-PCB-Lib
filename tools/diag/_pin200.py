# -*- coding: utf-8 -*-
"""200 训练集 **引脚** 为什么大面积缺失 —— 看清 GT/预测里 `pins` 的真实结构与总量。

要点：`components` / `pins` / `nets` 是 **以元件 id 为键的字典**，
`len()` 数的是元件数，不是引脚数；真引脚数要 `sum(len(v) for v in pins.values())`。

用法：
    python work\\tools\\diag\\_pin200.py 0001               # 单张，并排打结构
    python work\\tools\\diag\\_pin200.py --scan 20          # 扫 20 张，看真引脚数分布
    python work\\tools\\diag\\_pin200.py --keys 0001        # 只看 GT 的一根引脚长什么样
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402
from stage2_eval import build_index, resolve                               # noqa: E402


def lookup(cid):
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    case, _ = resolve(cid, exact, digits)
    if case is None:
        case, _ = resolve(cid.lstrip("0") or "0", exact, digits)
    return case


def pin_total(d):
    """`pins` 可能是 dict[cid] -> list，也可能是 list，也可能是 dict[pinid] -> dict。"""
    p = (d or {}).get("pins")
    if p is None:
        return 0, 0
    if isinstance(p, list):
        return len(p), 0
    if isinstance(p, dict):
        withp = [k for k, v in p.items() if isinstance(v, list) and v]
        n = sum(len(v) for v in p.values() if isinstance(v, list))
        if n:
            return n, len(withp)
        return len(p), len(p)
    return 0, 0


def pred_of(cid, out_dir):
    for name in (cid + "_Pred.json", cid + ".json"):
        q = Path(out_dir) / "final_results" / name
        if q.exists():
            return load_json(q), q
    return None, None


def show(cid, out_dir):
    case = lookup(cid)
    if case is None:
        print("找不到 case %s" % cid)
        return
    gt = load_json(Path(case.json_path))
    n, nc = pin_total(gt)
    print("== case %s  split=%s" % (case.case_id, case.split))
    print("   GT 元件 %d  真引脚 %d（%d 个元件带引脚） 网络 %d" % (
        len(gt.get("components") or {}), n, nc, len(gt.get("nets") or {})))
    pr, path = pred_of(case.case_id, out_dir)
    if pr:
        m, mc = pin_total(pr)
        print("   PR 元件 %d  真引脚 %d（%d 个元件带引脚） 网络 %d   <- %s" % (
            len(pr.get("components") or {}), m, mc, len(pr.get("nets") or {}), path))

    for tag, d in (("GT", gt), ("PR", pr)):
        if not d:
            continue
        pins = d.get("pins") or {}
        if isinstance(pins, dict):
            ks = [k for k, v in pins.items() if isinstance(v, list) and v][:2]
            for k in ks:
                print("   --- %s pins['%s'] 前 2 项 ---" % (tag, k))
                for x in pins[k][:2]:
                    print("      %s" % json.dumps(x, ensure_ascii=False)[:260])
        comps = d.get("components") or {}
        if isinstance(comps, dict) and comps:
            k0 = list(comps.keys())[0]
            print("   --- %s components['%s'] ---" % (tag, k0))
            print("      %s" % json.dumps(comps[k0], ensure_ascii=False)[:300])
        nets = d.get("nets") or {}
        if isinstance(nets, dict) and nets:
            k0 = list(nets.keys())[0]
            print("   --- %s nets['%s'] ---" % (tag, k0))
            print("      %s" % json.dumps(nets[k0], ensure_ascii=False)[:260])


def scan(n, out_dir):
    cs = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == "200_train_cases"][:n]
    print("%-8s %6s %7s %7s | %6s %7s   %s" % (
        "case", "元件", "引脚GT", "网络GT", "元件PR", "引脚PR", "引脚 召回"))
    tot_g = tot_p = 0
    for c in cs:
        gt = load_json(Path(c.json_path))
        gn, _ = pin_total(gt)
        pr, _ = pred_of(c.case_id, out_dir)
        pn, _ = pin_total(pr or {})
        tot_g += gn
        tot_p += pn
        print("%-8s %6d %7d %7d | %6d %7d   %5.1f%%" % (
            c.case_id, len(gt.get("components") or {}), gn,
            len(gt.get("nets") or {}), len((pr or {}).get("components") or {}), pn,
            100.0 * pn / max(1, gn)))
    print("合计 GT 引脚 %d，预测引脚 %d ⇒ %.1f%%" % (tot_g, tot_p, 100.0 * tot_p / max(1, tot_g)))


def flat_box(doc):
    """完全照抄 metrics.pin_f1 的 flat()：只看 GT 里 type∈IC_TYPES 的元件。"""
    from schematic_parser.metrics import IC_TYPES, canon_type, pin_key_to_number
    box = {k for k, v in (doc.get("components") or {}).items()
           if canon_type((v or {}).get("type")) in IC_TYPES}
    out = {}
    for comp, group in (doc.get("pins") or {}).items():
        if comp not in box or not isinstance(group, dict):
            continue
        for pk, info in group.items():
            out[(comp, pin_key_to_number(pk))] = info
    return box, out


def why_pins(cid, out_dir, top):
    """逐 IC 对照：GT 要多少根引脚、我们给了多少根、名字/坐标差在哪。

    **必须复刻 evaluate_case 的顺序**：先 assign_components 拿 keymap、
    再 rewrite_keys 改写预测键，最后才 pin_f1 —— 否则会把「位号读错但被
    坐标匹配救回来」的元件误报成「引脚全丢」。
    """
    from schematic_parser.metrics import (EvalConfig, assign_components,
                                          canon_type, normalise_doc, pin_f1,
                                          rewrite_keys)
    case = lookup(cid)
    if case is None:
        print("找不到 case %s" % cid)
        return
    cfg = EvalConfig()
    gt = normalise_doc(load_json(Path(case.json_path)))
    pr_raw, path = pred_of(case.case_id, out_dir)
    pr = normalise_doc(pr_raw or {})
    detail = {}
    _comp, keymap = assign_components(gt, pr, cfg, detail)
    pred2 = rewrite_keys(pr, keymap)
    f1 = pin_f1(gt, pred2, cfg, dict(detail))

    gbox, gf = flat_box(gt)
    pbox, pf = flat_box(pred2)
    print("== case %s（%s）" % (case.case_id, case.split))
    print("   keymap %d 条（例如 %s）" % (
        len(keymap), list(keymap.items())[:6]))
    print("   pin_f1 = %.4f  tp=%d fp=%d fn=%d" % (f1.f1, f1.tp, f1.fp, f1.fn))
    print("   GT 的 IC 元件 %d 个 / 引脚 %d 根 ｜ 改写后预测 IC 元件 %d 个 / 引脚 %d 根"
          % (len(gbox), len(gf), len(pbox), len(pf)))
    hit = set(gf) & set(pf)
    good = sum(1 for k in hit
               if isinstance(gf[k], dict) and isinstance(pf[k], dict)
               and pf[k].get("pinname") == gf[k].get("pinname"))
    print("   键相符 %d 根，其中 pinname 也相符 %d 根" % (len(hit), good))
    gc, pc = {}, {}
    for (c, _n) in gf:
        gc[c] = gc.get(c, 0) + 1
    for (c, _n) in pf:
        pc[c] = pc.get(c, 0) + 1
    rows = sorted(set(gc) | set(pc), key=lambda c: -(gc.get(c, 0) - pc.get(c, 0)))
    print("   %-14s %-10s %6s %6s %6s" % ("元件键", "type", "GT引脚", "PR引脚", "差"))
    for c in rows[:top]:
        t = canon_type(((gt.get("components") or {}).get(c) or {}).get("type"))
        print("   %-14s %-10s %6d %6d %6d" % (c, t, gc.get(c, 0), pc.get(c, 0),
                                              gc.get(c, 0) - pc.get(c, 0)))
    mism = [(k, gf[k].get("pinname"), pf[k].get("pinname")) for k in list(hit)[:800]
            if isinstance(gf[k], dict) and isinstance(pf[k], dict)
            and pf[k].get("pinname") != gf[k].get("pinname")]
    if mism:
        print("   pinname 对不上的样本（最多 8 条）：")
        for k, g, p in mism[:8]:
            print("      %-26s GT=%-14s PR=%s" % (str(k), g, p))
    # 坐标差
    from schematic_parser.metrics import _point
    import math
    far = []
    for k in hit:
        g, p = gf[k], pf[k]
        if not (isinstance(g, dict) and isinstance(p, dict)):
            continue
        if p.get("pinname") != g.get("pinname"):
            continue
        a, b = _point(g.get("point")), _point(p.get("point"))
        if a is None or b is None:
            far.append((k, "no_point", None))
        elif math.hypot(a[0] - b[0], a[1] - b[1]) > cfg.pin_tol:
            far.append((k, "%.1fpx" % math.hypot(a[0] - b[0], a[1] - b[1]), None))
    if far:
        print("   名字对但坐标超差 %d 根（样本）：%s" % (
            len(far), ", ".join("%s %s" % (k, w) for k, w, _ in far[:8])))


def funnel(n, out_dir, show_each):
    """★ 把引脚失分**拆成互斥的四段**（官方 Q2 口径，复刻 evaluate_case 顺序）：

        GT 的 IC 引脚总数
          ├─ A 丢了「元件键没配上」   → 位号/坐标匹配问题
          ├─ B 键配上了但我们没给这根 → 引脚漏检
          ├─ C 给了但 pinname 不同    → 引脚名读错
          └─ D 名字对但坐标超差       → 引脚坐标问题（tp = 其余）
    """
    import math
    from schematic_parser.metrics import (EvalConfig, _point, assign_components,
                                          canon_type, normalise_doc, pin_f1,
                                          rewrite_keys)
    cfg = EvalConfig()
    cs = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == "200_train_cases"][:n]
    T = dict(gt=0, A=0, B=0, C=0, D=0, tp=0, boxes=0, matched=0)
    print("%-8s %6s %6s %6s %6s %6s %6s  %7s" % (
        "case", "IC数", "GT根", "A键丢", "B漏检", "C名错", "D坐标", "pin_f1"))
    for c in cs:
        gt = normalise_doc(load_json(Path(c.json_path)))
        praw, _ = pred_of(c.case_id, out_dir)
        if praw is None:
            continue
        pr = normalise_doc(praw)
        detail = {}
        _comp, keymap = assign_components(gt, pr, cfg, detail)
        pred2 = rewrite_keys(pr, keymap)
        f1 = pin_f1(gt, pred2, cfg, dict(detail))
        gbox, gf = flat_box(gt)
        pbox, pf = flat_box(pred2)
        matched_gt = {v for v in keymap.values()}          # 目标=GT 键
        A = sum(1 for (comp, _n) in gf if comp not in matched_gt)
        both = set(gf) & set(pf)
        B = len(gf) - A - len(both)
        C = sum(1 for k in both
                if isinstance(gf[k], dict) and isinstance(pf[k], dict)
                and pf[k].get("pinname") != gf[k].get("pinname"))
        D = 0
        for k in both:
            g, p = gf[k], pf[k]
            if not (isinstance(g, dict) and isinstance(p, dict)):
                continue
            if p.get("pinname") != g.get("pinname"):
                continue
            a, b = _point(g.get("point")), _point(p.get("point"))
            if a is None or b is None or math.hypot(a[0] - b[0], a[1] - b[1]) > cfg.pin_tol:
                D += 1
        row = dict(gt=len(gf), A=A, B=B, C=C, D=D, tp=f1.tp,
                   boxes=len(gbox), matched=len(matched_gt & set(gbox)))
        for k in T:
            T[k] += row[k]
        if show_each:
            print("%-8s %6d %6d %6d %6d %6d %6d  %7.4f" % (
                c.case_id, row["boxes"], row["gt"], A, B, C, D, f1.f1))
    print("-" * 78)
    print("%-8s %6s %6d %6d %6d %6d %6d %6d" % ("合计", "", T["boxes"], T["gt"],
                                                T["A"], T["B"], T["C"], T["D"]))
    if T["gt"]:
        print("  占比： A 位号没配上 %.1f%% ｜ B 引脚漏检 %.1f%% ｜ C 名读错 %.1f%%"
              " ｜ D 坐标超差 %.1f%% ｜ tp %.1f%%" % (
                  100.0 * T["A"] / T["gt"], 100.0 * T["B"] / T["gt"],
                  100.0 * T["C"] / T["gt"], 100.0 * T["D"] / T["gt"],
                  100.0 * T["tp"] / T["gt"]))
    return T


def whyA(n, out_dir, show_each):
    """★ 把漏斗里的 **A（元件键没配上，占引脚失分 ~65%）** 再拆三份：

        A1  那张图上**根本没检出**这个 IC        → 检测问题，读什么位号都没用
        A2  检出了，但键是兜底键 `pNNN`（没读出位号） → **位号读取**问题
        A3  检出了、也给了位号，但**读成了别的**     → **位号读取**问题（读错）

    判据：GT 元件的 bbox 与预测元件 bbox 的**中心距 < 40px** 就算「同一个框」。
    """
    from schematic_parser.metrics import (EvalConfig, _centre, _box, assign_components,
                                          canon_type, is_unkeyed, normalise_doc)
    cfg = EvalConfig()
    cs = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == "200_train_cases"][:n]
    T = dict(gt=0, A1=0, A2=0, A3=0, n1=0, n2=0, n3=0)
    print("%-8s %6s %6s %6s %6s   %s" % ("case", "A根", "A1没检出", "A2兜底键", "A3读错",
                                         "样本（GT键←预测键）"))
    for c in cs:
        gt = normalise_doc(load_json(Path(c.json_path)))
        praw, _ = pred_of(c.case_id, out_dir)
        if praw is None:
            continue
        pr = normalise_doc(praw)
        _comp, keymap = assign_components(gt, pr, cfg, {})
        matched_gt = set(keymap.values())
        gbox, gf = flat_box(gt)
        pc = pr.get("components") or {}
        pcbox = {}
        for k, v in pc.items():
            ctr = _centre(_box((v or {}).get("bbox")))
            if ctr:
                pcbox[k] = ctr
        cnt = {}
        ex = []
        for (comp, _n) in gf:
            cnt[comp] = cnt.get(comp, 0) + 1
        a1 = a2 = a3 = 0
        n1 = n2 = n3 = 0
        for comp in gbox:
            if comp in matched_gt:
                continue
            npin = cnt.get(comp, 0)
            ctr = _centre(_box(((gt.get("components") or {}).get(comp) or {}).get("bbox")))
            best, bd = None, 1e18
            if ctr:
                for k, kc in pcbox.items():
                    d = ((ctr[0] - kc[0]) ** 2 + (ctr[1] - kc[1]) ** 2) ** 0.5
                    if d < bd:
                        best, bd = k, d
            if ctr is None or best is None or bd > 40.0:
                a1 += npin
                n1 += 1
                ex.append("%s←(无框)" % comp)
            elif is_unkeyed(best):
                a2 += npin
                n2 += 1
                ex.append("%s←%s(%d根)" % (comp, best, npin))
            else:
                a3 += npin
                n3 += 1
                ex.append("%s←%s" % (comp, best))
        T["gt"] += len(gf)
        T["A1"] += a1
        T["A2"] += a2
        T["A3"] += a3
        T["n1"] += n1
        T["n2"] += n2
        T["n3"] += n3
        if show_each:
            print("%-8s %6d %6d %6d %6d   %s" % (
                c.case_id, a1 + a2 + a3, a1, a2, a3, " ".join(ex[:4])))
    print("-" * 78)
    A = T["A1"] + T["A2"] + T["A3"]
    print("合计：IC 引脚 %d 根，未配上的 %d 根（%.1f%%）" % (
        T["gt"], A, 100.0 * A / max(1, T["gt"])))
    print("   A1 根本没检出  %5d 根（%d 个 IC）  占 A 的 %.1f%%" % (
        T["A1"], T["n1"], 100.0 * T["A1"] / max(1, A)))
    print("   A2 兜底键 pNNN %5d 根（%d 个 IC）  占 A 的 %.1f%%" % (
        T["A2"], T["n2"], 100.0 * T["A2"] / max(1, A)))
    print("   A3 位号读错    %5d 根（%d 个 IC）  占 A 的 %.1f%%" % (
        T["A3"], T["n3"], 100.0 * T["A3"] / max(1, A)))
    return T


def emit_scan(n, out_dir, split):
    """★ 我们**到底产出了多少 IC 引脚**（不看键、只看预测自己的 type）——
    用来区分「引脚根本没产出来」和「产出来了但名字/键不对」。
    """
    from schematic_parser.metrics import IC_TYPES, canon_type, normalise_doc
    cs = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == split][:n]
    tg = tp = tgi = tpi = 0
    print("%-8s %6s %6s %6s %6s %7s" % ("case", "IC-GT", "引脚GT", "IC-PR", "引脚PR", "召回"))
    for c in cs:
        gt = normalise_doc(load_json(Path(c.json_path)))
        praw, _ = pred_of(c.case_id, out_dir)
        if praw is None:
            continue
        pr = normalise_doc(praw)
        gc = {k for k, v in (gt.get("components") or {}).items()
              if canon_type((v or {}).get("type")) in IC_TYPES}
        pc = {k for k, v in (pr.get("components") or {}).items()
              if canon_type((v or {}).get("type")) in IC_TYPES}
        gp = sum(len(v) for k, v in (gt.get("pins") or {}).items()
                 if k in gc and isinstance(v, dict))
        pp = sum(len(v) for k, v in (pr.get("pins") or {}).items()
                 if k in pc and isinstance(v, dict))
        tg += gp
        tp += pp
        tgi += len(gc)
        tpi += len(pc)
        print("%-8s %6d %6d %6d %6d %6.1f%%" % (c.case_id, len(gc), gp, len(pc), pp,
                                                100.0 * pp / max(1, gp)))
    print("-" * 48)
    print("合计：GT IC %d 个 / 引脚 %d 根 ｜ 预测 IC %d 个 / 引脚 %d 根 ⇒ 产出率 %.1f%%" % (
        tgi, tg, tpi, tp, 100.0 * tp / max(1, tg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", help="case id")
    ap.add_argument("--out", default=str(ROOT / "work" / "_submit_out220r36"))
    ap.add_argument("--scan", type=int, default=0)
    ap.add_argument("--pins", action="store_true", help="逐 IC 对照引脚数（官方 Q2 口径）")
    ap.add_argument("--funnel", type=int, default=0,
                    help="对训练集前 N 张做引脚四段漏斗")
    ap.add_argument("--whyA", type=int, default=0,
                    help="把漏斗的 A 段再拆成 没检出/兜底键/读错")
    ap.add_argument("--emit", type=int, default=0, help="对前 N 张看 IC 引脚产出率")
    ap.add_argument("--split", default="200_train_cases")
    ap.add_argument("--each", action="store_true", help="逐张打印")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    if a.scan:
        scan(a.scan, a.out)
    if a.funnel:
        funnel(a.funnel, a.out, a.each)
    if a.whyA:
        whyA(a.whyA, a.out, a.each)
    if a.emit:
        emit_scan(a.emit, a.out, a.split)
    for cid in a.cases:
        if a.pins:
            why_pins(cid, a.out, a.top)
        else:
            show(cid, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
