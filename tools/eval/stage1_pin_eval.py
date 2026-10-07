"""量 stage1 的引脚点：位置精度 / 召回（**只比位置，不比引脚号**）。

为什么要分开量：`pin_f1` 的键是 `(component, pin_number)`，要拿分得**引脚号也对**，
而引脚号只能靠 OCR（第二阶段）。这个脚本先回答一个更基本的问题：
**「引脚点摆对地方了吗」** —— 如果位置本身就不对，OCR 再准也没用。

GT 的 `point` 是 y 向上的 JSON 坐标，这里统一转成图像坐标（`y_img = H - y`）再比。

匹配：先按元件中心贪心一对一（<=20px）配元件，再在配上的元件内部把引脚贪心一对一
配到 <=tol。位置口径忽略引脚号，所以比官方 `pin_f1` 宽松，是它的**上界**。

用法：python Y:\workspace\EDA\work\stage1_pin_eval.py --stage1 DIR [--limit N]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

from schematic_parser.data import DEFAULT_DATA_ROOT, discover  # noqa: E402
from stage1.pins import find_pins  # noqa: E402

TOLS = (3.0, 5.0, 8.0, 12.0, 20.0)


def gt_components(doc, H):
    out = {}
    for cid, c in (doc.get("components") or {}).items():
        bb = c.get("bbox")
        if not bb or len(bb) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bb]
        # JSON y 向上 -> 图像 y 向下；bbox = [左下, 右上]
        out[cid] = dict(type=c.get("type") or "?",
                        box=(x1, H - y2, x2, H - y1))
    return out


def gt_pins(doc, H):
    out = {}
    for cid, pl in (doc.get("pins") or {}).items():
        pts = []
        for pkey, pr in (pl or {}).items():
            pt = pr.get("point") or {}
            if pt.get("x") is None:
                continue
            pts.append((float(pt["x"]), H - float(pt["y"])))
        out[cid] = pts
    return out


def greedy_match(a, b, tol):
    """a/b 都是点列；贪心一对一，返回 (n_match, pairs)。"""
    if not a or not b:
        return 0, []
    A = np.asarray(a, np.float64)
    B = np.asarray(b, np.float64)
    D = np.hypot(A[:, 0][:, None] - B[:, 0][None, :], A[:, 1][:, None] - B[:, 1][None, :])
    pairs = []
    used_b = set()
    for i in np.argsort(D.min(axis=1)):
        order = np.argsort(D[i])
        for j in order:
            if D[i, j] > tol:
                break
            if j in used_b:
                continue
            used_b.add(int(j))
            pairs.append((int(i), int(j)))
            break
    return len(pairs), pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=r"Y:\workspace\EDA\work\stage1_all13")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default=None)
    ap.add_argument("--no-prior", action="store_true")
    ap.add_argument("--comp-tol", type=float, default=20.0)
    ap.add_argument("--gt-boxes", action="store_true")
    # 扫两个没定下来的参数：接触半径 / 同边聚类间隔
    ap.add_argument("--touch-tols", default="6")
    ap.add_argument("--cluster-rs", default="12")
    a = ap.parse_args()

    def _floats(s):
        return [float(v) for v in str(s).replace("，", ",").split(",") if v.strip()]

    combos = [(tt, cr) for tt in _floats(a.touch_tols) for cr in _floats(a.cluster_rs)]

    cases = sorted(discover(a.data_root), key=lambda c: c.case_id)
    if a.split:
        cases = [c for c in cases if c.split == a.split]
    if a.limit:
        cases = cases[:a.limit]

    tp = {t: 0 for t in TOLS}
    n_gt_pin = 0
    def _blank():
        return dict(tp={t: 0 for t in TOLS}, n_gt_pin=0, n_pred_pin=0, n_matched=0,
                    by_type=collections.defaultdict(
                        lambda: dict(gt=0, tp={t: 0 for t in TOLS})))

    acc = {c: _blank() for c in combos}
    n_gt_comp = 0
    n_case = 0
    for case in cases:
        sp = os.path.join(a.stage1, case.case_id + ".stage1.json")
        if not os.path.exists(sp):
            continue
        sd = json.load(open(sp, encoding="utf-8"))
        H = float(sd["image"]["h"])
        gdoc = json.load(open(case.json_path, encoding="utf-8"))
        gc = gt_components(gdoc, H)
        gp = gt_pins(gdoc, H)
        if not gc:
            continue
        n_case += 1
        comps = sd.get("components") or []
        segs = (sd.get("wire_detail") or {}).get("segments") or []
        preds = {c: find_pins(comps, segs, touch_tol=c[0], cluster_r=c[1],
                              use_prior=not a.no_prior) for c in combos}

        # 元件配对：中心 <= comp_tol 的贪心一对一（与参数无关，只算一次）
        gids = [cid for cid in gc if gp.get(cid)]
        gctr = []
        for cid in gids:
            x1, y1, x2, y2 = gc[cid]["box"]
            gctr.append(((x1 + x2) / 2, (y1 + y2) / 2))
        pid = [c["id"] for c in comps]
        pctr = [(c["center_img"][0], c["center_img"][1]) for c in comps]
        if not pctr or not gctr:
            continue
        G = np.asarray(gctr, np.float64)
        P = np.asarray(pctr, np.float64)
        D = np.hypot(G[:, 0][:, None] - P[:, 0][None, :], G[:, 1][:, None] - P[:, 1][None, :])
        used = set()
        pairs = []
        for i in range(len(G)):
            for j in np.argsort(D[i]):
                if D[i, j] > a.comp_tol:
                    break
                if int(j) in used:
                    continue
                used.add(int(j))
                pairs.append((gids[i], pid[int(j)]))
                break
        n_gt_comp += len(gids)

        # 关键对照：用 **GT 的框**代替检测框，看瓶颈是不是「元件定位误差被放大成引脚误差」
        if a.gt_boxes:
            use = {cid: cid for cid, _ in pairs}
            _comps = [{"id": cid, "type": gc[cid]["type"], "bbox_img": list(gc[cid]["box"])}
                      for cid, _ in pairs]
        else:
            use = {cid: pidj for cid, pidj in pairs}
            _comps = comps
        preds = {c: find_pins(_comps, segs, touch_tol=c[0], cluster_r=c[1],
                              use_prior=not a.no_prior) for c in combos}

        for combo in combos:
            r = acc[combo]
            pred = preds[combo]
            r["n_matched"] += len(pairs)
            for cid, pidj in pairs:
                t = gc[cid]["type"]
                gpts = gp[cid]
                ppts = pred.get(use[cid]) or []
                r["n_gt_pin"] += len(gpts)
                r["n_pred_pin"] += len(ppts)
                r["by_type"][t]["gt"] += len(gpts)
                for tol in TOLS:
                    n, _ = greedy_match(gpts, ppts, tol)
                    r["tp"][tol] += n
                    r["by_type"][t]["tp"][tol] += n

    for combo in combos:
        r = acc[combo]
        print("")
        print("=== touch_tol=%g  cluster_r=%g | cases %d / GT 有引脚的元件 %d / 配对 %d"
              " / GT 引脚 %d / 预测引脚 %d" % (combo[0], combo[1], n_case, n_gt_comp,
                                               r["n_matched"], r["n_gt_pin"], r["n_pred_pin"]))
        print("| tol | TP | R | P | F1 |")
        print("|---|---|---|---|---|")
        for tol in TOLS:
            R = r["tp"][tol] / max(1, r["n_gt_pin"])
            P = r["tp"][tol] / max(1, r["n_pred_pin"])
            F1 = 2 * P * R / max(1e-9, P + R)
            print("| %g | %d | %.4f | %.4f | **%.4f** |" % (tol, r["tp"][tol], R, P, F1))
        if len(combos) == 1:
            print("")
            print("按类型（tol=8）：")
            for t, v in sorted(r["by_type"].items(), key=lambda kv: -kv[1]["gt"])[:16]:
                print("  %-16s GT %5d  命中@8 %5d  R %.3f"
                      % (t, v["gt"], v["tp"][8.0], v["tp"][8.0] / max(1, v["gt"])))
    if len(combos) > 1:
        print("")
        print("=== 汇总（tol=8）：")
        for combo in combos:
            r = acc[combo]
            R = r["tp"][8.0] / max(1, r["n_gt_pin"])
            P = r["tp"][8.0] / max(1, r["n_pred_pin"])
            print("  touch %5.1f cluster %5.1f -> R %.4f P %.4f F1 %.4f (预测引脚 %d)"
                  % (combo[0], combo[1], R, P, 2 * P * R / max(1e-9, P + R), r["n_pred_pin"]))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
