# -*- coding: utf-8 -*-
"""红绿交点锚点探针（诊断工具，非管线源码）。

问题：如果新 GT 把 pin 锚点定义在「红引线 → 绿导线」的交点上（而不是引线根/端点），
     我们能不能 GT-blind 地把那个点找出来？

三条口径（都在图像坐标里、只看我们自己的像素分类）：
  jnear = 锚点邻域内离锚点最近的**绿像素**
  junc  = 离锚点最近的**绿且内侧 1~2px 贴 maroon** 的交点像素（更贴「红绿交点」原义）
与 `_pinanchor.py` 的「色转折射线」(ed) 口径并列，作为第三种/第四种独立估计。

三种模式：
  --mode agree （默认含）与 GT 锚点比：逐组中位是否一致(≤2px) + 把该估计当作平移施行后的 pin_f1
                            （GT 在这里只是量尺；施行分才代表「真能换成涨分吗」）
  --mode stats 覆盖率 / 交点距离中位·p90 / 组内稳定性(dev≤2 的组占比)。被统计的量只读图像
                与我们自己的锚点（**不读 GT 坐标**），但**配对与 (元件,边) 分组仍借 GT**
                （脚集合来自 GT↔我方配对，见 `schematic_parser.metrics.assign_components`）。
  --mode statsfree 与 stats 同口径，但**连分组也不碰 GT**：脚集合与 (元件,边) 全部来自
                我方产物 `<run-dir>/final_results/<cid>_Pred.json`（`components[*].bbox` +
                `pins[*][*].point`），只借数据集的**图像路径**。
                ⇒ 新 GT 落地、其锚点口径未知时，这一档可以先跑；GT 一到手再跑 stats 对照。

用法：
  python -X utf8 eda-pcb/tools/diag/_juncprobe.py --mode both
  python -X utf8 eda-pcb/tools/diag/_juncprobe.py --mode stats --min-n 3
  python -X utf8 eda-pcb/tools/diag/_juncprobe.py --mode statsfree --min-n 3   # 新 GT 未到手时先跑
  python -X utf8 eda-pcb/tools/diag/_juncprobe.py --mode stats --back 4   # 放宽「内侧贴 maroon」窗口
  python -X utf8 eda-pcb/tools/diag/_juncprobe.py --mode agree --show --cases 1,5,10
（--half 默认 26px、--lat 默认 ±2px、--back 默认 1..2px = 严格口径。放宽 --back 到 4 时覆盖与
  组内稳定性明显上升，而 --lat 放宽无收益 ⇒ 新 GT 上两种 --back 都要看，见 §5.152 补。）
"""
import argparse
import copy
import io
import json
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from schematic_parser.data import discover                                    # noqa: E402
from schematic_parser.metrics import (                                        # noqa: E402
    EvalConfig, assign_components, rewrite_keys, pin_f1,
)
import _pinanchor as PA                                                       # noqa: E402

DEFAULT_RUN = r"Y:\workspace\EDA\work\_fast_out_deterministic_default"
CASES = [str(i) for i in range(1, 11)]


def probe(img, q, u, half, want_maroon, lat=2, back=2):
    """在 q 邻域沿 ±u 搜索目标像素（图像坐标）。返回 (t, dx, dy) 或 None。

    lat  = 横向搜索半径（±px）；back = 「内侧贴 maroon」的窗口（内侧 1..back px）。
    默认 (2, 2) = §5.151/§5.152 用过的严格口径；放宽到 back=4 时覆盖 37.9%→53.2%、
    组内恒定 19.2%→50.0%（§5.152 补），所以新 GT 上两种口径都要看。
    """
    h, w = img.shape[0], img.shape[1]
    cands = []
    for t in range(-half, half + 1):
        for s in range(-lat, lat + 1):
            x = int(round(q[0] + u[0] * t - u[1] * s))
            y = int(round(q[1] + u[1] * t + u[0] * s))
            if x < 1 or y < 1 or x >= w - 1 or y >= h - 1:
                continue
            if PA._classify(img, x, y) != "g":
                continue
            if want_maroon:
                ok = False
                for b in range(1, back + 1):
                    if PA._classify(img, int(round(x - u[0] * b)),
                                    int(round(y - u[1] * b))) == "m":
                        ok = True
                        break
                if not ok:
                    continue
            # 选点次序：先「沿轴向最近」再「横向离射线最近」再「t 升序」——同一条 t 上有多个
            # 候选时，贴轴的那个才可能是真交点（旧版按 dx/dy 排序，等价于按 s 降序，是任意的）。
            cands.append((abs(t), abs(s), t, x - q[0], y - q[1]))
    if not cands:
        return None
    cands.sort()
    return cands[0][2:]


def scan(a, cfg, index, mode):
    """跑一遍：返回 (逐案 agree 行, 统计字典)。"""
    rows = []
    st = dict(n_pin=0, n_jn=0, n_jc=0, d_jn=[], d_jc=[],
              n_group=0, stable_jn=0, stable_jc=0,
              n_cmp=0, hit_jn=0, hit_jc=0)
    for cid in [s.strip() for s in a.cases.split(",") if s.strip()]:
        case = index.get(cid)
        if case is None:
            continue
        pp = os.path.join(a.run_dir, "final_results", "%s_Pred.json" % cid)
        if not os.path.exists(pp):
            continue
        gt_doc = json.load(io.open(case.json_path, encoding="utf-8"))
        pred_doc = json.load(io.open(pp, encoding="utf-8"))
        _, km = assign_components(gt_doc, pred_doc, cfg, {})
        pred2 = rewrite_keys(pred_doc, km)
        r = PA.analyse_case(cid, case, a.run_dir, cfg, ray=False)
        if r is None:
            continue
        img = np.array(Image.open(case.png_path).convert("RGB"))
        h = img.shape[0]
        groups = r["groups"]
        g_jn, g_jc = copy.deepcopy(groups), copy.deepcopy(groups)
        for (ck, edge), g in groups.items():
            u = PA._outward_img(edge)                     # 图像坐标外向单位向量
            v_jn, v_jc = [], []
            for num, pt, gpt, td in g["td"]:
                st["n_pin"] += 1
                q = (pt[0], h - pt[1])
                pn = probe(img, q, u, a.half, False, lat=a.lat, back=a.back)
                pc = probe(img, q, u, a.half, True, lat=a.lat, back=a.back)
                if pn:
                    st["n_jn"] += 1
                    st["d_jn"].append((pn[1] ** 2 + pn[2] ** 2) ** 0.5)
                    v_jn.append((pn[1], -pn[2]))
                    g_jn[(ck, edge)]["ed"].append((num, pt, (pn[1], -pn[2])))
                if pc:
                    st["n_jc"] += 1
                    st["d_jc"].append((pc[1] ** 2 + pc[2] ** 2) ** 0.5)
                    v_jc.append((pc[1], -pc[2]))
                    g_jc[(ck, edge)]["ed"].append((num, pt, (pc[1], -pc[2])))
            if len(g["td"]) >= a.min_n:
                st["n_group"] += 1
                if len(v_jn) >= a.min_n and PA._axis_dev(v_jn) <= 2.0:
                    st["stable_jn"] += 1
                if len(v_jc) >= a.min_n and PA._axis_dev(v_jc) <= 2.0:
                    st["stable_jc"] += 1
        # 与 GT 锚点比一致率
        for (ck, edge), g in groups.items():
            if len(g["td"]) < a.min_n:
                continue
            tdx = PA._med([d[0] for (_, _, _, d) in g["td"]])
            tdy = PA._med([d[1] for (_, _, _, d) in g["td"]])
            for gg, tag in ((g_jn, "jn"), (g_jc, "jc")):
                ds = gg[(ck, edge)]["ed"]
                if len(ds) < a.min_n:
                    continue
                mx = PA._med([e[2][0] for e in ds])
                my = PA._med([e[2][1] for e in ds])
                ok = max(abs(mx - tdx), abs(my - tdy)) <= 2.0
                st["n_cmp"] += 1
                st["hit_jn" if tag == "jn" else "hit_jc"] += 1 if ok else 0
                if a.show and tag == "jc":
                    print("   %-4s %-5s %-2s n=%d true=(%6.1f,%6.1f) junc=(%6.1f,%6.1f) %s"
                          % (cid, ck, edge, len(g["td"]), tdx, tdy, mx, my,
                             "OK" if ok else "no"))
        if mode == "stats":
            continue
        base = pin_f1(gt_doc, pred2, cfg, {}).f1
        o_true = pin_f1(gt_doc, PA._shift_pins(pred2, groups, "td", False), cfg, {}).f1
        o_jn = pin_f1(gt_doc, PA._shift_pins(pred2, g_jn, "ed", True), cfg, {}).f1
        o_jc = pin_f1(gt_doc, PA._shift_pins(pred2, g_jc, "ed", True), cfg, {}).f1
        o_jn_s = pin_f1(gt_doc, PA._shift_pins(pred2, g_jn, "ed", True, strict=True),
                        cfg, {}).f1
        o_jc_s = pin_f1(gt_doc, PA._shift_pins(pred2, g_jc, "ed", True, strict=True),
                        cfg, {}).f1
        rows.append((cid, base, o_true, o_jn, o_jc, o_jn_s, o_jc_s))
        print("case %-3s base %.4f | true %.4f | jnear %.4f junc %.4f | jnear_s %.4f "
              "junc_s %.4f" % (cid, base, o_true, o_jn, o_jc, o_jn_s, o_jc_s))
    return rows, st


def scan_free(a):
    """GT-free 统计：脚集合与 (元件,边) 分组全部来自我方产物，不读 GT。

    注意：只借数据集索引里的**图像路径**（`discover()`），不碰 GT 标注坐标/名字。
    """
    st = dict(n_pin=0, n_jn=0, n_jc=0, d_jn=[], d_jc=[],
              n_group=0, stable_jn=0, stable_jc=0,
              n_case=0, n_case_skip=0, n_comp=0)
    for cid in [s.strip() for s in a.cases.split(",") if s.strip()]:
        pp = os.path.join(a.run_dir, "final_results", "%s_Pred.json" % cid)
        if not os.path.exists(pp):
            st["n_case_skip"] += 1
            continue
        case = a.index.get(cid)
        if case is None or not os.path.exists(case.png_path):
            st["n_case_skip"] += 1
            continue
        pred_doc = json.load(io.open(pp, encoding="utf-8"))
        img = np.array(Image.open(case.png_path).convert("RGB"))
        h = img.shape[0]
        # ---- GT-free 分组：我方 box 元件的 bbox + 我方脚点 ----
        groups = {}
        for ck in sorted(PA.box_comps(pred_doc).keys()):
            op = PA._pin_map(pred_doc, ck)
            bbox = (pred_doc.get("components") or {}).get(ck, {}).get("bbox")
            if not op or not bbox:
                continue
            st["n_comp"] += 1
            for num, (_nm, pt) in sorted(op.items()):
                groups.setdefault((ck, PA._edge_of(pt, bbox)), []).append((num, pt))
        if not groups:
            st["n_case_skip"] += 1
            continue
        st["n_case"] += 1
        for (ck, edge), pins in groups.items():
            u = PA._outward_img(edge)
            v_jn, v_jc = [], []
            for num, pt in pins:
                st["n_pin"] += 1
                q = (pt[0], h - pt[1])
                pn = probe(img, q, u, a.half, False, lat=a.lat, back=a.back)
                pc = probe(img, q, u, a.half, True, lat=a.lat, back=a.back)
                if pn:
                    st["n_jn"] += 1
                    st["d_jn"].append((pn[1] ** 2 + pn[2] ** 2) ** 0.5)
                    v_jn.append((pn[1], -pn[2]))
                if pc:
                    st["n_jc"] += 1
                    st["d_jc"].append((pc[1] ** 2 + pc[2] ** 2) ** 0.5)
                    v_jc.append((pc[1], -pc[2]))
            if len(pins) >= a.min_n:
                st["n_group"] += 1
                if len(v_jn) >= a.min_n and PA._axis_dev(v_jn) <= 2.0:
                    st["stable_jn"] += 1
                if len(v_jc) >= a.min_n and PA._axis_dev(v_jc) <= 2.0:
                    st["stable_jc"] += 1
    return st


def report_rows(rows):
    if not rows:
        print("(无可判分的 case)")
        return
    m = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows]).mean(axis=0)
    print()
    print("均值 base %.4f | true %.4f (Δ%+.4f)" % (m[0], m[1], m[1] - m[0]))
    for tag, idx, idxs in (("jnear", 2, 4), ("junc", 3, 5)):
        win = sum(1 for r in rows if r[idx + 1] > r[1] + 1e-9)
        lose = sum(1 for r in rows if r[idx + 1] < r[1] - 1e-9)
        print("     %-5s %.4f (Δ%+.4f) | strict %.4f (Δ%+.4f) | 逐案 胜 %d / 负 %d"
              % (tag, m[idx], m[idx] - m[0], m[idxs], m[idxs] - m[0], win, lose))


def report_stats(st, min_n, free=False):
    print()
    if free:
        print("GT-free 统计（脚集合与 (元件,边) 分组均来自我方产物；不用 GT 标注，只借图像路径）"
              " 可用 case %d（跳过 %d） 我方 box 元件 %d"
              % (st["n_case"], st["n_case_skip"], st["n_comp"]))
    else:
        print("组统计（可比组 n>=%d） n=%d；中位与 GT 一致(≤2px)：jnear %d/%d (%.1f%%)  "
              "junc %d/%d (%.1f%%)"
              % (min_n, st["n_group"], st["hit_jn"], st["n_cmp"],
                 100.0 * st["hit_jn"] / max(1, st["n_cmp"]), st["hit_jc"], st["n_cmp"],
                 100.0 * st["hit_jc"] / max(1, st["n_cmp"])))
    print("GT-blind 体检：脚总数 %d" % st["n_pin"])
    for tag, k, dk in (("jnear", "n_jn", "d_jn"), ("junc", "n_jc", "d_jc")):
        d = st[dk]
        print("  %-5s 覆盖 %d (%.1f%%)  距离中位 %.2f p90 %.2f"
              % (tag, st[k], 100.0 * st[k] / max(1, st["n_pin"]),
                 float(np.median(d)) if d else -1,
                 float(np.percentile(d, 90)) if d else -1))
    print("  组内恒定（dev<=2 的组占比）：jnear %d/%d (%.1f%%)  junc %d/%d (%.1f%%)"
          % (st["stable_jn"], st["n_group"],
             100.0 * st["stable_jn"] / max(1, st["n_group"]),
             st["stable_jc"], st["n_group"],
             100.0 * st["stable_jc"] / max(1, st["n_group"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--mode", choices=("agree", "stats", "statsfree", "both"), default="both")
    ap.add_argument("--min-n", type=int, default=2,
                    help="成组/一致率所需最少脚数（stats 组统计通常用 3）")
    ap.add_argument("--half", type=int, default=26)
    ap.add_argument("--back", type=int, default=2,
                    help="「内侧贴 maroon」窗口 1..back px（默认 2 = 严格口径；放宽到 4 时"
                         "覆盖 37.9%%→53.2%%、组内恒定 19.2%%→50.0%%，见 §5.152 补）")
    ap.add_argument("--lat", type=int, default=2, help="横向搜索半径 ±px（默认 2；放宽无收益）")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    cfg = EvalConfig()
    index = {str(c.case_id): c for c in discover()}
    print("口径：half=%d, lat=±%d, back=1..%d, min-n=%d, run-dir=%s"
          % (a.half, a.lat, a.back, a.min_n, a.run_dir))
    a.index = index
    if a.mode == "statsfree":            # GT-free 档不借 GT 分组，也不跑 GT 配对
        report_stats(scan_free(a), a.min_n, free=True)
        return
    rows, st = scan(a, cfg, index, a.mode)
    if a.mode in ("agree", "both"):
        report_rows(rows)
    if a.mode in ("stats", "both"):
        report_stats(st, a.min_n)


if __name__ == "__main__":
    main()
