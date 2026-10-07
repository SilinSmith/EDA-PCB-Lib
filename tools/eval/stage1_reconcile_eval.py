"""评测「悬空交叉校验」的收益：把端点全悬空的导线分量删掉，看导线指标怎么变。

用法: python work\stage1_reconcile_eval.py [stage1产出目录] [--limit N]

两个臂用**完全同一套**判据（tol=5px，见 stage1_wires_eval.py）：
  base  = 原产物
  drop  = 删掉 reconcile 判为 suspect 的导线分量（端点两端都没锚在元件/文字上）
"""
import glob
import json
import os
import sys

import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
for _d in ("eval", "diag", "data", "gnd", "viz"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d))

from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402
from stage1_wires_eval import seg_arrays, dist_pts_to_segs, sample_polyline  # noqa: E402
from stage1.reconcile import reconcile, apply_drop  # noqa: E402
from cvparts.textbox import detect_boxes  # noqa: E402


def measure(GA, PA, tol):
    if GA.shape[0] and PA.shape[0]:
        gp = np.concatenate([sample_polyline(GA[i, :2], GA[i, 2:], 16) for i in range(GA.shape[0])])
        go = np.concatenate([np.full(16, i) for i in range(GA.shape[0])])
        dist = dist_pts_to_segs(gp, PA)
        worst = np.full(GA.shape[0], 0.0)
        np.maximum.at(worst, go, dist)
        cov = int((worst <= tol).sum())
        pp = np.concatenate([sample_polyline(PA[i, :2], PA[i, 2:], 16) for i in range(PA.shape[0])])
        po = np.concatenate([np.full(16, i) for i in range(PA.shape[0])])
        pdist = dist_pts_to_segs(pp, GA)
        pworst = np.full(PA.shape[0], 0.0)
        np.maximum.at(pworst, po, pdist)
        pcov = int((pworst <= tol).sum())
    else:
        cov = pcov = 0
    lg = float(np.hypot(GA[:, 2] - GA[:, 0], GA[:, 3] - GA[:, 1]).sum()) if GA.shape[0] else 0.0
    lp = float(np.hypot(PA[:, 2] - PA[:, 0], PA[:, 3] - PA[:, 1]).sum()) if PA.shape[0] else 0.0
    return cov, GA.shape[0], pcov, PA.shape[0], lg, lp


def f1_of(tp, ng, pp_, npd):
    r = tp / max(1, ng)
    p = pp_ / max(1, npd)
    return 2 * r * p / max(1e-9, r + p)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else r"Y:\workspace\EDA\work\stage1_all7"
    limit = 0
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    tol = 5.0

    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    files = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    if limit:
        files = files[:limit]
    print(f"stage1 产物 {len(files)} 份   tol={tol}px   目录 {out_dir}")

    arms = {"base": (None, False)}
    for m in (80.0, 160.0, 1e9):
        arms[f"drop<{m:g}"] = (m, False)
    for m in (80.0, 160.0, 1e9):
        arms[f"drop<{m:g}+txt"] = (m, True)

    acc = {k: [0, 0, 0, 0, 0.0, 0.0] for k in arms}
    dang_cls = {}
    dang_src = {}
    suspect_len = []
    n_groups = 0
    n_dang = 0

    for fi, f in enumerate(files):
        d = json.load(open(f, encoding="utf-8"))
        cid = d["case"]
        case = cases.get(cid)
        if case is None:
            for k, v in cases.items():
                if k.lstrip("0") == cid.lstrip("0"):
                    case, cid = v, k
                    break
        if case is None:
            continue
        H = float(d["image"]["h"])
        gt = json.load(open(case.json_path, encoding="utf-8"))
        gsegs = []
        for net in gt["nets"].values():
            for pts in (net.get("edges") or {}).values():
                for i in range(len(pts) - 1):
                    gsegs.append([[float(pts[i]["x"]), H - float(pts[i]["y"])],
                                  [float(pts[i + 1]["x"]), H - float(pts[i + 1]["y"])]])
        GA = seg_arrays(gsegs)

        bgr = cv2.imread(str(case.png_path), cv2.IMREAD_COLOR)
        tboxes = detect_boxes(bgr) if bgr is not None else []

        for name, (maxlen, use_txt) in arms.items():
            if maxlen is None:
                doc = d
            else:
                rep = reconcile(d, suspect_max_len=maxlen,
                                text_boxes=tboxes if use_txt else None)
                doc = apply_drop(d, rep["drop_segments"])
            PA = seg_arrays(doc["wire_detail"]["segments"])
            a = acc[name]
            cov, ng, pcov, npd, lg, lp = measure(GA, PA, tol)
            a[0] += cov; a[1] += ng; a[2] += pcov; a[3] += npd; a[4] += lg; a[5] += lp

        # 统计用（只算一次，用默认参数 + 文字锚）
        rep = reconcile(d, text_boxes=tboxes)
        st = rep["stats"]
        n_groups += st["n_wire_groups"]
        suspect_len.append(st["suspect_len_frac"])
        for c in rep["components"]:
            if c["dangling"]:
                n_dang += 1
                dang_cls[c["type"]] = dang_cls.get(c["type"], 0) + 1
                dang_src[c["source"]] = dang_src.get(c["source"], 0) + 1

    print("\n=== 导线指标（删掉端点全悬空的分量）===")
    print(f'  {"臂":12s} {"R":>7s} {"P":>7s} {"F1":>7s} {"提取段":>8s} {"总长比":>8s}')
    for name in arms:
        cov, ng, pcov, npd, lg, lp = acc[name]
        r = cov / max(1, ng); p = pcov / max(1, npd)
        print(f'  {name:12s} {r:7.4f} {p:7.4f} {f1_of(cov,ng,pcov,npd):7.4f} {npd:8d} {lp/max(1.0,lg):8.3f}')

    print(f"\n=== 悬空统计（{len(files)} 张）===")
    print(f"  导线分量合计 {n_groups}   悬空元件 {n_dang}")
    if suspect_len:
        a = np.asarray(suspect_len) * 100
        print(f"  suspect 分量占总长比例：p10 {np.percentile(a,10):.1f}%  p50 {np.percentile(a,50):.1f}%  "
              f"p90 {np.percentile(a,90):.1f}%")
    print("  悬空元件按 type:")
    for k, v in sorted(dang_cls.items(), key=lambda t: -t[1])[:15]:
        print(f"    {k:20s} {v}")
    print("  悬空元件按 source:")
    for k, v in sorted(dang_src.items(), key=lambda t: -t[1]):
        print(f"    {str(k):20s} {v}")


if __name__ == "__main__":
    main()
