"""给 stage1 提取出的导线线段做「假阳性归因」：没贴住 GT 的那些，究竟是哪来的？

判据与 stage1_wires_eval.py 完全一致（tol=5px，线段上 16 个采样点全部贴住才算 TP）。
对每条非 TP 线段，按其中点落在哪里归类：
    comp  —— 贴着某个**已检出元件框**（说明该元件的轮廓没被遮挡干净 / 框比本体小）
    text  —— 贴着某个**文字框**（文字被涂掉了却仍有残余笔画，或桥接段落在文字里）
    frame —— 贴着图纸边框 / 标题栏（drop_frame 没覆盖到的表格线）
    other —— 以上都不是

用法: python work\\stage1_wire_attrib.py [stage1产出目录] [--limit N]
"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
for _d in ("eval", "diag", "data", "gnd", "viz"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _d))

from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402
from stage1_wires_eval import seg_arrays, dist_pts_to_segs, sample_polyline  # noqa: E402


def near_box(px, py, boxes, pad):
    for b in boxes:
        if b[0] - pad <= px <= b[2] + pad and b[1] - pad <= py <= b[3] + pad:
            return True
    return False


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else r"Y:\workspace\EDA\work\stage1_all9"
    limit = 0
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    tol = 5.0
    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    files = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    if limit:
        files = files[:limit]

    tot = collections.Counter()
    by_src = collections.defaultdict(collections.Counter)
    seg_len = collections.defaultdict(float)
    lens = collections.defaultdict(list)
    pws = collections.defaultdict(list)
    n_files = 0
    for f in files:
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
        n_files += 1
        H = float(d["image"]["h"]); W = float(d["image"]["w"])
        gt = json.load(open(case.json_path, encoding="utf-8"))
        gsegs = []
        for net in gt["nets"].values():
            for pts in (net.get("edges") or {}).values():
                for i in range(len(pts) - 1):
                    gsegs.append([[float(pts[i]["x"]), H - float(pts[i]["y"])],
                                  [float(pts[i + 1]["x"]), H - float(pts[i + 1]["y"])]])
        GA = seg_arrays(gsegs)
        PA = seg_arrays(d["wire_detail"]["segments"])
        if PA.shape[0] == 0:
            continue

        if GA.shape[0]:
            pp = np.concatenate([sample_polyline(PA[i, :2], PA[i, 2:], 16) for i in range(PA.shape[0])])
            po = np.concatenate([np.full(16, i) for i in range(PA.shape[0])])
            pdist = dist_pts_to_segs(pp, GA)
            pworst = np.full(PA.shape[0], 0.0)
            np.maximum.at(pworst, po, pdist)
        else:
            pworst = np.full(PA.shape[0], 1e9)

        cboxes = [c["bbox_img"] for c in d.get("components", [])]
        tboxes = [t["bbox_img"] for t in d.get("texts", [])]
        png = os.path.basename(d["image"]["path"]).lower().replace(" ", "")
        src = ("kicad" if "kicad" in png else "altium" if "altium" in png
               else "datasheet" if "datasheet" in png else "jlc" if "jlc" in png else "other")

        for i in range(PA.shape[0]):
            L = float(np.hypot(PA[i, 2] - PA[i, 0], PA[i, 3] - PA[i, 1]))
            if pworst[i] <= tol:
                k = "TP"
            else:
                mx = float((PA[i, 0] + PA[i, 2]) / 2); my = float((PA[i, 1] + PA[i, 3]) / 2)
                if near_box(mx, my, cboxes, 3.0):
                    k = "comp"
                elif near_box(mx, my, tboxes, 3.0):
                    k = "text"
                elif (mx < 0.03 * W or mx > 0.97 * W or my < 0.03 * H or my > 0.97 * H
                      or (mx > 0.60 * W and my > 0.62 * H)):
                    k = "frame"
                else:
                    k = "other"
            tot[k] += 1
            by_src[src][k] += 1
            seg_len[k] += L
            lens[k].append(L)
            pws[k].append(float(pworst[i]))

    print(f"stage1 产物 {n_files} 份   tol={tol}px")
    n = sum(tot.values())
    print(f"\n=== 提取线段归因（合计 {n}）===")
    print(f'  {"类别":8s} {"条数":>8s} {"占比":>7s} {"总长px":>12s} {"长占比":>7s}')
    L = sum(seg_len.values()) or 1.0
    for k in ("TP", "comp", "text", "frame", "other"):
        print(f'  {k:8s} {tot[k]:8d} {tot[k]/max(1,n):7.3f} {seg_len[k]:12.0f} {seg_len[k]/L:7.3f}')

    print("\n=== 每个桶的内部结构（长度 / 到最近 GT 线段的最差采样距离）===")
    print(f'  {"类别":8s} {"len p50":>8s} {"len p90":>8s} {"len p99":>8s} '
          f'{"d p10":>7s} {"d p50":>7s} {"d p90":>7s}')
    np.set_printoptions(suppress=True)
    for k in ("TP", "comp", "text", "frame", "other"):
        if not lens[k]:
            continue
        a = np.array(lens[k]); b = np.array(pws[k])
        print(f'  {k:8s} {np.percentile(a,50):8.1f} {np.percentile(a,90):8.1f} '
              f'{np.percentile(a,99):8.1f} {np.percentile(b,10):7.1f} '
              f'{np.percentile(b,50):7.1f} {np.percentile(b,90):7.1f}')

    print("\n=== 按来源（占比）===")
    for s in sorted(by_src):
        a = by_src[s]
        m = sum(a.values()) or 1
        print(f'  {s:10s} n={m:7d}  TP {a["TP"]/m:.3f}  comp {a["comp"]/m:.3f}  '
              f'text {a["text"]/m:.3f}  frame {a["frame"]/m:.3f}  other {a["other"]/m:.3f}')


if __name__ == "__main__":
    main()
