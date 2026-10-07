# -*- coding: utf-8 -*-
"""检出框 vs 真值框的几何差：宽高比、中心偏移。看清「框到底差在哪」。

    cd Y:\\workspace\\EDA\\eda-pcb
    python Y:\\workspace\\EDA\\work\\_boxgeom.py [--limit 10] [--show 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from cvparts import textbox, textseg  # noqa: E402
from stage2.text import imread_unicode  # noqa: E402
from _textseg_eval import ROLE_OF, cases, iou_matrix  # noqa: E402

CFG = dict(ink_mode="max", min_dim=2, min_area=3, line_len=15,
           resid_min_h=4, resid_min_dim=1)


def gt_of(js):
    d = json.loads(Path(js).read_text(encoding="utf-8"))
    out = []
    for t in (d.get("texts") or {}).values():
        loc = t.get("location") or {}
        c = (t.get("content") or "").strip()
        role = ROLE_OF.get(str(t.get("details") or ""))
        if not role or not c or len(c) > 24:
            continue
        try:
            b = tuple(float(loc[k]) for k in ("x1", "y1", "x2", "y2"))
        except (TypeError, KeyError):
            continue
        if b[2] - b[0] < 2 or b[3] - b[1] < 2:
            continue
        out.append((b, c, role))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--show", type=int, default=20)
    args = ap.parse_args(argv)

    wr, hr, dxr, dyr, ious = [], [], [], [], []
    ious_pad = []
    ok = {"gt": 0, "pad": 0, "det": 0}
    cnt = 0
    shown = 0
    from collections import Counter
    ncomp = Counter()
    from stage2.text import TextEngine, pad_box
    rr = TextEngine().recognizer()
    for png, js in cases(args.limit):
        gts = gt_of(js)
        if not gts:
            continue
        bgr = imread_unicode(png)
        det = [tuple(float(v) for v in b) for b in textseg.detect_boxes(bgr, **CFG)]
        ncomp[png.parent.name] = len(det)
        M = iou_matrix([g[0] for g in gts], det) if det else np.zeros((len(gts), 0))
        jobs = {}
        for i, (b, c, role) in enumerate(gts):
            row = M[i] if M.size else np.zeros(0)
            j = int(np.argmax(row)) if row.size else -1
            if j < 0 or row[j] < 0.05:
                continue
            d = det[j]
            pd = pad_box(d)
            ious_pad.append(float(iou_matrix([b], [pd])[0][0]))
            jobs.setdefault(role, []).append((i, b, c, d, pd))
            gw, gh = b[2] - b[0], b[3] - b[1]
            dw, dh = d[2] - d[0], d[3] - d[1]
            wr.append(dw / max(1e-6, gw))
            hr.append(dh / max(1e-6, gh))
            dxr.append(((d[0] + d[2]) - (b[0] + b[2])) / 2.0 / max(1e-6, gw))
            dyr.append(((d[1] + d[3]) - (b[1] + b[3])) / 2.0 / max(1e-6, gh))
            ious.append(float(row[j]))
            if shown < args.show:
                shown += 1
                print(f"  {role:12s} {c[:14]!r:16s} gt=({b[0]:.0f},{b[1]:.0f},"
                      f"{b[2]:.0f},{b[3]:.0f}) {gw:.0f}x{gh:.0f}  det=({d[0]:.0f},"
                      f"{d[1]:.0f},{d[2]:.0f},{d[3]:.0f}) {dw:.0f}x{dh:.0f}  "
                      f"IoU {row[j]:.2f}  pad={pd[2]-pd[0]:.0f}x{pd[3]-pd[1]:.0f}")
        for role, lst in jobs.items():
            o1 = rr.read_many(bgr, [(x[1], role) for x in lst])
            o2 = rr.read_many(bgr, [(x[4], role) for x in lst])
            o3 = rr.read_many(bgr, [(x[3], role) for x in lst])
            for (i, b, c, d, pd), a1, a2, a3 in zip(lst, o1, o2, o3):
                cnt += 1
                ok["gt"] += (a1[0] or "").strip() == c
                ok["pad"] += (a2[0] or "").strip() == c
                ok["det"] += (a3[0] or "").strip() == c
    print(f"\n配对 {len(wr)}   检出框数按 case {dict(ncomp)}")
    print(f"  同一批 {cnt} 个框：GT原样 {ok['gt']/max(1,cnt):.4f}   "
          f"pad(det) {ok['pad']/max(1,cnt):.4f}   det原样 {ok['det']/max(1,cnt):.4f}")
    print(f"  pad(det) 与 GT 的 IoU: p10 {np.percentile(ious_pad,10):.3f} "
          f"p50 {np.percentile(ious_pad,50):.3f} p90 {np.percentile(ious_pad,90):.3f}")

    # ---- 固定 px 外扩扫描 + 干净子集 ----
    print("\n固定外扩扫描（同一批配对，只在外扩量不同）")
    print("| pad | 全部 exact | 干净子集 exact | 干净子集 n |")
    print("|---|---|---|---|")
    for pd_ in (0, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20):
        hits_all = hits_cln = n_cln = 0
        tot = 0
        for png, js in cases(args.limit):
            gts = gt_of(js)
            if not gts:
                continue
            bgr = imread_unicode(png)
            det = [tuple(float(v) for v in b) for b in textseg.detect_boxes(bgr, **CFG)]
            if not det:
                continue
            M = iou_matrix([g[0] for g in gts], det)
            jobs = {}
            for i, (b, c, role) in enumerate(gts):
                j = int(np.argmax(M[i]))
                if M[i][j] < 0.05:
                    continue
                d = det[j]
                gw, gh = b[2] - b[0], b[3] - b[1]
                dh = d[3] - d[1]
                hr_ = dh / max(1e-6, gh)
                clean = M[i][j] >= 0.40 and 0.5 <= hr_ <= 1.5
                grew = (d[0] - pd_, d[1] - pd_, d[2] + pd_, d[3] + pd_)
                jobs.setdefault(role, []).append((i, c, grew, clean))
            for role, lst in jobs.items():
                outs = rr.read_many(bgr, [(x[2], role) for x in lst])
                for (i, c, _g, clean), (txt, _cf) in zip(lst, outs):
                    tot += 1
                    good = (txt or "").strip() == c
                    hits_all += good
                    if clean:
                        n_cln += 1
                        hits_cln += good
        print(f"| {pd_:+d} | {hits_all/max(1,tot):.4f} | {hits_cln/max(1,n_cln):.4f} | {n_cln} |")
    for name, v in (("宽 det/gt", wr), ("高 det/gt", hr),
                    ("中心 dx/gt宽", dxr), ("中心 dy/gt高", dyr), ("IoU", ious)):
        a = np.asarray(v)
        print(f"  {name:12s} p10 {np.percentile(a,10):6.2f} p25 {np.percentile(a,25):6.2f}"
              f" p50 {np.percentile(a,50):6.2f} p75 {np.percentile(a,75):6.2f}"
              f" p90 {np.percentile(a,90):6.2f}")
    a = np.asarray(wr)
    print(f"  宽比 <0.8 占 {float((a<0.8).mean()):.3f}  <0.6 占 {float((a<0.6).mean()):.3f}")
    a = np.asarray(hr)
    print(f"  高比 <0.8 占 {float((a<0.8).mean()):.3f}  <0.6 占 {float((a<0.6).mean()):.3f}"
          f"  >1.3 占 {float((a>1.3).mean()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
