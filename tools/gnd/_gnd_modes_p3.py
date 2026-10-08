"""Part 3: use the Part-1 shape modes as a VALIDATOR for YOLO's existing gnd boxes.

Read-only w.r.t. the repo.  Same ruler, same normalisation (32x32 zero-mean
unit-var) as tools/gnd/gnd_modes.py Part 1.

For every gnd prediction written by stage1 we crop the predicted box from the
sheet, normalise it exactly like a GT crop, and take the max cosine similarity
to the k=10 mode centroids.  Then:
  arm a  baseline          : keep every YOLO gnd box
  arm b  mode validation   : drop the box when maxsim < thr   (thr swept)
and we report, per threshold, how many of the DROPPED boxes were true positives
versus false positives -- that is the number that decides the whole idea.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb\tools\gnd")
sys.stdout.reconfigure(encoding="utf-8")
import gnd_modes as GM  # noqa: E402
from schematic_parser import data as S  # noqa: E402

RUN = Path(r"Y:\workspace\EDA\work\_full220\runs\stage1")
TILE = GM.TILE
IOU = 0.25


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def norm_patch(g, box):
    """gnd_modes.grab()'s normalisation, on a box ALREADY in image coords (top-left).

    (bbox_img is top-left origin; only the GT `bbox` field is bottom-left and
    needs the `h - y` flip, which the caller does.)
    """
    h, w = g.shape[0], g.shape[1]
    x1, y1, x2, y2 = [float(v) for v in box]
    iy1, iy2 = sorted((y1, y2))
    x1, x2 = sorted((x1, x2))
    a, b = max(0, int(round(iy1))), min(h, int(round(iy2)))
    c, d = max(0, int(round(x1))), min(w, int(round(x2)))
    sub = g[a:b, c:d]
    if sub.size == 0 or min(sub.shape) < 5:
        return None
    n = np.asarray(Image.fromarray(sub).resize((TILE, TILE), Image.BILINEAR), dtype=np.float64)
    n = (n - n.mean()) / (n.std() + 1e-6)
    return n.reshape(-1)


def main() -> None:
    # ---------- rebuild the Part-1 modes (identical code path / seed) ----------
    normals, gt_by_case = [], {}
    for c in S.discover(S.DEFAULT_DATA_ROOT):
        try:
            doc = S.load_json(c.json_path)
        except Exception:
            continue
        gnds = [v for v in (doc.get("components") or {}).values()
                if str(v.get("type", "")).strip() == "gnd"]
        if not gnds:
            continue
        try:
            g = np.asarray(Image.open(str(c.png_path)).convert("L"), dtype=np.uint8)
        except Exception:
            continue
        h = g.shape[0]
        boxes = []
        for comp in gnds:
            b = comp.get("bbox")
            if not b or len(b) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in b]
            boxes.append((x1, h - y2, x2, h - y1))
            n, _ = GM.grab(g, b)          # identical shape gate as Part 1/2
            if n is not None:
                normals.append(n)
        if boxes:
            gt_by_case[str(c.case_id)] = boxes
            gt_by_case[Path(str(c.png_path)).stem] = boxes
    X = np.stack(normals).reshape(len(normals), -1).astype(np.float64)
    print(f"rebuilt modes from {len(X)} GT gnd crops; GT sheets {sum(1 for k in gt_by_case)}")
    lab, C, _ = GM.kmeans(X, 10)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    print("mode sizes", dict(Counter(lab.tolist())))

    # ---------- calibration of the ruler: how similar is a REAL GT gnd crop? ----------
    Xu = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    s_gt = (Xu @ Cn.T).max(1)
    print(f"\n  calibration on the {len(X)} GT gnd crops (max cosine to any mode centroid):")
    print("  " + "".join(f"{p:>8}" for p in (10, 25, 50, 75, 90))
          + f"{'mean':>8}   " + "".join(f"{p:>8.3f}" for p in np.percentile(s_gt, (10, 25, 50, 75, 90)))
          + f"{s_gt.mean():>8.3f}")

    # ---------- score every YOLO gnd box ----------
    rows = []
    n_sheets = 0
    for f in sorted(RUN.glob("*.stage1.json")):
        doc = json.load(open(f, encoding="utf-8"))
        case = str(doc.get("case"))
        img_path = (doc.get("image") or {}).get("path")
        gts = gt_by_case.get(case) or gt_by_case.get(Path(str(img_path)).stem)
        if gts is None or not img_path or not Path(img_path).exists():
            continue
        n_sheets += 1
        g = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
        preds = [c for c in doc.get("components", []) if str(c.get("type")) == "gnd"]
        # greedy 1-1 IoU match, preds sorted by score
        preds.sort(key=lambda c: -float(c.get("score") or 0.0))
        used = [False] * len(gts)
        for c in preds:
            box = [float(v) for v in c["bbox_img"]]
            bi, bj = -1, 0.0
            for gi, gb in enumerate(gts):
                if used[gi]:
                    continue
                v = iou(box, gb)
                if v > bj:
                    bi, bj = gi, v
            is_tp = bi >= 0 and bj >= IOU
            if is_tp:
                used[bi] = True
            v = norm_patch(g, box)
            if v is None:
                sims = np.zeros(10)
            else:
                sims = Cn @ (v / (np.linalg.norm(v) + 1e-9))
            rows.append({"case": case, "tp": is_tp, "maxsim": float(sims.max()),
                         "mode": int(sims.argmax()), "score": float(c.get("score") or 0.0),
                         "src": c.get("source")})
        # GT not matched by any pred
        for gi, gb in enumerate(used):
            if not gb:
                rows.append({"case": case, "tp": None, "miss": True})

    box_rows = [r for r in rows if "maxsim" in r]
    tp = sum(1 for r in box_rows if r["tp"])
    fp = sum(1 for r in box_rows if not r["tp"])
    n_gt = tp + sum(1 for r in rows if r.get("miss"))
    print(f"\n=== part 3 base: YOLO gnd boxes on {n_sheets} sheets ===")
    print(f"  boxes {len(box_rows)}  TP {tp}  FP {fp}  GT-missed {n_gt - tp}")
    print(f"  P {tp/max(tp+fp,1):.3f}  R {tp/max(n_gt,1):.3f}  FP/sheet {fp/max(n_sheets,1):.1f}")
    print(f"  reference (official val, printed by gnd_modes.py:284): P 0.851  R 0.762")

    s_tp = np.array([r["maxsim"] for r in box_rows if r["tp"]])
    s_fp = np.array([r["maxsim"] for r in box_rows if not r["tp"]])
    q = [10, 25, 50, 75, 90]
    print(f"\n  maxsim quantiles  {'':<12}" + "".join(f"{p:>8}" for p in q))
    print(f"  {'TP boxes':<26}" + "".join(f"{np.percentile(s_tp,p):>8.3f}" for p in q))
    print(f"  {'FP boxes':<26}" + "".join(f"{np.percentile(s_fp,p):>8.3f}" for p in q))
    print(f"  mean TP {s_tp.mean():.3f}  mean FP {s_fp.mean():.3f}")

    print(f"\n  mode histogram (n / how many of them are FP)")
    mt, mf = Counter(r["mode"] for r in box_rows if r["tp"]), Counter(r["mode"] for r in box_rows if not r["tp"])
    for m in range(10):
        n = mt[m] + mf[m]
        print(f"    mode {m}: n={n:4d}  TP={mt[m]:4d}  FP={mf[m]:4d}  FP-rate={mf[m]/max(n,1):.2f}")

    print(f"\n  {'thr':>5} {'kept':>6} {'P':>7} {'R':>7} {'FP/sheet':>9} {'dropped':>8} "
          f"{'dropTP':>7} {'dropFP':>7} {'prec.of.dropped':>16}")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        kept = [r for r in box_rows if r["maxsim"] >= thr]
        ktp = sum(1 for r in kept if r["tp"])
        kfp = len(kept) - ktp
        drop = [r for r in box_rows if r["maxsim"] < thr]
        dtp = sum(1 for r in drop if r["tp"])
        dfp = len(drop) - dtp
        print(f"  {thr:>5.2f} {len(kept):>6} {ktp/max(len(kept),1):>7.3f} {ktp/max(n_gt,1):>7.3f} "
              f"{kfp/max(n_sheets,1):>9.1f} {len(drop):>8} {dtp:>7} {dfp:>7} "
              f"{dfp/max(len(drop),1):>16.3f}")

    print("\n  --- arm c (spawn new boxes from mode kernels) NOT RUN:")
    print("      a sliding max-over-10-kernels search on 220 sheets costs ~20 min per")
    print("      kernel-set at 5 scales; deferred, see report.")

    json.dump({"rows": rows}, open(r"Y:\workspace\EDA\work\_gnd_p3_rows.json", "w", encoding="utf-8"),
              ensure_ascii=False)


if __name__ == "__main__":
    main()
