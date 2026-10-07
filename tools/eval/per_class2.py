# -*- coding: utf-8 -*-
"""按类别打印检测器在某个 split 上的 P/R/mAP50/mAP50-95/实例数，并按「漏检数」排序。

用法:
    python per_class2.py <weights.pt> [--data X.yaml] [--split val] [--imgsz 1024] [--out x.json]

注意 1: ultralytics 的 r.box.p / .r / .ap50 / .maps 都是「按 class id 索引、长度 nc」的数组，
        ap_class_index 只列出出现过的类。两者不能直接 zip —— 必须用 class id 去取。
注意 2: Windows 上 multiprocessing 是 spawn，脚本必须有 __main__ 守卫，否则
        worker 进程 import 主模块时会重跑整个脚本（RuntimeError: bootstrapping phase）。
        这里直接 workers=0 双保险。
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_DATA = r"Y:\workspace\EDA\trainset_v2\data.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    res = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
                    workers=args.workers, device=args.device, conf=args.conf, iou=0.7,
                    plots=False, verbose=False)

    b = res.box
    names = res.names if isinstance(res.names, dict) else {i: n for i, n in enumerate(res.names)}
    idx = [int(c) for c in getattr(b, "ap_class_index", range(len(b.p)))]
    n_ap = len(idx)
    NT = getattr(b, "nt_per_class", None)

    def pick(arr, i, c):
        """ultralytics 的版本之间不一致：p/r/ap50 有时按「出现顺序」、有时按 class id。"""
        if arr is None:
            return None
        if len(arr) == n_ap:
            return float(arr[i])
        if len(arr) > c:
            return float(arr[c])
        return None

    rows = []
    for i, c in enumerate(idx):
        n = int(NT[c]) if NT is not None and c < len(NT) else (int(NT[i]) if NT is not None and i < len(NT) else None)
        p_, r_ = pick(b.p, i, c), pick(b.r, i, c)
        a50, a = pick(b.ap50, i, c), pick(getattr(b, "maps", None), i, c)
        rows.append({
            "id": c, "name": names.get(c, str(c)), "instances": n,
            "P": p_, "R": r_, "mAP50": a50, "mAP50_95": a,
            "missed": (n * (1.0 - r_)) if (n is not None and r_ is not None) else None,
        })

    print("weights :", args.weights)
    print("data    :", args.data, " split:", args.split, " imgsz:", args.imgsz)
    print("overall : P %.4f  R %.4f  mAP50 %.4f  mAP50-95 %.4f  (instances %d)"
          % (float(b.mp), float(b.mr), float(b.map50), float(b.map),
             sum(r["instances"] or 0 for r in rows)))
    print()

    hdr = "%-20s %8s %6s %6s %7s %9s %9s" % ("class", "inst", "P", "R", "mAP50", "mAP50-95", "missed")
    for title, key in (("按「漏检数 instances*(1-R)」排序 —— 这就是补数据的优先级", lambda z: -(z["missed"] or 0)),
                       ("按实例数排序", lambda z: -(z["instances"] or 0))):
        print("=== %s ===" % title)
        print(hdr)
        for r in sorted(rows, key=key):
            print("%-20s %8s %6s %6s %7s %9s %9s"
                  % (r["name"], r["instances"],
                     "  -  " if r["P"] is None else "%.3f" % r["P"],
                     "  -  " if r["R"] is None else "%.3f" % r["R"],
                     "   -  " if r["mAP50"] is None else "%.3f" % r["mAP50"],
                     "    -   " if r["mAP50_95"] is None else "%.3f" % r["mAP50_95"],
                     "   -   " if r["missed"] is None else "%.1f" % r["missed"]))
        print()

    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
