"""「GT 有值 / 我们 None」的那 477 条，正确的值串到底**读没读出来**？

用法：
  python work\\tools\\diag\\_valread.py --pred work\\_submit_out220r27\\runs\\stage2 [--limit 60] [--pad 40]

对每个「GT `value` 非 None、我们的 `value` 是 None」的 r/c/l/... 配对，
拿 stage1 的全部文字框跑一次 `TextEngine.read(..., ['value'])`，然后看：
  A 值串在本 case 的**任意** value-role 读数里出现过  → 是「归属/绑定」问题
  B 值串只在本元件 bbox `--pad` 邻域内的读数里出现    → 是「距离闸门」问题
  C 压根没读出来                                      → 是「识别」问题
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "eda-pcb"))

import cv2                                                 # noqa: E402
from schematic_parser import data as D                     # noqa: E402
from schematic_parser import metrics as M                  # noqa: E402
from stage2.text import TextEngine, read_stage1_texts      # noqa: E402

VALUE_TYPES = ("r", "c", "l", "fuse", "crystal_2pin", "crystal_3pin",
               "crystal_4pin", "crystal", "d")


def dist_to_rect(px, py, b):
    dx = max(b[0] - px, 0.0, px - b[2])
    dy = max(b[1] - py, 0.0, py - b[3])
    return (dx * dx + dy * dy) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--stage1", default="", help="stage1 目录；默认 <pred>/../stage1")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--pad", type=float, default=40.0)
    args = ap.parse_args()

    s1 = args.stage1 or os.path.join(os.path.dirname(os.path.abspath(args.pred)), "stage1")
    idx = {c.case_id: c for c in D.discover(D.DEFAULT_DATA_ROOT)}
    cases = sorted([f[:-5] for f in os.listdir(args.pred)
                    if f.endswith(".json") and not f.startswith("_")])
    cases = [c for c in cases if c in idx][:args.limit]
    print("case 数：%d  stage1=%s" % (len(cases), s1))

    eng = TextEngine()
    cfg = M.EvalConfig()
    st = {"missing": 0, "A_anywhere": 0, "B_near": 0, "C_unread": 0}
    samples = []

    for cid in cases:
        c = idx[cid]
        gt = D.load_json(c.json_path)
        pr = D.load_json(os.path.join(args.pred, cid + ".json"))
        _f, km = M.assign_components(gt, pr, cfg)
        gc, pc = gt["components"], pr["components"]

        targets = []
        for pk, gk in km.items():
            g, p = gc.get(gk), pc.get(pk)
            if g is None or p is None:
                continue
            if M.canon_type(g.get("type")) not in VALUE_TYPES:
                continue
            gv, pv = g.get("value"), p.get("value")
            if gv is not None and pv is None:
                st["missing"] += 1
                targets.append((pk, gv, p.get("bbox")))
        if not targets:
            continue

        sp = os.path.join(s1, cid + ".stage1.json")
        if not os.path.exists(sp):
            continue
        doc = D.load_json(sp)
        boxes = read_stage1_texts(doc)
        if not boxes:
            continue
        img = cv2.imread(c.png_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        raw = eng.read(img, boxes, ["value"])
        reads = [{"box": r.get("bbox"), "value": (r.get("reads") or {}).get("value", [None])[0]}
                 for r in raw]
        allv = {r["value"] for r in reads if r["value"]}

        for pk, gv, pb in targets:
            if gv in allv:
                st["A_anywhere"] += 1
                if len(samples) < 12:
                    near = []
                    if pb:
                        # pred bbox 是 y-UP，转成图像系
                        pb_img = (pb[0], H - pb[3], pb[2], H - pb[1])
                        for r in reads:
                            if r.get("value") != gv:
                                continue
                            bb = r.get("box")
                            if not bb:
                                continue
                            d = dist_to_rect((pb_img[0] + pb_img[2]) / 2.0,
                                             (pb_img[1] + pb_img[3]) / 2.0, bb)
                            if d <= 200:
                                near.append((round(d, 1), bb))
                    samples.append((cid, pk, gv, "A", sorted(near)[:3]))
            elif pb:
                pb_img = (pb[0], H - pb[3], pb[2], H - pb[1])
                hit = None
                for r in reads:
                    bb = r.get("box")
                    if not bb or not r.get("value"):
                        continue
                    d = dist_to_rect((pb_img[0] + pb_img[2]) / 2.0,
                                     (pb_img[1] + pb_img[3]) / 2.0, bb)
                    if d <= args.pad:
                        hit = (round(d, 1), r["value"])
                        break
                if hit is not None:
                    st["B_near"] += 1
                    if len(samples) < 24:
                        samples.append((cid, pk, gv, "B_near", hit))
                else:
                    st["C_unread"] += 1
                    if len(samples) < 36:
                        samples.append((cid, pk, gv, "C_unread", None))

    print(dict(st))
    for s in samples:
        print(" ", s)


if __name__ == "__main__":
    main()
