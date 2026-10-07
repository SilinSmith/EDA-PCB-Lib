"""用官方评测口径（schematic_parser/metrics.py）评估 YOLO，并给出归因。

关键：官方 Component 匹配**按 key（位号）**，而 YOLO 不给位号。所以这里跑的是
**oracle-key 变体** —— 检测框从 YOLO 来（贴成规范图章），位号从 GT 借，用来把
「类型 + 坐标」这一段的损失单独量出来，并用 metrics 的归因计数器拆成
missing / extra / type_mismatch / bbox_mismatch / text_mismatch。

两个变体：
  V1 oracle-key + oracle-text : 只留 YOLO 的 type 和 bbox
  V2 oracle-key only          : 文本也留空，看纯文本惩罚有多大
对照组 gt : 全用 GT（应当 compF1 = 1.0）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser.stamp import snap_box  # noqa: E402
from schematic_parser.metrics import EvalConfig, evaluate_case  # noqa: E402
from schematic_parser.data import discover  # noqa: E402

DATA_ROOT = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)")
OFFICIAL_SPLITS = ("10GT", "10_GTcase", "200_train_cases")


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def to_img(b, H):
    """GT 的 bbox 是 y 向上（y_json = H - y_img）；转成左上角原点图像坐标。"""
    x1, y1, x2, y2 = (float(v) for v in b)
    return [x1, H - y2, x2, H - y1]


def to_json(b, H):
    x1, y1, x2, y2 = (float(v) for v in b)
    return [x1, H - y2, x2, H - y1]


def match_keys(gt_boxes, det_boxes):
    """同类一对一贪心：GT 框 -> 检测框索引。返回 {gt_idx: det_idx}"""
    cand = []
    for i, gb in enumerate(gt_boxes):
        gc = center(gb)
        for j, db in enumerate(det_boxes):
            dc = center(db)
            cand.append((((gc[0] - dc[0]) ** 2 + (gc[1] - dc[1]) ** 2) ** 0.5, i, j))
    cand.sort(key=lambda t: t[0])
    ug, ud, out = set(), set(), {}
    for d, i, j in cand:
        if i in ug or j in ud:
            continue
        ug.add(i); ud.add(j); out[i] = j
    return out


def nms(dets, iou_thr=0.5):
    """dets: list[(cls, box, conf)] -> 全局贪心 NMS（跨切片去重）"""
    out = []
    for cls in {d[0] for d in dets}:
        sub = sorted([d for d in dets if d[0] == cls], key=lambda d: -d[2])
        keep = []
        for d in sub:
            b = d[1]
            if all(((min(b[2], k[1][2]) - max(b[0], k[1][0])) <= 0 or
                    (min(b[3], k[1][3]) - max(b[1], k[1][1])) <= 0 or
                    (lambda iw, ih, u: (iw * ih) / u <= iou_thr if u > 0 else True)(
                        max(0, min(b[2], k[1][2]) - max(b[0], k[1][0])),
                        max(0, min(b[3], k[1][3]) - max(b[1], k[1][1])),
                        (b[2]-b[0])*(b[3]-b[1]) + (k[1][2]-k[1][0])*(k[1][3]-k[1][1])
                        - max(0, min(b[2], k[1][2]) - max(b[0], k[1][0]))
                        * max(0, min(b[3], k[1][3]) - max(b[1], k[1][1]))))
                    for k in keep):
                keep.append(d)
        out.extend(keep)
    return out


def predict_all(model, path, W, H, imgsz, conf, device, tile=0, overlap=256):
    """整图预测；tile>0 时切成 tile×tile（overlap 重叠）分别预测再全局 NMS。"""
    if not tile or (W <= tile and H <= tile):
        r = model.predict(str(path), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        return [(int(c), np.array(b, dtype=float), float(s)) for b, c, s in
                zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                    r.boxes.conf.cpu().numpy())]
    im = Image.open(str(path))
    step = max(1, tile - overlap)
    out = []
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            x1, y1 = min(x0 + tile, W), min(y0 + tile, H)
            if x1 - x0 < 64 or y1 - y0 < 64:
                continue
            crop = im.crop((x0, y0, x1, y1))
            r = model.predict(crop, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
            for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                               r.boxes.conf.cpu().numpy()):
                out.append((int(c), np.array([b[0]+x0, b[1]+y0, b[2]+x0, b[3]+y0], dtype=float),
                            float(s)))
    return nms(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default=r"Y:\workspace\EDA\eda-pcb\yolo_part\runs\detect\pcb43_v2_from_best\weights\best.pt")
    ap.add_argument("--root", default=r"Y:\workspace\EDA\trainset_v2")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--tile", type=int, default=0,
                    help="整图切成 tile×tile 分别预测再全局 NMS；0=关。大图小符号靠它救")
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--out", default=r"Y:\workspace\EDA\work\recon\official_eval.json")
    a = ap.parse_args()

    txt = (Path(a.root) / "data.yaml").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"names:\s*\[(.*?)\]", txt, re.S)
    names = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()] if m else \
        (Path(a.root) / "classes.txt").read_text().split()

    cases = {f"{c.split}/{c.case_id}": c for c in discover(str(DATA_ROOT), OFFICIAL_SPLITS)}
    val_lab = Path(a.root) / "labels" / "val"
    want = []
    for lp in sorted(val_lab.glob("*__full.txt")):
        sp, rest = lp.stem.split("-", 1)
        cid = rest.replace("__full", "")
        key = f"{sp}/{cid}"
        if key in cases:
            want.append(cases[key])
    print(f"val 全图 case: {len(want)}")

    from ultralytics import YOLO
    model = YOLO(a.weights)
    rows = []
    agg_detail = Counter()
    miss_by_type = Counter()
    gt_by_type_all = Counter()
    for n, c in enumerate(want, 1):
        gt = json.loads(Path(c.json_path).read_text(encoding="utf-8"))
        im = Image.open(str(c.png_path))
        W, H = im.size
        r = predict_all(model, c.png_path, W, H, a.imgsz, a.conf, a.device,
                        tile=a.tile, overlap=a.overlap)
        dets = defaultdict(list)  # type -> [(box_img, conf)]
        for ci, b, sc in r:
            nm = names[ci]
            dets[nm].append((np.array(snap_box(b, nm)), sc))

        gts = list(gt.get("components", {}).items())
        gt_by_type = defaultdict(list)
        for i, (k, comp) in enumerate(gts):
            gt_by_type[comp.get("type")].append((i, to_img(comp.get("bbox"), H)))

        comps_v1, comps_v2, comps_gt = {}, {}, {}
        for t, items in gt_by_type.items():
            dl = dets.get(t, [])
            assign = match_keys([b for _, b in items], [b for b, _ in dl])
            gt_by_type_all[t] += len(items)
            for local_i, (gi, gb) in enumerate(items):
                key, comp = gts[gi]
                j = assign.get(local_i)
                if j is not None:
                    box = dl[j][0]
                    comps_v1[key] = {"Name": comp.get("Name"), "type": t,
                                     "value": comp.get("value"), "bbox": to_json(box, H)}
                    comps_v2[key] = {"type": t, "bbox": to_json(box, H)}
                else:
                    miss_by_type[t] += 1
                comps_gt[key] = {"Name": comp.get("Name"), "type": t,
                                 "value": comp.get("value"), "bbox": comp.get("bbox")}

        cfg = EvalConfig()
        for tag, comps in (("gt", comps_gt), ("V1", comps_v1), ("V2", comps_v2)):
            pred = {"components": comps, "pins": gt.get("pins", {}), "nets": gt.get("nets", {})}
            s = evaluate_case(gt, pred, case_id=f"{c.split}-{c.case_id}", cfg=cfg)
            detail = dict(s.diagnostic)
            rows.append({"case": f"{c.split}-{c.case_id}", "arm": tag,
                         "score": s.score, "compF1": s.component.f1,
                         "pinF1": s.pin.f1, "netHF1": s.net_hypergraph.f1,
                         "detail": dict(detail)})
            if tag == "V1":
                agg_detail.update(detail)
        if n % 20 == 0:
            print(f"  {n}/{len(want)}")

    out = {"rows": rows, "detail_V1": dict(agg_detail), "conf": a.conf, "imgsz": a.imgsz,
           "miss_by_type": dict(miss_by_type), "gt_by_type": dict(gt_by_type_all)}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nV1 漏检按类分解（缺 = 该类没匹配上任何预测）:")
    for t in sorted(gt_by_type_all, key=lambda x: -miss_by_type[x]):
        tot, mm = gt_by_type_all[t], miss_by_type[t]
        if mm:
            print(f"  {t:<18} 漏 {mm:>4} / GT {tot:>4}  = {mm/max(1,tot)*100:5.1f}%")

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0
    print(f"\n{'arm':<6}{'score':>9}{'compF1':>9}{'pinF1':>9}{'netHF1':>9}")
    for tag in ("gt", "V1", "V2"):
        rs = [x for x in rows if x["arm"] == tag]
        print(f"{tag:<6}{mean([x['score'] for x in rs]):>9.4f}"
              f"{mean([x['compF1'] for x in rs]):>9.4f}"
              f"{mean([x['pinF1'] for x in rs]):>9.4f}"
              f"{mean([x['netHF1'] for x in rs]):>9.4f}")
    print(f"\nV1 归因计数（官方 metrics 的分项失败计数器）:")
    for k, v in sorted(agg_detail.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<34}{v}")
    print(f"\n写到 {a.out}")


if __name__ == "__main__":
    main()
