#!/usr/bin/env python3
"""把「两件叠画」做进 YOLO 训练集：真实符号 copy-paste（只叠墨、不遮墨）。

背景
----
主尺 case11 有一对叠画的 LED（GT `LED1 [1104,374,1145,425]` / `LED2 [1104,354,1145,406]`，
json 坐标），实测 `work\\_yolooracle.py`：imgsz=1600 下即使把 NMS 阈值放到 0.99，
该处也**只有一个 led 框**（[1116,370,1141,387]），剩下的全是同一实例的重复框
⇒ 是**模型识别不出两个叠在一起的符号**，不是我们的 NMS/后处理并框。

本脚本从池子里 200 张真实页里剪出真实符号，把它的**暗笔画**叠到另一颗符号上
（`dst[mask] = src[mask]`，只覆盖笔画像素，两套轮廓都还看得见），标签里就多出一个
与原来那颗 IoU 0.2~0.6 的框。风格 100% 真实（同一张图、同一套线宽与字体）。

用法
----
    python yolo_part\\aug_overlap.py --src training_data --out training_data_ovl --per 3
再
    yolo_part\\data_ovl.yaml 指向 training_data_ovl 即可训练。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "yolo_part" / "Dataset(waitforpreprocess)" / "200_train_cases"

# 拿来当「被叠上去的那颗」的类：小而常见、叠起来最考验实例切分
DONOR_TYPES = {"led", "r", "c", "d", "gnd", "net_input", "net_output", "pin", "v",
               "testpoint", "switch", "buzzer", "crystal", "opt", "mosfet", "bjt", "dc"}
MIN_SIDE = 9.0
MAX_SIDE = 130.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "yolo_part" / "training_data"))
    ap.add_argument("--out", default=str(ROOT / "yolo_part" / "training_data_ovl"))
    ap.add_argument("--per", type=int, default=3, help="每张页最多生成几次叠画")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_gt(stem: str):
    """stem 形如 `0001-KiCad` -> (png, components)。bbox 转图像坐标（y 向下）。"""
    case = stem.split("-")[0]
    d = POOL / case
    js = list(d.glob("*_target_named.json"))
    png = list(d.glob("*.png"))
    if not js or not png:
        return None, None
    gt = json.load(open(js[0], encoding="utf-8"))
    img = cv2.imread(str(png[0]))
    if img is None:
        return None, None
    h = img.shape[0]
    comps = []
    for k, v in (gt.get("components") or {}).items():
        bb = v.get("bbox")
        if not bb:
            continue
        x0, y0, x1, y1 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        comps.append({"key": k, "type": (v.get("type") or "").lower(),
                      "box": [x0, h - y1, x1, h - y0]})
    return str(png[0]), comps


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area = lambda z: max(0.0, z[2] - z[0]) * max(0.0, z[3] - z[1])
    u = area(a) + area(b) - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = parse_args()
    rng = random.Random(ap.seed)
    src, out = Path(ap.src), Path(ap.out)
    cls = [x.strip() for x in (src / "classes.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    idx = {c: i for i, c in enumerate(cls)}
    for sub in ("train", "val", "ovlplus"):
        (out / "images" / sub).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "classes.txt", out / "classes.txt")

    made = {"pages": 0, "pastes": 0, "skipped": 0}
    for split in ("train", "val"):
        for img_path in sorted((src / "images" / split).glob("*.png")):
            stem = img_path.stem
            lab = (src / "labels" / split / (stem + ".txt"))
            base = lab.read_text(encoding="utf-8").splitlines() if lab.exists() else []
            shutil.copy(img_path, out / "images" / split / img_path.name)
            (out / "labels" / split / (stem + ".txt")).write_text(
                "\n".join(base) + ("\n" if base else ""), encoding="utf-8")
            # 验证集只做「能力」用的那份（ovlplus），训练集既做训练份也做能力份
            png, comps = load_gt(stem)
            if not png or not comps:
                made["skipped"] += 1
                continue
            img = cv2.imread(png)
            h, w = img.shape[:2]
            cands = [c for c in comps if c["type"] in DONOR_TYPES
                     and MIN_SIDE <= (c["box"][2] - c["box"][0]) <= MAX_SIDE
                     and MIN_SIDE <= (c["box"][3] - c["box"][1]) <= MAX_SIDE]
            if len(cands) < 2:
                made["skipped"] += 1
                continue
            work = img.copy()
            extra = []
            for _ in range(ap.per):
                for _try in range(12):
                    donor = rng.choice(cands)
                    victim = rng.choice(cands)
                    if victim is donor:
                        continue
                    db, vb = donor["box"], victim["box"]
                    dw, dh = db[2] - db[0], db[3] - db[1]
                    # 让新框压住 victim 的 0.2~0.6（IoU），偏移是框尺寸的 20~45%
                    frac = rng.uniform(0.20, 0.45)
                    sx = rng.choice((-1, 1)) * frac * dw
                    sy = rng.choice((-1, 1)) * frac * dh
                    nb = [vb[0] + sx, vb[1] + sy, vb[0] + sx + dw, vb[1] + sy + dh]
                    if nb[0] < 0 or nb[1] < 0 or nb[2] > w or nb[3] > h:
                        continue
                    u = iou(nb, vb)
                    if not (0.20 <= u <= 0.60):
                        continue
                    if max(iou(nb, c["box"]) for c in comps if c is not victim) > 0.55:
                        continue
                    x0, y0 = int(round(db[0])), int(round(db[1]))
                    x1, y1 = int(round(db[2])), int(round(db[3]))
                    src_crop = img[y0:y1, x0:x1]
                    if src_crop.size == 0:
                        continue
                    gx0, gy0 = int(round(nb[0])), int(round(nb[1]))
                    gx1, gy1 = gx0 + (x1 - x0), gy0 + (y1 - y0)
                    if gx1 > w or gy1 > h:
                        continue
                    dst = work[gy0:gy1, gx0:gx1]
                    if dst.shape[:2] != src_crop.shape[:2]:
                        continue
                    g = cv2.cvtColor(src_crop, cv2.COLOR_BGR2GRAY)
                    mask = g < 128
                    if mask.sum() < 12:
                        continue
                    dst[mask] = src_crop[mask]
                    extra.append((donor["type"], nb))
                    break
            if not extra:
                made["skipped"] += 1
                continue
            made["pages"] += 1
            made["pastes"] += len(extra)
            lines = list(base)
            for typ, nb in extra:
                ci = idx.get(typ, idx.get("other", 0))
                cx, cy = (nb[0] + nb[2]) / 2.0 / w, (nb[1] + nb[3]) / 2.0 / h
                bw, bh = (nb[2] - nb[0]) / w, (nb[3] - nb[1]) / h
                lines.append("%d %.6f %.6f %.6f %.6f" % (ci, cx, cy, bw, bh))
            sub = "train" if split == "train" else "ovlplus"
            name = stem + "_ovl"
            cv2.imwrite(str(out / "images" / sub / (name + ".png")), work)
            (out / "labels" / sub / (name + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[aug] 生成页 %d / 跳过 %d / 叠画次数 %d" % (made["pages"], made["skipped"], made["pastes"]))
    print("[aug] 输出 %s" % out)


if __name__ == "__main__":
    main()
