#!/usr/bin/env python3
"""在 `pcb43_v2_from_best.pt` 上微调，加入「两件叠画」样本（见 `aug_overlap.py`）。

与 `train_yolo.py` 的差别（都是有意的）：
- **imgsz=1280**：正式管线推理用 1600，`train_yolo.py` 的 640 会让尺度分布对不上
  （小元件在大图上更小）。1280 是 16 GB 显存下 batch 8 能稳的上限。
- 从现役权重继续训，而不是从 yolo11s 从头 —— 目标是「多认一种叠画」，不是重学。
- lr0=3e-4 微调量级；patience=20 早停。

用法: python yolo_part\\train_ovl.py [--epochs=45] [--imgsz=1280] [--batch=8] [--device=0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr0", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--name", default="pcb43_ovl")
    ap.add_argument("--model", default=str(ROOT / "weights" / "pcb43_v2_from_best.pt"))
    ap.add_argument("--data", default=str(ROOT / "yolo_part" / "data_ovl.yaml"))
    args = ap.parse_args(argv[1:])

    from ultralytics import YOLO
    m = YOLO(args.model)
    m.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device=args.device, workers=args.workers, lr0=args.lr0, patience=args.patience,
            project=str(ROOT / "yolo_part" / "runs_ovl"), name=args.name, exist_ok=True,
            plots=False, verbose=True)


if __name__ == "__main__":
    main(sys.argv)
