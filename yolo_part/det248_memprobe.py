#!/usr/bin/env python3
"""量 batch 上限：临时探针，跑 fraction=0.05 一个 epoch，报显存峰值。用完可删。"""
import os
import sys
import tempfile

import torch
from ultralytics import YOLO

B = int(sys.argv[1]) if len(sys.argv) > 1 else 12
W = r"Y:\workspace\EDA\eda-pcb\weights\pcb43_v2_from_best.pt"
DATA = r"Y:\workspace\EDA\eda-pcb\yolo_part\training_data2\data.yaml"

torch.cuda.reset_peak_memory_stats()
free0, total = torch.cuda.mem_get_info()
print(f"batch={B} 开跑前 空闲 {free0/1024**3:.2f}GiB / 共 {total/1024**3:.2f}GiB", flush=True)
try:
    YOLO(W).train(
        data=DATA, epochs=1, imgsz=1024, batch=B, device="0", fraction=0.05,
        project=os.path.join(tempfile.gettempdir(), "det248probe"), name=f"b{B}",
        exist_ok=True, workers=2, amp=True, plots=False, val=True, verbose=False,
        optimizer="auto", mosaic=0.7, close_mosaic=0, fliplr=0.0, flipud=0.0, degrees=0.0,
    )
    print(f"OK batch={B}", flush=True)
except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
    print(f"FAIL batch={B}: {str(exc)[:200]}", flush=True)
print(f"峰值 allocated {torch.cuda.max_memory_allocated()/1024**3:.2f}GiB "
      f"reserved {torch.cuda.max_memory_reserved()/1024**3:.2f}GiB", flush=True)
free1, _ = torch.cuda.mem_get_info()
print(f"结束后 空闲 {free1/1024**3:.2f}GiB", flush=True)
