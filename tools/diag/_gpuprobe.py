# -*- coding: utf-8 -*-
"""判断「GPU 是不是被 VLM 占住导致 CUDA 变慢」：量一次小矩阵乘 + 一次 YOLO 推理的墙钟。"""
import sys
import time

import torch

sys.stdout.reconfigure(encoding="utf-8")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
free, total = torch.cuda.mem_get_info()
print("显存 free=%.2f GiB / total=%.2f GiB" % (free / 2**30, total / 2**30))
t0 = time.time()
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
t1 = time.time()
c = a @ b
torch.cuda.synchronize()
t2 = time.time()
print("4096^3 矩阵乘：分配 %.3fs 计算 %.3fs" % (t1 - t0, t2 - t1))
del a, b, c
torch.cuda.empty_cache()

t3 = time.time()
from ultralytics import YOLO
m = YOLO(r"Y:\workspace\EDA\eda-pcb\weights\pcb43_v2_from_best.pt")
print("加载模型 %.2fs" % (time.time() - t3))
for img in (r"Y:\workspace\EDA\work\_submit_in220\0002.png",
            r"Y:\workspace\EDA\work\_submit_in220\0003.png"):
    t4 = time.time()
    r = m.predict(img, imgsz=1600, device=0, verbose=False)
    print("%s 推理 %.2fs  box=%d" % (img.split("\\")[-1], time.time() - t4,
                                     len(r[0].boxes) if r else -1))
