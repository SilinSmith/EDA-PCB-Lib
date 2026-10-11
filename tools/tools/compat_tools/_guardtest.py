"""compat_guard 的自检样本 —— 故意踩 5 个「评测机 vs 队友机器」的差异点。

目的是证明：守卫能把这些**全部收集下来**，而不是崩在第一个上。
在容器里跑：
    python3 /audit/compat_audit.py guard --outdir /tmp/cg --report /out/guard.json \
        -- python3 /audit/_guardtest.py
"""
import os
import tempfile
import warnings

print("=== 1) np.cross 传 2 元向量（numpy>=2.0 抛 ValueError，第五次提交 0 分的真凶）===")
import numpy as np
try:
    print("   结果 =", np.cross([1.0, 0.0], [0.0, 1.0]))
except Exception as e:                                   # noqa: BLE001
    print(f"   抛错 {type(e).__name__}: {e}")

print("=== 2) np.array(..., copy=False) 语义收紧 ===")
try:
    print("   结果 =", np.array([1, 2, 3], copy=False))
except Exception as e:                                   # noqa: BLE001
    print(f"   抛错 {type(e).__name__}: {e}")

print("=== 3) numpy 2.x 删掉的名字 np.float_ ===")
try:
    print("   结果 =", np.float_)
except Exception as e:                                   # noqa: BLE001
    print(f"   抛错 {type(e).__name__}: {str(e)[:110]}")

print("=== 4) torch.load 不带 weights_only ===")
import torch
_p = os.path.join(tempfile.mkdtemp(), "t.pt")
torch.save({"a": 1}, _p)
try:
    print("   结果 =", torch.load(_p, map_location="cpu"))
except Exception as e:                                   # noqa: BLE001
    print(f"   抛错 {type(e).__name__}: {e}")

print("=== 5) DeprecationWarning ===")
warnings.warn("假装这里有个被弃用的调用", DeprecationWarning, stacklevel=1)

print("=== 自检样本跑完（能走到这一行，说明守卫没有把它拦死）===")
