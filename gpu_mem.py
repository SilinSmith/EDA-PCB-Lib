"""显存配额 —— 本地 LLM（LM Studio）和管线共用同一张卡，别把显存吃满。

背景（本机 RTX 4080 16 GB）：LM Studio 里跑视觉模型时**常驻占掉 ~10 GB**，
留给 YOLO（`imgsz=1600`）+ CRNN 的只有 5~6 GB。管线这边一旦按 PyTorch 默认
「能用多少用多少」去要显存，要么直接把 LLM 挤到 CPU（慢十倍），要么自己 OOM。

用法：在任何 **torch / ultralytics 被导入并初始化 CUDA 之前**调用一次::

    from gpu_mem import cap_vram
    info = cap_vram()          # 幂等，重复调用只生效一次

它做三件事（都不改精度、不改结果）：

1. 设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` —— 减少长跑时的碎片
   （原来一张 1600px 的图跑完留下碎片，下一张就可能 OOM）。**注意必须在
   `torch` 首次初始化 CUDA 之前设，进程起来后再设是无效的。**
2. `torch.cuda.set_per_process_memory_fraction(frac)` —— 给**本进程**的 caching
   allocator 设一个硬上限。超了会抛 `torch.cuda.OutOfMemoryError` 而**不会**去抢
   LLM 的显存，这是我们想要的失败方式。
3. 记一份 `describe()` 供日志打印。

配置（环境变量或 `submission/run.py` 的命令行，命令行优先）：

| 变量 | 含义 | 默认 |
|---|---|---|
| `EDA_GPU_FRAC` | 本进程可用显存 = 总量 × 该比例 | `0.30` |
| `EDA_GPU_HEADROOM_MB` | 改为「给 LLM 留出 N MiB」，二选一 | 未设 |
| `EDA_GPU_CAP` | 设 `0` / `off` 则完全不管显存（老行为） | 开 |

⚠️ `workers > 1` 时每个 stage1 子进程各有一份配额（`set_per_process_memory_fraction`
是**按进程**的），所以并发数要和 `EDA_GPU_FRAC` 一起看：`workers=4` × `0.30`
仍然会超。`submission/run.py` 里已经把两者的乘积写进日志。
"""
from __future__ import annotations

import os
from typing import Optional

_CAPPED = False
_INFO = ""

DEFAULT_FRAC = 0.30


def _truthy(v: Optional[str], default: bool = True) -> bool:
    if v is None:
        return default
    return v.strip().lower() not in ("", "0", "false", "off", "no", "none")


def _resolve_frac() -> tuple[float, str]:
    """返回 (fraction, 说明)。`EDA_GPU_HEADROOM_MB` 优先于 `EDA_GPU_FRAC`。"""
    head = os.environ.get("EDA_GPU_HEADROOM_MB", "").strip()
    if head:
        try:
            hmb = float(head)
        except ValueError:
            hmb = 0.0
        if hmb > 0:
            try:
                import torch
                total = float(torch.cuda.get_device_properties(0).total_memory)
                frac = max(0.05, (total - hmb * 1024.0 * 1024.0) / total)
                return frac, "headroom=%.0fMiB(总%.1fGiB)->frac=%.3f" % (
                    hmb, total / 2 ** 30, frac)
            except Exception as e:                                  # noqa: BLE001
                return DEFAULT_FRAC, "headroom 读显卡失败(%s)，回落 %.2f" % (
                    type(e).__name__, DEFAULT_FRAC)
    try:
        frac = float(os.environ.get("EDA_GPU_FRAC", str(DEFAULT_FRAC)))
    except ValueError:
        frac = DEFAULT_FRAC
    return min(max(frac, 0.05), 1.0), "EDA_GPU_FRAC=%.2f" % frac


def cap_vram(frac: Optional[float] = None) -> str:
    """给本进程设显存上限。幂等。返回一行可直接写日志的说明。

    `frac` 显式给定时覆盖环境变量。
    """
    global _CAPPED, _INFO
    if _CAPPED:
        return _INFO

    if not _truthy(os.environ.get("EDA_GPU_CAP"), True):
        _CAPPED, _INFO = True, "EDA_GPU_CAP=off（不限显存）"
        return _INFO

    # ① 分配器配置：必须在 CUDA 初始化之前设。
    # ⚠️ `expandable_segments` 在 Windows 上不被支持（会打一条 UserWarning 然后忽略），
    # 所以只在非 Windows 上设，免得每次跑都刷一行噪音。
    if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF") and os.name != "nt":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ② 每进程硬上限。
    why = "命令行/调用方指定"
    if frac is None:
        frac, why = _resolve_frac()
    frac = min(max(float(frac), 0.05), 1.0)
    try:
        import torch
        if not torch.cuda.is_available():
            _CAPPED, _INFO = True, "无 CUDA，跳过显存配额（%s）" % why
            return _INFO
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        _INFO = ("显存配额 %.0f%%（%s）≈ %.1f GiB / 卡共 %.1f GiB；"
                 "PID=%d" % (frac * 100, why, frac * total, total, os.getpid()))
    except Exception as e:                                          # noqa: BLE001
        _INFO = "显存配额设置失败 %s: %s（继续跑，不阻断）" % (type(e).__name__, e)
    _CAPPED = True
    return _INFO


def describe() -> str:
    """不设配额，只回报当前状态（日志用）。"""
    if _INFO:
        return _INFO
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return "显存 空闲%.1f/共%.1f GiB（尚未设配额）" % (
                free / 2 ** 30, total / 2 ** 30)
    except Exception:                                               # noqa: BLE001
        pass
    return "无 CUDA 或未初始化"


if __name__ == "__main__":                                          # pragma: no cover
    print(cap_vram())
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            a = torch.zeros(int(0.9 * total / 4), dtype=torch.float32, device="cuda")
            print("申请了 90%% 显存没被拦住 —— EDA_GPU_CAP 是关的？")
            del a
            torch.cuda.empty_cache()
    except Exception as e:                                          # noqa: BLE001
        print("按预期被拦住 ->", type(e).__name__)
