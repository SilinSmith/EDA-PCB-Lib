# 版本适配审计报告

仓库：`C:\Users\22647\Documents\deepseek-harness\default-workspace\pcb\eda-pcb-main`

判定口径：**评测机属性面清单**（C:\Users\22647\Documents\deepseek-harness\default-workspace\团队仓库_待提交\tools\版本适配\evalenv_manifest.json，导出于 2026-10-11T07:28:55）

## 目标环境快照（就是跑这次审计的解释器）

| 项 | 值 |
|---|---|
| `python` | `3.12.3` |
| `torch` | `2.5.1+cu121` |
| `torchvision` | `0.20.1+cu121` |
| `numpy` | `2.5.3` |
| `scipy` | `1.18.1` |
| `cv2` | `4.10.0` |
| `PIL` | `12.3.0` |
| `ultralytics` | `8.4.175` |
| `yaml` | `6.0.2` |
| `psutil` | `7.2.2` |
| `matplotlib` | `3.11.2` |
| `torch.cuda.is_available` | `True` |
| `torch.version.cuda` | `12.1` |
| `gpu` | `NVIDIA GeForce RTX 4070 Laptop GPU` |
| `cv2 有 ximgproc(contrib)` | `False` |
| `opencv dist-info` | `opencv_python_headless-4.10.0.84.dist-info` |

## 结论：扫到 183 条第三方属性链，其中 **0 条在目标环境不存在**，**2 条已接受**（不算失败）

**⇒ 未发现缺失属性。** 属性级兼容性通过（注意：这不等于签名级、语义级也通过）。

## 已接受（2 条，来自 --allow）

这些属性链在目标环境不存在，但已确认无害（例如调用点有 `hasattr` 兜底）。

| # | 属性链 | 理由 | 调用点 |
|---|---|---|---|
| 1 | `cv2.ximgproc` | tools/diag/_wiretj.py:72 调用点已用 hasattr(cv2,'ximgproc') 保护；且 tools/ 整个目录不进提交包 | `tools/diag/_wiretj.py:72` |
| 2 | `cv2.ximgproc.thinning` | 同上 | `tools/diag/_wiretj.py:72` |

## `np.cross` 调用点（0 处，需人工确认参数维度）

numpy ≥ 2.0 对 2 元向量 `np.cross` 会抛 `ValueError`（第五次提交阶段一 0/10 全 FAIL 的根因）。

- （无）
