#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提交前的运行环境自检：第三方依赖到底装上了没有。

为什么需要它：第一次提交时阶段一 10 张全部
    ModuleNotFoundError: No module named 'ultralytics'
而**日志里看不到任何原因**（`stage1/pipeline.py:426` 只打印 FAIL 和耗时），
唯一的线索是决策日志 `decision_log.json` 里的 `stage1.error`。
这个脚本把「装了没有」直接打到 stdout，让下一次提交一眼就能看出来。

导入顺序与运行期一致：
    sys.path = /submission/eda-pcb-main : /submission : 原有路径
    （`submission/agent_system` 把 HERE 与它的上一级塞进 sys.path，
      而 `stage2/`、`stage1/` 里还有 `import stage2_eval` / `import labels` 这类
      依赖 cwd 的裸导入，所以两个目录都要在前面。）
"""
import os
import sys

REPO = "/submission/eda-pcb-main"
for _p in ("/submission", REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# (显示名, 导入名) —— 覆盖运行路径上所有非标准库依赖。
# 分类依据：stage1=检测/几何，stage2=文字，stage3/fusion=仲裁，textrec/pipeline=底层。
MODULES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("cv2 (opencv-python)", "cv2"),
    ("PIL (pillow)", "PIL"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("ultralytics", "ultralytics"),
    ("psutil", "psutil"),
    ("yaml (PyYAML)", "yaml"),
    ("requests", "requests"),
    # ★ httpx 不进 MODULES：交付路径一行都不用它（只用标准库 urllib.request），
    #   基础镜像也不保证有。列在这里会被判成「第三方依赖缺失」而误导排查，
    #   所以挪到下面的 OPTIONAL 里只做展示。
    # 仓库内的模块（缺了同样会炸，但它们是「打包漏了」而不是「没装」）
    ("本地: schematic_parser", "schematic_parser"),
    ("本地: stage1.detect", "stage1.detect"),
    ("本地: stage2.text", "stage2.text"),
    ("本地: textrec.infer", "textrec.infer"),
    ("本地: vlm.client", "vlm.client"),
    ("本地: fusion.run", "fusion.run"),
    ("本地: gpu_mem", "gpu_mem"),
    ("本地: assets", "assets"),
]

print("=" * 66)
print("运行时依赖自检  verify_imports.py")
print("=" * 66)
print("python      =", sys.version.split()[0])
print("executable  =", sys.executable)
print("cwd         =", os.getcwd())
print("sys.path[0:3] =", sys.path[:3])
print()

ok_list, bad_list = [], []
for label, name in MODULES:
    try:
        mod = __import__(name, fromlist=["__name__"])
        ver = getattr(mod, "__version__", "")
        where = getattr(mod, "__file__", "") or ""
        if not ver and where:
            ver = os.path.basename(where)
        print("  OK    %-24s %s" % (label, ver))
        ok_list.append(label)
    except BaseException as e:                      # noqa: BLE001
        print("  FAIL  %-24s %s: %s" % (label, type(e).__name__, e))
        bad_list.append("%s (%s: %s)" % (label, type(e).__name__, e))

print()
print("-" * 66)
# 可选包：缺失**不算失败**，只提示（交付路径不 import）。
OPTIONAL = [("httpx (可选, 交付路径不用)", "httpx")]
for label, name in OPTIONAL:
    try:
        mod = __import__(name, fromlist=["__name__"])
        print("  可选  %-24s %s" % (label, getattr(mod, "__version__", "")))
    except BaseException as e:                      # noqa: BLE001
        print("  可选  %-24s 缺失（无影响）: %s" % (label, type(e).__name__))
print("通过 %d / %d" % (len(ok_list), len(MODULES)))
if bad_list:
    print("缺失：")
    for b in bad_list:
        print("  -", b)
    # 只把**第三方包**缺失算硬失败；本地模块缺失在 `assets` 那条下面已经报得很清楚
    hard = [b for b in bad_list
            if not b.startswith("本地:")]
    if hard:
        print()
        print("!! 第三方依赖缺失 %d 个 —— 阶段一会在每张图上抛 ModuleNotFoundError，"
              "整批 0 分。" % len(hard))
        print("!! 修法：在 Dockerfile 里补 pip3 install --break-system-packages "
              "<包名>，重新 build 后再提交。")
    print("=" * 66)
    sys.exit(1)
print("全部通过。")
print("=" * 66)

# torch 与 CUDA 单独看一眼 —— 阶段二要在 cuda:0 上加载 CRNN
try:
    import torch
    print("torch       =", torch.__version__)
    print("cuda 可用   =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda 版本   =", torch.version.cuda)
        print("显卡        =", torch.cuda.get_device_name(0))
except BaseException as e:                          # noqa: BLE001
    print("torch 检查失败：%s: %s" % (type(e).__name__, e))
