"""纯 CV 部件检测器集合。

与 `pipeline/` 的分工：
  - `pipeline/pskel.py` —— 导线骨架与成网
  - `cvparts/*`          —— 单个符号类的纯 CV 检测器（形状文法 + 上下文门控）

每个模块都要求：
  1. `detect_boxes(gray) -> list[[x1,y1,x2,y2]]`（图像坐标，左上原点）
  2. `python <module>.py` 能在官方 220 个 case 上跑出 P/R/F1 与耗时自检

已实现：
  - `testpoint` —— 同心双圆环靶心，43/43 能找到，97.7% 落在官方 ±20px 内
"""
