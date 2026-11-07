# 归档说明（archive）

本仓库是 EDA 精英挑战赛 · 赛题六「AI 驱动的 PCB 原理图自动解析」工作仓库
`SilinSmith/eda-pcb` 的**归档副本**，用于长期留存源码、权重与赛题材料。

- 归档分支：`main` = 源码、随仓库交付的权重，以及在开发仓库 `main` 之上追加的
  `_workspace/`（赛题文档）与本说明。源码部分与 `SilinSmith/eda-pcb` 的 `main` 逐提交一致，
  归档提交 `5847e8d` 是 `main` 之上唯一的额外提交。

## 分支总览

除 `main` 外，开发仓库的 8 个分支也已同步：
`codex/combined-pin-recognition`、`codex/improve-monochrome-pin-geometry`、
`codex/improve-pin-network-parsing`、`tools/compat-audit`、`codex-merge-test`、
`pr2-test`、`pr3-on-main`、`pr3-port`。

（`improve-pin-network-parsing` 与 `codex/improve-pin-network-parsing` 提交相同，
`codex/pcb-current-20261105` 只存在于开发仓库的 origin，两者未纳入归档；
赛题材料已并入 `main`，不再单设材料分支。）

## 实际随包交付的权重（`weights/`，三个都在 `main` 中）

| 文件 | 大小 | 用途 |
| --- | --- | --- |
| `weights/pcb43_v2_from_best.pt` | 38.8 MB | YOLO 元件检测 |
| `weights/sym16_1280_best.pt` | 5.3 MB | 16 类小符号检测（imgsz 1280） |
| `weights/textrec_best.pt` | 29.6 MB | CRNN 文字识别 |

解析链路（`assets.py` 的候选表）只认这三个文件；`yolo_part/runs/**`、
`textrec/runs/**`、`work/**` 里的其余 checkpoint 均为训练过程产物，不参与推理，
故不归档。

`weights/textrec_pre_mix_20261016.pt`（29.6 MB，`textrec_best.pt` 的前身）与
`weights/yolo26n.pt`（5.3 MB，COCO 预训练基座）同样不是推理输入，未归档。

## `_workspace/` 归档内容（随本归档提交并入 `main`）

来自开发机工作区 `EDA/`（`eda-pcb` 的上一级目录）的赛题材料：

- `赛题资料/` —— 官方赛题 PDF、数据收集指南、`cpntLibrary` / `BoxICLibrary` 元件库、`open_GT/`
- `评测要求.pdf`、`赛题Q&A(1).docx`
- `SchLib需求单.md`、`RESULTS.md`
- `open-gt1106.zip`

## 未归档的内容（有意排除）

- `eda_yolo/`（3.5 GB）—— Python 虚拟环境，含 torch 的 CUDA DLL，可由
  `requirements.txt` / `requirements.lock.txt` 重建
- `yolo_part/runs`（2.5 GB）、`yolo_part/runs_ovl`（0.7 GB）、`textrec/runs`（1.5 GB）、
  `work/`（2.3 GB）—— 训练日志、逐 epoch checkpoint 与中间产物
- `yolo_part/symds`（0.17 GB）、`textrec/data`（0.17 GB）、训练集图片（0.34 GB）
- 工作区里的数据集：`训练集_v3` / `trainset_v3`（各 19 GB）、`work/`（50 GB）、
  `delete/`（18.8 GB）、`models/`（2.9 GB）
- 交付包与历史快照：`提交第十七次.zip`（131 MB）、`submission.zip`（65 MB）、
  `eda-pcb-full-20261104.bundle`（332 MB）
- 4 个超过 GitHub 单文件 100 MB 上限的实验 checkpoint：
  `yolo_part/runs/detect/pcb43_v4_im1536/{best,last}.pt`、
  `yolo_part/runs_ovl/pcb43_ovl/{best,last}.pt`（各 154 MB，仅 `train_ovl.py` 的输出目录名，
  推理不使用）

## 未提交的密钥

`.lmstudio_key`（本机 LM Studio token）从未进入任何一次提交，历史亦无泄漏。
