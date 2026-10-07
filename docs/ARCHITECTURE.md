# 架构说明

> 配套阅读：`README.md`（做什么 / 怎么跑）、`RESULTS.md`（实测数字）、
> `DIAGNOSTICS.md`（怎么量）、`workflow.html`（交互式流程图）。
> 全部环境开关的清单见 `KNOBS.md`（由 `tools/dev/gen_knobs.py` 生成）。

---

## 0. 一句话架构

```
PNG ──► stage1 几何 ──► stage2 文字与角色 ──► stage3 位号仲裁 ──► fusion 融合 ──► 官方 JSON
        (元件/导线/引脚)    (CRNN + 分配)          (重装配)          (先验/LLM)
```

分层的理由是**每一层都可以被真值替换**，从而把总分归因到具体一层：
`pipeline/prun.py` 里的 ablation ladder 就是干这个的（每一级换成 oracle，看分数变化）。

---

## 1. `stage1/` —— 几何层

第一阶段只做「不读字」的事：**元件在哪、导线怎么走、引脚在哪、谁连谁**。

| 文件 | 职责 | 关键点 |
|---|---|---|
| `detect.py` | 元件几何 | YOLO 通用检测 + 「规范图章」（固定尺寸符号）+ `cvparts` 的 CV 专类部件，输出**多假设槽位**而不是单一定论 |
| `wires.py` | 导线骨架 | 复用 `pipeline/pskel.py`，补三件事：降采样（`pskel` 在 1675×1190 的图上单张 88.5 s，赛题预算 2.4~4 min/case）、按颜色取线、线段拆分 |
| `labels.py` | 段 → 连通分量 | `wire_groups()`：5 条并网判据（端点-端点、端点-线段、`weld`、`ext`、**规则 5 穿越式微越过**） |
| `pins.py` | 引脚点 | 纯几何，不训练不 OCR。官方 pin 0.25 + netH 0.35 + netLine 0.10 = **0.70 分全部以引脚点为最小单位** |
| `nets.py` | 引脚点 × 导线图 → 网络 | 段连通分量 → 引脚聚类 → 网络 |
| `pincol.py` | 引脚列枚举 | 按格点补脚（默认关，实测 −0.63） |
| `pipeline.py` | 总装 | **进程池**而非线程池：`pskel` 是纯 Python + numpy、GIL 绑死 |
| `llm.py` | LLM 裁决器 | 只做**多选**（候选 + `"none"`），不让模型自由生成 |

### 为什么 `stage1` 必须存在

在 `stage1` 之前，产物里只有元件框：**没有引脚点就没有网络**，
`pin + netH + netLine = 0.70` 在结构上就拿不到。

### 规则 5：并网判据的教训

**距离是坏门限，形状关系是好门限。**

一根导线在**串联元件**处天然断成两截，两端共线且相对，缺口正好是元件本体
（电容 ~28 px、电阻 ~59 px）—— 与「结点圆点把导线切断」**纯几何不可分**，
但语义上是两个不同的网。抬 `weld`/`ext` 会短路所有串联元件（实测 −1.98 ~ −4.45 分）。

规则 5 换成**形状关系**：端点落在另一段的**内部**（而不是端点）附近，且**两段不平行**
（`|cos| < 0.9`），才算穿越式微越过。落地后官方 55.73 → 56.31、派生 34.69 → 35.23，
**两个档都赚**。

---

## 2. `stage2/` —— 文字层

| 文件 | 职责 | 关键点 |
|---|---|---|
| `text.py` | 文字框定位 + CRNN | 同一张裁片在 **5 个角色下各读一遍**再选；`tight_ink_box()` 描边剔除 |
| `assign.py` | 角色分配 | 位号 / 型号 / 引脚名 / 引脚号；归属判据 `dist_rect_to_rect` + `OWN_BBOX` |
| `structmatch.py` | 结构化重绑定 | 单调序列对齐 `align_monotone` + 阶梯重排 `_ladder`，两个整体方案取优 |
| `assemble.py` | 组装官方 JSON | **坐标系在这一层翻一次**（`y_json = H − y_img`） |

### 为什么「按角色各读一遍」

「先要知道角色才能读对」与「先要读出来才知道角色」互相依赖。
用「全都读一遍再选」把鸡生蛋问题绕过去；一张图几百个框，代价可忽略。

### 文字的几何先验（实测）

* `pinName` 100% 落在元件框**内**（1020/1020）
* `pinNumber` 95% 落在框**外**，p50 距离 5 px
* 位号 p50 距离 6 px

### 描边剔除（`STAGE2_PINNUM_TIGHT`）

图上 **绿色 = 导线与 `no-connect` 叉**、**暗红 = 元件外框**、**蓝 = 位号/型号**、
**黑 = 引脚号**。文字检测会把绿色叉和数字圈进同一个框，CRNN 于是把叉读成 `K`：
`4 → K4`、`14 → L14`、`10 → L10`、`2 → X1`。

对策：取框内**低饱和暗墨**像素的紧致外接框（`tight_ink_box`），**再照常外扩**
（外扩是承重的：去掉 `pad_box` 从 0.6866 崩到 0.4135）。收益 **+0.83**（56.71 → 57.54）。
**只对 `pin_number,pin_name` 生效** —— 位号/型号是蓝字，同法处理掉 3.8 个 component 分。

---

## 3. `stage3/` —— 位号仲裁

读 `stage2` 的产物与 `runs/stage2/_assigned/`，仲裁位号、重新装配。
`STAGE3_OWN=on` / `STAGE3_AUTO_LLM=0`。

---

## 4. `fusion/` —— 融合与仲裁（workflow.html 右侧那一列）

**为什么单独一层。** 现状是 `stage1/detect.py` 写 `components[i].type`、
`cvparts/textbox.py` 写 `texts[j].bbox`、`textrec/infer.py` 写 `texts[j].reads[role]`、
`vlm/enrich.py` **直接改** `assigned.comps[cid].designator.text` —— 每层都在改同一个文档，
出问题无法定位是谁改的。`fusion` 把改动收敛成一条**有序、可回放、可审计**的链。

| 文件 | 职责 |
|---|---|
| `priors.py` | **常识与先验**：不读图就能说出来的东西，且**会自己造候选**（与「只给现有候选打分」的本质区别）。含位号连号、电源轨、封装引脚数等 |
| `axis.py` | 全局轴解码：按「长度律」把同一轴上的位号/引脚号整体铺号 |
| `cropfix.py` | 裁切方向反馈：读不全时问方向、做外扩阶梯 |
| `judge.py` | LLM 多选裁决（`FUSION_JUDGE`、`FUSION_LLM`） |
| `iterate.py` | 迭代消解（`_smooth_numbers` 等） |
| `audit.py` | 复核：对可疑网络/键做二次读图 |
| `rescan.py` | 重扫可疑槽位 |
| `run.py` | 总装 |

---

## 5. 辅助包

| 包 | 职责 |
|---|---|
| `textrec/` | CRNN+CTC 识别器：`model.py`（CNN 宽下采样 8× → 角色嵌入 → BiLSTM → CTC）、`dataset.py`、`train.py`、`infer.py`、`charset.py` |
| `vlm/` | OpenAI 兼容本地视觉端点（LM Studio）的极薄客户端 + `enrich.py`（补 IC 位号与引脚号） |
| `cvparts/` | 单个符号类的纯 CV 检测器（形状文法 + 上下文门控），与 `pipeline/` 分工 |
| `pipeline/` | 纯 CV 的 P 流水线（**`pskel.py` 被 `stage1/wires.py` 复用**）；`prun.py` 是带 oracle 消融梯的端到端原型 |
| `schematic_parser/` | 官方评测器逆向实现（`metrics.py`）+ 数据读取（`data.py`） |
| `synth/` | 合成原理图生成（按真实 213 张语料的统计特性） |
| `submission/` | 竞赛容器入口 `run.py` + `agent_system` |

---

## 6. 关键约定

1. **坐标系**：stage1 全程**图像坐标**（左上原点，y 向下）；只在 `assemble.py` 出口
   翻一次成 JSON 坐标（左下原点）。翻两次必然出错。
2. **`nets[k]` 的成员在 `hyperGraph` 里**：`{"hyperGraph": "(U1.8)", "edges": {...}, "pins": []}`
   —— `pins` 是空的，别去读它。
3. **环境开关**：假值词表（`0/off/false/none/no`）唯一实现在 `envflag.py`。
   PowerShell 里 `$env:X=''` 是**删除**变量而不是关功能。
4. **代码即文档**：每个非平凡决策都把实测数字写进模块 docstring / 注释，
   包括**失败的尝试和它的数字**，避免后人重犯。
