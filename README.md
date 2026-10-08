# EDA 精英挑战赛 · 赛题六：原理图图像 → 结构化 JSON

给一张 PCB 原理图 PNG，输出官方口径的电路结构 JSON（`components` / `pins` / `nets`）。
本仓库是一套**可离线运行的完整推理管线**：YOLO 检测元件 → 纯 CV 提导线骨架与引脚点 →
CRNN 读文字 → 结构与先验分配角色 → LLM/VLM 融合仲裁 → 官方格式导出与自评。

> 竞赛评分（`schematic_parser/metrics.py` 逆向实现，与官方 Q&A 对齐）：
> `100 × (0.30·ComponentF1 + 0.25·PinF1 + 0.35·NetHypergraphF1 + 0.10·NetLineF1)`

---

## 一、实测成绩

| 评测集 | 分数 | 说明 |
|---|---:|---|
| 官方公开 GT（`10GT`，9 张） | **58.63** | **交付口径**（VLM `auto` + `-fusion on`，全部默认值）；同一批图本条优化线起点是 44.16 |
| 混合派生集（19 张 = 官方 9 + 派生 10） | **36.35** | 同上口径；派生集含大量 `Datasheet` 空位号图，分数天然偏低 |

四项分数（官方 9 张，满分 30 / 25 / 35 / 10）：

| component | pin | net_hypergraph | net_line |
|---:|---:|---:|---:|
| 28.21 | 9.80 | 18.74 | 1.88 |

**分会丢在哪**（实测归因，见 `docs/reports/引脚号回归与桩偏移.md`）：
`∅NNN` 电源/地符号的引脚号在 GT 里是**官方随机数**，占网络成员的 22%，数学上不可赢；
剩下的大头是**引脚号识别**（CRNN 单字混淆）与**导线端点精度**（逐条差 5~13 px，容差 5 px）。
这三件事决定了 58 分附近的天花板形状。

---

## 二、目录结构

```
eda-pcb/
├── submission/           竞赛容器入口：run.py + agent_system（-input/-output）
├── stage1/               几何层：元件几何 + 导线骨架 + 引脚点 + 多假设槽位
│   ├── detect.py         YOLO 通用检测 + 规范图章 + CV 专类部件
│   ├── wires.py          导线骨架（复用 pipeline/pskel，加降采样与加速）
│   ├── labels.py         段→连通分量（并网判据）、网络标签定位
│   ├── pins.py           引脚点提取（纯几何，不训练）
│   └── nets.py           引脚点 × 导线图 → 网络
├── stage2/               文字层：读字 → 定角色 → 组装
│   ├── text.py           文字框定位 + CRNN（角色条件读法、描边剔除）
│   ├── assign.py         位号 / 型号 / 引脚名 / 引脚号的分配与仲裁
│   ├── structmatch.py    单调序列对齐 + 阶梯重排（防整排错位）
│   └── assemble.py       拼官方 JSON（坐标系在这一层翻一次）
├── stage3/               位号仲裁与重装配
├── fusion/               融合与仲裁总装（先验 / 冲突 / LLM 裁决 / 迭代 / 复核）
├── vlm/                  OpenAI 兼容本地视觉端点客户端（LM Studio 等）
├── textrec/              CRNN+CTC 文字识别器（模型、数据、训练、评测）
├── pipeline/             纯 CV 的 P 流水线（pskel 导线骨架被 stage1 复用）
├── cvparts/              单个符号类的纯 CV 检测器（形状文法 + 上下文门控）
├── schematic_parser/     官方评测器逆向实现 + 数据读取
├── synth/                合成原理图数据生成
├── weights/              推理权重（**随仓库提交**，见下）
├── tools/                诊断与评测脚本（打分尺子、错例归因、A/B 工具）
└── docs/                 架构说明、评测结果、诊断报告、流程图
```

---

## 三、环境与安装

* Python ≥ 3.10，**推荐 NVIDIA GPU**（`imgsz=1600` 的 YOLO + CRNN；CPU 可跑但很慢）
* 参考环境：Python 3.13 + PyTorch 2.13 + ultralytics 8.4 + OpenCV 5.0（Windows / CUDA 13）

```powershell
# 1) 核心依赖
pip install -r requirements.txt

# 2) PyTorch（按你的 CUDA 版本选 index，不要照抄版本号）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 3) 自检：两个权重是否都能解析到
python assets.py
```

`requirements.txt` 只列**直接依赖**；完整冻结版本（本机实测跑通的那一套）在
`requirements.lock.txt`。

---

## 四、快速开始

### 4.1 批量推理

```powershell
python -m submission.run -input <输入图目录> -output <输出目录> -device 0 -imgsz 1600
```

输出 `<输出目录>/<case>_Pred.json`，即官方提交格式。自定义参数：

```powershell
python -m submission.run -input in -output out `
  -limit 10 `                 # 只跑前 10 张
  -cases 1,2,10 `             # 只跑指定 case
  -no-wires `                 # 跳过导线网络（只交元件+引脚）
  -mode off `                 # VLM 关闭；默认 auto
  -fusion on `                # 融合仲裁层（默认开）
  -gpu-frac 0.45              # 给本地 LLM 留显存
```

### 4.2 启用 VLM / LLM（默认端口 1234）

管线里的「融合仲裁」需要一个 **OpenAI 兼容的本地视觉端点**（默认 `http://127.0.0.1:1234/v1`，
模型默认 `qwen3.5-9b`）。推荐 LM Studio 加载一个带视觉的模型并 start server。
端点不可用时 `-vlm auto` 会自动降级，管线仍能跑完（分数约降 5 分）。

```powershell
$env:FUSION_LLM='0'      # 关掉 LLM 裁决，只跑纯几何 + VLM（做 A/B 时的基线口径）
$env:STAGE1_LLM_BASE='http://127.0.0.1:1234/v1'
```

> 所有开关都是 `STAGE1_* / STAGE2_* / FUSION_*` 环境变量，假值词表统一在 `envflag.py`。
> 注意：`$env:STAGE1_DROP_TYPES=''` 在 PowerShell 里是**删除**变量，要关功能请写具体假值。

### 4.3 官方口径自评

```powershell
# 单 case 打分（用仓库自带的逆向评测器）
python -m schematic_parser.metrics --gt <GT目录> --pred <预测目录>

# 批量 A/B（tools/eval/fast.py，带结果归档）
python tools\eval\fast.py --cases 1,2,3,4,5,7,8,9,10 --save off9_base
python tools\eval\fast.py --list                # 列出可用 case
```

`fast.py` 打开关请用 `$env:X='1'`，**不要**用 `--extra`（该参数会把多词参数按字符拆开）。

---

## 五、流水线架构

```
PNG
 ├─ stage1  元件几何（YOLO + 图章 + CV 专类） → 多假设槽位
 │          导线骨架（pskel 复用 + 降采样）   → 段 / 连通分量 / 网络标签
 │          引脚点（纯几何，不训练）          → pins
 │          labels.wire_groups 并网判据 → nets（段连通分量 → 网络）
 ├─ stage2  文字框定位 + CRNN（5 种角色各读一遍）
 │          assign   位号 / 型号 / 引脚名 / 引脚号（几何 + 字符串模式 + 归属判据）
 │          structmatch  单调序列对齐 + 阶梯重排（防整排错位）
 │          assemble 组装官方 JSON（坐标系在此翻一次）
 ├─ stage3  位号仲裁与重装配
 └─ fusion  常识先验（不读图就能说的东西）→ 冲突检测 → LLM 多选裁决
            → 迭代消解 → 复核 → 导出
```

设计动机与每个模块的「为什么」都写在**各文件的模块 docstring 里**（本项目的约定：
每个非平凡决策都要把实测数字写进代码注释，避免后人重犯）。

---

## 六、关键设计决策（含实测结论）

| 决策 | 依据 |
|---|---|
| `stage1` 用**进程池**而非线程池 | `pskel` 是纯 Python 循环 + numpy，GIL 绑死，多线程无效 |
| 同一张裁片在 **5 个角色下各读一遍**再选 | 「先知道角色才读得对」与「先读出来才知道角色」互为前提，全读一遍绕开鸡生蛋 |
| 并网判据用**形状关系**而非距离 | 距离门限是坏门限：串联元件处导线天然共线断开、缺口正好是元件本体，各 28~59 px；抬距离门限会短路所有电容电阻（实测 −1.98 ~ −4.45 分） |
| 引脚号裁片先做**描边剔除** | 图上绿色 `no-connect` 叉与数字同框被 CRNN 读成 `K4`/`L14`；按颜色（低饱和暗墨）取紧致框后 +0.83（官方 9 张 56.71 → 57.54） |
| 描边剔除**只对 `pin_number,pin_name`** | 位号/型号是**蓝字**，同法处理会掉 3.8 个 component 分 |
| 归属判据用**矩形间距**而非文字中心距 | 原判据随文字宽度漂移，三位球号因「中心更远」被整排丢出候选池；改后官方 +0.14 / 派生 +0.59 |
| `PINNUM_DECAY=10`（沿边距离衰减） | 衰减太慢时 DP 宁可整排错位（多配一对赚得比错位亏得多） |
| 导线端点做**同网吸附**（`STAGE2_EDGE_SNAP=20`，只吸近似垂直的邻居） | `net_line` 的瓶颈不是边形状而是**端点精度**：骨架落在一条描边上、分叉点没解算到交叉处，逐条差 **5~13 px**，而官方 `line_tol=5` 且判据是「两端点距离取 max」。同网内吸附后导线 F1 **0.065 → 0.165**（官方 9）、**0.036 → 0.084**（派生 19） |
| 不并「同一根导线的两条描边」 | 并成中心线只值 **+0.02 导线分**（+0.2 总分），却把 case 2 的导线打成 0 |
| boxscan（给零候选元件建框去问一遍）**默认关** | 类型闸门挡掉 `gnd`/`v`/`net_*` 后每张只建 1 个槽位，好/坏就是那 1 颗元件的自由读数对不对 ⇒ 官方 9 **−0.01**、派生 19 **+0.05**，两档不都赚 |
| `own.assign`（全局一对一归属）**默认关** | 一对一不变式在改造前就已成立（`owned` 本来就是划分），端到端两档**逐位相同**；天花板测量显示 19 颗失配里 **只有 1 颗**是候选池已有正确读法 ⇒ 剩下的分在 CRNN 上，不在归属上 |
| 形状惩罚（`FUSION_SHAPE_W`）**默认 0.0** | 官方 9 张 `0/0.35/1.0` **逐位相同**；派生 19 张 `W=1.0` 比 `W=0` **低 0.57** |
| 不实现「把 GND 系引脚并成一条网」 | 电气上成立，但**官方 GT 不这么编码**：case 1 = 7 个引脚散在 1 条（8 成员）、case 0001 = 5 散 4、case 0042 = 9 散 2 ⇒ 按它并网是负向操作 |
| **改完先证明分支执行了，再解释分差** | boxscan 第一版有差（52.48 → 52.47），但把 `_fusion_report.json` 的 `notes` 打出来才看到每张只建了 1 个槽位 —— 分差根本不是「boxscan 整体」的效果。详见 `docs/DIAGNOSTICS.md` |

**被证伪的路线**（都留了实测数字，别再试）：
墨迹尺度归一（三版全败，除非连同模型重训）、二次外扩修正（去掉 `pad_box` 崩 39%）、
`weld/ext` 长缺口桥接（净亏，见上表）、引脚号候选去重（单点对、指标亏 −1.80）、
`STAGE2_PINCOL` 引脚列引导（−0.63）、`net_line` 并描边/焊共线（见上表）、
boxscan（−0.01 / +0.05）、`own.assign`（端到端零效果）、形状惩罚（−0.57）、
GND 互证先验（官方 GT 不这么编码）、自由读数压分（`FUSION_FREE_SCORE` 逐位无差别）、
`--no-vlm` 与 VLM 开的两组分数**不可同表比较**（差 5.2 分）。

---

## 七、数据与权重

### 权重（已随仓库提交，约 68 MB）

| 文件 | 用途 | 大小 |
|---|---|---:|
| `weights/pcb43_v2_from_best.pt` | YOLO 元件检测（`stage1/detect.py`） | 38.8 MB |
| `weights/textrec_best.pt` | CRNN 文字识别（`stage2/text.py`） | 29.6 MB |

`assets.py` 按**相对路径**解析，可用环境变量覆盖：`YOLO_WEIGHTS` / `STAGE2_CKPT`。
源码里**没有功能性绝对路径**（仅少数 docstring 示例保留本机路径）。

### 数据集（**不在仓库内**）

图片与标注体积过大，已 gitignore，需要另行准备：

| 名称 | 用途 |
|---|---|
| `yolo_part/training_data/` | YOLO 元件检测训练集（41 类，`data.yaml` 已入库） |
| `yolo_part/Dataset(waitforpreprocess)/` | 原始 PNG（评测脚本按此目录找图） |
| `textrec/data/manifest.jsonl` | CRNN 训练清单（85 MB，未入库） |
| 官方 `10GT` | 公开 GT，评测基准 |

重新生成合成数据：`python -m synth.make_demo --n 24 --out <目录>`。

---

## 八、已知限制（诚实清单）

1. **`∅NNN` 不可赢**：电源/地符号的引脚号与名在官方 GT 中随机生成，占网络成员 22%，
   任何方法都拿不到；`net_hypergraph` 的实际上限因此明显低于 1。
2. **单字混淆**：`9↔8`、`6↔8`、`2↔3` 一类错误只能靠识别器本身（重训 / TTA 投票）。
   已经量过天花板：官方 9 张里配得上 GT 位号元件的 124 颗中，**读对 104 / 什么都没读 11 /
   读错 9**，而这 20 颗失配里**只有 1 颗**的候选池里本来就有正确读法
   （`tools/diag/_ownasgn.py --headroom --gt`）⇒ **位号这一路剩下的分全在识别率上，
   不在归属/重排上**，任何更聪明的一对一最多值 1 颗元件。
3. **位号不可找回**：约 44% 的派生图**图上根本不印位号**，GT 却要求 `U1`；这部分无解。
4. **`Datasheet` 类图**分数天然低，不要用它和普通图混在一个平均数里比较。
5. **VLM 需要外部端点**，默认配置假设本机 1234 端口有 LM Studio 在跑。
6. **导线端点仍有 5~13 px 量级的误差**：骨架落在导线的一条描边上、分叉点没解算到交叉处。
   官方判据是「两端点距离取 max」+ `line_tol=5`，所以 `net_line` 天然吃亏；
   同网吸附（`STAGE2_EDGE_SNAP=20`）只把导线 F1 从 0.065 抬到 0.165，离满分很远。
7. **LLM 轮是纯开销还是收益取决于图**：官方 2 张实测产物与关掉 LLM 时**逐字节相同**，
   但墙钟 +29.6 s/张（220 张 ≈ +1.8 h）。目前保留默认开启是 **owner 的决定**
   （`submission/run.py` 注释），不是实测结论。

---

## 九、文档索引

| 文档 | 内容 |
|---|---|
| `docs/ARCHITECTURE.md` | 分层架构、模块职责、四条关键约定 |
| `docs/RESULTS.md` | 分数轨迹、全部 A/B 表、负结果归档 |
| `docs/DIAGNOSTICS.md` | 怎么做测量：诊断尺子、五步工作流、复现命令 |
| `docs/KNOBS.md` | **全部 265 个环境开关**（自动生成，带默认值与 `文件:行号`） |
| `docs/TOOLS.md` | **全部 483 个工具脚本**按用途分组（自动生成） |
| `docs/workflow.html` | 交互式流程图（31 节点 / 65 连线，含每个节点的实测笔记） |
| `docs/reports/*.md` | 逐主题诊断报告（引脚号、并网、位号上限、尺度归一、GT 保真度…） |

两个索引是**从代码生成**的，改了开关或加了脚本请重跑：

```powershell
python tools\dev\gen_knobs.py --write      # → docs/KNOBS.md
python tools\dev\gen_toolindex.py --write  # → docs/TOOLS.md
```

---

## 十、许可

仅用于 EDA 精英挑战赛赛题六的参赛与研究用途。第三方组件（Ultralytics YOLO、PyTorch
等）遵循其各自许可；`weights/` 下的模型为本队训练产物。
