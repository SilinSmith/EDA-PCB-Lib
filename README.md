# EDA 精英挑战赛 · 赛题六：原理图图像 → 结构化 JSON

给一张原理图 PNG，输出官方口径的电路结构 JSON（`components` / `pins` / `nets`）。
本仓库是一套**可离线运行的完整推理管线**：YOLO 检测元件 → 纯 CV 提导线骨架与引脚点 →
CRNN + PP-OCR 双识别器读文字 → 结构与先验分配角色 →（可选）VLM 仲裁 → 官方格式导出与自评。

> 评分（`schematic_parser/metrics.py`，与官方 Q&A 对齐）：
> `100 × (0.30·ComponentF1 + 0.25·PinF1 + 0.35·NetHypergraphF1 + 0.10·NetLineF1)`
>
> 口径分三档（见 §1.2）：**内部尺**把非 box 元件的引脚号折成通配 `*`
> （`net_pin_collapse_nonbox=True`，历史基线一直用它）；**官方 / 判题机口径不折叠**
> （`EvalConfig()` 默认）。这一条是拿官方真实产物标定出来的——折算会在官方 10 张上虚高 7.4 分，
> 所以 README 两边都给。另外 `pin_f1` 只统计 box/IC 类元件（`pin_box_only=True`，
> 评测 Q2：非 box 的 pin name/number 不参与评分）。

**当前交付档（2026-11-04，`main@f6eb6d2`）**

| 评测集 | 分数 |
|---|---:|
| 13 页内部复测集（内部尺 / 官方口径 / 判题机口径） | **85.58 / 84.53 / 81.83** |
| 30 案派生池（同一内部尺） | **29.85** |
| 四轴损失（内部尺 85.58 的分解，满分 100） | comp 2.84 · pin 3.51 · netH 3.85 · **line 4.22** |

> ⚠️ **旧版 README 的头条 68.29（官方 10GT 9 张）是 2026-10 的旧口径 + 旧默认值**，
> 与本表不可直接比较（中间换过识别通道、改过引脚法向归位、翻过几十个默认值），
> 轨迹见 `docs/RESULTS.md`，按主题索引查 `docs/RESULTS_INDEX.md`。

---

## 一、实测成绩

### 1.1 13 页内部复测集（逐页）

图片 2200×1017 / 1469×770 / 1658×752 一类的大图（公开样张重新渲染，GT 与图片不在本仓库内）。
内部尺 = `EvalConfig(net_soft=True, bbox_tol=20, pin_tol=5, line_tol=5,
net_pin_collapse_nonbox=True, unkeyed_match='gt', coord_gate=20, coord_center=True)`。

| case | 总分 | component | pin | netH | net_line |
|---:|---:|---:|---:|---:|---:|
| 1 | **66.55** | 0.696 | 0.629 | 0.743 | 0.395 |
| 2 | 89.55 | 1.000 | 0.875 | 0.956 | 0.424 |
| 3 | 97.92 | 1.000 | 1.000 | 1.000 | 0.792 |
| 4 | 93.90 | 0.958 | 1.000 | 0.942 | 0.718 |
| 5 | 87.50 | 1.000 | 0.727 | 0.942 | 0.635 |
| 6 | 92.82 | 0.935 | 0.933 | 0.973 | 0.739 |
| 7 | 96.87 | 0.958 | 1.000 | 1.000 | 0.812 |
| 8 | 84.42 | 1.000 | 0.759 | 0.840 | 0.606 |
| 9 | 96.67 | 1.000 | 1.000 | 1.000 | 0.667 |
| 10 | 92.80 | 0.983 | 0.870 | 0.975 | 0.743 |
| 11 | **60.48** | 0.769 | 0.485 | 0.567 | 0.542 |
| 12 | **71.20** | 0.650 | 0.941 | 0.744 | 0.213 |
| 13 | 81.85 | 0.821 | 0.957 | 0.888 | 0.224 |
| **均值** | **85.58** | 0.905 | 0.860 | 0.890 | 0.578 |

三页（11 / 1 / 12）占掉全部损失的 53%：case11 是电源符号抢 IC 脚 + 触点缺失，
case1 是电源符号整颗没检出 + 位号读错，case12 是 `net_input/net_output` 标签符号缺件 + 导线端点偏。

### 1.2 三把尺

同一次产物会同时打三次分，差值是**被宽松口径掩盖的损失**：

| 尺 | 分数 | 口径 |
|---|---:|---|
| 内部尺 | **85.58** | 协议4（非 box 脚号折成 `*`，无键件走坐标匹配） |
| 官方口径 | **84.53** | `评测要求.pdf` 口径：自动编号件的 `value` 仍严格比 |
| 判题机口径 | **81.83** | `EvalConfig()` 原样 —— **不**折叠非 box 脚号（用官方真实产物标定过，见 `docs/JUDGE.md`） |

### 1.3 30 案派生池

从 `yolo_part/Dataset(waitforpreprocess)/200_train_cases` 抽 30 案（`docs/JUDGE.md` 的池清单），
**29.85**。这个数长期在 29~30 徘徊，原因不是这轮改动掉了分，而是**结构性的**：

- 池图很小（1127×509 / 1052×642 / 871×431），字号 ~10 px；
- GT 里 box/IC 件 143 个、在册脚 **1508** 根，而 13 页只有 191 根；
- **判分按键精确匹配**（`metrics.py:335`），只有 GT 侧 `∅NNN` 才走坐标匹配
  ⇒ 位号读错/读不到 = 该件的 comp、pin、net 三轴同时作废；
- 池页上 **约 45% 的 GT 位号在图上根本找不到**（KiCad 导出成 `U?`/`C?` 未标注、
  Altium 的 `R.23` 被导线端头 `>` 粘连成 `>R24>R25`），这部分无论检测多准都拿不到分。

按「未标注件按得分处理」重算，池的上限也只有 **36.18**（`work/_deanno.py`，内部第二读数，
不入正式口径）；池 pin 轴在当前匹配下几乎恒为 0 —— 已匹配件上 GT 1093 根脚、我们发 388 根，
其中同件同号 710 根的**点对点距离中位数 20.1 px**（`pin_tol=5`）⇒ 是「脚号/脚名覆盖率」+
「20 px 级位置误差」双重卡死，不是某一处小 bug。

### 1.4 复现

本仓库内可直接打分的是三个 split（`10GT` 10 张 / `10_GTcase` 10 张 / `200_train_cases` 200 张），
GT 标注随仓库提交、图片需自备（见 §七）：

```powershell
# 1) 自检尺子（不需要图，只用 GT 跑协议自洽性）
python -m schematic_parser.evaluate --self-test

# 2) 跑一批图，再打分
python -m submission.run -input  <图片目录> -output out\run1 -device cpu -lean
python -m schematic_parser.evaluate --pred out\run1 --split 10GT

# 3) 单臂快跑（跑管线 + 打分一体，最常用）
python tools\eval\fast.py --cases 1,2,3 --no-vlm --extra "-device cpu" --save arm1
python tools\eval\fast.py --cases 1,2,3 --score-only --cmp arm1
```

13 页那套图不在仓库内，用内部探针跑（同样单进程、CPU）：
`work/_cv.py <tag> --run --cases 1,11,12,13`。

---

## 二、仓库结构

```
eda-pcb/
├─ submission/     官方入口：run.py（编排 + 导出）、agent_system（容器入口，薄壳）
├─ stage1/         检测 + 骨架：detect.py(YOLO) wires.py extract_wires sym16.py pins.py labels.py
├─ pipeline/       纯 CV 底座：pskel.py(墨迹/细化/建图/RDP) prun.py pcore.py …
├─ stage2/         结构与文字：text.py(CRNN+PP-OCR) assign.py assemble.py partname_ocr.py
│                  libpins.py rowbind.py genocr.py
├─ stage3/         位号仲裁（LLM，默认 off）：arbitrate.py
├─ fusion/         常识先验 + 冲突消解：pwrclip.py edgecal.py priors.py run.py
├─ vlm/            视觉模型客户端：client.py（OpenAI 兼容端点）
├─ schematic_parser/ 官方格式与评分：metrics.py（评分器） official.py data.py evaluate.py
├─ cvparts/        图像基础操作     textrec/ CRNN 训练与推理     synth/ 合成数据
├─ tools/          评估与诊断脚本（eval/ diag/ dev/）           tests/ 自检
├─ packaging/      容器与提交包：Dockerfile run.sh install_deps.py verify_imports.py
├─ weights/        推理权重（§七）   docs/ 文档（§九）
└─ yolo_part/      GT 标注、合成脚本、检测训练集（图片不入库）
```

---

## 三、环境与安装

```powershell
python -m pip install -r requirements.txt      # 直接依赖（不含 torch）
python -m pip install torch torchvision        # 按 CUDA 版本自选 index；CPU-only 直接装
python -m pip install rapidocr_onnxruntime     # 可选：PP-OCR 第三识别通道（缺包会自动跳过）
```

- 本机实测：**Python 3.11.6**，`requirements.lock.txt` 是跑通那一套的精确冻结版本。
- **纯 CPU 可跑全流程**（本仓库所有公开数字都是 CPU 单进程测的）；有 GPU 时 `-device 0` 自动用。
- 首次运行会加载 `weights/`（§七），无网络依赖；只有 VLM 通道需要外部端点。

### 3.1 必填项 / 可选环境变量

| 变量 | 作用 | 缺省行为 |
|---|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | VLM（`-vlm auto`）读取 IC 位号与引脚号/名 | 未配置 ⇒ VLM 自动跳过，纯离线跑完 |
| `DASHSCOPE_API_KEY` | 容器里作为 `LLM_API_KEY` 的别名兜底（`packaging/run.sh`） | — |
| `YOLO_WEIGHTS` / `STAGE2_CKPT` | 覆盖检测/识别权重路径 | 用 `assets.py` 的相对路径 |
| `EDA_DATA_ROOT` | 覆盖评分用的数据集根 | `yolo_part/Dataset(waitforpreprocess)` |

> 坑：平台注入**空串**（`LLM_MODEL=""`）时 `os.environ.get(k, 默认)` 返回空串而不是默认值，
> 会导致 VLM 静默降级。打包时务必确认三个变量要么齐全、要么整组不设。

---

## 四、快速开始

### 4.1 跑一个目录

```powershell
python -m submission.run -input samples\ -output out\run1 -device cpu
# 产出 out\run1\final_results\<case>_Pred.json（+ decision_log.json + run.log）
```

常用参数（全部实现在 `submission/run.py`，`-h` 有逐条实测说明）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `-input` / `-output` | 必填 | 输入图片目录 / 输出目录 |
| `-device` | `0` | `cpu` 或卡号；无 GPU 用 `cpu` |
| `-imgsz` | `1600` | 检测输入边长（实测 1200/2048 都更差） |
| `-mode` | `off` | 阶段三 LLM 仲裁；`llm` 需显式开（实测零收益且 161.7 s/张） |
| `-vlm` | `auto` | `auto/on/off`：端点可达才开 |
| `-lean` | 关 | 「省 token 不掉分」的通道组合（容器默认档） |
| `-offline` | 关 | 强制断网路径 |
| `-no-wires` | 关 | 跳过骨架提线（最慢的 CPU 段） |
| `-cases` | 空 | 只跑指定 case（逗号分隔） |

### 4.2 自评 / 诊断

见 §1.4。**改任何默认值之前先读 `docs/KNOBS.md`**（全部开关 + 默认值 + `文件:行号`），
它和 `docs/TOOLS.md`（脚本索引）都是从代码生成的。

### 4.3 容器 / 提交包

`packaging/run.sh` 是官方镜像里的入口包装：它把 `/submission/eda-pcb-main/submission/agent_system`
接上平台给的参数，并在 `EDA_TIER_ARGS`（默认 `-lean`）里塞比赛日档位；启动时先跑
`verify_imports.py` 自检依赖。本地没有 docker CLI，因此容器镜像的验证只能在评测机上完成，
源码级验证请用 `python submission\agent_system -input ... -output ...`（与容器同一份代码）。

---

## 五、流水线

```
图 ──► stage1  检测(YOLO pcb43_v2) ─┐
       骨架(pskel: 墨迹→细化→建图→RDP)├─► stage2  文字(CRNN + PP-OCR 双通道)
       引脚/符号几何(sym16,pins)     ─┘           角色分配(assign) → 组装(assemble)
                                                    ↑ libpins 官方库 / partname_ocr 型号拼接
   ──► stage3  位号仲裁（LLM，默认关）
   ──► fusion  常识先验 → 冲突检测 →（可选 LLM）→ 复核 → 导出
   ──► submission  组装官方 JSON（坐标系在此翻一次）+ decision_log
```

| 阶段 | 入口 | 干什么 |
|---|---|---|
| stage1 | `stage1/detect.py`、`stage1/wires.py:702 extract_wires` | 元件框与类别；导线骨架→线段；符号几何 |
| stage2 | `stage2/text.py`、`assign.py`、`assemble.py` | 文字识别、角色指派、引脚编号/命名、组装 |
| stage3 | `stage3/arbitrate.py` | 位号仲裁（默认 `off`） |
| fusion | `fusion/run.py` | 先验修正、冲突消解、复核 |
| 导出 | `submission/run.py` | 官方 JSON + `decision_log.json` |

每个模块的「为什么」写在文件头 docstring 里，并附当轮实测数字——本项目的约定是
**没有数字的决策不许进代码注释**。

---

## 六、关键设计决策（当前有效）

| 决策 | 依据（实测） |
|---|---|
| 导线走**骨架**而不是检测 | 骨架覆盖 GT 边长度均值 **0.994**、骨架点精度 0.870 ⇒ 墨迹→矢量已经做完；损失在**端点/切点**（GT 边 744 条里端点偏 5~40 px 的占 14.8%） |
| 切点回到**墨迹**判分支，不在线段几何上补切 | 事后切分三连负（T 形 −0.061 / X −0.008 / 圆点门控 −0.008）；真分支检测器实测**切了 0 条边**（我们的段本来就断在分支点）。GT 只在**真结点**处断：圆盘墨量 `r=4 ≥0.95` 判真结点，命中 98%（169 个结点 vs 27 处纯视觉交叉） |
| 文字走**双识别器**：CRNN + PP-OCR（rec-only） | 池 82 件「CRNN 全页读不到」的：CRNN exact 20/82 → PP-OCR **52/82**；13 页 21 条错配里 PP-OCR 精确读对 **12/21** ⇒ 端到端 **+0.4518**（池 **+0.7513**）。只用识别头（`use_det=False`）≈10 ms/框；全页检测版要 29.7 s/页 |
| IC 引脚做**同侧法向归位** | 同侧 ≥3 根时把法向坐标归到该侧中位 ⇒ **+0.4808**（pin 轴 4.15→3.67，只 case4 +2.08 / case9 +4.17）。**不做**沿边吸附、不补脚、**不**把脚硬放到「框边外 20 px」（后者 −6.29） |
| 位号是**第一性**的（键错 = 三轴归零） | 判分按键精确匹配；池上 comp/netH 一半以上的损失是被位号连带打死的（换对键后 comp 0.556→0.911、netH 0.197→0.420） |
| 默认值必须**双向**验证 | 4 页只能筛「有没有效果」，正负必须 13 页定（`PIN_XSTUB` 4 页 +0.51 → 13 页 −1.49）；**默认开**的开关也要反扫（45 个逐个关掉，最后只留一个 `STAGE2_PIN_NUMBER_REPAIR=0`，+0.1033） |
| 每个改动配**默认关开关 + 惰性证明** | 默认关时 13/13 `_Pred.json` sha256 与基线逐字节相同；默认开要 13 页 + 30 案池两把尺同时达标，且**无单页掉 >0.05** |

**已判死（都留了数字，别再试）**：
检测器参数（conf 0.15/0.10/0.35、imgsz 2048/1200、tile 1024 —— 全部打平或更差）、
事后切分/合并（三连负）、把导线端点吸到 GT 引脚点（−0.063；GT 的端点本来就不在引脚点上）、
按 YOLO 框做 naive 格点重排（−1.95 / −2.72）、近重复框合并（−0.45）、
两脚件脚序推断（31 个反向件里 24 个是左右对称的 C/R，无观测线索）、
用体内印的脚名补缺失触点（0 命中）、位号编号规律（8 种几何排序最高 10.7%）、
「未标注位号按得分」重算之外的位号通道（池 +0.66 上限、13 页 0）。

---

## 七、数据与权重

### 7.1 权重（随仓库提交，约 79 MB）

| 文件 | 用途 | 大小 |
|---|---|---:|
| `weights/pcb43_v2_from_best.pt` | YOLO 元件检测（`stage1/detect.py`） | 38.8 MB |
| `weights/textrec_best.pt` | CRNN 文字识别（`stage2/text.py`） | 29.6 MB |
| `pipeline/glyph_gallery.pkl` | 字形先验（比对用） | 11 MB |

`weights/textrec_pre_mix_20261016.pt`（29.6 MB）与 `weights/yolo26n.pt`（5.3 MB）是备选/小模型；
检测训练集权重 `yolo_part/weights/` 被 `.gitignore` 忽略（体积大，只在本机）。
`assets.py` 按相对路径解析，可用 `YOLO_WEIGHTS` / `STAGE2_CKPT` 覆盖。

> 2026-11-04 做过一次**历史重写**：`>4 MB` 的历史大 blob（`RESULTS.md` 407 个版本、
> 旧权重、`docs/media` 的 gif/mp4）已从历史剔除，`.git` 从 355 MB 降到 165 MB。
> 队友必须重新 clone 或 `git reset --hard origin/main`，详见 `docs/GIT_HISTORY_20261104.md`。

### 7.2 数据集（图片不入库，GT 标注入库）

| 位置 | 入库 | 用途 |
|---|---|---|
| `yolo_part/Dataset(waitforpreprocess)/10GT/**/*.json` | ✅ | 10 张官方公开 GT + `type_list.md`（34 类） |
| `yolo_part/Dataset(waitforpreprocess)/10_GTcase/**/*.json` | ✅ | 10 张 |
| `yolo_part/Dataset(waitforpreprocess)/200_train_cases/**/*.json` | ✅ | 200 张派生（30 案池从中抽） |
| `yolo_part/Dataset(waitforpreprocess)/**/*.png` | ❌ | 原图，跑流水线才需要 |
| `yolo_part/training_data/`、`textrec/data/manifest.jsonl` | ❌ | 训练侧 |

图片就位规则：`<data_root>/<split>/<case>/<case>.png`，或 `--data-root` 指向别处。
合成数据：`python -m synth.make_demo --n 24 --out <目录>`。

---

## 八、已知限制（诚实清单）

1. **池 29.85 是结构性的**：约 45% 的池 GT 位号在图上不存在（未标注/被粘连），
   判分按键精确匹配 ⇒ 这些件连同引脚与网络全部作废；按宽松口径重算上限也只有 36.18。
2. **池 pin 轴≈0**：已匹配件上 GT 1093 脚 / 我们 388 脚；同件同号 710 根的点对点距离中位数
   20.1 px、≤5 px 只占 2.1% ⇒ 位置精度与脚名覆盖率都不够，不是单点 bug。
3. **13 页损失集中**：case11（60.48）、case1（66.55）、case12（71.20）三页占 53%。
4. **`net_line` 是最弱轴（0.578）**：GT 边与我们边两端 5 px 内配上的只有 65.7%，
   端点偏 5~40 px 的另占 14.8%；网络配对闸门只影响 0.4%，不是配对问题。
   已证明事后切分/合并无效，要修必须回到骨架阶段（见 §六）。
5. **`∅NNN` 不可赢**：电源/地符号的引脚号与名在 GT 中随机生成，任何方法都拿不到满分。
6. **VLM 需要外部端点**；环境变量为空串时会静默降级；VLM 开/关两组分数**不可同表比较**。
7. **PP-OCR 通道有代价**：+6.9 s/张（CPU，13 页 75.5 s → 195.6 s）。要更快用
   `STAGE2_GEN_OCR=0` 关掉（会掉 +0.4518，池掉 +0.7513）。
8. **`Datasheet` 类图**分数天然低，别和普通图混在一个平均数里比。
9. **判题机口径比内部尺低 3.75 分**（81.83 vs 85.58）：官方不折叠非 box 脚号，rcl 的脚号错位会实扣。

---

## 九、文档索引

| 文档 | 内容 |
|---|---|
| `docs/ARCHITECTURE.md` | 分层架构、模块职责、四条关键约定 |
| `docs/JUDGE.md` | **判题机说明书**：怎么打分、口径开关、参考分、最短路径 |
| `docs/RESULTS.md` | 分数轨迹与全部 A/B（§5.5xx 是最近一轮；42k 行） |
| `docs/RESULTS_INDEX.md` | `RESULTS.md` 的目录（先查这里再定位小节） |
| `docs/DIAGNOSTICS.md` | 怎么做测量：诊断尺、五步工作流、复现命令 |
| `docs/KNOBS.md` | 全部环境开关（自动生成，带默认值与 `文件:行号`） |
| `docs/TOOLS.md` | 全部工具脚本按用途分组（自动生成） |
| `docs/TRAINING.md` · `docs/RETRAIN.md` · `docs/DETECTOR_RETRAIN.md` | 重训 CRNN / YOLO |
| `docs/PACKAGING.md` | 打包、上传、把判题机单独交给队友 |
| `docs/GIT_HISTORY_20261104.md` | 2026-11-04 历史重写说明（**队友必读**） |
| `docs/workflow.html` · `docs/导线workflow.html` | 交互式流程图 / 导线专页 |
| `docs/reports/` | 逐主题诊断报告 |

索引类文档是从代码生成的，改了开关或加了脚本请重跑：

```powershell
python tools\dev\gen_knobs.py --write      # → docs/KNOBS.md
python tools\dev\gen_toolindex.py --write  # → docs/TOOLS.md
```

---

## 十、许可与仓库历史

仅用于 EDA 精英挑战赛赛题六的参赛与研究用途。第三方组件（Ultralytics YOLO、PyTorch、
RapidOCR 等）遵循其各自许可；`weights/` 下的模型为本队训练产物。

仓库在 2026-11-04 做过一次强制历史重写（剔除 >4 MB 历史 blob），
如果你手上的 clone 是旧的，请按 `docs/GIT_HISTORY_20261104.md` 重新同步。
