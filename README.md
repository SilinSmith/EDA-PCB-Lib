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
| 官方公开 GT（`10GT`，9 张） | **68.3** | **交付口径**（VLM `auto` + fusion on，全部默认值；`fast.py` 不加 `--no-vlm`）+ 最新 QA 的净图标口径（非 box 元件引脚号不参与比较，见 `docs/RESULTS.md` §0）。**68.29** 是含今日三个新默认值的单臂（`docs/RESULTS.md` §2.6 表 + §4.8(b)）；关掉官方库那一项的单臂重复是 **66.79 / 66.60**（VLM 噪声约 ±0.19，导线项逐位可复现）。本条优化线起点（旧版口径 + 旧默认）是 44.16 |
| 混合派生集（19 张 = 官方 9 + 派生 10） | 36.78 ⚠️ | **这个数还没按当前默认值重跑**（是「电源节点默认口径翻转 + 导线旋钮 + 官方库」三件事之前/之中的旧数）。派生集含大量 `Datasheet` 空位号图，分数天然偏低 |

> ⚠️ **口径分水岭（2026-10 最新 QA）**：官方答疑明确「非 box 元件的 pin number 不计分 ⇒
> rcl/gnd/vcc 在净图里相当于**不区分引脚**」。评分器默认已改成把非 box 元件的引脚号折成
> 通配符（`metrics.py: EvalConfig.net_pin_collapse_nonbox=True`），**本 README 与
> `docs/RESULTS.md` 里 2026-10 之前的数字都是旧口径**（`stage2_eval.py --net-literal`
> 可复现旧数）。同一批产物重打后，各条臂 **Δ 近似常数、名次不变**。
>
> ⚠️ **2026-10-09 的三个默认值改动**：① `STAGE1_WIRE_THIN_W` 1.6 → **2.6**（导线 F1
> 0.0978→0.1469 的纯 CPU 口径、真交付口径 0.1929→0.2317，见 §4.8）；② **`STAGE2_LIB_PINS`
> `0` → `1`**（官方元器件库的引脚补号/改名，交付口径官方 9 张 **+1.60 九张零回归**、
> 220 张拆分回归官方 20 张 **+1.3045** / 派生 200 张 +0.0609，见 §2.6）；③ 电源/地节点的净图口径
> （`STAGE2_NET_DROP_UNKEYED=0` + `STAGE2_NET_DROP_PWRONLY=1`，见 §0）。
> 三项都可一键关回去：`STAGE2_LIB_PINS=0` / `STAGE1_WIRE_THIN_W=1.6`。
>
> ⚠️ **2026-10-14**：库通道的**默认库**从 `cpntLibrary.json`（337 键）换成官方 box/IC 库
> `BoxICLibrary_200cases.json`（360 键，用户 m42932）—— 只换库文件的三臂 A/B 官方 9 张
> 68.7995 → **69.0883**、派生 220 张 33.2120 → **33.4364**（两库合并无额外收益，见 §2.6 尾）。

四项分数（官方 9 张，满分 30 / 25 / 35 / 10；当前默认 = `THIN_W=2.6` + 官方库开）：

| component | pin | net_hypergraph | net_line |
|---:|---:|---:|---:|
| 28.21 | **13.55** | **24.21** | **2.32** |

（四项加起来 **68.29**。同一批里关掉官方库的两臂是 28.09~28.32 / 12.26 / 23.81~23.89 / 2.32
= 66.60 ~ 66.79。旧记录 28.21 / 9.80 / 19.70 / 1.83 = 59.54 是「电源节点默认口径
翻转 + 导线旋钮改默认 + 官方库」之前的状态；口径只动净图那一项。）

**分会丢在哪**（实测归因，见 `docs/reports/引脚号回归与桩偏移.md`）：
`∅NNN` 电源/地符号的引脚号在 GT 里是**官方随机数**，占网络成员的 22%，数学上不可赢；
剩下的大头是**引脚号识别**（CRNN 单字混淆）与**导线端点精度**（逐条差 5~13 px，容差 5 px）。
这三件事决定了 66~68 分附近的天花板形状。

**全量 220 的阶段账**（`work\_full220`，交付口径；见 `docs/RESULTS.md` §2.3）：

| 阶段 | 用时 | 关键数字 |
|---|---:|---|
| 阶段一 | **416.5 s**（220/220，1.89 s/张） | 元件 43.1、文字框 249.5、引脚触点 154.1、网络 63.6（逐张平均）；分段 **wires 157.2 s** 最大 |
| 阶段二 | **347.1 s** | **9472 个元件里位号有值只有 3161 = 33.4%**；33899 个引脚槽位读出号 41.5%、读出名 40.2% |
| 阶段三 | 7.0 s | `-mode off` |
| VLM 补强 | 6 并发约 8~9 s/张 | 交付口径的 `comp` 分主要靠它 |

> **踩过的坑**：stage1 的 `_hint_wire_hue`（`stage1/wires.py:230`）曾经对**整张原图**发一次
> VLM 询问、用全局 600 s 超时、重试 4 次、异常还被吞成 `"ERR:..."` 字符串 ⇒ `0003`
> 单张 **315.73 s**（`timing` 里各段加起来只有 4.31 s）。修法 = 先缩到 1024 +
> 独立 30 s 超时 + 连续失败熔断，**315.73 s → 8.3 s**。细节见 `docs/DIAGNOSTICS.md`。

**按官方权重的可回收空间**（`WEIGHTS = {component 0.30, pin 0.25, net_hypergraph 0.35, net_line 0.10}`）：
**网络 +27.9 ＞ 引脚 +23.1 ＞ 元件 +11.4 ＞ 导线 +9.7** —— 网络与引脚合计 **60% 权重**，
而且是同一个根因：**IC 引脚我们没有**。且上帝键实验证明：**位号全修对也只值引脚 +0.0112**。

**引脚通道三分账**（官方 9 张交付产物；`_pinledger.py` / `_ocrstat.py` / `_ocroracle.py`，见 `docs/RESULTS.md` §2.4）：

| 项 | 数字 | 该去改哪里 |
|---|---:|---|
| GT 的 IC(box) 引脚 / 我们发出的 | **120 / 75** | —— |
| ├ **号读对了，名字没读到 ⇒ `stage2/assemble.py:759` 整条不发** | **23（51%）** | **要补的是「名字」，不是把闸门关掉**（见下） |
| ├ 号读错（`113→13`、`39→3`、`48→46`） | 22 | CRNN / 裁剪 / VLM |
| └ 附近 45px 内没有任何号槽位 | 6 | 检测 |
| 45px 内**有**槽位的 GT IC 引脚（按位置查） | **135/136 = 99%** | 几何不缺 ⇒ 分丢在**字符串**上 |
| **把号与名全部改对的上界**（`_ocroracle.py`） | **58.6304 → 63.2307 = +4.60** | **动权重/重训之前先看这个数** |

> **那 23 根不是「一行开关」就能拿回来的**（这条是量过之后才敢写的）：
> 把 `STAGE2_BOX_PIN_NEED_NAME=0`（读对号就发）做**离线单变量 A/B**
> （`tools/diag/_gateab.py`：直接用已落盘的 `runs/stage1` + `_assigned` 重跑 `build_doc`，
> 不需要 GPU/OCR）⇒ **官方 9 张 52.6510 → 52.4987（−0.15，判负）**；
> 机制是名字只能回退成引脚号，而官方档的 GT 引脚名 86.5% 是真信号名 ⇒
> `pin.name_mismatch` **16 → 41**、`pin.extra` 6 → 11，多出来的引脚全是 fp
> （换来的只有 `net_hypergraph` +0.030）。而 **全量 220 张上反而是 +0.10**（26.0962 → 26.1978）
> —— 因为派生档有一部分引脚名本来就是数字，回退成号也能对上。
> ⇒ **闸门保护的是精度**；要真拿那 23 根，必须**把名字读出来**
> （VLM 引脚通道 `VLM_PIN_CHANNEL=1` + `VLM_PIN_MERGE=1`，`_pinsweep` 在正确引脚点上
> 读号+名 0.826），而不是把闸门关掉。

> **两个评测集的引脚约定不同**（`_pinconv.py` 全 split 普查 —— **这一条推翻了本节早先写的相反结论**）：
> 官方 `10GT` 的 21 个 IC 里 **20 个（95.2%）的 GT 引脚键就是「1..N 顺序号」**（`10GT/3/U1` keys
> `[1..6]` names `BS/GND/FB/EN/IN/LX`），IC 引脚的 `pinname` **纯数字占 0.0%**，唯一例外是
> `10GT/1/U1`（n=26，物理号 `7/8/…/101/113/115/122`）；派生档 `200_train_cases` 恰好相反：
> 946 个 IC 里只有 **22.6%** 是顺序键，**50.9% 的 IC 引脚键 >8**（物理封装号）、`pinname`
> 纯数字占 **20.1%**（`10_GTcase` 居中：14 个 IC、28.6% 顺序键）。⇒ **官方档的号是小顺序号、
> 缺口全在「名字」上；派生档的号是大物理号，且名字常常就是号本身。**
> 另一面（IC 引脚口径，本轮实测）：我们产出的 `pinname` 是**空或纯数字**的比例 —— 官方 9 张
> **2.5%**、派生 220 张 **17.5%**，而 GT 派生档自己是 **20.1%** ⇒ 派生档上「号当名」本来就接近
> GT 的习惯，官方档却几乎全靠真信号名，同一个「名」通道因此在两档上必然一正一负。
> （`_pinname_stat.py` 报的 78.2% 是**全部元件**的预测引脚、且读的是旧产物
> `work\_submit_out220r36`，口径不同别混用。）
> 这直接定了引脚通道的出厂配置：官方 9 张**只补号 +1.95 / 只补名 +1.32 / 全开 +2.88**；
> 全量 220 张按 split 拆开是**三档**：官方 `10GT` 10 张 **+2.59**、`10_GTcase` 10 张 **+0.76**、
> `200_train_cases` 200 张 **−0.25**（合计才 −0.07，负号只在自动命名那一档）；而「只补号」
> 在官方 9 张上**没有任何一张变差**（`docs/RESULTS.md` §2.5）。
>
> 截断的成因已用框宽定位：`39→3` 的框宽 **24 px**、`113→13` **34 px**、`115→15` **35 px**，
> 都够装下原字 ⇒ **是 CRNN 少读字符**，不是检测框切字；另一小撮（`GND→D` 10 px、
> `IS→S` 8 px）才是检测框只框住一个字符。
> 而识别器的瓶颈在**裁剪分布**：真值紧框 exact **0.843** / stage1 紧框 **0.197** / 外扩 +3 → **0.654**。

### 1.1 LLM 轮到底值多少（这是「LLM 有没有用」的第一个有前提答案）

| 口径（官方 9 张） | LLM 关 | LLM 开 | LLM 开 + **上下文块** |
|---|---:|---:|---:|
| **交付口径**（VLM `auto` + `-fusion on`，全默认值） | **58.63** | **58.63** | **58.63** |

> ⚠️ **2026-10-11 起交付 CRNN 权重换了**：`weights/textrec_best.pt` 从 `ocr_real`(e51) 换成
> `ocr_mix`(e57)。上表与本文其它 58.63/58.6304 都是**旧权重**下量的；阶段级与管线内文字两项都已确认
> 变好（stage2 官方 9 张 57.26→62.80、派生 80 张 23.81→26.14、全 220 张文字 exact 38.2%→40.9%、
> 无 VLM 的 fusion 层 220 张 +0.19），**交付口径（VLM 开）的 220 张 A/B 待补**（排在 ②b 训练后），
> 复测后本文头条与四项表要一起改。见 `docs/RESULTS.md` §2.0、`docs/DIAGNOSTICS.md`。
| VLM 关（`--no-vlm`，同一批图） | 52.48 | 52.48 | **53.27（+0.79）** |

- 交付口径三臂**逐位相同**（逐张 `1=47.24 2=53.50 3=68.25 4=66.60 5=52.20 7=62.12 8=62.71 9=57.13 10=57.93`；
  三项 `comp 0.9403 / pin 0.3920 / netH 0.5355` 也不动，只有 `netL 0.1876`）。
  已用 `python tools/diag/_jrep.py --cmp <三个产物目录>` 证明三臂**确实是三套配置**
  （`reg9` 有 judge 块、无 `ctx` 键；`reg9ctx` `ctx: True`；`reg9nollm` **没有 judge 块**）。
- ⇒ **LLM 轮是「没有 VLM 时的补强」，不是交付口径的增量**；而它要 **+1.8 h / 220 张**。
  在 VLM 关的口径下它确实赚（**+0.79**，无一张变差），赚的是 case 5 / 7 / 8。
- **上下文块**（`FUSION_JUDGE_CTX`，默认开）= 把「元件类型 / 同件别的字段读数 / 邻近元件 /
  图上已定位号 / 已判可疑的原因」拼进提示词，并要求模型回一句 `依据=<…>`。
  这就是「LLM 没有上下文环境」那句批评的代码级修复。
- 要省时间：`$env:FUSION_LLM='0'`（**默认开是 owner 的决定，不要改默认值**）。

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

### 3.1 ★ 你必须自己填的东西（仓库里不含任何密钥）

克隆下来就能跑判题机（§4.3），但要跑出 84.95 那条链路，**大模型 API key 必须你自己提供** ——
仓库里没有、也不会有。漏填的失败方式很安静：VLM 通道整体跳过，官方 10 张从 **84.95 掉到 ~70**，
日志里只有一行「跳过」。

| 要填的 | 变量 / 位置 | 怎么填 | 漏填后果 |
| --- | --- | --- | --- |
| 大模型 API key | `LLM_API_KEY`（`vlm/client.py:62-102` 四级回退） | ① 环境变量 `LLM_API_KEY`；② `LLM_API_KEY_FILE=<文件路径>`；③ 仓库根 `eda-pcb/.lmstudio_key`（已 gitignore）；④ `~/.lmstudio/api_key`。文件内容可裸 token，也可写 `LLM_API_KEY=...` 一行 | VLM 通道静默降级：84.95 → ~70 |
| 接口地址 | `LLM_BASE_URL`，默认 `http://127.0.0.1:1234/v1`（`vlm/client.py:59`） | 用官方 API 时设 `https://dashscope.aliyuncs.com/compatible-mode/v1` | 连不上端点 → 同上 |
| 模型名 | `LLM_MODEL`，默认 `qwen3.5-9b`（`vlm/client.py:106`） | 改成你账号下可用的模型（例如平台给的 `qwen3.8-flash`） | 模型不存在 → 同上 |
| scipy | `pip install "scipy>=1.11"` | 判题机 `schematic_parser/metrics.py:42` 的三处匈牙利匹配要用；`requirements.txt` 已列 | 判题机 import 失败 |
| torch | 按 CUDA 版本自选 index | `requirements.txt` **故意不列** torch（见文件头注释） | 装错 CUDA → 检测链路跑不动 |

**已经随仓库提交、不用你准备**：三个权重（`weights/pcb43_v2_from_best.pt` 38.8 MB、
`weights/textrec_best.pt` 29.6 MB、`yolo_part/weights/yolo11m.pt` 38.8 MB，合计约 68 MB）与
**221 个 GT 标注 json**（220 个用例 + `10GT/type_list.md`，11.5 MB）。
**没有入库**：GT 的 PNG、训练数据（见 §七）。

两条最小验证（在 `eda-pcb/` 目录下跑）：

```powershell
# ① key 到底读到了没有：期望打印的不是 none、不是空串
python -X utf8 -c "from vlm import client as c; print(c.KEY_SOURCE, '|', c.BASE, '|', c.MODEL)"
# ② 判题机环境是否完整（220 例自检，不需要 key）
python -X utf8 -m schematic_parser.evaluate --self-test
```

纪律：**不要把 key 写进任何被 git 跟踪的文件**（`.lmstudio_key`、`*.key` 已在 `.gitignore`，
但别把 key 贴进 README、脚本或 Dockerfile）。容器里用
`docker run -e LLM_API_KEY=... -e DASHSCOPE_API_KEY=...` 注入：官方示例注入的是
`DASHSCOPE_API_KEY`，本仓库代码读的是 `LLM_API_KEY`，`packaging/run.sh` 已加别名兜底
（只在 `LLM_API_KEY` 为空时映射，不覆盖显式设置）。

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
  -vlm-pin on `               # VLM 引脚通道（**默认开**：官方 9 张交付口径 +2.88）
  -vlm-pin-name off `         # 只补引脚「号」不补「名」（更保守的一档，官方 9 张 +1.95 零回归）
  -fusion on `                # 融合仲裁层（默认开）
  -gpu-frac 0.45              # 给本地 LLM 留显存
```

> 引脚通道的两个开关见 `docs/RESULTS.md` §2.5：默认「号+名」全开（官方 10GT 9 张上
> 58.6304 → 61.5112）；`-vlm-pin-name off` 是**防回归**档（编号那半在官方 9 张上没有
> 任何一张变差）；`-vlm-pin off` 则完全回到 58.6304 的旧行为。

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

### 4.3 官方口径自评（判题机）

```powershell
# 官方 10 张 GT 一对一打分 —— 只需要 numpy + scipy，不要 GPU / 图片 / 权重
python -X utf8 -m schematic_parser.evaluate --pred submission\final_results --split 10GT

# 逐案 JSON 落盘
python -X utf8 -m schematic_parser.evaluate --pred <目录> --split 10GT --out scores.json

# 判题机自检（不需要预测）
python -X utf8 -m schematic_parser.evaluate --self-test

# 端到端「跑 + 判分」（tools/eval/fast.py，带结果归档）
python tools\eval\fast.py --cases 1,2,3,4,5,7,8,9,10 --save off9_base
python tools\eval\fast.py --list                # 列出可用 case
```

`fast.py` 打开关请用 `$env:X='1'`，**不要**用 `--extra`（该参数会把多词参数按字符拆开）。

> 完整说明书（口径开关、已知性质、参考分、给队友的最短路径）见 **`docs/JUDGE.md`**。
> GT 标注已随仓库提交（只有 `*.json`，没有 PNG），clone 下来就能直接跑分。

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
| LLM 提示词补**上下文块**（类型/同件别的字段/邻近元件/已定位号/已判可疑 + 要求回 `依据=`） | 「LLM 没有上下文环境」是**代码事实**：原提示词只有裁片 + A/B 两个字符串，且「给依据」这一件事从未做过。补上后 VLM 关的口径 **52.48 → 53.27**，且 `ctx=1` 时 **12/12** 行都回了 `依据=`、`ctx=0` 时 0/12（证明它真的进了提示词） |
| **整表裁决（contact sheet）默认关**（`FUSION_JUDGE_SHEET_MIN=9999`） | 入口原来是 `if mode == "sheet" and SHEET_ON:` 而 `MODE` 默认 `auto` ⇒ 那条路**一次都没跑过**。接通后 4 张 A/B：**55.32 vs 逐题 59.62（−1.07）**，且更慢（160.5 s vs 83.6 s） |
| `judge` / `rescan` **各自包 `try`** | 原来 `LJ.judge` 不在任何 try 里，一抛异常该 case **一份融合产物都没有**（`AU.run` 早就包了） |
| 收敛判据改用**本轮结束**的题面（`FUSION_CONV_EXACT=1`） | 旧判据看的是上一轮的 `q_first`，「第 2 轮到底跑不跑」不可复现；现在报告里有 `n_q_next` |
| 位号取景框取**最高分**那个（`FUSION_REGION_TOPK=1`） | `stage2` 从不保证 `desig_slots` 按分数降序，`boxed[0]` 只是列表顺序 |
| **改完先证明分支执行了，再解释分差** | boxscan 第一版有差（52.48 → 52.47），但把 `_fusion_report.json` 的 `notes` 打出来才看到每张只建了 1 个槽位 —— 分差根本不是「boxscan 整体」的效果。详见 `docs/DIAGNOSTICS.md` |
| **stage1 的 VLM 色相询问必须有界**（`STAGE1_WIRE_HUE_MAXSIDE=1024` / `_TIMEOUT=30` / `_MAXFAIL=3`） | 原来**无条件对整图**发一次询问、用全局 600 s 超时、重试 4 次，且 `vlm/client.py` 把异常吞成 `"ERR:..."` **字符串正常返回** ⇒ 失败不报错、只表现为「这张图莫名多了 5 分钟」。`0003` 实测 **315.73 s**（`timing` 各段之和只有 4.31 s）⇒ 修后 **8.3 s**；220 张 stage1 一共 **416.5 s** |
| **框外引脚名走 late pass**（`STAGE2_PINNAME_OUTSIDE`，默认 0 = 关）：引脚号那一轮**跑完之后**，只往还空着的名字槽里补框外的名字 | GT 的 IC 引脚 `point` **100% 在元件框之外**，而 `pin_name` 角色有 `want_inside` 硬门槛 ⇒ 框外的名字根本进不了候选池。第一版「放宽限制、仍在引脚号之前跑」**判负**（19 张 30.81 → 30.09，tp 引脚 44 → 42）；改动顺序后对引脚号那一轮**零影响**。闸门只压「适配度×置信度」不压距离（`_score` 的 `-0.035*d` 会让 30px 外的合法名字算成负分），且**拒绝纯数字文本**（GT 的引脚名有 13.5% 本来就是数字，`assemble.py:761` 的兜底已经能填对） |
| **要动识别器/权重之前，先量「全改对」的上界**（`tools/diag/_ocroracle.py`） | 引脚这一路全对只值 **+4.60**（官方 9 张 58.6304 → 63.2307），而且其中 **23/45 的缺口根本不是识别问题**（号读对了、名字没读到，被 `stage2/assemble.py:759` 的 `BOX_PIN_NEED_NAME` 整条丢掉）。先算天花板，再决定投不投训练 |
| **同一批缺号要按「位置」查，不能按「号」查** | 按号查只能找到 94/136 个槽位（读错的被算成「没槽位」），按位置查是 **135/136 = 99%** ⇒ 几何不缺、分丢在字符串上。绑定方式换了结论就换了（`tools/diag/_ocrstat.py` vs `_pinnamepair.py`） |

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

> `weights/textrec_best.pt` = `work/ocr_mix/runs/best.pt`（**epoch 57、`val_cer` 0.1698、md5 `7559096C9979`**，
> 2026-10-11 换入；此前是 `work/ocr_real/runs/best.pt`，e51、`val_cer` 0.1871、md5 `0CD23402952D`，
> 旧文件留在 `work/weights_backup/textrec_best.real_e51.pt`）。

`assets.py` 按**相对路径**解析，可用环境变量覆盖：`YOLO_WEIGHTS` / `STAGE2_CKPT`。
源码里**没有功能性绝对路径**（仅少数 docstring 示例保留本机路径）。

### 数据集（图片不在仓库内，**GT 标注在**）

图片体积过大已 gitignore；**GT 标注 `*.json` 随仓库提交**（判题机只读标注，不读图）：

| 名称 | 是否入库 | 用途 |
|---|---|---|
| `yolo_part/Dataset(waitforpreprocess)/10GT/**/*.json` | ✅ 入库 | 10 张官方公开 GT（评测基准）+ `type_list.md`（34 类） |
| `yolo_part/Dataset(waitforpreprocess)/10_GTcase/**/*.json` | ✅ 入库 | 10 张 |
| `yolo_part/Dataset(waitforpreprocess)/200_train_cases/**/*.json` | ✅ 入库 | 200 张派生 |
| `yolo_part/Dataset(waitforpreprocess)/**/*.png` | ❌ 未入库 | 原始图（跑流水线才需要） |
| `yolo_part/training_data/` | ❌ 未入库 | YOLO 元件检测训练集（41 类，`data.yaml` 已入库） |
| `textrec/data/manifest.jsonl` | ❌ 未入库 | CRNN 训练清单（85 MB） |

只想跑判题机的话，把图片准备到 `yolo_part/Dataset(waitforpreprocess)/<split>/<case>/<case>.png` 即可
（也可以 `--data-root` 指向别处）。

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
7. **LLM 轮在交付口径下是纯开销**：官方 9 张的 **LLM 关 / LLM 开 / LLM 开+上下文 三臂全是 58.63
   且逐位相同**（已证明三臂确实是三套配置），而墙钟要 **+1.8 h / 220 张**。
   它在**没有 VLM** 时才赚（52.48 → 53.27，+0.79）。保留默认开启是 **owner 的决定**
   （`submission/run.py` 注释），不是实测结论；要省时间用 `FUSION_LLM=0`。
8. **上下文的「冲突内容」在 round 1 是空的**：`judge` 跑在 `pool.apply()` / `plausible.check()`
   **之前**，而 `solve` 已经把重号解掉了 ⇒ `n_ctx_slot: 0`（实测）。LLM 实际吃到的是
   类型 / 同件别的字段 / 邻近元件 / 已定位号这四样。
9. **两份同名脚本会让改动看起来「没生效」**：`work\flow\_wfbullets.py` 才是构建真正 import 的那份
   （`_wfbuild.py` 做 `sys.path.insert(0, FLOW)`），`eda-pcb\tools\diag\` 那份是仓库快照；
   `_wfhead.py` 则**只有在 `work\tools\diag\` 下**才解析得到 `work\flow\workflow.html`。
   改完必须两处同步。
10. **IC 引脚是最大的结构性缺口，且不在位号上。** 官方 9 张：GT 的 IC(box) 引脚 **120 根我们只发 75**，
    缺的 45 = **号已读对但名字没读到、被 `stage2/assemble.py:759` 整条丢掉 23（51%）** + 号读错 22
    + 附近没槽位 6。**但那 23 根不能靠关闸门拿回来**（`STAGE2_BOX_PIN_NEED_NAME=0` 离线 A/B：
    官方 9 张 −0.15 判负，全量 220 +0.10）—— 闸门保护的是精度，要补的是**名字**。
    派生档更极端：2220 根 GT IC 引脚里 **64.9%** 在「配对上的元件键」下一根都没有，
    有可比的 780 根里名字逐字符相等的只 **3.1%**（**这些比例只适用于派生档**，官方档的引脚键是纯数字）。
    上帝键实验证明位号全修对也只值引脚 **+0.0112**（元件那 +0.1310 是强行配对的作弊分）。
11. **220 张全量是纯诊断口径**：题目约束是「最多 50 样例 / 总时间 2 小时」，
    不是 220 张 / 2 小时。全量跑一轮的墙钟（含 VLM 补强）远超 2 小时。
12. **「把引脚号与名全部读对」的上界只有 +4.60 分**（官方 9 张 58.6304 → 63.2307，
    `tools/diag/_ocroracle.py` 的上帝视角改写）。⇒ 重训识别器 / 换模型之前先拿这个数当预算上限；
    而这条路现在的瓶颈在**裁剪分布**（真值紧框 exact 0.843 / stage1 紧框 0.197 / 外扩 +3 → 0.654），
    不在权重上。另一半缺口（23/45）**跟识别无关**，是一行开关的事。

---

## 九、文档索引

| 文档 | 内容 |
|---|---|
| `docs/ARCHITECTURE.md` | 分层架构、模块职责、四条关键约定 |
| `docs/JUDGE.md` | **判题机说明书**：怎么用 `schematic_parser.evaluate` 打分、口径开关、已知性质、参考分、给队友的最短路径 |
| `docs/RESULTS.md` | 分数轨迹、全部 A/B 表、负结果归档 |
| `docs/DIAGNOSTICS.md` | 怎么做测量：诊断尺子、五步工作流、复现命令 |
| `docs/KNOBS.md` | **全部 458 个环境开关**（自动生成，带默认值与 `文件:行号`） |
| `docs/TOOLS.md` | **全部 720 个工具脚本**按用途分组（自动生成） |
| `docs/TRAINING.md` | **怎么重训**：数据集重建（`trainset_v4[_merged]` 的开关与默认值）、YOLO 训练、CRNN 混合清单与热启动微调、训练后必跑的两把尺子 |
| `docs/workflow.html` | 交互式流程图（**37 节点 / 86 连线**，含每个节点的实测笔记；判题机节点已画进去） |
| `docs/PACKAGING.md` | **怎么打包 / 上传 GitHub / 把判题机单独交给队友**（含上传前自检清单与密钥红线） |
| `docs/导线workflow.html` | **导线识别专页**（14 阶段图 + 「容差 vs tp」主图 + 网级配对口径 + 空网抢配对 + 6 张爆 0），由 `tools/diag/_wirepage.py --write-docs` 生成 |
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
