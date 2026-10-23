# 重训运行手册（RETRAIN）—— 用外部新标注集微调 CRNN

> 面向场景：明天拿到一份**新的人工标注集**，当天就要把它变成 CRNN 权重并过验收闸门。
> 本文只讲「明天要敲什么、期望看到什么、会怎么翻车」；识别器结构、损失、序列化细节见
> `textrec/README.md`，历史实测见 `docs/RESULTS.md`、`docs/TRAINING.md`。
> 所有数字都是本机实测（2026-11，CPU-only），不是估计。

## 0. 出发状态（先确认这三件事）

| 项 | 现值 | 证据 |
| --- | --- | --- |
| 交付权重 | `eda-pcb/weights/textrec_best.pt`（31 MB） | 存在，`LastWriteTime 2026/10/16` |
| ckpt 内 `epoch` | **69** | `torch.load(...)["epoch"] == 69` |
| ckpt 自带字表 | **71 字**：` %+-./0-9A-Z_a-zΩ∅` | `len(ck["charset"]) == 71` |
| ckpt 验证读数 | `val_cer 0.15355` / `val_exact 0.73677` | ckpt 字段 |
| 模型输入 | `img_h 32 / max_w 384 / width_reduction 8` | `ck["config"]` |

权重解析顺序（`assets.resolve` ← `stage2/text.py:28 _default_ckpt`）：
`$env:STAGE2_CKPT` → `eda-pcb/weights/textrec_best.pt`（随仓库） → `EDA/work/ocr_mix/runs/best.pt`（本机训练产物）。
出厂权重 epoch 是 **69**，这条决定了 §3 的 `--epochs` 写法。

## 1. 数据格式：今天能吃什么，要向厂商要什么

### 1.1 三条入口（都已经在跑）

| 入口 | 标签来源 | 有逐条 bbox？ | 产出 | 用法 |
| --- | --- | --- | --- | --- |
| 合成 | 渲染器自己知道每个字符串的框 | ✅ 精确 | `textrec/manifest.py::build_synth` | `python -X utf8 -m textrec.manifest` |
| Dataset300 | `*_CVJsonStd.json` 的 `texts[].location` | ✅ 精确 | `build_real` | 同上（`--include-real` 才进训练） |
| **页面级 GT**（明天最可能给的就是这个） | `components[键].bbox` + `pins[键].pin_N.point` | ❌ 只有元件框 / 引脚点 | `tools/dev/build_manifest_from_gt.py` | §2 |
| **框级 JSONL**（首选） | 厂商逐条给的框 | ✅ 精确 | 同上 `--boxes` | §2.3 |

### 1.2 向标注厂商要什么（可直接复制给厂商）

要**框级**，不要页面级。每一行一个文字实例，UTF-8 JSONL（一行一个 JSON 对象）：

```json
{"case": "0007", "bbox": [412, 88, 486, 104], "text": "10uF", "role": "value"}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `case` | string | 页/案编号，和图片文件同名（`0007.png`） |
| `bbox` | `[x0,y0,x1,y1]` | 图像像素、左上原点、浮点或整数；就是**这个字符串自己**的外接框 |
| `text` | string | 框里那个字符串本身，原样（不要补单位、不要猜）；读不清就整条不要出 |
| `role` | enum | `desig` / `value` / `pin_num` / `other`（`other` 含网络名、引脚名、注释） |
| `image` | string（可选） | 图片绝对/相对路径；缺省按 `case` 到图片根目录里找 `<case>.png` |

前置条件：**图片必须和给人看的截图完全同一张（同分辨率、同裁剪）**。框是图像像素坐标，
一旦图片被重新渲染/缩放，框就全废了。

页面级 GT（只有 `components[].bbox` + 字符串、没有每个字符串自己的框）也能吃，但会退化成
**弱配对**（§2.2），实测精度不足以当训练标签，只能用来验通路。

### 1.3 页面级 GT 的实测损失（为什么非要框级）

用 `tools/dev/build_manifest_from_gt.py` 在本仓库 10GT（10 页 KiCad 原理图）上跑出来的：

| `--radius` | 配上 / GT 字符串 | 占比 |
| ---: | ---: | ---: |
| 40（默认） | 93 / 865 | **10.8%** |
| 80 | 151 / 865 | 17.5% |
| 120 | 169 / 865 | 19.5% |
| 200 | 175 / 865 | 20.2% |

丢因（`--radius 40`）：无框在半径内 **648**、框侧并列 29、GT 侧并列 95。

更关键的是**配上的是不是对的**。用一个不依赖 OCR 的代理指标
`(框宽/框高) / len(text)`（真框清单里它应当≈单字符宽高比，约 0.52）做对照：

| 清单 | 中位 `(w/h)/len` | 结论 |
| --- | ---: | --- |
| `textrec/data/manifest.jsonl`（合成，真框） | **0.524** | 基准 |
| `textrec/data/manifest_ho.jsonl`（真框） | **0.542** | 基准 |
| `work/_real_only.jsonl`（真框） | **0.524** | 基准 |
| `work/_man_10gt.jsonl`（本次弱配对，10 页） | **1.348** | **2.5× 偏离 ⇒ 多数配对是错的** |
| `work/_man_pool1.jsonl`（弱配对，池页 `0001`） | **0.667** | 1.3× 偏离 —— 池页更疏、字号分层，但**仍不等于真框** |

原因很直接（诊断图 `work/_diag_zoom_5.png` 可见）：页面级 GT 的锚点是「元件符号 bbox 的中心」
与「引脚连接点」，而位号/值的**字形框**普遍在旁边 20–130 px；再加同级 60+ 个引脚号互相抢框，
「唯一最近 + 4 px 并列即弃」的规则几乎不可能解对。

> 一句话：**页面级 GT 只够验通路，不够训模型。** 明天要么拿到框级标签，要么先按 §2.2 跑出
> 统计、把这张表给厂商看。

## 2. 转换：GT → CRNN 清单

### 2.1 今天的 10GT（自检命令，两条都要跑）

```powershell
cd Y:\workspace\EDA
python -X utf8 eda-pcb\tools\dev\build_manifest_from_gt.py --gt "eda-pcb\yolo_part\Dataset(waitforpreprocess)\10GT" --out work\_man_10gt.jsonl --stats
python -X utf8 eda-pcb\tools\dev\build_manifest_from_gt.py --gt "eda-pcb\yolo_part\Dataset(waitforpreprocess)\10GT" --out work\_man_10gt.jsonl
```

`--stats` 只打印、不落盘；去掉它才写 `--out`。期望输出（实测）：

```
[gt] case 成功 10 / 跳过 0
[gt] split: {'val': 44, 'train': 49}
  role               gt   paired   med_h px
  designator        146        4       10.0
  value             225       10       10.0
  pin_number        494       79       10.0
[gt] 丢弃：无框在半径内 648 / 框侧并列 29 / GT 侧并列 95
[gt] 记录 93 条（10.8% 的 GT 字符串被配上）
[manifest] 写出 93 行 -> work\_man_10gt.jsonl
```

### 2.2 新目录（明天的主路径，页面级 GT）

```powershell
# 目录约定：<根>\<case>\  里面一张图 + 一个标签 json
#   标签优先级：<case>*target_named.json  ->  <case>.json  ->  目录里唯一的 .json
#   图片名**不必**等于目录名：`200_train_cases\0001\0001-KiCad.png` 这种也能被认出来
python -X utf8 eda-pcb\tools\dev\build_manifest_from_gt.py `
    --gt "D:\newset" --out work\_man_new.jsonl --radius 120
```

池子格式的自检（单页 340 条 GT -> 99 条记录 = **29.1%**，`med_h` 9/13/8 px 按角色分层，
说明池页比 10GT 稀疏、字号分级更清楚）：

```powershell
python -X utf8 eda-pcb\tools\dev\build_manifest_from_gt.py `
    --gt "eda-pcb\yolo_part\Dataset(waitforpreprocess)\200_train_cases" `
    --out work\_man_pool1.jsonl --cases 0001 --stats
```

| 参数 | 默认 | 什么时候改 |
| --- | --- | --- |
| `--gt` | — | 可给多个根目录；也接受单个 case 目录 |
| `--out` | `work/_manifest_gt.jsonl` | 训练入口用绝对/相对仓库路径都行 |
| `--radius` | 40 px | 字号大/元件符号离文字远时调大（§1.3 表：40→200 命中 10.8%→20.2%，之后再大只涨并列丢弃） |
| `--cases "1,2,3"` | 全部 | 先小批试跑 |
| `--limit-cases N` | 0（不限） | 同上 |
| `--val-every N` | 5 | 按 case 名 md5 抽 1/N 进 `val`（**按 case 切，不按裁剪切**，防止同页泄漏） |
| `--split-mode real` | `hash` | 全部 `split="real"`（= `build_real` 口径，只能用于 eval，`train.py` 不吃） |
| `--stats` | 关 | 只看统计 |

记录格式与 `textrec/manifest.py::build_real` **逐字段一致**（8 个字段），一行实例：

```json
{"image": "Y:\\workspace\\EDA\\eda-pcb\\yolo_part\\Dataset(waitforpreprocess)\\10GT\\1\\1.png", "case": "1", "src": "real", "split": "val", "role": "pin_number", "role_id": 3, "text": "2", "bbox": [983.0, 223.0, 1030.0, 241.0]}
```

`role` 用 `textrec/config.py::ROLES = ("designator","value","pin_name","pin_number","net_label","note")`，
`role_id` 是它的下标；**别写 `desig`/`pin_num`**——不在词表里，会静默落到兜底 `note`，把角色嵌入训歪。

### 2.3 框级 JSONL（厂商先给这个，就走这条）

```powershell
python -X utf8 eda-pcb\tools\dev\build_manifest_from_gt.py `
    --boxes "D:\vendor\labels.jsonl" --gt "D:\newset" --out work\_man_vendor.jsonl
```

`--boxes` 给了就不走弱配对：逐条信任厂商的框，`role` 经 `ROLE_CANON` 归一
（`pin_num→pin_number`、`other→note`…）；找不到图片或 bbox 非法的行会计数后丢弃。

## 3. 明天在 GPU 上训练

### 3.1 命令（照抄，注意 cwd 与 `-m`）

```powershell
cd Y:\workspace\EDA\eda-pcb
python -X utf8 -u -m textrec.train `
    --manifest ..\work\_man_new.jsonl `
    --resume weights\textrec_best.pt `
    --epochs 74 --save-every 1 `
    --batch 64 --workers 6 `
    --out ..\work\crnn_retrain_new
```

* **必须 `cd eda-pcb` 且用 `-m textrec.train`**。`python eda-pcb\textrec\train.py` 会直接
  `ImportError: attempted relative import with no known parent package`（`train.py:27 from . import config`）。
  这不是本次改出来的问题，是包结构本来的样子。
* **`--epochs` 必须大于 ckpt 里的 `epoch`**（今天 = 69）。`train.py:275-281` 用
  `start_epoch = int(ck.get("epoch", 0))`，`range(69, 74)` 才非空；写 `--epochs 40` 会一轮都不跑就以 0 退出。
* **`--batch` 必须小于训练条数**。`drop_last=True`（`train.py:256`），条数不足一个 batch 就是 0 步：
  实测 49 条 train + 默认 batch 得到 `1 轮 × 0 步`，`loss 0.0000`。新集只有几百条时给 `--batch 8/16/32`。
  旁证：真跑了 6 步的 `last.pt` 是 93 MB，0 步的那个只有 31 MB（`opt.state_dict()` 为空）。
* 存档语义（`train.py:344-361`）：`last.pt` **每轮无条件写**（带优化器状态，续训用它）；
  `best.pt` 只在 val CER 改善时写；`--save-every N` 额外每 N 轮存 `ckpt_e%03d.pt` 快照。
  所以 `--save-every 1` 是「留痕」，不是「开存档」。
* 产出：`--out` 目录下 `best.pt` / `last.pt` / `history.json`（每轮 loss、CER、exact）。

### 3.2 字表（最容易翻车的一环）

* `--charset` **留空 + 有 `--resume`** ⇒ 自动用 ckpt 自带字表（71 字，`train.py:159-173`），这是唯一安全的默认。
* 显式 `--charset` 且与 ckpt 不符 ⇒ 直接拒绝（`train.py:155-167`），因为输出层对不上。
* 仓库默认 `textrec/data/charset.json` 是 **67 字**（缺 `% / q v w Ω` 等），**只该用于从零训练**。
  真要扩表：新表必须 = 旧 71 字 ∪ 新增字符，且只能从零训（或改输出层），并同步
  `stage2/text.py` 侧的解码表。
* 混语料（真实+合成）用 `tools/dev/mix_crnn_manifest.py --synth <合成清单> --real <新清单> --out <目录>
  --charset-from <ckpt>`，它按角色配额抽（`--pin-per-len/--designator/--value/...`）。

### 3.3 其它开关

| 开关 | 默认 | 取舍 |
| --- | --- | --- |
| `--pin-num-balance equal` | `off` | 线上失败模式是「3 位号读成 2 位」（`113→13`），合成里 2 位占 79%；若新集 3 位偏多再开 |
| `--include-real` | 关 | 清单里 `split="real"` 的记录默认只做验证不进训练 |
| `--limit / --val-limit` | 0 | 调试时截断 |
| `--smoke` | 关 | 强制 `epochs=1, limit=2000, val_limit=500, batch=min(batch,64), workers=min(workers,2)` |

### 3.4 本机实测：单域微调会灾难性遗忘（2026-11，CPU，`--resume weights/textrec_best.pt`）

语料：`work/_man_new_ocr.jsonl` = 新训练集 48 页收的字（8665 条，`tools/dev/harvest_labels.py`，
见 `RESULTS.md` §5.355）；`work/_real_only.jsonl` = 官方真实语料 6642 条。

| 臂 | 语料 | lr / 轮 | 新域 val exact | 官方真实 位号 exact | GT10 `protocol4` |
| --- | --- | --- | ---: | ---: | ---: |
| 交付权重 | — | — | 0.8142 | 0.9375 | 84.43 |
| new1 | 只新集 | 1e-4 / 1 | **0.9042** | **0.7934** | **70.39** |
| new5 | 只新集 | 1e-5 / 1 | 0.8793 | 0.9045 | — |
| mix1 | 真实+新集 | 1e-5 / 1 | 0.8812 | 0.9253 | 85.05 |
| mix3 | 真实+新集 | 2e-5 / +3 | 0.9080 | **0.9462** | 84.81 |

* **只喂新集＝把旧域挤掉**：新域 val 涨 9 个点，官方真实位号掉 14 个点，GT10 从 84.43 塌到 70.39。
  这不是「学坏了」，是纯粹的遗忘 —— 混训（mix1）用同样的 lr 只掉 1.2 个点，GT10 反而 +0.62。
* **官方 200 案（`200_train_cases`）不要拿来收字**：那批 GT 本身标注错误多，`near` 层
  （编辑距离 ≤2）会把「错位的真值串」扣到相邻文字框上，等于教模型错误的邻域关系；实测 mix2
  （真实+新集+官方 200 案 3435 条）虽然 GT10 到 85.74，但 comp 轴从 mix1 的 0.9335 掉到 0.9133，
  正是这个机理的可观测痕迹，故不作为交付配方。
* **安全的旧域锚点＝合成语料**（`work/textrec_merged.jsonl` 里 `src=="synth"`，36.6 万条）：
  它由渲染器生成、标签无歧义，用它代替官方 200 案来防遗忘。

## 4. 换权重与验收闸门

### 4.1 三个环境变量（`stage2/text.py`）

| 变量 | 作用 | 代码 |
| --- | --- | --- |
| `STAGE2_CKPT` | 主识别器权重 | `_default_ckpt()` L28-49 |
| `STAGE2_CKPT_DESIG`（别名 `STAGE2_CKPT_ROLES_CKPT`） | 角色专用第二把权重；未设或文件不存在 ⇒ 不启用 | L505-513 |
| `STAGE2_CKPT_ROLES` | 哪些角色走第二把权重，逗号分隔，默认 `designator` | L551 |

```powershell
$env:STAGE2_CKPT      = "Y:\workspace\EDA\work\crnn_retrain_new\best.pt"
# 想「位号钉旧权重、其余用新权重」才设下面两个；只换主权重时两个都别设
# $env:STAGE2_CKPT_DESIG = "Y:\workspace\EDA\eda-pcb\weights\textrec_best.pt"
# $env:STAGE2_CKPT_ROLES = "designator"
```

三者互不修改文件、只读环境变量；跑完记得 `Remove-Item Env:STAGE2_CKPT`，否则同会话里所有
eval/提交都会带着它。

### 4.2 闸门 A：官方 10 张，`protocol4`（提交尺，5 px）

```powershell
cd Y:\workspace\EDA
$env:CUDA_VISIBLE_DEVICES=''; $env:FUSION_LLM='0'
python -X utf8 eda-pcb\tools\eval\fast.py --out <tag> --engine protocol4 --no-vlm --cases "1,2,3,4,5,6,7,8,9,10"
```

今天基线：**88.73**（component 0.9505 / pin 0.8381 / netH 0.9486 / line 0.6060；2026-10-23 换权重后，旧基线 84.43）。
判据：总分与四个分项都不低于基线 −0.3；`component`/`pin` 掉分即视为识别器没变现，回退。

### 4.3 闸门 B：池 50 案，`protocol3`（历史对齐尺，20 px）

`fast.py` 只吃逗号列表（`--cases`），要把 `work/_eval50.txt` 拼进来：

```powershell
cd Y:\workspace\EDA
$c = ((Get-Content work\_eval50.txt) | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() }) -join ','
$env:CUDA_VISIBLE_DEVICES=''; $env:FUSION_LLM='0'
python -X utf8 eda-pcb\tools\eval\fast.py --out <tag>_50 --engine protocol3 --no-vlm --cases $c
```

今天基线：**36.84**（0.5808 / 0.1999 / 0.3620 / 0.1851，2026-10-22 口径）。换权重后池子 36.36 / 33.97（protocol3/protocol4），与旧基线配对 t −0.90，属噪声内微负。
两把尺**不可混用**（同一批预测换尺总分差 4.96），对外只报 `protocol4`。

## 5. 风险清单

| # | 风险 | 现象 / 处置 |
| --- | --- | --- |
| 1 | **字表漂移** | ckpt 71 字 vs `data/charset.json` 67 字。混用 ⇒ 未知字符被静默丢弃（`[data] ... 丢弃未知字符 N>`0`）或输出层维度不符。处置：`--charset` 留空、让 ckpt 自带表接管 |
| 2 | **样本量 / batch 陷阱** | `drop_last=True` ⇒ 条数 < batch 就是 0 步（实测 `1 轮 × 0 步`，且以 0 退出、`best.pt` 照写，看着像成功）。处置：`--batch` 调小、看 `[sched] ... = 0 步` 这行 |
| 3 | **页面级 GT 弱配对** | §1.3：命中 10.8%、配对精度代理 2.5× 偏离；肉眼可见的错配例：`text="+5V"` 配到 `[629,398,734,408]`（105×10 的长框）。处置：只做通路验证；训练等框级标签（§2.3） |
| 4 | **`--pin-num-balance`** | `equal` 会改 pin_number 的长度分布（1/2/3 位对齐）。与本数据集分布不符时反而拉低 pin 分项；改动前后各跑一次闸门 A |
| 5 | **角色专用 ckpt** | 设了 `STAGE2_CKPT_DESIG` 而 `STAGE2_CKPT_ROLES` 不含该角色 ⇒ 第二把权重静默不生效（L552）；文件不存在也静默退回主权重 ⇒ 「换了没反应」 |
| 6 | **合成风格迁移** | 只在合成/单一样式上训出来的识别器，换字体/换渲染器会掉；`--include-real` + 真实语料混训才稳 |
| 7 | **小 val 噪声** | 10GT 弱配对只有 44 条 val，CER 抖动大，别用它做早停判断；`patience` 早停看 val CER，val 太小时容易早停在第 1 轮 |
| 8 | **CPU/GPU 差异** | 本机验证全程 CPU（`--device cpu`）；GPU 上 cuDNN+LSTM 曾以 `-1073740791` 退出（已在 `textrec/model.py` 用 `cudnn disabled` 的 RNN 包住），换机器时留意 |

## 6. 明天的最小清单

1. 拿数据 → 首行先看有没有框：有 ⇒ §2.3 `--boxes`；只有页面级 ⇒ §2.2 并接受降级。
2. `--stats` 跑一遍，核对：case 数、`role` 计数、`med_h`（和肉眼字号对比）、丢弃计数。
3. `--out` 落盘 → `python -X utf8 -m textrec.train --smoke --device cpu --batch 8` 验通路（§3.1）。
4. GPU 续训（§3.1，`--epochs` > 69）→ 看 `history.json` 的 val CER 是否真的降。
5. `$env:STAGE2_CKPT` 指到新 `best.pt` → 闸门 A（84.43）→ 闸门 B（36.840）→ 都过再谈交付；
   掉分就 `Remove-Item Env:STAGE2_CKPT` 回退，一句话说明是哪把尺掉的。

## 7. 已经发生过的一次成功替换（2026-10-23，可照抄）

下面的读数与配方就是本书 §3/§4 流程的一次完整执行记录，细节见 docs/RESULTS.md §5.357。

旧权重（epoch 69）在 GT10 protocol4 上是 **85.90**；换成新权重后 **88.73**。

§5.356 的 mix5（87.01）之后又在 GPU 上把语料换成「合成 12 万 + 真实 6.8 万」，得两条臂
（armB 3 轮 / armB2 +1 轮）。本节的数都是**当前 HEAD 代码 + 干净环境**复现的：
每次运行前显式 `Remove-Item Env:STAGE2_CKPT`，不留任何 `STAGE1_*`/`STAGE2_*` 残留
（§5.338.1 的串号坑，本段又踩了一次，见 7.3）。

### 7.1 读数

| 权重 | GT10 `protocol4` | comp | pin | netH | line |
| --- | ---: | ---: | ---: | ---: | ---: |
| 交付旧权重（epoch 69） | 85.90 | 0.9044 | 0.8042 | 0.9318 | 0.6054 |
| **armB2（epoch 72，新交付）** | **88.73** | **0.9505** | **0.8381** | **0.9486** | 0.6060 |

逐案（`work/_paircmp.py work/_gt10base_s.txt work/_gt10new_s.txt`）：

```
1  73.74 -> 84.17  +10.43    6  86.22 -> 94.01  +7.79
2  90.37 -> 94.60   +4.23    7  91.03 -> 93.21  +2.18
3  96.17 -> 96.17   +0.00    8  68.03 -> 68.17  +0.14
4  84.88 -> 86.13   +1.25    9  94.41 -> 96.17  +1.76
5  89.27 -> 89.27   +0.00   10  84.92 -> 85.36  +0.44
paired n10  base 85.904  new 88.726  delta +2.822  t +2.47  win 8  loss 0
```

8 胜 2 平 0 负，且原先最差的两案（1 号 73.74、8 号 68.03）分别 +10.43 / +0.14 —— 收益不是单案翻转。
收益全部来自文本轴：comp 0.9044→0.9505、pin 0.8042→0.8381（脚号/脚名的字读对了）、
netH 0.9318→0.9486（键对了网才对得上），导线轴 0.6060 不动。

### 7.2 语料与配方（可复现）

```
# 1) 收字：检测框用我们自己的 detect_texts，标签用 GT 串的 exact/near 两层匹配
cd eda-pcb; python -X utf8 tools\dev\harvest_labels.py --out work\_man_new_ocr.jsonl
# 2) 组语料：合成 12 万（锚点）+ 真实 300 案 6642 + 新集 7621×7 过采样 = 18.0 万条
python -X utf8 tools\dev\build_arm_manifest.py --synth ..\work\textrec_merged.jsonl \
       --synth-n 120000 --real ..\work\_real_only.jsonl --new ..\work\_man_new_ocr.jsonl \
       --new-repeat 7 --out ..\work\_man_armB.jsonl
# 3) 微调（GPU 约 10 分钟/轮；CPU 同命令换 --device cpu，约 17 分钟/轮）
python -X utf8 -m textrec.train --manifest ..\work\_man_armB.jsonl \
       --resume weights\textrec_pre_mix_20261016.pt --epochs 72 --batch 256 --lr 1.5e-5 --device cuda
```

* `--epochs` 是**绝对**轮号（ckpt 内 `epoch=69`，所以从 70 起算），不是相对轮数。
* 字表必须留空（`--resume` 自动取 ckpt 自带 71 字），显式给 `--charset` 会被拒。
* 旧权重备份在 `eda-pcb/weights/textrec_pre_mix_20261016.pt`（epoch 69，回退只需拷回）。
* 交付那次的 armB 清单是 18.8 万条（新集过采样 7.16×、且未丢 val 行）；上面这条命令产
  18.0 万条（丢 val），同配方、行不逐条相同，2 轮内的读数差在 0.1 分内。

### 7.3 两个坑（都真实踩到）

1. **`STAGE2_CKPT` 的绝对/相对路径不影响读数，但环境里残留的 `STAGE1_*`/`STAGE2_*` 会**：
   同一条权重两次跑分 88.96 / 88.73（pin 0.8327 vs 0.8381）。凡是对外公布的数字，
   必须在新 shell 里显式清空相关环境变量后重跑；`88.73` 是本节的权威值。
2. **收字工具的 `near` 层会把错位真值串扣到邻框上**：官方 200 案带进语料时 comp 掉到 0.9133
   （不走它 0.9544，§5.356.2），故交付配方里只留合成 + 真实 + 新集。

### 7.4 池子侧（`protocol3`，50 案）

armB2 在同一批 50 案上 **36.36**（基线 36.840，配对 t −0.90，25 胜 22 负），`protocol4` 池
33.97（基线 34.375）。都在噪声内、方向略负；池页来自官方 200 案（用户判定标注不可靠），
两把尺页源不同，交付以 GT10 为准，但池子这一负号记在这里，不隐藏。
