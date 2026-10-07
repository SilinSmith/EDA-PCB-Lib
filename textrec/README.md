# textrec —— 原理图文字识别器

把位号（`R1` / `U1` / `C87`）、引脚号、引脚名、参数值从原理图里读出来。

这个包存在的理由，写在 `Y:\workspace\EDA\docs\算法路线设计.md` §4.2 / §5.4 里：

> 现有 `pipeline/` 原型的四个 `read-*` 变体端到端得分**恰好 0.0000**。
> 原因不在检测、不在骨架——文字行都找到了（0025 找到 19 行、0036 找到 32 行、
> 0059 找到 21 行），而是**模板匹配这一步既拒识又误读**（`exact` 0.207、
> 切格 `ccells` 上限 0.476）。一个位号都没读出来，`read_components` 就跳过该
> 组件，组件集合为空，component / pin / net 三块**同时归零**。
>
> 而全部 100 分里有 70 分要经过位号传导。

同一份文档 §5.4 还证伪了「换成系统字体渲染模板」这条补救路线：语料抠模板确实
不跨厂商（留一 char 0.493 vs 留库 0.997），但作者为救它写的字体渲染库在同口径下
**严格更差**（exact 0.012 / char 0.095 vs 语料 0.153 / 0.424）。

**所以要换的是「模板匹配」这一类方法，不是模板来源。** 这个包就是替换品：
CRNN + CTC，不切格、不做模板比对，直接在图上对整行文字建模。

---

## 1. 数据从哪来（不用标注）

`合成原理图数据集/texts.jsonl` 里已经有 **381876 条**文字实例，每条形如：

```json
{"role": "pin_number", "content": "18", "comp": "U1", "bbox": [937.13, 592.67, 944.8, 597.0]}
```

`bbox` 是左上角原点的图像像素。语料实测的尺寸分布：

| role | n | 高 p50 | 宽 p50 | 长度 p50 | 长度 max |
|---|---|---|---|---|---|
| value | 116127 | 5.7 px | 19.7 | 4 | 16 |
| pin_number | 90987 | **4.3 px** | 7.7 | 2 | 3 |
| pin_name | 90987 | 5.0 px | 17.7 | 4 | 8 |
| designator | 76175 | 5.7 px | 10.0 | 2 | 4 |
| net_label | 7600 | 6.7 px | 22.0 | 5 | 8 |

**字小是这件事的本质困难**：`pin_number` 恒定只有 **4.3 px 高**。所以所有裁剪都
统一缩放到高 32 px（放大约 6~7 倍）再送网络。

字符表从语料频次自动生成（`min_freq=20`），语料里共出现 68 个不同字符，包括
`∅`（电源符号位号用）和 `螺丝` 这两个汉字。低频字符默认丢弃并记在
`data/charset.json` 的 `meta.dropped` 里。

---

## 2. 快速开始

```powershell
cd Y:\workspace\EDA\eda-pcb

# 1) 建裁剪清单 + 字符表（约 1~2 分钟，产出 textrec/data/）
python -m textrec.manifest

# 2) 先冒烟验管线（1 轮 / 2000 条 / 几十秒）
python -m textrec.train --smoke

# 3) 正式训练
python -m textrec.train --epochs 40 --batch 256 --workers 6

# 4) 真实语料上验一次（唯一诚实的泛化检查）
python -m textrec.eval_real --ckpt textrec/runs/best.pt --dump textrec/runs/errors.jsonl
```

---

## 3. 在管线里怎么接

```python
from PIL import Image
from textrec.infer import Recognizer

rec = Recognizer.load(r"Y:\workspace\EDA\eda-pcb\textrec\runs\best.pt")

img = Image.open(png_path)          # RGB / L 都能吃，会自动转灰度
text, conf = rec.read(img, [x1, y1, x2, y2], role="designator")

# 一张图几十上百行文字时，批量跑比逐条快一个数量级：
results = rec.read_many(img, [
    (bbox_designator, "designator"),
    (bbox_pin_number, "pin_number"),
    (bbox_pin_name,   "pin_name"),
])
```

`bbox` 用**左上角原点的图像像素**，和 `pipeline/` 的约定一致。

### 关于 `conf`——请务必用它

`conf` 是**非空白字符对数概率的几何平均**（每个字平均有多确定），取值 (0, 1]。

位号读错一个字符会连带丢掉该组件、它的引脚、以及挂在它身上的所有网络；而
读不出位号只是丢掉这一个组件。**所以宁可拒识，不要误读。** 建议在管线里卡一个
阈值，低于阈值就当作"没读出"，走别的路径（比如从网络标签反推）。

`eval_real` 会直接打印阈值扫描表，用真实数据告诉你该卡在哪。

---

## 4. 设计要点

| 选择 | 理由 |
|---|---|
| **CRNN + CTC** | 不需要切字。切格错误正是现有管线失败的直接原因（`ccells` 上限 0.476），CTC 直接对整行建模。 |
| **统一高 32 px** | 语料文字只有 4.3~6.7 px 高，必须上采样才有信息。 |
| **宽度下采样 8 倍** | 语料最长字符串 16 字符、宽高比约 7.2，缩到高 32 后宽约 230，时间步 T≈28，够 CTC 表达 16 个字符。`config.WIDTH_REDUCTION` 与 `model.py` 必须一致。 |
| **角色作为条件输入** | 调用方已经知道这一段是位号还是引脚号。六类角色的字符集差异极大（`pin_number` 基本纯数字、长度 ≤3），把 role 嵌入拼进每个时间步是白捡的准确率。 |
| **运行时裁剪，不预落盘** | 38 万条裁剪写成 PNG 又慢又占地方；改成每 worker 一个图片 LRU 缓存，同一 case 的裁剪连续命中。 |
| **按 case 抽 val** | 同一张图里的文字高度相关，按裁剪抽会泄漏。`config.VAL_EVERY=25`。 |
| **真实语料只验证、不训练** | 32 个真实 case 是唯一诚实的泛化检查，拿它训练就没意义了。想混训要显式加 `--include-real`。 |
| **丢弃 >24 字符的文本** | 真实语料里有 58 字符的整段说明文字，既不是位号也不是参数，缩到固定高度会严重失真。`config.MAX_TEXT_LEN`。 |

---

## 5. 文件

| 文件 | 作用 |
|---|---|
| `config.py` | 路径、角色表、图像超参、训练默认值 |
| `charset.py` | CTC 标签空间（**0 号永远是 blank**）、编解码、从语料建表 |
| `manifest.py` | 汇总合成 `texts.jsonl` + 真实 `CVJsonStd` → `data/manifest.jsonl` |
| `dataset.py` | 裁剪/缩放/归一化/增广。`crop_gray()` **训练与推理共用**，保证预处理一致 |
| `model.py` | CRNN：CNN → 拼角色嵌入 → 2 层双向 LSTM → 线性头 |
| `train.py` | CTC 训练、按角色分桶的 CER/完全正确率、早停、checkpoint |
| `infer.py` | `Recognizer.load / read / read_many`，带置信度 |
| `eval_real.py` | 真实语料评测 + 置信度阈值扫描 |

产物：

* `data/manifest.jsonl`、`data/charset.json` —— 清单与字符表
* `runs/best.pt`、`runs/last.pt`、`runs/history.json` —— 权重与训练历史

---

## 6. 还没做的（如实）

* **没有和 `pipeline/` 真正接起来**。`prun.py` 的 `read_components` 目前走的是
  `pread`（模板匹配）。把 `pread` 换成 `Recognizer.read` 是一次小改动，但
  `pipeline/` 里部分文件带 UTF-8 BOM，改动前要小心；而且那批代码目前处于
  「每阶段可换 GT oracle」的原型状态，接口还在动。
* **没有跑过完整训练**。这个包交付的是代码，`runs/best.pt` 需要你自己训。
  冒烟只能证明管线通，不能证明精度。
* **检测框的边界问题没解决**。`docs/算法路线设计.md` 指出 IoR@0.1 时 F1=0.760
  而 @0.5 只有 0.280，即文字行**位置找对了、边界不对**。识别器吃的是检测框，
  框太松会把邻居的笔画带进来。可以先用 GT bbox 训、再看检测框下的掉分幅度。
* **合成字体与真实字体有域差**。合成图用 PIL 内置字体渲染，真实图纸是 KiCad /
  Altium / JLC / Datasheet 各自的矢量字体。`eval_real` 就是用来量这个差距的，
  如果差距大，正解是在真实图上做自监督微调，而不是继续加大合成量。

---

## 7. 一个必须知道的环境坑：cuDNN 与退出码

**症状**：训练跑完、`best.pt` / `last.pt` / `history.json` 都写对了，但进程退出码是
`-1073740791`（= `0xC0000409` `STATUS_STACK_BUFFER_OVERRUN`），stderr 为空。
这个码很容易被误判成"训练失败"。

**根因**：本机 `torch 2.13.0+cu130`（系统 python 与 `eda_yolo` venv 都是这一版）上，
**只要 cuDNN 参与过 LSTM 的前向/反向**，进程结束时 Windows 的 `DLL_PROCESS_DETACH`
就会 fail-fast。这不是本包的逻辑 bug：

* `os._exit(0)` **躲不开**（它照样走 DLL 卸载）——已实测。
* 显式 `del` + `gc.collect()` + `torch.cuda.synchronize()` + `empty_cache()` **躲不开**。
* 换成 CPU 就正常。

**定位过程**（batch 64、裁剪宽 112）：

| 条件 | 退出码 | 速度 |
|---|---|---|
| 纯 `nn.Linear` 前向反向 | 0 | — |
| CRNN + CTC，cuDNN 全程参与 | **-1073740791** | 45.0 ms/step |
| CRNN + CTC，**只 RNN 关 cuDNN** | **0** | 79.1 ms/step |
| CRNN + CTC，全局关 cuDNN | 0 | 168.4 ms/step |
| CRNN + CTC，CPU | 0 | — |

**已采用的修法**：`model.py` 里只把 `self.rnn(feat)` 包进
`torch.backends.cudnn.flags(enabled=False)`。卷积仍走 cuDNN，所以只比全速慢
1.76 倍，而不是全局关掉的 3.7 倍。这个动作是无条件的（Linux 上同理），
保证开发与部署行为一致。

想换回 cuDNN 版 RNN（更快，但退出码非零，会被脚本判成失败）：
`set TEXTREC_CUDNN_RNN=1`。

**验证环境问题时的陷阱**：`drop_last=True` 意味着 `--limit` 小于 `--batch` 时
**一个 batch 都不会跑**，训练循环根本没执行、退出码当然正常。要确认真的跑了
至少一个 batch，否则结论是假的。（我一开始就被这个骗了：`--limit 50 --batch 256`
看起来"训练了但退出码正常"，其实一步没跑。）

**另**：在 PowerShell 里 `& python ... > $null` 或 `| Select-Object` 量不出真退出码
（连 `print("hi")` 都报 exit 1）。要量退出码必须用
`Start-Process -Wait -PassThru -RedirectStandardOutput ...`，读 `$p.ExitCode`。

---

## 8. 另一个必须知道的坑：**批内补白必须在解码时截掉**（2026-10-05 修，220 张 +0.565）

**症状**：`R2` 读成 `R2441`、`C4` 读成 `C44`、`1` 读成 `111`、`L2` 读成 `L21`。
**尾巴上多出来的字符，恰好是把真结果最后一位又抄了一遍。**

**根因**在 `infer.Recognizer`：推理把整批补白到本批最大宽度（`max_w = max(widths)`），
然后**解码全部 T 个时步**：

```python
max_w = max(widths)
batch = torch.full((len(tensors), 1, h, max_w), 1.0)     # 补白
...
col_ids = idx[:, b].tolist()                              # ← 旧的：连补白那段一起解码
```

而训练侧做的是**另一回事**：`dataset.collate` 同样补白，但
`ctc_input_lens_for(widths, out_len)` 把每个样本的 CTC 输入长度设成
`max(1, w // WIDTH_REDUCTION)`，**补白那段被 `CTCLoss(input_lengths=...)` 整个屏蔽掉**。
⇒ 模型对补白那段**没有任何监督**，它在那里吐什么字符完全靠猜；而这些字符会直接
连在真结果后面。

**量化**（`work\tools\diag\_crnneval.py`，训练用的 `work\ocr_real\manifest.jsonl`，
val 785 条——原先用 38 万条的 `textrec/data/manifest.jsonl` 得到的
`val_exact 0.662` 就是这么来的，那份不是训练用的）：

| 批大小 | 解码方式 | exact | 其中位号 exact |
|---:|---|---:|---:|
| 32 | 全部 T | 0.6815 | 0.8407 |
| 32 | 截到 `w // 8` | 0.6879 | 0.8761 |
| **256**（= 线上 `stage2/text.py` 的 `CHUNK`） | 全部 T | **0.6573** | **0.7522** |
| **256** | 截到 `w // 8` | **0.6866** | **0.8850** |

⇒ **批越大、补白越宽，全 T 解码错得越狠**（位号 0.8407 → 0.7522），
截断把这个损失**整个收回**（位号 **+13.3pp**），而且批大小 256 截断后反而**最好**。

**修法**（`infer.py::_forward_padded`，默认生效）：

```python
tmax = max(1, min(T, widths[b] // WIDTH_REDUCTION + EXTRA_STEPS))
out.append(self._decode(idx[:tmax, b].tolist(), logp[:tmax, b].tolist()))
```

`EXTRA_STEPS`（`TEXTREC_EXTRA_STEPS`，默认 0）是留的余量旋钮，正常不要动。

**220 张端到端 A/B（官方口径，无 VLM）**：`23.8329 → 24.3976`（**+0.565**）——
`component F1 0.5268 → 0.5365`、`tp 5033 → 5125`、`missing 2955 → 2838`、
`netH 0.2007 → 0.2062`。墙钟 617.5s（比基线 699s 还快，因为不用为补白那段跑 RNN）。

**试过但判负的两条**（都留着代码，默认关）：

* **按裁片宽度分桶**（`TEXTREC_BUCKET=1`，让补白为零）：220 张 **23.6422**（−0.19）。
  原因：训练时几乎所有样本都带着大片补白，模型习惯了那个分布；喂无补白的裁片
  反而是分布外。**不要开。**
* **全局补白到固定宽度**（如 384）：快评 exact 掉到 0.4586，位号 0.0708。

**顺带确认的两件事**（都不用改）：

* `stage2/text.py` 的 `pad_box` 默认 `PAD_FRAC=0.30` **就是最优**。用真值框按
  `det/gt` 实测的 `0.77 x 0.54` 收缩来模拟我们的检测框，再扫 `PAD_FRAC`：
  `0.0 → 0.6497`、`0.15 → 0.6510`、**`0.30 → 0.6701`**、`0.45 → 0.6637`、
  `0.6 → 0.5669`、`0.8 → 0.3529`。
* 单条 `read`（批大小 1）本来就不受这个 bug 影响：`max_w == w` ⇒ `tmax == T`。
  出问题的只有 `read_many` 的批量路径，也就是线上。


