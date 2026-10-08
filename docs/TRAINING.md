# 训练手册：数据集 → YOLO / CRNN → 两把尺子

管线里有两个**可训**模型，本文件给出它们的完整重建路径与实测数字：

| 模型 | 权重落点 | 训练入口 | 数据集 |
|---|---|---|---|
| 元件检测器 YOLO11 | `weights/best_component_<name>.pt` | `yolo_part/train_pcb43.py` | `work/trainset_v4[_merged]` |
| 文字识别 CRNN+CTC | `weights/textrec_best.pt` | `textrec/train.py` | `work/crnn_mix/manifest.jsonl` |

所有命令都从 `eda-pcb/` 下敲。**改了数据集或权重，§4 的两把尺子必须重测**——否则「新模型更好」只是感觉。

---

## 1. 数据集（YOLO 检测）

### 1.1 三份数据集

| 目录 | nc | 类别差异 | train | val | test |
|---|---:|---|---:|---:|---:|
| `Y:\workspace\EDA\trainset_v3`（junction → `训练集_v3`） | 43 | 上一版基线 | 1624 图 / 73675 框 | 77 图 / 2431 框 | 10 图 / 254 框 |
| `work/trainset_v4` | 43 | +296 张新语料、清重复框、修墨迹 | 1920 图 / **84760 框** | 77 / 2431 | 10 / 254 |
| `work/trainset_v4_merged` | **42** | 再把 `pin`+`testpoint` 合成 `tp` | 1920 / 84760 | 77 / 2431 | 10 / 254 |

`v4_merged` 的 `tp`（id 34）= `pin` 1371 + `testpoint` 338 = **1709** 框，val 里 95 = 1 + 94。
合并的意义：这两类在像素上本就是一回事（都是焊盘/探针点），分辨它们该靠**文本绑定**（有没有位号/引脚号）而不是检测头——见 `docs/RESULTS.md`。

> v4 的第一版重建是 84774 框，加上 `--drop-dup on` 后是 **84760**（少 14 = 6 个 `reason=DUP` + 8 个新语料内部重复）。
> **v4 的 val/test 与 v3 逐字节相同**（`build_trainset_v4.py` 的 `transfer` 显示 87 个基础标签是「原样链接」= val 77 + test 10），所以 `mAP50 0.6208` 这条基线仍然可比。

### 1.2 构建命令

```powershell
cd Y:\workspace\EDA\eda-pcb
python -X utf8 tools\dev\build_trainset_v4.py --force --fix-ink recenter --drop-dup on --out ..\work\trainset_v4
python -X utf8 tools\dev\build_trainset_v4.py --force --fix-ink recenter --drop-dup on `
    --merge-classes pin,testpoint --merged-name tp --out ..\work\trainset_v4_merged
```

跑之前先 `--dry-run`（只统计不落盘，几秒钟），它会打印每一类修了多少：

```
base: dup dropped 6        # = work/label_audit.csv 里那 6 个 reason=DUP，一个不多一个不少
base: ink moved 170        # 只有真踩在笔画缺口上的框才重定位
new:  ink moved 1030       # 新增语料是「先 snap 成图章尺寸」，重心最容易跑偏
new:  size snapped to stamp 4023
new:  empty tiles dropped 28
TOTAL train 1920 val 77
```

外加**每次运行都打**的一行硬指标（非 0 会标 ✗，见 1.4）：

```
空框自检（框内一点墨迹都没有的框）：new 3 + base 9 = 12   [v3 基线 27/13620 = 0.2%]
```

### 1.3 三个开关的默认值与理由（都别乱开）

| 开关 | 默认 | 为什么 |
|---|---|---|
| `--fix-ink {off,recenter,blob}` | `off` | `recenter` 在框周围搜索窗里用积分图枚举所有落点，取「框内墨迹够多且**位移最小**」的那个；窗里连一个墨迹像素都没有才退回最近墨迹像素，空心符号（框内本就无墨迹）判 `noink_noblob` 原样留下。**老版本把中心挪到最近连通域的质心**，对「一整条网络」那种大连通域质心可能落在空白里 ⇒ 修完仍是空框，见 `docs/DIAGNOSTICS.md` 第 6 条 |
| `--fix-tpl {off,snap,drop}` | `off` | 按「图章尺寸」改框宽高看着诱人，但**真实图章偏差的典型值不是 0**：`net_input` 的 `w/stamp` 中位数 **1.3201**、`v` **1.2000**，而 `c/r/d` 是 1.0000。25851 个「待修」里 11329 个只是这两个类的固有差异，按图章修就是**改坏**。要用得先排除这两类或改成按类众数 |
| `--drop-dup {off,on}` | `off` | 同类且中心/宽高都在 `--dup-tol`（默认 2.0）内即判重。v3 里有 6 个 |
| `--tpl-tol` / `--dup-tol` | 1.0 / 2.0 | 判定容差 |

### 1.4 标签体检（决定「要不要重标」而不是凭感觉）

两把尺子，先量数字再动手：

```powershell
python -X utf8 tools\dev\audit_labels.py --root ..\work\trainset_v4 --csv ..\work\label_audit.csv
python -X utf8 tools\diag\_inkcheck.py --sample 300 --roots "v3=..\trainset_v3,v4=..\work\trainset_v4"
```

逐框查 `ZERO_INK / DEAD / TPL_MISMATCH / SIZE_OUTLIER / TEXT_BOX / DUP / BAD_NAME`。全量结论（`docs/RESULTS.md` §5.2）：

* 真·错标 23 框 = **0.24%**；硬缺陷（无墨迹且 25px 内也没墨迹）70 框 = **0.73%**
* 没有**任何一类**整体无效——「训练集被错误标号污染」在数量上不成立
* 41 个越界框是裁剪管线缺陷（该从矢量源重导，不是标注错）
* `pin`/`testpoint` 与 `d`/`led` 存在类型重合 ⇒ 这正是 `v4_merged` 合并 `tp` 的依据

**空框（框内一点墨迹都没有）是最省事的硬指标**，`_inkcheck.py` 同一批 300 张 train 图的读数：

| 数据集 | 框数 | 空框 | 居中偏移（中位） |
|---|---|---|---|
| `trainset_v3` | 11375 | 50 | 0.096 |
| `trainset_v4`（老 `ink_recenter`，有 bug） | 10491 | **290** | 0.097 |
| `trainset_v4`（现版本） | 10493 | **12** | 0.097 |

⇒ 机制：`snap()` 只改尺寸不改中心，老 recenter 又把一批空框挪到**同一个连通域质心**（随后被 `--drop-dup` 当重复框删掉，真丢标注：框总数 84760 → **84781**）。现在构建脚本**每次都打**「空框自检」，残留非 0 就是有框没修好，先别开训。

当前 v4 指纹：43 类（`v4_merged` 42 类）、train 1920 图 / **84781** 框、val 77 / 2431、test 10 / 254；有 bug 的老版留在 `work\_v4_prev_buggy` / `work\_v4m_prev_buggy` 备回滚。队列 ②/②b 两臂读的就是这一版。

---

## 2. YOLO 检测器

```powershell
python yolo_part\train_pcb43.py                                   # 43 类 / yolo11m / imgsz 1024 / 200 epoch
python yolo_part\train_pcb43.py --data ..\work\trainset_v4\data.yaml --nc 42   # 42 类合并集
python yolo_part\train_pcb43.py --smoke                           # 1 分钟冒烟：1 epoch / 5% 数据 / imgsz 640
python yolo_part\train_pcb43.py --resume                          # 断点续训
```

* `--data` 不传时按 `trainset_v3 → 训练集_v3 → trainset_v2 → 训练集_v2 → PCB_trainset → PCB训练集 → work/trainset_v4 → work/trainset_v4_merged` 找第一个存在的 `data.yaml`（**重建版排在最后，不会悄悄改掉默认基线**）。
* `--nc` 覆盖类别数检查（默认要求 43）。`nc=42` 是**已知例外**（`pin+testpoint` 合并），不传 `--nc` 时脚本会打印一行「注意」并继续；传了但和 data.yaml 不符则直接报错退出。
* **权重陷阱**：`weights/yolo11m.pt`、`yolo11l.pt` 曾是**团队自己 41 类成品**覆盖官方文件名。从那种权重续训，本次 val 的图纸它早见过——实测同为 1 epoch/5% 数据，污染起点 `mAP50 0.758` vs 干净起点 `8.6e-05`，差四个数量级。`check_checkpoint()` 会读 `train_args` 认出来并警告。
* 历史基线：`training_data` 那轮 11 组实验里 11m 每组都赢 11l（11m im1536 `mAP50 0.5761`，11l im1536 `0.5583`）；本仓库 v3 数据集上 `mAP50 0.6208`。
* 显存（RTX 4080 16 GB）：`imgsz 1024 + yolo11m` → batch 8~12 稳，16 可能 OOM。脚本**OOM 会自动 batch 减半重试**（最多 4 次），不会整晚训练死在一次 OOM 上。
* 训完自动把 `best.pt` 复制成 `weights/best_component_<name>.pt`，并做一次 `conf=0.1` 的**低阈值复验**（看真实召回，而不是只看 mAP）。
* 合 class 之后要验证的是「检测 f1 不降、绑定阶段的 pin/testpoint 归类不降」——两边一起看，别只看 mAP。

---

## 3. CRNN 文字识别器

### 3.1 为什么需要混合清单

出厂 `weights/textrec_best.pt`（= `work/ocr_real/runs/best.pt`，`epoch 51`，`val_cer 0.1871`）**只在 2977 条真实裁片上**训了 111 遍，没见过合成语料；而线上最痛的一类（3 位引脚号被 CTC 吞一位：`113` → `13`）在合成语料里有 87k 条，真实语料里却极少。纯合成训出来的模型又过不了域差。所以两边都要。

```powershell
python -X utf8 tools\dev\mix_crnn_manifest.py --out ..\work\crnn_mix --charset-from weights\textrec_best.pt
```

产出（已实测，可复核）：

| 文件 | 内容 |
|---|---|
| `work/crnn_mix/manifest.jsonl` | train **28033** 条：`pin_number 15841` / `designator 4463` / `value 3341` / `pin_name 2811` / `net_label 1577`；val **796** 条 = **真实语料**（`pin_number 274` / `pin_name 203` / `designator 113` / `net_label 112` / `value 94`） |
| `work/crnn_mix/charset.json` | 71 字 + blank = 72 类，`meta.from` 记着来源 ckpt（**必须与 `--resume` 的权重是同一份**，否则输出头与字符表错位） |

`pin_number` 按长度分层采样（1/2/3 位每层 4000）把「2 位占 79%」的先验扳平：train 长度分布 1 位 **5580** / 2 位 **5202** / 3 位及以上 **5059**（合成原始分布是 8909 / 70694 / 7660）。val 的 274 条里 1 位 187 / 2 位 74 / 3 位及以上 13。

### 3.2 训练 / 微调

```powershell
# 从零（只用混合清单）
python -X utf8 -m textrec.train --manifest ..\work\crnn_mix\manifest.jsonl `
    --charset ..\work\crnn_mix\charset.json --out ..\work\ocr_mix\runs

# 热启动微调（从出厂权重继续）
python -X utf8 -m textrec.train --resume Y:\workspace\EDA\eda-pcb\weights\textrec_best.pt `
    --manifest ..\work\crnn_mix\manifest.jsonl --charset ..\work\crnn_mix\charset.json `
    --out ..\work\ocr_mix\runs --epochs 66 --batch 128 --workers 6 --patience 4 --lr 0.0001 --seed 0
```

启动时应看到这三行，**对不上就别继续**：

```
[charset] 71 chars + blank = 72 classes
[data] train 27958 (丢弃未知字符 75)  val 785 (丢弃 11)     # ← val 是真实语料，与出厂模型同一把尺子
[resume] 从 ...\weights\textrec_best.pt 的第 51 轮继续
```

（28033 − 75 = **27958**，796 − 11 = **785**，与清单统计正好对上；被丢弃的是含字符表外字符的裁片。）

坑：

* **`--epochs` 必须大于 ckpt 里的 `epoch`**。`textrec/train.py:224-230` 用 `start_epoch = int(ck.get("epoch", 0))` 接着数，出厂权重是 `epoch 51` ⇒ `--epochs 66` 就是再训 15 轮；写成 40 会**一轮都不跑**直接退出。
* 出厂 ckpt 没有 `opt` 键（只有 `charset/config/epoch/model/val_cer/val_exact`），`--resume` 容忍这一点（`if "opt" in ck`）。
* 调度是 OneCycle，总步数按 `--epochs` 算；微调时 `--lr` 给默认值的一半（`1e-4`）更稳。
* 显存：默认 `--batch 256`；本机 LM Studio 的 `llama-server` 常驻约 12 GB，与 CRNN 同跑时用 `--batch 128`（实测在剩下 ~4 GB 里正常起跑，无 OOM）。
* 每轮打印的 val 就是那 785 条真实裁片，`best.pt`/`history.json` 落在 `--out` 目录；**别把 `--out` 指到 `eda-pcb/textrec/runs`**（那是历史目录，容易和自己打架）。

---

## 4. 两把尺子（训练完必须跑）

### (a) 每角色 exact / CER —— 离线，最便宜

```powershell
python -X utf8 -m textrec.eval_real --ckpt <新 ckpt> --manifest ..\work\ocr_real\manifest.jsonl
```

* `textrec/eval_real.py:48` 只取 `split == 'real'` 的记录 ⇒ **必须传 `work/ocr_real/manifest.jsonl`**；混合清单 `crnn_mix` 里没有 `real` 这一档，传了会得到空结果。
* 出厂基线（每角色 exact）：`pin_number 0.7196` / `value 0.663` / `designator 0.5398` / `pin_name 0.3317` / `net_label 0.4091`。

### (b) 端到端引脚号 —— 唯一算数的尺子

```powershell
$env:STAGE2_CKPT = 'Y:\workspace\EDA\work\ocr_mix\runs\best.pt'   # 默认 weights\textrec_best.pt（stage2/text.py:31,43，装载点 :375）
# 之后重跑官方 9 张的 stage1+stage2（VLM/LLM 可关，--no-vlm/--no-llm，做纯模型 A/B）
python -X utf8 work\tools\diag\_numread.py --dir <新产物目录>      # 引脚号绑定准确率
```

出厂基线：引脚号准确率 **0.6767**。每角色 exact 涨了但这里不涨，等于没赢——引脚号是唯一进分的通道（`∅NNN` 电源/地符号的引脚号在 GT 里是官方随机数，占网络成员 22%，数学上不可赢）。

### (c) 检测器

`train_pcb43.py` 训完自带 `conf=0.1` 复验；要官方口径就用 `python tools\data\official_eval.py`（9 张 GT）。

---

## 5. 环境与纪律

* 解释器：本机 `python`（`K:\python\311`）或项目 venv `eda-pcb\eda_yolo`；`train_pcb43.py` 在缺 `ultralytics` 时会自动用 venv 重启自己，所以只敲一条命令即可。
* 中文路径是故障源，优先用 ASCII 路径（`trainset_v3` 是指向 `训练集_v3` 的 junction）。
* PowerShell 里加 `$env:PYTHONIOENCODING='utf-8'`，否则中文日志会乱码；本机**没有 `pwsh`（7.x）**，
  跑 `.ps1` 要用 `powershell -NoProfile -ExecutionPolicy Bypass -File x.ps1`，且脚本存 **UTF-8 with BOM**
  （5.1 按 ANSI 解析无 BOM 的脚本，中文字面量会变乱码）。详见 `docs/DIAGNOSTICS.md`。
* 跑 GPU 前先 `python eda-pcb\tools\dev\gpu_gate.py status`（它自己读 `nvidia-smi`）：本机最常抢卡的是 LM Studio 的 `llama-server.exe`（~12 GB），其次才是别的训练。
* **GPU 必须排队跑，不许并发抢显存** —— 见下面 5.1（用户 m34654 的硬要求）。
* 每次训练/评测的数字都写进 `docs/RESULTS.md`，**负结果也要归档**（「修了但没用」和「没用所以修坏了」是两种结论）。
* 改了开关或加了脚本，重生成两个索引：`python tools\dev\gen_knobs.py --write`、`python tools\dev\gen_toolindex.py --write`。

### 5.1 GPU 纪律：一次只跑一个吃卡任务（`tools/dev/gpu_gate.py`）

用户 m34654 的原话：「**能不能不要爆显存，显卡同时跑多任务会导致缓存命中降低，这对 LLM 和各种训练是灾难性的，
能不能分开来跑这些内容，不要一起抢显存**」。

多进程分显存的代价不是「都慢一点」：LLM 的**前缀缓存 / KV 被挤掉要重算**，训练侧的 cudnn workspace 与
pinned buffer 也要重新分配，两边墙钟都可能翻倍，而且崩的时候通常**崩在后起的那个**身上（谁先占谁赢）。
所以处理方式是**排队**，不是「小心一点」。

```powershell
cd Y:\workspace\EDA\eda-pcb
# 0) 先看：整卡占用 / 谁在吃卡（自己人 vs 外人）/ 锁归谁 / 最近台账
python tools\dev\gpu_gate.py status

# 1) 排队后运行（拿到锁、等到显存够、跑完自动释放，并把峰值占用写进台账）
python tools\dev\gpu_gate.py run --tag crnn_mix66 --need-mb 2000 -- `
  python -X utf8 -u -m textrec.train --resume ..\weights\textrec_best.pt `
  --manifest Y:\workspace\EDA\work\crnn_mix\manifest.jsonl `
  --charset Y:\workspace\EDA\work\crnn_mix\charset.json `
  --out Y:\workspace\EDA\work\ocr_mix\runs --epochs 66 --batch 64 --workers 4 --lr 0.0001

# 2) 台账（谁在什么时候占了多少、跑了多久、退出码）
python tools\dev\gpu_gate.py ledger --tail 20
```

规则与语义：

* **所有吃卡的东西都从这里起**：`tools\eval\fast.py`、`submission\run.py`、`textrec.train`、`train_pcb43.py`、
  以及 `fusion.run` 的 LLM 轮（它会调 LM Studio 的本地模型）。
* **闸门只保证「自己人」串行**：锁是 `work\_gpu.lock`（JSON，持锁进程死了会被自动接管），
  台账是 `work\_gpu_ledger.jsonl`（含 `peak_used_mb`、`sec`、`exit`）。
* **外人（LM Studio / `llama-server.exe`）在场时先问用户**，不要硬上；确实要独占就加 `--exclusive`
  （外人在场就等），别用别的手段把别人的模型挤掉。
* 要多臂对比就**串行**跑；能做成离线（纯 CPU）复算的（例如只换 `assemble.build_doc` 里的模块级闸门、
  或只换 CRNN 权重重跑 stage2）就**别占卡**——`work\_ckptab\<tag>\runs\stage1` 是指向已落盘 stage1 产物的
  junction，换了权重只需重跑 `python -X utf8 -m stage2.run … --ckpt <新权重> --device cpu`。
* 起任务之前先估 `--need-mb`：CRNN 微调 batch 64 ≈ 1.5~2 GB，YOLO 从零训练 ≈ 4~6 GB，
  LM Studio 的 9B 模型常驻 ≈ 12 GB ⇒ 16 GB 卡上「9B 常驻 + 训练」几乎必然互相拖慢甚至 OOM。
* 判读 `status`：**「自己人的吃卡进程」> 1 个就是违反纪律**，要先停掉一个；「外人的吃卡进程」非 0 时，
  要么问用户，要么用 `--exclusive` 等它走。

### 5.2 串行训练队列（`work\_gpu_queue.ps1`）

把一组 GPU 活按顺序摆在一个脚本里，每一步都经 `gpu_gate`，**共用同一把锁 ⇒ 天然串行**：

| 步 | 内容 | gate 参数 | 日志 |
|---|---|---|---|
| ① | CRNN 微调续跑（e55 → e66，约 45 min） | `--tag crnn_mix66 --need-mb 2500 --headroom 12000` | `work\ocr_mix\train_log_r2.txt` |
| ② | YOLO v4 重训（43 类 / imgsz 1536 / 60 轮，约 2.2 h） | `--tag yolo_v4_im1536 --need-mb 13000 --headroom 1000` | `work\_yolo_v4_log.txt` |
| ②b | YOLO 合并集重训（42 类 `tp`，同一批图同一 warm start，约 2.2 h） | `--tag yolo_v4m42_im1536` | `work\_yolo_v4m42_log.txt` |
| ③ | 两把尺子（纯 CPU：`eval_real` → 换权重 stage2 → `_numread` → 抄 `results.csv`） | 不占卡 | `work\_post_train_*.txt` |

* **headroom 故意开大，就是「不许和 LLM 并发」的机器保证**：空闲 ≥ `need + headroom`（①要 14.5 GB、
  ②要 14.0 GB）才放行，而 LM Studio 载着模型时只剩 ~5.4 GB、卸载后 ~15.3 GB ⇒ 只有它真的卸载了才会
  跑起来；这个条件不依赖任何人记得。**不要用 `--exclusive`**（`LM Studio.exe` 命中 `HEAVY_NAMES`，永远在场）。
* 启动方式（本机没有 `pwsh`，且默认执行策略拒绝直接跑 `.ps1`）：
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File Y:\workspace\EDA\work\_gpu_queue.ps1 *> Y:\workspace\EDA\work\_gpu_queue_log.txt
  ```
  脚本必须存成 **UTF-8 with BOM**，否则 PowerShell 5.1 按 ANSI 解析、中文字面量全成乱码
  （Python 子进程写的日志文件不受影响，只有脚本自己的 `Write-Host` 会乱）。改完脚本用
  `work\_syntax_check.ps1` 做一次只解析不执行的语法检查，再重启队列。
* 每一步开跑前都 **fail-fast**（`last.pt` / `manifest.jsonl` / `charset.json` / warm-start 权重 /
  `data.yaml` 逐个 `Test-Path`）。2026-10-08 就靠这个抓到一次：`--model weights\best_component_pcb43_v3_im1536.pt`
  **这个路径不存在**（v3 生产权重的真身是 `yolo_part\runs\detect\pcb43_v3_im1536\weights\best.pt`），
  不检查的话会在 CRNN 跑完 45 min 之后才失败，白等一整轮。
* ② 与 ②b 是**同一批 1920 张图、同一个 warm start**，唯一变量是类别布局（43 类 vs 42 类 `tp`）
  ⇒ 只有这一对能回答「`pin`+`testpoint` 在检测阶段合并到底有没有用」
  （官方 9 张里 testpoint 实例为 0，只能在派生 6 张与 220 全量上看）。
* 跑完记得**两条线都要看**：CRNN 的 `eval_real`（真实集 6642）+ 端到端引脚号；YOLO 的自带
  `{name}_conf010` 复验（`train_pcb43.py:383-387`），并把数字写进 `docs/RESULTS.md`（负结果也要归档）。
