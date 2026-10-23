# PCB 元件检测器重训记录（det248：官方 200 页 + 新补充 48 页）

> 本文只记录一次可复现的实验：数据怎么来的、怎么转的坐标、怎么训的、新旧权重在同一份 held-out 上的差距，以及**结论**。
> 本次**没有**修改 `stage1/`、`stage2/`、`textrec/` 下任何文件，也**没有**覆盖 `eda-pcb/weights/` 里的任何旧权重，流水线默认权重仍指向 `pcb43_v2_from_best.pt`。

---

## 0. 一句话结论

新权重 `det248_best.pt` 在同一份 held-out 20 页上 **mAP50 0.820 / mAP50-95 0.600**，低于老权重 `pcb43_v2_from_best.pt` 的 **0.860 / 0.637**。
即：**本轮"加 48 页新数据 + 50 epoch 微调"没有提升 held-out 指标，反而是回退**。原因分析见 §6，不建议直接替换默认权重。

---

## 1. 数据来源与去重

### 1.1 官方 200 页（沿用既有划分）

- 原始目录：`eda-pcb/yolo_part/Dataset(waitforpreprocess)/200_train_cases/`
- 既有产物：`eda-pcb/yolo_part/training_data/`（`images/train` 180 张、`images/val` 20 张，41 类）
- 本轮处理：**原样复制**到 `training_data2/`，训练/验证划分与文件名完全不变（`0001-KiCad.png`、`0002-Datasheet.png` …）。
  这样 `val` 与老模型评估时看到的是**同一批 20 页**，新旧 mAP 可直接比较。

### 1.2 新补充 48 页

- 原始目录：`Y:\workspace\EDA\新补充训练集2\`
  - `dataset\`：34 个 `case_*` 组（每组 1 个 json + 1 个同名 png）
  - `新补充训练集(1)\NN\`（NN = 01..60 的子目录）：48 个 `case_*` 组 + 10 个 `SCH_*` 组
- **去重**：对每组按 `(json 的 sha256, png 的 sha256)` 建内容指纹，跨目录全局去重：
  - 候选 `case_*` 组 82 个 → **去重后 48 个**，删除 **34 个逐字节重复组**（`dataset\` 里的那 34 组与 `新补充训练集(1)\NN` 中的同内容组重复，仅文件名/路径不同）
  - `SCH_*` 前缀的 **10 组全部不用**（按任务要求排除）
- 落盘命名：`case_0001__ds.png` … `case_0048__ds.png` + 同名 `.txt` 标签，**全部进 train**（新集没有任何页面参与 val）。

### 1.3 最终数据集 `eda-pcb/yolo_part/training_data2/`

| 项 | 数量 |
|---|---|
| `images/train` | **228**（180 官方 + 48 新集） |
| `images/val` | **20**（仅官方原 20 页） |
| `labels/train` | 228 |
| `labels/val` | 20 |
| 框数 train | **12316**（官方 8265 + 新集 4051） |
| 框数 val | **906** |
| 类别数 | 41（`nc=41`） |

目录结构与既有 `training_data/` 一致：`images/{train,val}`、`labels/{train,val}`、`classes.txt`、`data.yaml`。

`training_data2/classes.txt` 与 `training_data/classes.txt` **逐字节一致**（sha256 `44198e1b27d26e5d…`，41 行），类别序号没有变。

---

## 2. 坐标翻转公式（新集）

新集 GT json 的坐标系已核实：**像素单位、原点左下、Y 向上**；PNG 尺寸为 `W × H`（H 为 png 像素高）。
`components[键]["bbox"] = [x1, y1, x2, y2]` 是元件的**符号体框**。

转到图像坐标（原点左上、Y 向下）只需镜像 Y：

```
y_img = H - y_label            # x 不变，无缩放
```

写 YOLO 归一化标签（cx, cy, w, h 均为 0..1）：

```
cx = ((x1 + x2) / 2) / W
cy = ((H - y1) + (H - y2)) / 2 / H = 1 - (y1 + y2) / (2H)
w  = |x2 - x1| / W
h  = |y2 - y1| / H
```

- 写入前对 `cx, cy, w, h` 做 clip 到 `[0, 1]`，并丢弃退化框（翻转后 `w <= 0` 或 `h <= 0`）。
- 目视核对：把翻转后的框画回原图，样本见 `eda-pcb/yolo_part/runs/det248/preview/case_0001__ds_boxes.png`（框与原理图符号一一对齐，未发现整体偏移或上下颠倒）。

官方 200 页沿用 `prepare_yolo_dataset.py` 既有转换，未改动。

---

## 3. 类别映射与丢弃统计

- 映射规则：新集 json 的 `components[键]["type"]` 字符串**直接按名字**匹配 `classes.txt` 里的类别名，命中则写该类序号；**不做任何改名/合并/重排**，因此序号与老数据集完全对齐。
- **丢弃统计（本次）**：

| 项 | 数量 |
|---|---|
| 新集解析到的框总数 | 4051 |
| 命中 41 类的框 | **4051** |
| **`type` 不在类别表而被跳过的框** | **0**（无任何未知 type） |
| 被整组排除的 `SCH_*` 组 | 10 组（非类别原因） |
| 被去重丢掉的重复组 | 34 组 |

- 新集**完全缺席**的 14 个类别（这些类的训练信号仍只来自官方页）：
  `amp_3pin, amp_5pin, ant, bjt_pnp, block, circle_header, crystal_2pin, crystal_3pin, crystal_4pin, mosfet_pnp, motor, pin, testpoint, transfomer`
  其中 `pin`（官方 train 159 框 / val 11 框）、`testpoint`（34 / 9）、`crystal_2pin`（40 / 1）在 val 里有实例却拿不到新数据，属于本轮的结构性缺口。
- 逐类框数（官方 train / 新集 train / val）：

| class | off_train | new_train | val | | class | off_train | new_train | val |
|---|---|---|---|---|---|---|---|---|
| gnd | 1888 | 964 | 248 | | r | 1395 | 740 | 152 |
| c | 1399 | 723 | 171 | | v | 963 | 567 | 130 |
| box | 862 | 320 | 98 | | net_input | 318 | 154 | 5 |
| d | 182 | 50 | 11 | | led | 167 | 47 | 13 |
| switch | 154 | 18 | 3 | | net_short | 161 | 66 | 2 |
| pin | 159 | 0 | 11 | | net_output | 71 | 69 | 1 |
| other | 63 | 70 | 3 | | bjt | 46 | 26 | 5 |
| mosfet | 56 | 12 | 10 | | m3螺丝 | 53 | 65 | 5 |
| crystal_2pin | 40 | 0 | 1 | | testpoint | 34 | 0 | 9 |
| l | 95 | 38 | 8 | | fuse | 13 | 3 | 7 |
| crystal | 1 | 14 | 0 | | net_bidirection | 11 | 51 | 2 |

---

## 4. 训练配置

初始化权重：`eda-pcb/weights/pcb43_v2_from_best.pt`（41 类，40.7 MB）

命令（实际执行的那一条）：

```bat
python eda-pcb/yolo_part/train_pcb43.py ^
  --model  Y:\workspace\EDA\eda-pcb\weights\pcb43_v2_from_best.pt ^
  --data   Y:\workspace\EDA\eda-pcb\yolo_part\training_data2\data.yaml ^
  --epochs 50 --batch 8 --imgsz 1024 --device 0 --cache ram --workers 4 ^
  --lr0 0.0002 --patience 40 --name det248
```

关键超参（取自 `runs/det248/args.yaml`）：

| 参数 | 值 | 说明 |
|---|---|---|
| epochs | 50 | 任务要求 40–60 |
| batch | **8** | 见下方显存说明 |
| imgsz | 1024 | 与老模型一致 |
| optimizer | AdamW | `auto` 解析结果 |
| lr0 / lrf | **2e-4** / 0.01 | 微调用较小初始学习率 |
| cos_lr | true | 余弦退火 |
| warmup_epochs | 3 | |
| weight_decay | 5e-4 | |
| box / cls / dfl | 10.0 / 1.5 / 2.5 | 沿用 `pcb43_v2` 的值 |
| mosaic / close_mosaic | 0.7 / 15 | |
| copy_paste | 0.3 | 小目标增强 |
| seed / amp / cache | 0 / true / ram | |

显存：batch=8 训练期间 `GPU_mem` 峰值 10.3–11.0 GB。事后用 `det248_memprobe.py 12` 复测 batch=12 也能跑（峰值 allocated **12.88 GiB**、reserved **13.73 GiB**，开跑前空闲 14.72 GiB），因此 12 在**独占** GPU 时可用；本轮取 8 是保守值——训练期间有其它进程共用 GPU，晚段单 epoch 从 ~7 s 拖到 25–45 s。

耗时：**50 epochs 共 0.327 小时（约 19.6 分钟）**，29 iters/epoch（228 张 / batch 8）。

### 训练前 preflight（脚本自带检查）

```
====================================================================
data.yaml : Y:\workspace\EDA\eda-pcb\yolo_part\training_data2\data.yaml
数据集根   : Y:\workspace\EDA\eda-pcb\yolo_part\training_data2
类别       : nc=41
图像       : train=228  val=20  合计 248
最多 6 类  : gnd=3100  c=2293  r=2287  v=1660  box=1280  net_input=477
少于 20 框 : opt=8  crystal_3pin=6  transfomer=5  ant=4  circle_header=3  seg=2  bjt_pnp=1  mosfet_pnp=1
====================================================================
```

### 每 epoch 关键指标（`runs/det248/results.csv`，节选）

| epoch | box_loss | cls_loss | dfl_loss | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| 1 | 1.470 | 1.920 | 1.437 | 0.630 | 0.663 | 0.6721 | 0.4676 |
| 2 | 1.357 | 1.670 | 1.385 | 0.714 | 0.640 | 0.6677 | 0.4395 |
| 5 | 1.243 | 1.455 | 1.377 | 0.899 | 0.642 | 0.7830 | 0.5166 |
| 10 | 1.079 | 1.240 | 1.325 | 0.913 | 0.625 | 0.7661 | 0.5365 |
| 15 | 1.052 | 1.195 | 1.324 | 0.933 | 0.655 | 0.8160 | 0.5758 |
| 20 | 1.011 | 1.126 | 1.302 | 0.945 | 0.649 | 0.7936 | 0.5736 |
| 25 | 0.935 | 1.054 | 1.308 | 0.920 | 0.622 | 0.7822 | 0.5799 |
| 30 | 0.899 | 1.003 | 1.294 | 0.868 | 0.662 | 0.7997 | 0.5431 |
| 35 | 0.843 | 0.964 | 1.288 | 0.726 | 0.758 | 0.7884 | 0.5814 |
| 40 | 0.796 | 0.912 | 1.293 | 0.717 | 0.790 | 0.8108 | 0.5900 |
| 45 | 0.753 | 0.879 | 1.288 | 0.703 | 0.789 | 0.8031 | 0.5825 |
| 48 | 0.744 | 0.869 | 1.292 | 0.712 | 0.795 | 0.8068 | **0.5916** |
| 50 | 0.762 | 0.877 | 1.296 | 0.708 | 0.790 | 0.8046 | 0.5911 |

（best mAP50-95 = 0.5967 @ep41；best mAP50 = 0.8160 @ep15。曲线显示 **ep15 之后基本平台化**，后 35 个 epoch 只把 recall 从 0.655 换到 0.79，代价是 precision 从 0.933 掉到 0.708。）

---

## 5. 新旧 mAP 对照（同一份 `training_data2/data.yaml` 的 val，20 页 / 906 框 / imgsz 1024）

| 权重 | P | R | **mAP50** | **mAP50-95** |
|---|---|---|---|---|
| 老：`eda-pcb/weights/pcb43_v2_from_best.pt` | 0.811 | 0.829 | **0.8603** | **0.6371** |
| 新：`eda-pcb/yolo_part/weights/det248_best.pt` | 0.712 | 0.813 | **0.8201** | **0.5998** |
| 差值 | −0.099 | −0.016 | **−0.0402** | **−0.0373** |

两个权重都用 `ultralytics.model.val()`、同一 `data.yaml`、同一 `split=val`、`imgsz=1024`、`batch=8`，脚本：`eda-pcb/yolo_part/det248_eval.py`（结果存 `runs/det248/eval/old_pcb43_v2.json`、`new_det248.json`）。

### 逐类 mAP50（老 → 新，节选；全表见 `runs/det248/eval/perclass.json`）

| class | 老 | 新 | 变化 |
|---|---|---|---|
| led | 0.845 | **0.919** | ↑ 明显 |
| fuse (mAP50-95) | 0.575 | **0.647** | ↑ |
| crystal_2pin (mAP50-95) | 0.597 | **0.796** | ↑ |
| box | 0.985 | 0.994 | ≈ |
| r | 0.795 | 0.791 | ≈ |
| v | 0.769 | 0.779 | ≈ |
| gnd | 0.964 | 0.929 | ↓ |
| bjt | 0.962 | **0.795** | ↓ 明显 |
| net_input | 0.715 | **0.455** | ↓ 明显 |
| net_bidirection | 0.663 | **0.247** | ↓ 严重 |
| net_output | 0.000 | 0.000 | 持平（两类都学不到，val 仅 1 个实例） |

---

## 6. 为什么会回退（结合证据的推断）

1. **尺度分布冲突**：新集页面明显更大（中位 **2362×1672**，官方中位 **1190×827**）。letterbox 到 imgsz 1024 后，新集页上的符号被压缩得更狠，小目标的像素尺度与官方页不一致；同时官方页里 `pin` / `testpoint` / `crystal_2pin` 这类小目标在新集**完全缺席**（见 §3），于是模型在这几类上只被官方数据塑造、却被更大页面的整体分布拉偏。
2. **标注口径冲突集中在 `net_*`**：新集给 `net_bidirection` 新增 51 框（官方 train 只有 11 框）、给 `net_input` 新增 154 框，而这两类恰好是本轮跌得最凶的（0.663→0.247、0.715→0.455）。新集的框是 KiCad GT 的**符号体框**，与官方 200 页沿用 `prepare_yolo_dataset.py` 得到的框在"网络标号该框到哪儿"这个问题上口径不同——两个口径混训，模型学成"要么框大要么框小"，于是 val 上两头都不对。
3. **训练预算与工作点**：`lr0=2e-4` + 50 epoch + cosine 的配置让曲线在 ep15 就摸到 mAP50 0.816 的平台，之后没有再涨，只是把 precision 换成 recall（0.933→0.708）。老权重显然在 200 页上训练得更充分。也就是说本轮更像是"用新数据把模型推到一个更宽松的判定阈值"，而不是"学到了新东西"。

**下一步建议（未执行，留作后续）**：
- 把新集的 `net_*` 口径先跟官方对齐（或在混训时给 `net_input`/`net_bidirection` 降权/降采样），再谈加数据；
- epoch 提到 120–200、`lr0` 降到 1e-4，并开 `--patience`（本轮 40 太大，50 epoch 用不上早停）；
- 把新集大页先按官方页的短边尺寸缩放，或提高 `imgsz` 到 1536（显存需同时降 batch）；
- 若目标是"只补官方稀缺类"，优先补 `pin` / `testpoint` / `crystal_*`，而不是再补大量 `gnd` / `c` / `r`。

---

## 7. 如何用新权重替换默认值（仅说明，本次未改）

流水线通过 `eda-pcb/stage1/` 引用 `weights/pcb43_v2_from_best.pt`；要把默认换成新权重，只需把该引用路径改成 `eda-pcb/yolo_part/weights/det248_best.pt`（或按你们既有部署方式把新文件改名覆盖）——**本次一律没有改动默认值、也没有覆盖任何旧权重**。

---

## 8. 产物清单与复现

| 产物 | 路径 |
|---|---|
| 数据集 | `eda-pcb/yolo_part/training_data2/`（228 train / 20 val，41 类） |
| 制备脚本 | `eda-pcb/yolo_part/det248_prepare.py` |
| 显存探针 | `eda-pcb/yolo_part/det248_memprobe.py` |
| 评估脚本（整体） | `eda-pcb/yolo_part/det248_eval.py` |
| 评估脚本（逐类） | `eda-pcb/yolo_part/det248_eval_perclass.py` |
| 训练输出 | `eda-pcb/yolo_part/runs/det248/`（`weights/best.pt`、`last.pt`、`results.csv`、`args.yaml`、PR/F1 曲线、混淆矩阵，以及 `eval/`、`preview/`、`smoke/` 子目录） |
| 新权重 | `eda-pcb/yolo_part/weights/det248_best.pt`（=`runs/det248/weights/best.pt` 的副本，40.7 MB） |

复现三步：

```bat
python eda-pcb/yolo_part/det248_prepare.py
python eda-pcb/yolo_part/train_pcb43.py --model Y:\workspace\EDA\eda-pcb\weights\pcb43_v2_from_best.pt --data Y:\workspace\EDA\eda-pcb\yolo_part\training_data2\data.yaml --epochs 50 --batch 8 --imgsz 1024 --device 0 --cache ram --workers 4 --lr0 0.0002 --patience 40 --name det248
python eda-pcb/yolo_part/det248_eval.py "Y:\workspace\EDA\eda-pcb\weights\pcb43_v2_from_best.pt" old_pcb43_v2
python eda-pcb/yolo_part/det248_eval.py "Y:\workspace\EDA\eda-pcb\yolo_part\weights\det248_best.pt" new_det248
```
