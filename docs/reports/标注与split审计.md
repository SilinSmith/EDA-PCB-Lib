# 标注与 split 审计 —— 你说的「训练集本身有问题 + 一直从里面切图」已坐实

审计对象：`Y:\workspace\EDA\trainset_v3`（`data.yaml` **43 类**，train 1624 张 / 73675 框，val 77 张 / 2431 框，test 10 张）。
脚本：`work\tools\diag\_labelaudit.py`（标注分布）、`work\tools\diag\_splitaudit.py`（按源原理图分组）、`work\tools\diag\_leakpair.py`（train/val 同源成对对照）。
起因：`User said (m02619)` 逐图核对 `work\preview\` 后提出「**可能是训练集本身一开始就有几个问题，然后你又一直从里面切图出来组训练集**」。

---

## 一、逐条核对你的判断（全部成立）

| 你说的 | 核对结果 | 证据 |
|---|---|---|
| `fn_r` 里漏检的**全都是电感，而不是电阻** | **成立** | `fn_r.png` 8 格**全部**来自 `200_train_cases-0050__full`，红框都是 `r 8x20`，扣在同一张图里一个**锯齿周期约 40px 的大锯齿**的单根笔画上 —— 那是电感的画法。`0050__full` 的标注是 `r=28, l=0`（见下表） |
| `fn_net_input` 的框**没框住任何有效东西** | **成立** | 红框为 `net_input 30x10`，分别扣在普通文字 `SET` / `erface` / `t Down` 上；另有 `net_input 10x29` 扣在**导线拐角**上。来源图 `0011` / `0064` / `0027` |
| `fp_r` 的错检**本质上都是对的，确实是 R** | **成立** | 见第二节：`r` 的标注框宽度**恒为 8px**，是固定模板框，套不上真实符号 ⇒ 同一颗器件同时记一笔 FN 和一笔 FP |
| `fp_net_input` 框歪了但**没有错检** | **成立** | 同上，`net_input` 只有 `(30,10)` 与 `(44,11)` 两种模板尺寸 |
| `fp_c`/`fp_gnd`/`fp_testpoint` **略微歪但都对** | **成立** | 同上机制 |
| `fn_v` 部分框歪了但确实有问题 | 成立 | `v` 的 GT 框切断连续墨迹 98.6%，模板化最严重 |
| `fn_box` / `fp_c` 两张**没看懂** | 我的图没画好 | `box` 是容器/块级大框（val 中位 59×54、train 68×50），`fp_c` 是电容错检接触表；图里只给了 2.2× 局部裁片，没有上下文，**这两张需要重画** |

---

## 二、根因一：标注不是「贴合符号」，是**固定模板框**

| 类 | 模板尺寸（train） | 计数 |
|---|---|---|
| `r` | **宽恒为 8px**，高 18–23 | `(8,20)=2703` `(8,22)=809` `(8,21)=806` `(8,19)=787` `(8,18)=678` `(8,23)=382` ⇒ **≈6165 / 12334（一半）** |
| `r`（val） | `(8,20)` | 234 |
| `net_input` | `(30,10)` / `(44,11)` | 682 / 347 |

**后果**：一个固定尺寸的框套不住尺寸各异的真实符号。真实符号比模板大 → 预测框一出界就被判 FP；比模板小/偏 → GT 判 FN。**同一颗器件因此同时产生一笔 FN 和一笔 FP** —— 这正是你看到的「`fp_*` 全都歪但都对、没有错检」，也让 FN/FP 计数从根本上**不能用来衡量模型好坏**。

GT 框切断连续墨迹的比例（即框住的是符号的一部分而不是整个符号）：
`gnd 58.7% / v 98.6% / r 92.8% / c 99.6% / testpoint 100% / box 94.2% / net_input 85.8%`。

---

## 三、根因二：val 不是 77 张独立图，而是 **19 张原理图的切片**

`trainset_v3\images\val` = 77 `.png`（+77 `.npy` 缓存）。按源原理图归并后：

| 源 | 文件数 | 占比 |
|---|---|---|
| `200_train_cases-0050` | **24** | **31%** |
| `200_train_cases-0011` | **16** | **21%** |
| 0101 / 0166 / 0127 / 0151 / 0117 / 0176 / 0002 / 0190 / 0027 / 0029 | 各 3 | 各 4% |
| 0023 / 0026 / 0047 / 0064 / 0121 / 0163 / `10_GTcase-0037-Datasheet` | 各 1 | 各 1% |

每张源图以 `__full` + 若干 `__tXX_YY` 切片**重复出现**。实测 `0027__full` 与 `0027__t00_00` 的类别直方图**逐项相同**。

**⇒ 两张图就占 val 的 52%。** 之前那条「77 张里约 5 张贡献了 44%~94% 的漏检、是少数渲染风格模型处理不了」是**重复计数的假象**，不是风格问题。

---

## 四、根因三：train / val **同源泄漏**（你猜的那条链）

7 个源 id 同时出现在 train 与 val：

| 源 id | train 侧文件名 | val 侧文件数 |
|---|---|---|
| `0050` | `synth_0050` | 24 |
| `0011` | `synth_0011` | 16 |
| `0002` | `synth_0002` | 3 |
| `0151` | `synth_0151` | 3 |
| `0027` | `synth_0027` | 3 |
| `0029` | `synth_0029` | 3 |
| `0026` | `synth_0026` | 1 |

合计 **53 / 77 = 68.8%** 的 val 文件，其源原理图在训练集里有合成版本（同一张原理图的另一种渲染）。

对照组（**没有** `synth_` 对照的 val 源）：`0101`、`0127`、`10_GTcase-0037` —— 这几个才是真正 held-out 的。

**⇒ 现行 val 指标不是 held-out 估计，不能用来判断模型好坏。**

---

## 五、对项目的影响与下一步

1. **必须重建 split**：按**源原理图**切分（`source_id(stem)` = `__` 前那段的最后一串数字），且把 `synth_<id>` 与 `200_train_cases-<id>*` 视为**同源**，禁止跨 split。重建前，任何 val 指标（包括我正在做的 FN/FP 归因）都只能当**定性线索**，不能当分数。
2. **标注需要复核**，优先三类：
   - `0050__full` 的 **28 个 `r`**（应为 `l` 或干脆不该有）—— 这是 `fn_r` 那一整张图的来源，也是 val 里 31% 的内容；
   - `net_input` 的 `30x10` 模板框扣在普通文字/导线拐角上的那些（`0011` / `0064` / `0027`）；
   - `testpoint`（val 共 94 个，高度集中在 `0011`/`0050`，而在 `0011` 上模型 **100% 漏检**）。
3. **模板框问题不只是标注问题，也是评估口径问题**：既然 GT 本身就是「随便套一个 8×20」，那么用 corner 容差 20px 判命中，等于在评一个和真实目标无关的东西。评估应当允许「同一颗器件被记成 FN+FP 各一笔」互相抵消，或者改判据。
4. `fn_box.png` / `fp_c.png` 重画：加**整图缩略 + 红框定位**，而不是只给 2.2× 局部裁片。

---

## 附：val 各源的完整标注直方图（`_leakpair.py` 实测）

```
val  0050__full        n=102  r=28 gnd=23 testpoint=20 c=13 box=9 v=6 block=2 jumper=1   ← l=0
train synth_0050       n= 23  gnd=7 c=5 box=3 net_input=2 r=2 testpoint=1 switch=1 net_short=1 net_output=1
val  0011__full        n= 61  testpoint=12 c=12 v=11 gnd=9 net_bidirection=6 box=5 r=4 block=1 net_input=1
train synth_0011       n= 19  gnd=5 r=4 box=3 c=2 battary=1 net_input=1 v=1 opt=1 d=1
val  0027__full        n= 45  c=11 gnd=11 net_input=9 r=7 l=2 v=2 box=1 buzzer=1 led=1
train synth_0027       n=111  c=26 gnd=24 r=15 v=13 box=12 net_input=4 net_short=3 l=2 ...
val  0002__full        n= 12  c=6 gnd=4 crystal_4pin=1 box=1
train synth_0002       n=  6  c=2 crystal_2pin=1 r=1 gnd=1 net_output=1
val  0029__full        n= 36  c=10 gnd=10 v=6 r=5 box=4 buzzer=1                       ← l=0
train synth_0029       n=130  gnd=29 r=24 v=19 box=18 c=16 pin=4 ...
val  0026__full        n= 29  c=11 gnd=10 r=4 box=2 buzzer=2
train synth_0026       n= 82  gnd=23 r=16 box=14 c=7 v=5 d=3 net_input=3 other=3 ...
val  0151__full        n= 66  c=20 v=13 gnd=9 r=9 box=9 d=3 l=3
train synth_0151       n= 98  gnd=25 c=16 r=13 box=8 v=8 d=6 net_input=4 led=3 l=2 ...
val  0101__full        n= 67  gnd=21 v=17 c=10 box=8 r=7 switch=2 crystal_2pin=1 led=1   （train 无 synth_0101 → 未泄漏）
val  0127__full        n= 38  r=11 gnd=8 switch=4 led=4 box=3 c=3 v=2 esd=2 net_input=1 （train 无 synth_0127 → 未泄漏）
val  0037-Datasheet    n= 13  c=4 r=3 bjt_npn=1 l=1 crystal_2pin=1 gnd=1 pin=1 v=1     （train 无 synth_0037 → 未泄漏）
```
