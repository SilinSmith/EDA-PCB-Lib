# LLM 循环与引脚链路（m25661 那批改动的交付账）

> 一切数字都是 **19 张定点档**（`work\tools\eval\fast.py`，外部时钟单轮 ≤20 分钟）与
> **220 张全量**（`work\_fusion_out220no4`）两档；单张只用来定位机理，不做判据。

---

## 0. 一句话

owner m25661 说的「LLM 根本没放进去」**是真的，而且根因是一行没有默认值的环境变量**。
修好之后 LLM 轮第一次真的跑起来了；它 + 引脚回传（`iterate`）合计 **+1.02**，
并行导线对齐与整表裁决 **中性**。真正的大头不是这些，而是**位号（组件键）**：
GT 的大 IC **全都被我们检出**（IoU 0.97~0.99，A 类 100% / B 类 0%），
死的是「这个框叫什么名字」。

---

## 1. ★★ 为什么 LLM 从来没进过生产档

`eda-pcb\stage1\llm.py:74-86`：

```python
def base_url():
    return os.environ.get("LLM_BASE_URL", "").rstrip("/")      # ← 空默认

def available():
    return bool(os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_MODEL"))
```

本机不设这两个变量 ⇒ `available()` **恒为 False** ⇒ 阶段三 LLM 仲裁
（`stage3\arbitrate.py`）与融合层 LLM 轮（`fusion\llm_judge.py`）**一次都没执行过**。
而**同一个端点**在 `eda-pcb\vlm\client.py:32-34` 是有默认值的，所以 VLM 一直好好的：

```python
BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen3.5-9b")
```

这解释了 `work\reports\节点活性体检.md` 里那两个「怪现象」：220 张
`stage3` 与 `stage2` **逐位相同**（不是模型不行，是根本没调用）。

**修法**：`base_url()` 与 `vlm\client.py` 对齐；新增 `model()`；
`available()` 改成**真探一次 `/models`**（`PROBE_TIMEOUT = env "LLM_PROBE_TIMEOUT" 5`）。
自检：`llm.available()` → `True`，`describe()` →
`model=qwen3.5-9b url=http://127.0.0.1:1234/v1 batch=8 workers=4 max_tokens=4096`。

**顺带堵上第二个坑**：`eda-pcb\submission\run.py` 原来写死

```python
os.environ["FUSION_LLM"] = "1" if a.fusion == "llm" else "0"
```

`-fusion` 默认是 `on` ⇒ **只要不显式写 `-fusion llm`，LLM 轮就是关的**
（`-mode` 只管阶段三，救不了它）。现在：

```python
FUSION_CROPFIX = "1" if a.fusion == "llm" else "0"
if a.fusion == "off":   os.environ["FUSION_LLM"] = "0"
elif a.fusion == "llm": os.environ["FUSION_LLM"] = "1"
# `on` 不覆盖 —— 尊重外部的 FUSION_LLM，而 fusion\run.py 的默认就是 1
_do_llm = os.environ.get("FUSION_LLM", "1").strip().lower() \
          not in ("0", "off", "no", "false", "none", "")
```

⇒ **现在 `-fusion on` 默认就带 LLM 轮。**

`-vlm` 侧可用模型（`http://127.0.0.1:1234/v1/models`）：
`qwen3.5-9b`、`qwen3.8-27b@q4_0`、`qwen3.8-27b@q4_k_m`、`qwen3.8-27b@iq3_s`、
`mistral-small-3.1-24b`、`qwen3-coder-next`。

---

## 2. 整表上下文裁决（`fusion\llm_judge.py` 的 `judge_sheet`）

### 为什么改

原来每道题**单独裁一张小图、单独发一次请求**（`MAX_ASK=64`，两轮最多 128 次）。
模型看不见同一张图上别的元件位号、看不见自己上一轮的判断、也看不见候选的来源。
这正是 owner 说的「你根本就没有准备可以好好判断的上下文环境」。
证据：自由读数模式的净效果是 **−0.765**（22.9943 → 22.2291），幻觉串靠
`LLM_SCORE=1.5` 盖过 `vlm:enrich`(1.15) 与 `ocr`(~1.0) 一票通吃
（`comp.c0018.key` 候选 `C13/C12` 却答 `C151`）。

### 怎么改

| 设施 | 值 |
|---|---|
| 模式 | `FUSION_JUDGE_MODE`（`auto\|read\|choose\|sheet`），新增 `sheet` |
| 开关 | `FUSION_JUDGE_SHEET` 默认 `1`；`FUSION_JUDGE_CELL` 260、`FUSION_JUDGE_COLS` 3、`FUSION_JUDGE_SHEET_MAX` 36 |
| 取景 | `_sheet_image()` 用 `reg.crop(bgr, scale=SCALE, focus_rect=False, mode=CROP_MODE)` 取**模型实际会看到的那张图**排成拼图；**格子上只印 `#k`，不印候选**（印了模型直接抄标签） |
| 全局约束 | `_known_designators(pool, qs)` 把「同一张图上已经确定、不要重复」的位号拼成一行（只列 `_is_read` 来源且非 `PLACEHOLDER` 的 `comp.*.key`，最多 60 个） |
| 一次请求 | `judge_sheet()` 复用 `judge()` 的跳过规则（无取景框 / 位号是占位符），**一次 `client.ask` 答完** |
| 回包解析 | `SHEET_LINE = ^#?\s*(\d{1,3})\s*[:：.、]\s*(.*)$`；`A`→top、`B`→alt、`OTHER xxx`→原文 |
| 共用结算 | `_settle(pool, q, ans, raw, dt, reg, row, rep, log=None)` —— 单题与整表两条路径共用 `_admit` + 「不许给先验加冕」+ `pool.add`，**抄一份必然发散** |

`judge()` 顶部接线：

```python
if mode == "sheet" and SHEET_ON:
    return judge_sheet(pool, qs, limit=min(limit, SHEET_MAX), client=client, log=log)
```

### 实测（0143）

```
[judge:sheet] 9 题一张表 · 回 9 行 · 采信 9 · 拒 0 · 0.6s
'rounds[0].judge' = {'mode':'sheet','n_q':20,'n_asked':1,'n_answered':9,'n_new':0,
                     'n_agree_top':9,'n_agree_alt':0,'n_other':0,'n_reject':0,
                     'crop':'tight','sec':0.6,'has_image':True,'n_skip_ph':11,
                     'sheet':'work\\preview\\_judge_sheet.png','n_raw':9}
```

9 道题全是位号二选一（`comp.c0004.key top=C6 alt=C16 -> vlm=C6 [agree_top]`、
`comp.c0003.key top=U1 alt=U11 -> vlm=U1`…），模型 **9/9 同意 top** ⇒ 分数不动。

### 19 张定点档 A/B（都是 VLM 开）

| 臂 | 均值 | 10GT(9) | 200_train(10) | comp | pin | netH | netL |
|---|---|---|---|---|---|---|---|
| `iter_on`（`FUSION_LLM=0`）| 30.05 | 44.32 | 17.21 | 0.6489 | 0.0344 | 0.2670 | 0.0375 |
| **`llm_sheet`（`FUSION_LLM=1` + sheet）** | **29.94** | 44.20 | 17.10 | 0.6477 | 0.0344 | 0.2643 | 0.0397 |

⇒ **−0.11，落在跑间噪声 ±0.3 内。**
对比历史 −0.765 ⇒ **换成「A/B 候选 + 整表上下文 + 拒收 OTHER」之后，
LLM 轮从拖后腿变成不亏**。这是它**第一次**真的接进生产档，值钱的是
「不再关着」，不是这 0.11。

基线：`work\_fast_base\llm_sheet.json`。

---

## 3. `fusion\iterate.py`：把 LLM 读到的引脚号/名回传

**动机**：`pin_f1` 占 **25%** 权重，220 张只拿到 **0.92/25**。
`stage1` 给 `0143` 的 `U1` 只发了 **8** 个引脚（GT 要 64 个）。

**通道**：`stage1\pincol.py` 枚举出的框外「号码窗」`out_win` 与框内「名字窗」`in_win`
交给 LLM 读 → 写回证据池 → `stage2` 用**真读数**重新装配。

### 三个必须踩过的坑

1. **必须按元件分批**。原来把 `c0001`(12 窗) + `c0003`(54 窗) 拼成一个平表切批，
   一张拼图里混进两个元件 ⇒ 模型把 `VBAT` 读成 `VDD`。改成
   `_collect` 返回 `[(cid, [(k, 裁剪), …]), …]`，`_ask` 逐组再按 `PER=8` 切批后，
   **只喂 `c0003` 的 54 个窗是 54/54 全中**。
2. **不能用最小二乘**。一个离群读法（L 边 index 8 读成 `10`、真值 `9`）会把整条边的
   截距拉偏 1，实测把 `1:VBAT` 拉成 `2:VDD`，tp 从 20 掉到 11。
3. **不能逐点照抄**。0143 的 R 边（GT 自上而下 `64,63,…`）只枚举出 22 个格点、
   中间有洞，照抄在洞那里错位。

### 定稿：锚点 + 梯子，且只补洞不外推

```
num_k = num_anchor + round((pos_k − pos_anchor) / pitch) · step      step ∈ {+1, −1}
```

锚点就是读到的号本身 ⇒ 永不错位；离群读法只影响自己那一格。
`pitch` 取该边各记录 `pitch` 的中位；步长取「能让最多锚点自洽」的方向
（L 边自上而下升序、R 边自下而上升序都自动正确）；
要求 `支持数 ≥ max(2, 0.5×读法数)`，否则整条边不拟合。

**新增 `GAP_MAX = env "FUSION_ITER_GAP" 默认 2`**：格点离**最近锚点**超过
`GAP_MAX` 格就留白。理由 —— 全边外推只对**稠密升/降序**符号成立
（0143 的 STM32 LQFP），而 19 张里的 GT 号大多是**稀疏 BGA 球号**：

| case | GT 引脚号 | 我们（全边外推）|
|---|---|---|
| `0001 U1` | `1,2,4,5,6,7,33,34,59,61..64,68,69,EP,391..400` | 稠密 `1,2,3,4,6,7,8,10,13,16,…,38`（凭空造号）|
| `0011 U5` | `2..9,21,31,39,40,41,1_1,1_2,317..327` | `1,2,3,4,6,7,8` |
| `0050 U1` | `2..9,447,450..458,461..463,MCPF1412M06` | `2..22` |

`FUSION_ITER_GAP ≥ 999` 即退回旧行为。

`pin_fields()` = `pin_numbers`（读号）→ `_smooth_numbers`（补洞）→ `pin_names`（读名）；
`fusion\run.py` 的调用点由 `IT.pin_names(...)` 换成 `IT.pin_fields(...)`。
`_selftest` 扩到 **23 过 / 0 不过**（含 L 边升序 + 一个离群读法、R 边降序、
只有一个读法时不拟合三条纯逻辑用例）。

### 实测

| 臂 | 均值 | pin F1 | tp 引脚 | 说明 |
|---|---|---|---|---|
| `vlm_ours`（无 iterate）| 29.03 | 0.0088 | 1 | 基线 |
| **`iter_on`（全开）** | **30.05** | **0.0344** | **18** | **净 +1.02** |

14 张阶梯（0143，`-mode off`）：

| 臂 | 总分 | pin F1 |
|---|---|---|
| `_pin1_iter`（只补名字，号走位置兜底）| **46.85** | 0.2532 |
| `_pin1_iter2`（号逐点照抄）| 44.77 | 0.1401 |
| `_pin1_iter3`（号最小二乘外推）| 42.35 | 0.0506（`1:VBAT`→`2:VDD`）|
| `_pin1_iter4`（按元件分组 + 照抄）| 46.83 | 0.2152 |
| `_pin1_iter5`（按元件分组 + 梯子）| 45.84 | 0.1899 |

⚠ **0143 的 pin F1 与 19 张均值方向相反**（0143 上 iterate 让 pin 从 0.2532 掉到 0.19，
19 张均值却从 0.0088 涨到 0.0344）——
因为 0143 只有一个大 IC、而它的 GT 本身**错位且有噪声**
（GT 键 `1 VBAT / 2 PC13/TRC_AF1 / … / [没有 5、6] / 7 NRST`，图上却是
`5 PH0/OSC_IN / 6 PH1/OSC_OUT / 8 PC0`；另有 `021`、`PAT3`、`BOOTO`、`1_2`、
`56_1`、`co`、`o`、`65/66/67` 这些不是引脚号的键）。
**判据一律看 19 张均值。**

新诊断：`work\tools\diag\_pinmatch1.py`（逐脚打印「GT 号 名 | 我们 号 名」）。

---

## 4. 并行导线等距对齐（`pin_synth` → 紧接线半径）

**机理**：IC 引脚列实测间距 **10.0 px**（0143 `U1`：GT 64 脚 = 左 32 + 右 32，
同一个 y 序列，间距精确 10.0），而 `eda-pcb\stage1\nets.py` 的
`PIN_JOIN_DIST = env "STAGE1_PIN_JOIN_DIST" 12` **比这一格还大** ⇒ `pincol`
按格点**补出来的悬空脚**会认领**隔壁脚的导线**，把两条网络并成一条。

**改法**：

* `stage1\pincol.py` 的 `merge_into(comp_pins, cols, snap=MERGE_SNAP)` 现在返回
  **(新 comp_pins, 每脚 synth 标志, 统计)**，标志与 `comp_pins[cid]` **逐位对齐**
  （`1` = 格点补的、框边外侧 3px 内没有墨 ⇒ 没有导线；`0` = 有真实触点，
  或末尾那个没人认领的老触点）；
* `stage1\nets.py` 新增 `PIN_JOIN_SYNTH = float(env "STAGE1_PIN_JOIN_SYNTH" 3)`，
  签名扩成 `build_nets(..., report=None, pin_synth=None, join_synth=PIN_JOIN_SYNTH)`，
  收集引脚时建 `JD = np.asarray(jd)`，`_pick(m)` 用 `jdm = float(JD[m])`；
* `stage1\pipeline.py` 解包三元返回进 `pin_synth` 并透传。
  `PIN_JOIN_SYNTH >= join_dist` 即退回历史行为。

自检 `python eda-pcb\stage1\pincol.py` → 7 组全过，含
`7) merge_into  synth 标志 got [0, 1, 0] want [0, 1, 0]`。

**实测**：

| 场景 | JOIN_SYNTH=12（旧） | JOIN_SYNTH=3（新） |
|---|---|---|
| 0143 单张（隔离 stage1）| 39.0581（orphans 8、netH 0.4088 tp37/fp44/fn63）| 38.9940（orphans 21、netH 0.4070 tp35/fp37/fn65）|
| 19 张（VLM 开，`FUSION_LLM=0 FUSION_ITERATE=1`）| **29.94** | **29.98** |

⇒ **中性（+0.04）**。0143 上多认出的 13 个悬空脚里只有 2 个本来是真 tp。
逐张：`0001 +0.95`、`0092 +0.92`、`8 −0.39`、`0086 −0.45`、`0103 −0.29`。
基线：`work\_fast_base\jsyn12.json` / `jsyn3.json`。

保留默认 `3` 的理由：它**不会变坏**（−0.06 单张 / +0.04 均值），
而且把「补出来的假脚不许抢隔壁导线」这条**正确的物理约束**写进了代码。

---

## 5. ★★ 引脚失分到底卡在哪：**位号**，不是枚举

新工具 `work\tools\diag\_pinloss.py`（口径与官方 `evaluate_case` 一致，
认三种产物目录布局：`<dir>/<case>.json`、`<dir>/final_results/<case>_Pred.json`、
`<dir>/runs/fusion/<case>.json`）。

### 19 张

```
GT box 元件 93，按键匹配上 33 (35.5%)
GT box 引脚 1211：匹配上的元件占 583（48.1%），没匹配上的元件占 628（51.9%）
[键已对上] tp 18  名字错 220  位置错 23  没这个号 322  多出来的号(fp) 241
按官方事件口径: tp 18 fp 484 fn 1193  P 0.0359 R 0.0149 F1 0.0210
匹配上的 box 里「一个引脚都没发」的有 0 个 / 33
```

⇒ **一半的箱体引脚死在「元件键没对上」**，且**匹配上的元件没有一个「不发引脚」**。

### 220 张（同一口径）

```
GT box 元件 996，按键匹配上 402（40.4%）
GT box 引脚 9356：没匹配上的元件占 4779（51.1%）
[键已对上] tp 206  名字错 620  位置错 195  没这个号 3556  多出来的号(fp) 834
```

### 新工具 `work\tools\diag\_boxkey.py`：**A 类 100%，B 类 0%**

```
GT box 元件 93 · 键对上 36 (38.7%) · 键没对上 57
  A 类（框位置上有我们的预测，只是键不同）: 57  (100.0%)
  B 类（框位置上一片空白 ⇒ 根本没检出）   : 0   (0.0%)

  0001 U1    A 检出但键错  GT框 167x667   与它重叠的: [(0.979, 'USB3')]
  0002 IC2   A 检出但键错  GT框 247x372   与它重叠的: [(0.977, 'U0')]
  0011 U1    A 检出但键错  GT框 238x358   与它重叠的: [(0.984, 'U4')]
  0013 U202  A 检出但键错  GT框 100x280   与它重叠的: [(0.988, 'U22')]
  0019 u1    A 检出但键错  GT框 173x224   与它重叠的: [(0.980, 'U11'), (0.448, 'U1')]
  0019 U2    A 检出但键错  GT框 22x66     与它重叠的: [(0.965, 'p000')]
  0019 U4    A 检出但键错  GT框 22x108    与它重叠的: [(0.968, 'p006')]
```

**大 IC 一个都没漏检（IoU 0.97~0.99），死的全是位号。**
其中 `p000`/`p001`/… 是**占位符**（`PLACEHOLDER = ^(?:p\d{3,4}|∅\d+)$`），
意思是**根本没读到任何位号**；`U11`（真值 `u1`）、`USB3`（真值 `U1`）、
`U22`（真值 `U202`）是**读错**。

⇒ **P0 = 位号（组件键）**，它就是 `pin_f1` 与 `net_hypergraph_f1` 的**共同闸门**。
新工具 `work\tools\diag\_boxcrop.py`（`--case/--key/--pad/--scale`）把 GT 框连同
四周留白裁出来放大，用来肉眼确认「位号到底画没画在图上」——
`0019 u1` 的裁片显示 `U1` **画在框内左上角**（黄色），
`0002 IC2` 的在框**下方**居中，`0143 U1` 的在框**上方**偏左。

---

## 6. 官方评测口径（照抄，用于判优先级）

* **分数**：`score = 100 × (0.30·component_f1 + 0.25·pin_f1 + 0.35·net_hypergraph_f1 + 0.10·net_line_f1)`
  （用 `work\_fusion_out220no4\_score.json` 的 220 个 case 做**无截距最小二乘**精确解出
  `[30, 25, 35, 10]`；回代 case `0001` 得 31.8310，与其 `score` 完全一致。）
  当前 220 张 **28.4973** = comp 19.17 + pin 0.92 + netH 7.82 + netL 0.30。
  **可提升空间：pin 24.08、netH 27.18、component 10.83。**
* `pin_f1`（`schematic_parser\metrics.py:426-464`）：**只统计 GT 里 `canon_type(type) ∈ IC_TYPES = {box, ic, boxic}` 的元件**；
  配对键 = `(元件键, pin_key_to_number(引脚号))`；逐对要
  `pinname` **字符串精确相等**、点距 `≤ cfg.pin_tol = 5.0`；
  `flat()` 会**丢弃键不在 box_comps 里的预测引脚**（⇒ 键没对上的引脚连 fp 都不算）。
* `net_line_f1`（`metrics.py:522-562`，权重仅 10%）：只统计**匈牙利配对成功的网**的边，
  每条预测边要有一条 GT 边 `_seg_distance ≤ cfg.line_tol = 5.0` 才算 tp；
  没配上的网把它的边全算 fp/fn。220 张 tp 774 / fp 17673 / fn 32263
  ⇒ **我们只产出 GT 边数的约 56%。低价值，先不动。**

---

## 7. 没有转正的开关（以及为什么）

| 开关 | 默认 | 19 张实测（VLM 开，基线 `vlm_on` 29.93） | 结论 |
|---|---|---|---|
| `FUSION_JUDGE_MODE=sheet` | 已是新默认 | 29.94（vs 30.05）| **保持开**：中性，但它是 owner 要的「LLM 进框架」 |
| `FUSION_ITERATE` | `1` | 29.03 → 30.05 | **保持开**（+1.02）|
| `FUSION_ITER_GAP` | `2` | 29.93（vs 旧代码 29.98）| **保持 2**：中性且不造号 |
| `STAGE1_PIN_JOIN_SYNTH` | `3` | 29.98（vs 12 的 29.94）| **保持 3**：中性且物理正确 |
| `STAGE2_PREFIX_SKEL` + `STAGE1_CV_WIN=1` + `STAGE1_DROP_TYPES=none` | 关 | **29.97**（vs 29.93，+0.04；tp 元件 382→384、网络 247→251、导线 59→63）| 见下 |

三项一起开（`tpvlm` 臂）**在 19 张上中性偏正**（+0.04，且 tp 元件/网络/导线都 +2~+4）。
但 VLM 关时的同一对照是 **24.46 vs 24.73（−0.27）** ⇒ **这一项对 VLM 开关敏感**，
而且它在 19 张里几乎没有可受益的对象（这批图里 testpoint 极少）。

**22 张定点档（19 + `0011/0050/0102`，全部 VLM 开）—— 最终判据：**

| 臂 | 均值 | 10GT(9) | 200_train(13) | comp | pin | netH | netL | tp 元件/引脚/网络/导线 |
|---|---|---|---|---|---|---|---|---|
| `c22_base`（默认）| **28.95** | 44.27 | 18.34 | 0.6331 | 0.0297 | 0.2542 | 0.0316 | 480/18/330/59 |
| `c22_tp`（三开关全开）| **28.93** | 44.27 | 18.31 | 0.6289 | 0.0297 | 0.2572 | 0.0321 | **494/18/347/61** |

逐张差值：`0050 **+2.66**`、`0092 +0.88`、`0086 +0.78`、`0001 +0.38`、`0075 −0.01`、`0019 −0.03`、
`0024 −0.15`、`0011 **−1.64**`、`0102 **−3.18**`，10GT 九张**逐位相同**。

⇒ **判决：不转正。** 均值 −0.02（中性），tp 计数全面上涨（元件 +14、网络 +17、导线 +2），
**但收益最佳一例只有 +2.66，亏损尾部却有 −3.18 与 −1.64**。
在「测试集构图未知」的比赛里，「零期望 + 肥负尾」是坏赌注。
开关全部保留（`STAGE2_PREFIX_SKEL=0`、`STAGE1_CV_WIN=off`、
`STAGE1_DROP_TYPES=pin,testpoint,mosfet_npn` 为默认），谁想复现只改环境变量。
基线：`work\_fast_base\c22_base.json` / `c22_tp.json`。

**根因（`0102`）**：新增的 8 个 testpoint 在原图上**没有任何 `TPn` 文本**，
而官方 `metrics.py` 只按元件 key 匹配（`is_unkeyed` 只认 `∅NNN`/`pNNN`）
⇒ **检出一律算 fp**。默认丢 testpoint 反而规避了这一点。
⇒ 想让这项转正，**必须先解决「匿名元件怎么起键」，也就是下一节那个大闸门。**

`STAGE2_PREFIX_SKEL` 的来历（testpoint 子代理 `ba57a3f6-…` 交付）：
`stage2\assign.py:316-322` 加一档 —— 读出的前缀是某个允许前缀的**后缀**时补全成最短的那个
（`P21`→`TP21`，保真度 0.7）。单张实测 `0050` **+3.89**（键交集 0/20→15/20）、
`0102` **−1.03**（新增 8 个匿名 testpoint 全是 fp）、`0011` 不变。
`0011`/`0102` 的 testpoint 在原图上**没有任何 `TPn` 文本**，而官方 `metrics.py`
只按元件 key 匹配（`is_unkeyed` 只认 `∅NNN`/`pNNN`）⇒ **默认丢 testpoint 反而规避了这点**。

---

## 8. ⚠⚠ 方法学教训：`--no-vlm` 是 5.2 分的陷阱

**这是本轮最容易骗到自己的一处。**

`work\tools\eval\fast.py --no-vlm` 会传 `-vlm off`。同一份代码、同一天：

| 命令 | 均值 |
|---|---|
| `fast.py --save vlm_on`（**不**加 `--no-vlm`）| **29.93** |
| `fast.py --no-vlm --save recheck` | **24.73** |

⇒ **VLM 位号/值读取值 5.2 分**（component F1 0.6483 → 0.5260）。后果：

* 两批臂（VLM 开 29.9x / VLM 关 24.4~24.7）**绝对不能混在一张表里比**；
* 中途我拿 `jsyn3`(29.98，VLM 开) 当 `skel`(24.46，VLM 关) 的对照，
  差点得出「改动掉了 5.5 分」的假结论；
* `0143` 单张的 component 也随 VLM 开关在 `0.6667`（关）与 `0.8750`（开）之间跳。

**规矩**：写进臂名（`vlm_on` / `novlm`），且**一次 A/B 里 VLM 开关必须保持一致**。

另外两条同源教训：

* **管线有跑间噪声**，同配置复跑 19 张均值抖 ±0.1~0.5
  （`iter_on` 30.05 / `llm_sheet` 29.94 / `vlm_on` 29.93 / `jsyn12` 29.94 / `jsyn3` 29.98）。
  **只看均值方向，不看单张。**
* 子代理实测：同配置复跑 27 个 case 里 **13 个**的 `nets` 数/引脚名不同
  ⇒ **整档 A/B 只能比分数，不能比文件哈希；`work\_fast_out` 不能当基准。**

---

## 9. 复现命令

```powershell
cd Y:\workspace\EDA
$env:PYTHONIOENCODING='utf-8'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$env:FUSION_LLM='0'; $env:FUSION_ITERATE='1'
python work\tools\eval\fast.py --save vlm_on --cmp jsyn3        # VLM 开基线，~6 分钟

# LLM 轮（整表）对照
$env:FUSION_LLM='1'; $env:FUSION_JUDGE_MODE='sheet'
python work\tools\eval\fast.py --save llm_sheet --cmp iter_on

# 单张探针（直调 fusion.run.run_case，先打印 DO_LLM/MAX_ROUNDS/JUDGE/MODE/SHEET_ON）
python work\tools\diag\_judgeprobe.py 0143 work\_fast_out

# 引脚失分 / 位号归因 / 裁图
python work\tools\diag\_pinloss.py --dir work\_fast_out --show 6
python work\tools\diag\_boxkey.py  --dir work\_fast_out --show 20
python work\tools\diag\_boxcrop.py --case 0019 --key u1 --pad 0.35 --scale 1.6
python work\tools\diag\_pinmatch1.py 0143
```

⚠ PowerShell 坑：`$env:STAGE1_DROP_TYPES=''` 是**删除**变量（回落默认），要写 `'none'`；
`-cases 0011,0050,0102` **必须加引号**，否则被 `fast.py` 拼成 `'11,50,102'`。

---

## 10. 还没做的

1. **位号（组件键）**：A 类 100% ⇒ 这是唯一还在的大头。方向：
   ①`box` 元件的位号**专门走一次 LLM 取景读**（框内左上 / 框上 / 框下三条带，
   带框线一起喂）；②`iterate` 已经证明「按元件分组 + 只问一个字段」可行；
   ③大小写（GT 里真有 `u1`、`u2` 这种小写键）与 `PLACEHOLDER` 占位符的处置。
2. `fusion\__init__.py` 的 `__all__` 已补 `iterate`/`run`，并把「谁真的在干活」的
   实测表写进 docstring；`know.py` / `conflict.py` **仍未写**。
3. `iterate._smooth_numbers` 只补 ≤2 格已落地并自检，但**只量了 19 张均值（中性）**，
   没有在 220 张上复核。
