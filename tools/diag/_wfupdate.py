# -*- coding: utf-8 -*-
"""把本会话（2026-10-07）的改动与实测写进 work/flow/workflow.html 的 graphdata。

用法：python work/tools/diag/_wfupdate.py
会自动备份成 workflow.html.bak7。
"""
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "work" / "flow" / "workflow.html"
BK = P.with_suffix(".html.bak7")

NAME = ("目标架构 · 证据层 → 约束求解 → 迭代重扫"
        "（实测：官方 10GT 9 张 56.71 / 派生 19 张 35.39 · 2026-10-07 · VLM 开；"
        "同一批官方 9 张在本会话起点是 44.16）")

NOTE_HEAD = """**★ 第六批（本会话 2026-10-07）的四项改动与一条方法学**

本会话把官方 `10GT` 的 9 张从 **44.16 拉到 56.71（+12.55）**，派生 19 张 34.77 → 35.39。
四次改动全部落在**证据层与归属层**，方向是一致的：**用「形状/拓扑关系」替换「距离」当闸门。**

| # | 改了什么 | 文件 | 官方 9 | 派生 19 |
|---|---|---|---|---|
| 1 | 稀疏封装引脚号不再被位置先验覆盖（`FUSION_BOX_PIN_PRIOR=0`） | `fusion/priors.py` | 44.16 → **53.41** | — |
| 2 | 并网改看「端点微越过 + 不平行」（规则 5，`STAGE1_WIRE_OVER=12`，`WIRE_TOL` 回退 1.5） | `stage1/labels.py` · `stage1/nets.py` | 55.73 → **56.31** | 34.69 → **35.23** |
| 3 | 文字归属改量「框到框」而不是「文字中心到框」（该分支） | `stage2/assign.py` | 56.31 → **56.45** | 34.69 → **35.28** |
| 4 | 引脚号候选的沿边衰减加快（`STAGE2_PINNUM_DECAY` 20 → 10） | `stage2/assign.py` | 56.45 → **56.71** | 35.28 → **35.39** |

**★ 方法学（本会话踩出来的，比上面四条更值钱）**
1. **`--no-vlm` 值 5.2 分。** 同一份代码、只差一个开关，均值 29.93 vs 24.73；`component_f1`
   0.6483 vs 0.5260。**两批实验绝对不能同表比**。凡报数字先写清 VLM 是开是关。
2. **共用输出目录会让诊断漂移。** `work/_fast_out` 每次跑都被覆盖，同一条命令隔一次跑就能得到
   57 与 60 两个「未匹配元件数」。现在一律用 `fast.py --out <名>` 起独立目录。
3. **自造的 GT 会骗你。** `10GT`（竞赛格式）box 键对上 **21/21 = 100%**；派生 `200_train_cases`
   按画图工具是 `jlc` 46.6% / `Altium` 35.3% / `Datasheet` **21.2%** / `other` **7.3%**。
   594 个未对上的 box 里 **80.3% 数学上不可赢**（占位符 294 · GT 里压根没这个键 183）。
   在派生档上追分 = 对着噪声调参。详见 `reports/GT保真度与位号上限.md` 与「GT 保真度检查」节点。

---

"""

NOTE_TAIL_APPEND = """

---

**★★ 第六批（本会话）在 `note` 顶部，此处以下为历史批次记录。**
"""

EV_WIRE_ADD = """

**【本会话 · 规则 (5)：导线图过碎的修法不是「放大容差」，而是「看形状关系」】**
- **病灶现场（官方 case 1）**：GND 母线整条检测正确（`net_31`，25 段，含 5 个结点圆点），6 个电容的
  引脚点与 GT **逐位吻合**，但其中 4 个电容的引线段**下端比母线低 5px**（238 vs 233）⇒ 与母线不在同一
  连通分量 ⇒ 各自成 3 段的孤立网。
- **`STAGE1_GROUP_TOL` 是空挥**：扫 6 / 12 逐位相同（`labels.py:47` 的 `GROUP_TOL` 只给
  `label_candidates` / `free_ends` 用）。并网真正的入口是 **`stage1/nets.py:132`：
  `tol = labels_mod.WIRE_TOL if tol is None else tol`**。
- **直接抬 `WIRE_TOL` 到 10 = 零和交换**：官方 9 张 55.07 → 55.89（+0.82），但派生 10 张 **−0.75**。
  因为抬容差是**用一个各向同性的半径去补一个有方向的几何缺陷**，必然把**电容的两块平行极板**
  （相隔 4~7px、平行）一起并上 ⇒ case 10 长出 **11 脚巨块**（横跨 GT 的 `net_2`/`net_5`/`net_8`/`net_13`）。
- **落地**：`labels.py:_overshoot_unions(segs, ends, own, over, edge=0.03, cross_only=True, cos_par=0.9)`
  —— A 的端点到 B 的**线段**距离 ≤ `WIRE_OVER=12`、垂足落在 B 内部（3% 边距）、且**两段不平行**
  （`|cos| >= 0.9` 即排除）。`WIRE_TOL` 回退历史值 **1.5**。env：`STAGE1_WIRE_OVER` /
  `STAGE1_WIRE_OVER_CROSS`（默认 0）/ `STAGE1_WIRE_OVER_COS`。
- **记分**：官方 9 **55.73 → 56.31**（netH 0.5190 → 0.5339）、派生 19 **34.69 → 35.23**
  ⇒ **两个档都赚**（钝补丁是一赚一亏）。`OVER` 平台 12~14，取 12。`WIRE_OVER_CROSS=1`
  （改成要求两端点在目标段支撑线异侧）反而 55.38 —— **同一个错误的镜像，别开**。
- **同轮证伪的第二条路（`weld` / `ext` 桥接）**：`WIRE_WELD` 24 → 54.33、32 → 51.86；
  `WIRE_EXT` 20 → 53.55、30 → 50.48。机制（`work\\tools\\diag\\_weldprobe.py` 的 `comp_between()`）：
  40 个「共线、相对」的长缺口候选**几乎全部坐在元件本体上**（`c0006(c)` / `c0013(gnd)` / `c0026(r)`…），
  只有 2 个中间是空的 ⇒ **抬 `weld` = 把所有串联元件短路**，与「结点圆点把导线切断」纯几何不可分。
  另：`wire_detail` 的 `junctions = 0`，我们**根本没检出结点圆点**，这条路也不通。
- **教训：「距离」是坏闸门，「形状关系」才是好闸门。**

**实测开关（`work/_fast_out_off9_ov`）**：`WIRE_TOL = 1.5`（回退）、`WIRE_OVER = 12.0`、
`WIRE_OVER_CROSS = False`、`WIRE_WELD = 16.0`（未动）、`WIRE_EXT = 8.0`（未动）、
`STAGE1_SPLIT_JUNCTIONS = 1`（扫过 0，+0.14，未转正）。"""

EV_PIN_ADD = """

**【本会话 · 引脚点比真脚多 = 归属错】**
- 官方 case 2 的 `U1`：GT bbox JSON `[629,721,687,819]`，**GT 只声明 4 个脚**（img y = 316/335/356/376），
  我们给了 **18 个点**（左边 7 个、上边 3 个 y=285、下边 8 个 y=396~408 —— 上下两排 y **已在 bbox 之外**，
  是邻居元件的触点混进来）。左边 7 个 = 307/316/326/336/354/370/376，真只有 316/336/354/376 对上 GT，
  **多出 307/326/370 卡在两个真脚中间**；文字框只有 3 块（`2`@330 `3`@350 `4`@370）却**每块被重复检出两次**。
  单调对齐在「7 点 × 6 候选」上把 `cover = PIN_COVER = 0.15` 算进去后，最优解就是**整排挪一格**：
  `pin_2`(y=336) 写上了 `3`。
- ⇒ **归属错的根因是「我们的引脚点比真脚多」，不是「谁离得近」。**
- 引脚列枚举的旋钮全在 `stage1/pincol.py`（`STAGE1_PINC_*` 约 22 个），`merge_into` 返回的
  `pin_synth` 逐位标志经 `stage1/nets.py` 的 `PIN_JOIN_SYNTH=3` 控制「格点补出来的悬空脚」能否认领导线
  （要防它认领隔壁脚的导线，因为 IC 引脚列实测间距只有 **10.0px**）。"""

OWN_NOTE = """**★★ 归属判据的真凶（本会话修复）：`FAR` 量的是「文字**中心**到元件框」，于是归属随文字长度变化。**

- **现场（官方 case 1 的 `U1`，框 `[961,56,1314,821]` = 353x765）**：框外 70px 内**正好 26 个小文字框**
  （左 19 / 右 7），与 GT 的 26 个脚**一一对应**；但 `[pin:pin_number] nplist=34 pool=21 cov=0.15 bound=16 amb=0`
  —— **候选池只有 21**。
- **根因**：`stage2/assign.py:54 FAR = 40.0`，归属用 `ds = [dist_to_rect(cx, cy, c["bbox"]) ...]`
  （**文字中心**到元件框）。于是 1 位号 `7`（宽 12）中心距 28 ⇒ 进；2 位号 `26`（宽 24）距 35 ⇒ 进；
  **3 字符 BGA 球号**（`101/113/114/115/122`，框宽 32~36px）中心距 **40~45 > 40** ⇒ **整排被判成 `free`（网络标签）**。
- **修法（`eda-pcb/stage2/assign.py`）**
  1. 新增 `def dist_rect_to_rect(a, b) -> float`：两矩形间距，重叠即 0
     （`dx = max(b[0]-a[2], 0.0, a[0]-b[2])`，`dy` 同理）。
  2. 新增 `OWN_BBOX = True`。
  3. 归属判定 `d_own = dist_rect_to_rect(b, comps[j]["bbox"]) if OWN_BBOX else d`，
     `rec = dict(t, d=d, own=d_own, ...)`，`if d_own <= far: owned[...] else: free.append(rec)`；
     **`rec["d"]`（中心距）原样保留**，因为评分与 `NEAR` 判据用的是它。`OWN_BBOX=0` 退回历史行为。
- **效果**：候选池 **21 → 26**、`bound` **16 → 24**；官方 9 **56.31 → 56.45**、派生 19 **34.69 → 35.28**
  （两个档都赚）。但 **`pin_f1` 一分未动**（增量全在 netH）—— 新进池的 `113/114/115` 只读出 `13/14/15`，
  是 CRNN 丢数字，不是归属错。
- **这是同一类错误**：位号那边表现为「把**型号**（`LM385` / `MAX232`）当位号」「把**邻居的位号**
  （`K11` / `H1` / `Q1`）抢来」；引脚号这边表现为「文字一宽就被踢出归属」。共性是
  **用了与「文字本身有多宽/多长」耦合的度量，去判「它贴不贴在这颗元件上」**。"""

SOLVE_ADD = """

**【本会话 · 稀疏封装引脚号被位置先验覆盖（已修）】**
- **`fusion/priors.py` 的 `prior:pinrun`**（`:698-728`）原来给没有读数或撞号的引脚槽位补号
  （`slots = ["pin.%s.%d.number" % (cid,k) …]`、`taken = {n for n,c in cnt.items() if c==1}`、
  `pool.add(sid, str(num), score=_cap(...,0.5), source="prior:pinrun", …)`）。
- **代价**：官方 case `1` 的 `U1` GT 是**稀疏 BGA 球号** `pin_7,8,9,10,16,17,26,28,39,47,48,51,52,58,101,113,114,115,122…`
  （26 个）。`runs/stage2/_assigned/1.json` 里只有 15 个脚有号、且是真稀疏号；到 `runs/fusion/_assigned/1.json`
  变成 **`1..34` 连续**、全部标 `_by = "solve:prior:pinrun"`。补齐后 `stage2/assemble.py:640-645` 的
  `ok` 闸门（`all(n.isdigit()) and 互不相同 and 1<=v<=999`）由 **False 变 True** ⇒ 整颗 IC 的键被换掉。
- **记分**：官方 `10GT` 9 张 pin tp **49 → 3**，总分 **58.15 → 44.16**。这是本会话前半段「派生档涨、官方档崩」的元凶。
  （`-fusion off` 也躲不掉 —— 稠密号在 stage2 就已经出现。）
- **修法**：`BOX_PIN_PRIOR = env_on("FUSION_BOX_PIN_PRIOR", "0")` 与
  `_PRIOR_IC_TYPES = ("box", "ic", "boxic", "bigic")`；`pinrun` 循环体开头
  `if not BOX_PIN_PRIOR:` 取 `pool.meta["comp.%s.key" % cid]["type"]`，落在 IC 类型就
  `st["pin_prior_skip_ic"] += 1; continue`。非 box 引脚号不进 `pin_f1`（官方评测 Q2），所以只跳过 IC。
- **效果**：官方 9 张 **44.16 → 53.41**（pin F1 0.0350 → 0.2942、netH 0.4105 → 0.4866）。
- **教训**：IC 的引脚号是**图上印出来的封装号**，天然稀疏（BGA / QFN 只画一部分球）；先验「位置即编号」
  只适用于那些**位置就是编号**的符号。"""

METRICS_ADD = """

**【本会话 · GT 保真度：自造的 GT 会骗你】**
- **同一天、同一份代码**：官方 `10GT`（竞赛格式，`10GT\\1\\{1.png,1.json}`，文件名**不带** `-<工具>` 后缀）
  9 张 **58.15**；派生 `200_train_cases` 的 box 键对上率按画图工具（`work\\tools\\diag\\_desigsrc.py`）：

  | 来源 | case | box 元件 | 对上 | 对上率 |
  |---|---:|---:|---:|---:|
  | `jlc` | 99 | 648 | 302 | **46.6%** |
  | `Datasheet` | 69 | 179 | 38 | **21.2%** |
  | `Altium Designer` | 23 | 85 | 30 | 35.3% |
  | `other` | 13 | 41 | 3 | **7.3%** |
  | **`10GT(?)`（竞赛格式）** | 10 | 21 | **21** | **100.0%** |
  | `KiCad` / `kicad` | 6 | 22 | 8 | 36.4% |

- **594 个未对上的 GT box 元件的失效模式**：占位符（我们没读到）**294** · 前缀不同（GT 里没这个键）**183** ·
  前缀同（号读错）69 · 我们的键在 GT 里（分配错位）45 · 没检出 3
  ⇒ **可赢 ≈ 117（19.7%），不可赢 ≈ 477（80.3%）**。
- **肉眼铁证**：`200_train_cases\\0024` 的 GT 键 `U5` 是个 33×33 的四脚盒，**图上画的是 `K11`**
  （`work\\preview\\_boxcrop_0024_U5.png`），而该 GT 的键集合是 `C1,C2,R1..R8,U1..U16,∅337`
  —— **根本没有 `K11`**（GT 把 16 颗各种元件一律记成 `U1..U16`）。
  `0048` / `0057` / `0062` / `0102`（全是 `Datasheet` 档）的裁片里**图上根本没有任何位号**：
  `0048` 只有引脚名（`VCC/VSEN/IBUS/COMP1/DRV/ISEN/VDD/GND/CF`）、`0057` 只有型号 `S-8225B`、
  `0102` 上画的是符号名 **`Flash`**、下表 `W25Q16JVUXIQ`，**没有 `U3`**。
- **`∅` 的号是随机生成的（赛题 Q6 明说）**：`∅202 pin_202 ('P_31',{1079,276})`、
  `∅121 pin_119 ('P_20',{77,897})`、`∅113 pin_111 ('P_27',{872,72})` … 号与名都是人造 ⇒
  419 个 GT 网成员里 **92 个（22.0%）数学上不可赢**。
- ⇒ **在派生档上调参前必须先量 GT 保真度。** 报告：`work\\reports\\GT保真度与位号上限.md`、
  `work\\reports\\引脚号回归与桩偏移.md`。"""

EV_OCR_ADD = """

**【本会话 · 按框宽分桶的引脚号识别】**（尺子 `work\\tools\\diag\\_numread.py`，官方 9 张 **133 个样本**）

| 桶 | exact | n |
|---|---:|---:|
| 1 字（框宽 < 20px） | **0.7653** | 98 |
| 2 字（20~30px） | 0.4194 | 31 |
| 3 字+（>= 30px） | 0.5000 | 4 |
| **整体** | **0.6767** | 133 |

43 个错例 = **归属错 15（34.9%） + 识别错 28（65.1%）**（归属错的机制见「独家归属」节点）。

- **剩余识别错**：① 单数字混淆约 **20** 个（`9>8`×3、`6>8`×3、`6>0`×2、`2>3`、`1>2`、`8>1`、`5>1`、
  `9>1`、`79>7`、`115>15`、`113>13`）；② **宽框吃进邻居字约 8 个**（集中 case 8 / case 10：
  `14>L14`、`10>L10`、`1>K1`、`2>K2`×2、`1>1L`、`4>K4`，框宽 25~29px 而实际只有一个数字）。
  ⇒ ② 的修法是**对宽框拆分重读**（左半 / 右半各读一次，取纯数字且置信度高的一侧），**未做**；
  ① 只能靠识别器本身（重训 / TTA 投票）。
- **外扩网格（同一把尺子）**：`RAW`（连 `pad_box` 都不做）**0.4135** · `(2,0,3,.20)` 现默认 0.6617 ·
  **`(2,0,3,0)` 0.6767（网格最优）** · `(4,0,4,0)` 0.6316 · `(1,0,1,0)` 0.5865 但 **3 字+ 桶 = 1.0000**。
  ⇒ 外扩不是杠杆。

**【本会话 · 位号那条路的尺子】**（`work\\tools\\diag\\_desigsweep.py`，官方 9 张 **101 个样本**，
按 bbox IoU ≥ 0.5 配对）：基线（产物里已写下的读数 vs GT 键）exact **0.9307**；14 组外扩网格最好
`(4,0,4,0)` = **0.9109**，**低于基线** ⇒ 外扩不是杠杆。7 个错例**全在 case 1**：
`C160>C10`、`C154>C15A4`、`C153>C53`、`C155>C15`、`C159>C153`、`C162>C182`、`R164>R134`
（喂进去的裁片肉眼完全清晰，见 `work\\preview\\_c1_desigread.png`）。
⇒ 位号上限约 **+1.79 分**（component 30% 里 0.9403 → 1.0）。"""

EV_CROP_ADD = """

**【本会话 · 「训练/推理二次外扩」假设被两条独立的尺子证伪】**
怀疑 `stage2/text.py` 的 `pad_box`（水平 `+2`、垂直 `max(3, 0.20h)`）与
`textrec/dataset.py:143 crop_gray`（`PAD_X_FRAC=0.12` / `PAD_Y_FRAC=0.25`，docstring 写着
「训练与推理必须共用这一个函数」）叠成二次外扩、把字形压小 40%。实测：

| 路径 | 尺子 | 样本 | 基线 exact | 关掉 / 缩小外扩 |
|---|---|---:|---:|---:|
| 位号 | `_desigsweep.py` | 101 | **0.9307** | 网格最好 0.9109、`RAW` 更低 |
| 引脚号 | `_numread.py` | 133 | **0.6767** | `RAW` **0.4135**（崩 39%） |

⇒ **`pad_box` 是承重的**：文本检测给出的框比训练标注框**更紧**，`pad_box` 在补回被检测切掉的字形；
二次外扩**不是**线上掉分的原因。位号那 7 个错例、引脚号那 28 个错例都是**识别器本身**的问题
（裁片肉眼完全清晰）。**主战场是识别器，不是预处理。**"""


def main():
    html = P.read_text(encoding="utf-8")
    m = re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', html, re.S)
    if not m:
        raise SystemExit("graphdata not found")
    d = json.loads(m.group(2))

    shutil.copyfile(P, BK)

    d["name"] = NAME
    d["note"] = NOTE_HEAD + (d.get("note") or "") + NOTE_TAIL_APPEND

    by = {n["id"]: n for n in d["nodes"]}
    for nid, add in (("ev_wire", EV_WIRE_ADD), ("ev_pin", EV_PIN_ADD), ("solve", SOLVE_ADD),
                     ("metrics", METRICS_ADD), ("ev_ocr", EV_OCR_ADD), ("ev_crop", EV_CROP_ADD)):
        by[nid]["note"] = (by[nid].get("note") or "") + add
    by["own"]["note"] = OWN_NOTE

    new_nodes = [
        dict(id="ev_bridge", group="ev", kind="new", title="⑥b 并网的形状判据（规则 5）",
             sub="labels.py:_overshoot_unions：端点微越过目标段 + 不平行才并",
             x=60, y=840, w=280, file="stage1/labels.py · stage1/nets.py",
             ins=[dict(id="segs", label="线段")],
             outs=[dict(id="segs", label="并好的段")],
             params=[["WIRE_OVER", "12（env STAGE1_WIRE_OVER；平台 12~14）"],
                     ["WIRE_OVER_CROSS", "0（=1 要求异侧穿过，反而 55.38，别开）"],
                     ["WIRE_TOL", "1.5（回退历史值；抬到 10 是零和交换）"],
                     ["WIRE_WELD / WIRE_EXT", "16 / 8（都未动，动了两头净亏）"]],
             note=EV_WIRE_ADD.strip()),
        dict(id="pincol", group="ev", kind="new", title="⑦b 引脚列枚举（格点补脚）",
             sub="pincol.py：按引脚列等距格点补出漏检的脚，再交给并网",
             x=60, y=940, w=280, file="stage1/pincol.py",
             ins=[dict(id="det", label="元件框"), dict(id="pins", label="引脚点"),
                  dict(id="segs", label="线段")],
             outs=[dict(id="pins", label="补全后的引脚点")],
             params=[["格点间距实测", "IC 引脚列 10.0px"],
                     ["PIN_JOIN_DIST", "12（stage1/nets.py，比一格还大 ⇒ 会跨脚认领）"],
                     ["PIN_JOIN_SYNTH", "3（格点补出来的脚只能用更紧的半径认领）"],
                     ["旋钮数", "STAGE1_PINC_* 约 22 个"]],
             note=("**为什么单独画出来**：这一层会**凭空造出 GT 里不存在的引脚点**，而这正是"
                   "「归属错」的上游（见「独家归属」与「⑦ 引脚点」）。\n\n"
                   "- 官方 case 2 的 `U1`：GT 只有 4 个脚，我们给了 18 个点（含 bbox 之外的两排）。\n"
                   "- `merge_into(comp_pins, cols, snap=MERGE_SNAP)` 返回 `(新 comp_pins, 每脚 synth 标志, 统计)`，"
                   "标志与 `comp_pins[cid]` **逐位对齐**（`1` = 格点补出来的、框边外侧 3px 内没有墨）。\n"
                   "- 自检 `python eda-pcb\\stage1\\pincol.py` → **23 过 / 0 不过**。")),
        dict(id="ownnum", group="fz", kind="new", title="★★ 引脚号候选：归属 + 沿边对齐",
             sub="assign.py / structmatch.py：谁贴在这颗元件上，以及号怎么排",
             x=1540, y=1140, w=300, file="stage2/assign.py · stage2/structmatch.py",
             ins=[dict(id="reads", label="文字读法"), dict(id="pins", label="引脚点"),
                  dict(id="det", label="元件框")],
             outs=[dict(id="reads", label="引脚号归属")],
             params=[["OWN_BBOX", "1（归属量框到框，不量文字中心到框）"],
                     ["PINNUM_DECAY", "10（沿边距离衰减；20 → 10 是 +0.26）"],
                     ["PINNUM_DUP", "0（去重实测 −1.80，关掉）"],
                     ["PIN_COVER", "0.15 · PINNUM_NEAR 60 · PINCOL_GUIDE 0"]],
             note=("本节点是**两条**踩坑合并出来的。\n\n"
                   "**坑一：`FAR` 量的是「文字中心到框」⇒ 归属随文字宽度变化。**\n"
                   "3 字符 BGA 球号（框宽 32~36px）中心距 40~45 > `FAR=40` ⇒ 整排被踢成 `free`。"
                   "修法是 `dist_rect_to_rect` + `OWN_BBOX`（见「独家归属」节点）。\n\n"
                   "**坑二：`score_of` 的沿边衰减 `/(1+d/20)` 太慢 ⇒ 单调对齐宁可整排挪一格。**\n"
                   "case 2 的 `U1` 上 `pin_2` 被写上 `3`。改成 `/(1+d/10)`（`PINNUM_DECAY` 默认 10）"
                   "⇒ 官方 9 张 56.45 → **56.71**、派生 19 → **35.39**。\n\n"
                   "**同轮证伪**：按「同 text 且中心差 ≤ 3px」去重是 **−1.80**（`PINNUM_DUP`）—— 因为重复副本在"
                   "多占坑（同一块文字绑两根脚，一 tp 一 fp），而 `pin_f1` 里丢 tp 比多 fp 更贵。\n\n"
                   "**没做的**：宽框拆分重读（见「④ CRNN 识别」）。")),
        dict(id="boxpinprior", group="kn", kind="new", title="★ IC 不补位置号",
             sub="priors.py：稀疏封装号是印出来的，位置先验会覆盖它",
             x=1140, y=40, w=300, file="fusion/priors.py",
             ins=[dict(id="det", label="元件框+类型")],
             outs=[dict(id="priors", label="先验规则")],
             params=[["FUSION_BOX_PIN_PRIOR", "0（默认关掉对 IC 的补号）"],
                     ["_PRIOR_IC_TYPES", "box / ic / boxic / bigic"]],
             note=SOLVE_ADD.strip()),
        dict(id="gtcheck", group="out", kind="new", title="★★ GT 保真度检查（自造 GT 会骗你）",
             sub="在派生档上调参前先量：这份 GT 到底能不能赢",
             x=1140, y=340, w=300, file="work/tools/diag/_desigsrc.py · _boxkey.py · _desigabsent.py",
             ins=[dict(id="file", label="GT + 我们的产物")],
             outs=[dict(id="faith", label="可赢/不可赢账")],
             params=[["10GT（竞赛格式）", "box 键对上 21/21 = 100%"],
                     ["jlc / Datasheet / other", "46.6% / 21.2% / 7.3%"],
                     ["不可赢占比", "477 / 594 = 80.3%"],
                     ["∅ 随机号", "92 / 419 网成员 = 22.0%（官方 Q6）"]],
             note=METRICS_ADD.strip()),
    ]
    have = {n["id"] for n in d["nodes"]}
    for n in new_nodes:
        if n["id"] in have:
            raise SystemExit("node already exists: " + n["id"])
        d["nodes"].append(n)

    new_links = [
        ("ev_wire.segs", "ev_bridge.segs"),
        ("ev_bridge.segs", "ev_net.segs"),
        ("ev_pin.det", "pincol.det"),
        ("ev_pin.pins", "pincol.pins"),
        ("ev_wire.segs", "pincol.segs"),
        ("pincol.pins", "ev_net.pins"),
        ("own.det", "ownnum.det"),
        ("ev_ocr.reads", "ownnum.reads"),
        ("ev_pin.pins", "ownnum.pins"),
        ("ownnum.reads", "pool.reads"),
        ("ev_pin.det", "boxpinprior.det"),
        ("boxpinprior.priors", "solve.priors"),
        ("metrics.file", "gtcheck.file"),
    ]
    seen = {(l["from"], l["to"]) for l in d["links"]}
    for a, b in new_links:
        if (a, b) not in seen:
            d["links"].append(dict(from_=a, to=b, kind="data")) if False else \
                d["links"].append({"from": a, "to": b, "kind": "data"})

    # 保持 JSON 的键顺序：dict 里 "from" 是关键字不能用 kwargs
    body = json.dumps(d, ensure_ascii=False, indent=1)
    P.write_text(html[:m.start(2)] + body + html[m.end(2):], encoding="utf-8")
    print("nodes", len(d["nodes"]), "links", len(d["links"]))
    print("backup ->", BK.name, BK.stat().st_size, "new size", P.stat().st_size)


if __name__ == "__main__":
    main()
