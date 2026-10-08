# 诊断手册：怎么量，以及为什么这么量

> 全部脚本索引见 `TOOLS.md`（483 个，自动生成）。
> 本文只讲**方法**：拿到一个笼统的「某项失分」，怎么把它拆到可行动的最小单元。

---

## 0. 三条纪律（都是踩坑换来的）

1. **不要用自造的 GT 判断对错。** 我们的评测器是逆向实现，官方 Q&A 之外的部分可能与
   真评测器有偏差。先跑 `tools/debug/ledger.py` 把官方打分过程**逐条摊开**再下结论。
2. **共用输出目录会让诊断漂移。** `fast.py` 默认写 `work/_fast_out`，每次运行都覆盖；
   做 A/B 必须 `--out <独立目录>`，否则你比的是上一次的残留。
3. **不要跨口径比分数。** `--no-vlm` 与 VLM 开差 5.2 分；不同 `--cases` 集合的均分不可比。
   报数时必须写清「评测集 + VLM 开关 + LLM 开关」。

---

## 1. 标准工作流

```
① 量出这块失分有多大        → 上层归因（_netloss / _pinloss / _des_ceiling）
② 找出失分的机制            → 现场核对（_netside / _pinband / _widebox / _numread --dump）
③ 在离线尺子上验证假设      → 不动管线的纯 CPU 实验（_numread / _desigsweep / _netmerge）
④ 端到端 A/B（两个评测集）  → fast.py --save，**两个档都赚才采纳**
⑤ 把数字写进报告与代码注释  → docs/reports/*.md + 模块 docstring
```

**离线尺子的价值**：它让你在几分钟内证伪一个假设，而不是每次改代码跑 9 张图。
但它**不能替代端到端** —— 例：`_crnneval.py --shrink` 在离线尺子上看着合理，
实际把「裁掉字形」与「多加白底」混在一起，结论完全错。

**采纳门槛**：官方 9 与派生 19 **两个档都赚**才落地。只赚一个档的改动一律记为负结果
（`WIRE_TOL=10` 就是这么被否掉的）。

### 口径必须配对，还要看「开关有没有生效」

同一块失分可能只在某个口径下存在，**对照臂必须同口径**：

* `boxscan`（`FUSION_BOXSCAN`）在 **VLM 开着时基本是 no-op**（`_add_vlm` 已把
  `comp.<cid>.key` 覆盖到 82/83、12/12、22/22），而它的第 ② 条又要求 LLM 轮真跑
  ⇒ 只有 **`--no-vlm` + `$env:FUSION_LLM='1'`** 这个口径量得到它本身。
  goal 里那两个基线（57.54 / 35.75）是**另一个口径**，不可同表比。

⚠️ **「A/B 有差别」不等于「你改的那条分支生效了」。** boxscan 第一版实测有差
（52.48 → 52.47），但把 `_fusion_report.json` 的 `notes` 打出来才看到
`{"kind":"boxscan","n_slot":1,"n_skip_type":11}` —— **每张图只建了 1 个槽位**，
分差根本不是「boxscan 整体」的效果，而是那 1 颗元件的自由读数对不对。
**改完先证明分支执行了、执行了几次，再解释分差。**

```powershell
# 打印建池阶段的计数笔记（boxscan / shape_gate / run_prior …）
python -c "import json;d=json.load(open(r'work\_fast_out_<名>\runs\fusion\_fusion_report.json',encoding='utf-8'));[print(c['case'],c['notes']) for c in d['cases']]"
```

同一份报告里还有 `rounds[*].judge_rows`（每一问的 `slot/ans/res/mode`）—— `res` 是
`agree_top / agree_alt / new / reject / agree_prior / empty`，`mode` 是
`choose / read / **free**`（`free` 就是 boxscan 的自由读数）。**看不到行就等于没问。**

### 旧记录里的「要修」也要先复核

**审计报告里的「这里有个 bug」四个字不等于这里真有 bug。** boxscan 节点的待办第 ③ 条写着
「`claims.py` 用 `pinpts.get(cid)`，而 `doc["pins"]` 的键是位号 ⇒ 引脚取景框恒为 None」。
拿 `tools/diag/_pinbox.py` 一量：融合层读的是 `runs/stage1/<case>.stage1.json`，那份文档的
`pins` **就是按 `cid`** 键的（case 1/4/7/10 = 22/22、25/25、24/24、30/30 命中，
113/90/67/85 个点全是真坐标），反过来拿位号当键**命中 0**。只有 `runs/stage2/<case>.json`
的 `pins` 才以位号为键 —— 旧记录把两份文档搞混了。**照它去改会直接引入一个 bug。**

⇒ 动手前先用一行脚本把「症状」复现出来；**复现不出来就不是 bug**。

---

## 2. 上层归因：把失分拆开

### 网络（netH，权重 0.35 —— 四项里最重）

```powershell
python tools\diag\_netloss.py --dir work\_fast_out_off9_tightpn
```

把 419 个 GT 网成员分成五类：

| 类 | 含义 | 官方 9 张的量 |
|---|---|---:|
| ④ 正确 | tp | 173 ~ 175 |
| ③ 拓扑错 | 引脚内容对但网络连错/碎裂 | **74（17.7%）** |
| ②b `∅` 缺脚号 | `∅NNN` 随机脚号，**不可赢** | 92（22.0%） |
| ②a 有名元件缺脚号 | | 58（13.8%） |
| ① 缺键 | 元件键没配上 | 22（5.3%） |

**上限推算**：修光 ③ 一处，netH F1 0.5339 → ~0.677 ≈ **+6 分**。

看具体碎裂在哪：

```powershell
python tools\diag\_netside.py --case 10      # GT 网 vs 我们的网并排
python tools\diag\_segap.py  --case 10       # 碎片到最近段的缺口距离
```

### 引脚（pin，权重 0.25 —— 剩余空间最大）

```powershell
python tools\diag\_pinloss.py --dir work\_fast_out_off9_tightpn
python tools\diag\_numprobe.py --dir work\_fast_out_off9_tightpn
```

`_numprobe` 把「IC 引脚点」分成：有小字且交出号 / 有小字却没交出号 /
交出号却没小字 / 两头都没有 —— 决定是「字没检出」还是「检出没认领」。
实测（官方 9 张）：引脚点 322 · 有小字且交出号 **79** · 有小字没交出 **72** ·
没小字却交出 22 · 都没有 149。

### 元件（component，权重 0.30）

```powershell
python tools\diag\_keytype.py     # 按元件类型量键命中率
python tools\diag\_des_ceiling.py # 位号链的天花板：认不出 vs 归错元件，分开量
python tools\diag\_desigring.py   # GT 位号在周围 R 像素的文字框里到底有没有
```

### `net_line`（权重 0.10，长期爆 0）

```powershell
python tools\diag\_linewhy.py     # 逐条看边的端点距离
python tools\diag\_lineweld.py    # 「共线且首尾相接」焊成整条直线值不值
```

---

## 3. 离线尺子（不动管线，纯 CPU）

### 引脚号：按**框宽分桶**的准确率

```powershell
python tools\diag\_numread.py --dir work\_fast_out_off9_tightpn
python tools\diag\_numread.py --dir work\_fast_out_off9_tightpn --dump work\preview\_numwrong.png
```

三个设计要点：

1. **按框宽分桶**（`<20px` = 1 字 / `20~30` = 2 字 / `>=30` = 3 字+）——
   只看整体 exact 会把「单字混淆」和「宽框吃邻居字」混成一个数，两者对策完全不同。
2. **归因**：再问「文字框**中心**离哪个 GT 脚点最近」，若比配给它的那个脚更贴近**别的**脚
   ⇒ 是**归属错**，不是识别错。实测错例 43 个 = 归属错 15（34.9%）+ 识别错 28（65.1%）。
3. `--dump` 把错例裁片（外扩 0.3h 横 / 0.6h 竖、红线画原框、6×）竖排拼图，人眼兜底。

**当前基线**（官方 9 张，133~134 个可对上 GT 脚点的读数）：
1 字 0.7700 · 2 字 0.5333 · 3 字+ 0.5000 · 整体 0.7090。

### 位号

```powershell
python tools\diag\_desigsweep.py    # 位号识别精度基准 + 裁剪外扩网格
python tools\diag\_numread.py       # 同一把尺子
```

实测：位号路径上**外扩不是杠杆**（基线 0.9307，网格最好 0.9109）。
7 个错误全在 case 1，与位号无关。

### 宽框视觉核对

```powershell
python tools\diag\_widebox.py --dir work\_fast_out_off9_tightpn --minw 25 --out work\preview\_widebox.png
```

⚠️ **取图必须用原始 PNG**（`tools/diag/_boxread.py` 的 `find_png(case)`），
**不要**用 `runs/stage1/<case>.png` —— 那是 stage1 的调试叠图，上面有 overlay，看到的图形是假的。

---

## 4. 端到端 A/B

```powershell
# 归档一次评测（不加 --save 就不会写 work/_fast_base/<名字>.json）
python tools\eval\fast.py --cases 1,2,3,4,5,7,8,9,10 --save off9_xxx --out _fast_out_off9_xxx

# 只看已有产物（不重跑管线）
python tools\eval\fast.py --score-only --out work\_fast_out_off9_xxx --save off9_xxx

# 与基线比
python tools\eval\fast.py --cmp off9_base --out work\_fast_out_off9_xxx

# 汇总所有归档
python tools\diag\_basemean.py
```

**打开关**：

```powershell
$env:STAGE2_PINNUM_TIGHT='0'      # 关整流剔除
$env:FUSION_LLM='0'               # 关 LLM 裁决
```

❌ **不要**用 `--extra "--env X=Y"`：`fast.py` 的 `--extra` 会把多词参数**按字符拆开**，
得到 `run.py: error: unrecognized arguments: --env ...`（exit 2）。

⚠️ PowerShell 里 `$env:X=''` 是**删除**变量，不是关功能；要关请写具体假值。

---

## 5. 常用现场核对工具

| 问题 | 脚本 |
|---|---|
| 这张图的网络长什么样 | `tools/diag/_netside.py --case N` |
| 某个元件的框外文字带 | `tools/diag/_pinband.py --case N --cid b0000` |
| `_assigned/<case>.json` 里有什么 | `tools/diag/_asgkeys.py --case N` |
| 官方打分过程逐条摊开 | `tools/debug/ledger.py` → `tools/debug/report.py`（生成交互式 HTML） |
| 两个融合产物到底改了哪些键 | `tools/diag/_keydiff.py` |
| 把 GT 框画回原图 | `tools/diag/_gtoverlay.py --case N` |
| GT 里某类元件长什么样 | `tools/data/gt_look.py` |
| 引脚号读数对不对（按框宽分桶 + 归属错/识别错归因） | `tools/diag/_numread.py [--dump out.png]` |
| **位号这一路还剩多少可救（候选池天花板）** | `tools/diag/_ownasgn.py --headroom --gt` |
| **`own.assign` 全局一对一离线重跑 + GT 裁决** | `tools/diag/_ownasgn.py --gt --show` |
| **GND 系引脚在 GT 里到底几条网** | `tools/diag/_gndnet.py` |
| **`net_line` 边能不能配上 GT（端点距离）** | `tools/diag/_nledge.py --case N --net net_9` |

---

## 6. 报告在哪

逐主题诊断报告在 `docs/reports/`（约 20 篇）。最常引用的几篇：

| 报告 | 内容 |
|---|---|
| `引脚号回归与桩偏移.md` | 引脚号全链路：回归根因、桩偏移、并网规则、`OWN_BBOX`、`PINNUM_DECAY`、描边剔除 |
| `分数天花板_四指标分解.md` | 四项指标各自的结构性上限 |
| `官方口径_∅NNN与坐标系.md` | `∅NNN` 随机性、坐标系与 JSON 形状的官方约定 |
| `GT保真度与位号上限.md` | GT 本身有多可信、位号到底能救回多少 |
| `工程哲学.md` | 这个项目怎么工作（测量优先、负结果也记录） |
| `漏检错检_归因.md` | 检测层的漏检 / 虚警归因 |
| `尺度归一_诊断.md` | 墨迹归一三版为什么全败 |
