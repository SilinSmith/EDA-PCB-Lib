# 诊断手册：怎么量，以及为什么这么量

> 全部脚本索引见 `TOOLS.md`（512 个，自动生成）。
> 本文只讲**方法**：拿到一个笼统的「某项失分」，怎么把它拆到可行动的最小单元。

---

## 0. 三条纪律（都是踩坑换来的）

1. **不要用自造的 GT 判断对错。** 我们的评测器是逆向实现，官方 Q&A 之外的部分可能与
   真评测器有偏差。先跑 `tools/debug/ledger.py` 把官方打分过程**逐条摊开**再下结论。
2. **共用输出目录会让诊断漂移。** `fast.py` 默认写 `work/_fast_out`，每次运行都覆盖；
   做 A/B 必须 `--out <独立目录>`，否则你比的是上一次的残留。
3. **不要跨口径比分数。** `--no-vlm` 与 VLM 开差 5.2 分；不同 `--cases` 集合的均分不可比。
   报数时必须写清「评测集 + VLM 开关 + LLM 开关」。
4. **「慢」先量余额、再上 profiler、最后才怀疑硬件。** 阶段耗时里 `total` 与各分段之和的
   **差额**永远指向进程外调用（网络/IO/锁）；一次 cProfile 就能落到调用行。
   任何发到进程外的调用都要自带超时、上界与熔断 —— 见 §1「一次无界的远程调用会把整个阶段
   变成黑洞」。**失败如果被吞成字符串，它就不会以报错的形式出现。**
5. **在决定「重训模型 / 换模型」之前，先量上帝视角上界。** 把手上的产物按理想值改写一遍
   再打分（`tools/diag/_ocroracle.py`），得到的是「这条路**全部**能值多少分」。
   如果上界只有 +4.6，而花在别处的力气能拿 +2，那就先做后者、别去动权重。
   **先算天花板，再谈投入** —— 这条纪律本轮直接改变了一个「要不要重训 CRNN」的决定。
6. **把「读错」和「没产出」分开量，否则你会去修错的东西。** 同一批缺号，成因可能是
   「识别错了」「策略把它丢了」「压根没检测到」——三者的修法完全不同（换模型 / 改一行开关 /
   改检测）。判据见 §1「按位置查，不要按号查」。

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

### 「这个开关开着」不等于「这条路被执行过」

本轮最贵的一次误判：`llm_judge` 节点从立项起就写着「给 LLM **整表上下文**」，
`SHEET_ON` 默认 `1`、`_sheet_image`/`_known_designators`/`judge_sheet` 全都是为它写的
（`fusion/llm_judge.py`），**但 `judge()` 里的入口是 `if mode == "sheet" and SHEET_ON:`，
而 `MODE` 的默认值是 `"auto"`** ⇒ 那条路**一次都没跑过**，默认配置下每次都是逐题一张小图 + A/B。
「没有上下文环境」这句话因此有了**代码级坐标**，而不是一句感觉。

**怎么才发现**：不是读代码读出来的，是**把报告里的开关字段打出来**——
`work/_fast_out_*/runs/fusion/_fusion_report.json` 的 `rounds[0].judge` 里有
`'mode': 'auto'`、`'ctx': True/False`、`'n_ctx_slot': 0`、`'n_q'`、`'n_agree_top'` 这些计数。

⇒ 两条可复用的尺子：

1. **凡「A/B 没差别」或「这条新路好像没用」，先证明它被执行了。** 在报告里加一个
   `rep["mode"]` / `rep["ctx"]` / `rep["n_ctx_slot"]` 这样的计数字段，然后在
   `--no-vlm` 单张上把 `judge` 那一行整个打出来：
   ```powershell
   python tools\diag\_jrep.py --case 1 --out work\_fast_out_ctxA   # 打 rounds[].judge 全部字段
   ```
2. **默认值 ≠ 生效值。** `SHEET_ON=1` 是「允许」，`MODE` 才是「选择」。凡是
   双开关（允许 + 选择）的模块，两个都要看。
3. **A/B 的「臂」本身要能证明它和基线不一样。** 同一个坑的第二形态：`work/tools/diag/_vlmab2.py`
   里那组 `ARMS` 有一个 `"pins": dict(designator=True, value=False)`——**`pin=True` 根本没设**
   ⇒ 这一臂等价于 `desig` 臂，等于「拿基线比基线」。凡是要量某条新通道，先在代码里确认
   那一臂真的传了那个开关（比 `_jrep.py` 更省事的办法：让 A/B 脚本把每臂的开关 `print` 出来）。

### 按位置查，不要按号查（否则你会去修错的东西）

同一批「缺引脚号」，成因完全不同，修法也完全不同。**按号查会把读错的混进「没有这个槽位」**：

| 尺子 | 绑定方式 | 官方 10GT 结果 | 结论 |
|---|---|---|---|
| `_pinnamepair.py` | **按号** | 136 根里只能查到 **94** 个槽位 | 44 根被算成「没槽位」，其中一半只是**号读错了** |
| `_ocrstat.py` | **按位置**（45px 内取最近槽位） | 135/136 = **99%** 有槽位 | 几何绑定几乎不缺 ⇒ **分丢在字符串上** |

而「字符串读错」和「根本没产出」必须再分一层（`_pinledger.py`，官方 9 张缺的 51 个号）：

| 成因 | 根数 | 该去改哪里 |
|---|---:|---|
| **号读对了，名字没读到 ⇒ `stage2/assemble.py:759` 整条不发** | **23（45%）** | **改一行开关**，跟识别模型无关 |
| 号读错（113→13、39→3…） | 22 | CRNN / 裁剪 / VLM |
| 附近 45px 内没有任何号槽位 | 6 | 检测 |

⇒ **先分层，再决定投哪。** 本轮如果没有这一步，会直接去重训识别器，而其中 45% 的分
根本不在识别器上。

### 「上帝视角上界」：动权重之前先算天花板

把产物按理想值改写一遍再打分，得到「这条路的**全部**能值多少」：

```powershell
python work\tools\diag\_ocroracle.py --out work\_fast_out_reg9 --cases 1,2,3,4,5,7,8,9,10
```

它只改**点已经落在 GT 点 5px 内**的那些引脚：先只换 `pinname`（`name` 臂），再连**引脚键**
一起换成 GT 的并同步改写 `nets[*].hyperGraph`（`num` 臂）。

官方 9 张：**58.6304 → 61.2888（名全对）→ 63.2307（号+名全对）= +4.6003**，
其中引脚 0.3920→0.5515、tp 37→52。两个臂的**差**还能告诉你「名」和「号」各值多少
（+2.66 / +1.94）—— 而「这一臂能不能落地」是另一个问题，别把上界当承诺。

**写法上的两个坑**（都在 `_ocroracle.py` 里踩过）：① 改写引脚键之后，
`nets[*].hyperGraph` 是**字符串**，必须用 `parse_hypergraph` / `build_hypergraph` 重建，
否则网络分不动；② 匹配要在 `assign_components` + `rewrite_keys` **之后**做，
不然预测的元件键和 GT 对不上，点匹配全是空的。

### 上界告诉你「值多少」，**不等于**「一行开关就能拿到」

`_ocroracle.py` 说的是**上帝视角**的天花板。把「某条策略丢掉的 N 个东西」说成
「改一行就能拿回来」是**两个不同的命题**，必须单独 A/B。本轮就有一个反面教材：

- 官方 9 张缺的 45 根 IC 引脚里 **23 根是「号已读对、名字没读到 ⇒ `stage2/assemble.py:759`
  的 `BOX_PIN_NEED_NAME` 整条不发」**。看起来这是「改一行开关白拿 23 根」。
- 真去量：`tools/diag/_gateab.py`（**用已落盘的 `runs/stage1` + `runs/stage2/_assigned`
  重跑 `assemble.build_doc`，改模块全局切开关 ⇒ 纯 CPU、不占 GPU、不重跑 OCR**），
  再 `tools/eval/stage2_eval.py work\_gate_ab\{on,off}`：
  **官方 9 张 52.6510 → 52.4987（−0.15，判负）**，`pin.name_mismatch` **16→41**、
  `pin.extra` 6→11，而 tp **一根没多**（34→34）；换来的只有 `net_hypergraph` +0.030。
  机制：关了闸门，多发的引脚名字只能回退成**引脚号本身**，官方档的 GT 引脚名 86.5% 是真信号名。
- 而**全量 220 张反而是 +0.10**（26.0962 → 26.1978）—— 派生档的引脚名常常本来就是数字。
  ⇒ 这类开关的效果**是分档相关的**，只在一档上量会得出相反的结论。

**「离线单变量 A/B」这个手法值得记下来**：`assemble.build_doc(stage1, assigned, case)`
的两个输入**都写在盘上**，而装配开关都是**模块级全局、在函数里按调用时读**
⇒ `assemble.BOX_PIN_NEED_NAME = False` 就能切，不用碰 GPU 和 OCR。
能这么做的开关要优先用这种方式量（秒级、可以和别的长跑并行）。

### 提示词改动必须回读「模型到底收到了什么」

「把上下文拼进 prompt」这类改动，**分不变也可能是真的生效了**（模型看到了但答案没变）。
所以除了分数，还要看 `judge_rows[].raw` —— 上下文生效的直接证据在**原始回答**里：

```powershell
python tools\diag\_jrep.py --case 1 --out work\_fast_out_ctxA --rows
```

本轮实测：`ctx=1` 时 **12/12 行都有 `依据=…`**（`raw` 里带解释），`ctx=0` 时 **0/12**。
分差则是官方 9 张 **53.27 vs 52.48（+0.79）**。**两样都要，一样都不能省**：
只报分数会说不清为什么变，只报 `why` 会说不清值不值。

### 「我改了这个文件」不等于「改的是被读的那个文件」

同一个形状的误判再犯一次：**仓库里有同名文件的副本，而构建读的不是我改的那份**。

| 脚本 / 产物 | 真正被读的那份 | 容易改错的那份 |
|---|---|---|
| `_wfbullets.py` | **`work\flow\_wfbullets.py`**（`work\tools\diag\_wfbuild.py:30` 先 `sys.path.insert(0, FLOW)` 再 `import _wfbullets`） | `eda-pcb\tools\diag\_wfbullets.py`（仓库快照） |
| `_wfhead.py` | **`work\tools\diag\_wfhead.py`**（它的 `_find("../../flow/workflow.html", "workflow.html")` 只有在 `work\tools\diag\` 下才解析得到 `work\flow\workflow.html`；从 `eda-pcb\tools\diag\` 跑会直接 SystemExit） | `eda-pcb\tools\diag\_wfhead.py` |
| `workflow.html` | `work\flow\workflow.html` 生成后**复制**到 `eda-pcb\docs\workflow.html` | — |

**症状**：改完重建，输出文件**大小一模一样**（228599 B → 228599 B）—— 这就是最好的证据。
**尺子**：

```powershell
python -c "import sys; sys.path.insert(0,'work/flow'); import _wfbullets as W; print(W.__file__)"
Get-FileHash work\flow\_wfbullets.py, eda-pcb\tools\diag\_wfbullets.py   # 两份必须一致
```

⇒ 规矩：**先打印「被 import 的到底是哪个路径」**，再把两份同步成逐字节一致。
（反过来也有价值：`work\flow\` 那份是本会话**维护中**的版本，`eda-pcb\` 那份常常是旧的；
不要凭「我上次改过」推断哪份新，要比 hash。）

**同坑的变体：替换脚本必须能被重复执行。** `_patch8.py` 的 `batch` 那条替换，
新文本里**包含了旧文本**（徽标行 + 原样那一行）⇒ 对同一文件跑第二次又多插了一行徽标。
⇒ 字面替换要么保证 `new` 不含完整的 `old`，要么先查标记再替换。

---

### 一次**无界的远程调用**会把整个阶段变成黑洞（先看 `timing` 的「余额」）

**症状**：`work\_full220` 首跑里 `0003`（4491×3173）单张 `total_s` = **315.73 s**，而同一张
的 `timing` 里 yolo 0.11 + cv 0.215 + wires 1.49 + rays 0.28 + text 1.66 + bigic 0.15 +
pins 0.4 = **4.31 s**。⇒ **311 s 不知道去哪了**。

**尺子 1（免费）**：把 `timing` 各段加起来与 `total_s` 比。**差多少就是「没记名的等待」**，
它只可能来自网络/IO/锁，不可能来自算力。这一步不需要 profiler。

**尺子 2（一次 cProfile 定位到调用行）**：

```powershell
python work\tools\diag\_prof1.py 0003 --top 12      # profile 落在 work\_prof1\<case>\_profile.txt
```

实测（320.85 s，20995097 次调用）：

```
stage1/pipeline.py:118 run_case
  └ stage1/wires.py:265 extract_wires           308.374 s
      └ stage1/wires.py:230 _hint_wire_hue      304.151 s
          └ vlm/client.py:161 ask               303.516 s
              └ vlm/client.py:128 _chat_once → urllib urlopen → socket.recv_into  303.377 s
```

**三处叠加才成灾**（每一处单独看都「有理由」）：

| 位置 | 事实 | 为什么危险 |
|---|---|---|
| `stage1/wires.py:311` | `_hint_wire_hue(bgr)` 在 `extract_wires` 里**无条件调用**，发的是**整张原图** | 一张 17.4 MPx 的图也照发 |
| `vlm/client.py:38` | `VLM_TIMEOUT` 默认 **600 s**，且 `ask()` 最多**重试 4 次** | 单张最坏 40 min |
| `vlm/client.py:161-178` | `ask()` 把一切异常吞成 `"ERR:..."` **字符串正常返回** | **不抛异常 ⇒ 不会出现在 `errors` 里，只会表现为「这张图莫名多了 5 分钟」** |

**规模先量再改**：`python work\tools\diag\_huetime.py --count` ⇒ 220 张里 **168 张**会触发
（52 张灰度图跳过），>2000px 的 **23 张**，最大 **17.4 MPx**。
延迟实测：`0043` 整图首发 **121.98 s 换来 `HTTPError 400`**，缩到 1024 后 **0.55 s**（16×）。

**修法（保留功能，只把它变有界）**：`_hue_small()` 先缩到 `STAGE1_WIRE_HUE_MAXSIDE=1024`；
独立超时 `STAGE1_WIRE_HUE_TIMEOUT=30`（不再用全局 600）；连续失败
`STAGE1_WIRE_HUE_MAXFAIL=3` 次后**整进程熔断**（端点没起来时不该每张都等一次超时）。
效果：`0003` **315.73 s → 8.3 s**；220 张 stage1 从「前 10 张 356.8 s」变成 **416.5 s**。

⇒ 规矩：**任何发到进程外的调用，都要自己带超时、带上界、带熔断，并且失败必须计数**。
只要「失败被吞成字符串」，它就会以「这里有段时间不见了」的形式存在，而不是以报错的形式。

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
| **融合层 `rounds[].judge` 的全部开关与计数**（开关到底生没生效） | `tools/diag/_jrep.py --out <产物目录> [--rows] [--case N]`；多臂对比用 `--cmp A B C` |
| **融合层读的 `doc["pins"]` 是按 `cid` 还是按位号键** | `tools/diag/_pinbox.py` |
| **每个 `kind='new'`（★要新写）节点的 note/bullets/params 全文** | `tools/diag/_wfnew.py [--all] [<id>…]` |
| **改 workflow 节点的徽标（`kind`）** | `tools/diag/_wfkind.py --list` / `--set exec <ids…>` / `--set dead --all-new`（默认 dry-run，`--write` 才落盘） |
| **改 `name`/`note`（长 JSON 字符串，别手改）** | `tools/diag/_wfhead.py --write`（**必须在 `_wfbuild.py --write` 之前**） |
| **按字面子串替换文件内容**（绕开 PowerShell 引号地狱） | `python -c "import sys;sys.path.insert(0,'work/tools/diag');from _subst import subst;subst(path,[(old,new)])"` |
| **单张 stage1 的 cProfile**（找「黑洞」到底在哪一行） | `tools/diag/_prof1.py <case> [--top 12]`，profile 落 `work\_prof1\<case>\_profile.txt` |
| **VLM 色相询问的触发张数与延迟**（整图 vs 缩到 1024） | `tools/diag/_huetime.py --count` / `tools/diag/_huetime.py <case> --reps 2` |
| **GPU / torch / YOLO 健康探针**（判断「慢」是不是硬件） | `tools/diag/_gpuprobe.py` |
| **一次全量跑的分阶段统计**（阶段一 stats+timing、阶段二读出率、按官方权重打分与最差 N 张） | `tools/diag/_stage220.py --run work\_full220 [--split 10GT] [--worst 12]` |
| **引脚四段漏斗**（键没配上 / 键配上没这根 / 名字不同 / 坐标超差） | `tools/diag/_pin200.py --n 60`；`--emit` 看产出率 |
| **上帝键上界**（位号全修对能涨几分；顺手看引脚涨多少） | `tools/diag/_keyoracle.py --n 60` |
| **把「绑定」用上帝视角消掉后引脚还剩什么**（同元件键下取离 GT 点最近的预测引脚） | `tools/diag/_pinoracle.py --n 60 [--examples 8]` |
| **逐根引脚的名字与距离归因**（名字逐字符相等的比例、距离分位） | `tools/diag/_pindist.py --n 40` |
| **GT 引脚名有多少是真信号名** / **我们产出的 `pinname` 有多少只是引脚号** | `tools/diag/_gtpinname_stat.py` / `tools/diag/_pinname_stat.py` |
| **GT 引脚点离元件框有多远**（判「框内硬门槛」是不是根本错） | `tools/diag/_gtpinplace.py` |
| **GT 引脚点附近有没有我们的文字、文字在框内还是框外** | `tools/diag/_pinlabel.py --n 60` |
| **GT 引脚点是文字框心吗 / 在 bbox 边线上吗**（两个假说都证伪） | `tools/diag/_pintext.py --run --split --n` |
| **引脚槽位的原始结构**（`number`/`name` 的 `{text, conf, box}`） | `tools/diag/_pinslot.py <case>` |
| **官方档的引脚键形状**（10GT 全是纯数字） | `tools/diag/_gtpin10.py --show 4` |
| **缺号三分账**（策略 / OCR / 检测） | `tools/diag/_pinledger.py --run --split --show` |
| **按位置对账引脚号与名的读法**（完全对 / 读错 / 空 / 截断） | `tools/diag/_ocrstat.py --run --split --n --show` |
| **读法全对的写实上界**（上帝视角改写再打分） | `tools/diag/_ocroracle.py --out <产物> --cases …` |
| **装配开关的离线单变量 A/B**（不占 GPU、不重跑 OCR） | `tools/diag/_gateab.py --run <产物> [--all]` → `tools/eval/stage2_eval.py work\_gate_ab\{on,off}` |
| **引脚邻域肉眼核对**（十字位置、号在框外上方、名在框内） | `tools/diag/_pincropgt.py` |

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
