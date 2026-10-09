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
7. **官方 Q&A 一变，先把「不可赢的口径」找出来，再谈优化。** 2026-10 的答疑明确
   「非 box 元件的 pin number 不计分 ⇒ rcl/gnd/vcc 在净图里不区分引脚」，而 GT 的电源引脚号
   是**随机生产**的（`∅113.111` / `pinname "P_27"`）⇒ 旧口径下这一项**任何队伍都不可能对上**。
   判据：拿一批**已归档的产物**（不重跑管线）同时用新旧两套口径重打
   （`tools/diag/_rescore.py --preset pin,order,f220,tp19,ckpt`），看 ① 分差有多大、
   ② 有没有 A/B 结论翻盘、③ 用**独立实现**再算一遍能否逐位复现
   （`tools/diag/_netconv.py` 的预处理版 vs 评分器内部默认值：9 张逐位相同）。
   本轮结果 +1.08（官方 9）/ +1.7（220 张）、**名次一个都没变** ⇒ 历史实验不用重跑，
   但报告里所有旧数字都必须挂上「旧口径」标签。**口径是配置项，不是事实**：改默认值的同时
   必须留一条回旧口径的开关（这里是 `stage2_eval.py --net-literal`）。

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

### 阈值是量出来的，不是猜出来的（浅灰笔画）

**这一条差点让我写进一份错误结论。** 量「GT 框里到底有没有符号」时我把墨迹定义成灰度 `<128`
（想着「白底黑线」），结果 gnd 报出 **14.5% 空框**、还顺手得出「错标集中在 jlc（28.8%）」。
真相是**这份数据集里的符号笔画是浅灰**：GND 三横线在图上灰度 **151/152**，背景 255
⇒ `frac(pixel<128)=0.0000`，正常框被判空。

**发现方式只有一种：把被判「空」的框裁出来看。** `work\preview\_gnd_empty_check.png`
（0186 ∅406 / 0101 ∅399 / 0094 ∅96 / 0122 ∅103 / 0112 ∅597 五个框）里**每个红框都包着完整符号**；
一量像素：`min=151/152`、`frac<180=0.1789`。阈值改 **200** 后：`≤2%` 墨迹的框从「324 个」变成 **6 个（0.3%）**，
外扩 25px 后**一个空框都没有**，按图源也**无差异**（jlc 2/1364、Datasheet 2/491、Altium 0/196）。

⇒ 三条：①**任何「占比 / 阈值」类判据，先把被判异常的那批裁出来肉眼核对**，再写结论；
②阈值要留成环境变量（`GNDBOX_INK`，默认 200）并注明它为什么是这个值；
③同一份证据里**作废与仍成立要分开写**——这次「14.5% 空框」「错标集中在 jlc」作废，
而「2229 个 GT gnd 框全是 `20x19`/`19x20` 固定图章」与阈值无关，仍然成立。

**同一批证据还教了一条「钝器」判据**：用同一把像素尺子量**公认单一形状**的类，
silhouette 反而是 `r 0.252 > c 0.207 > d 0.199 > gnd 0.141 > v 0.092 > net_input 0.086`
⇒ 像素 k-means 的 silhouette **分不开「多形状」与「单形状」**，只能用来**否定**
「k-means 构成某类多模态的证据」，**不能**用它断言「某类没有模态」。做这类对照（拿一个已知答案的类试尺子）
比多跑十个 k 值有用。

### bbox 是哪种约定：**别用眼睛判**，用「引脚对称性」判

新补充训练集（`Y:\workspace\EDA\新补充训练集`，58 张）的 `components[cid].bbox` 是归一化浮点
`[?, ?, w, h]`，前两位到底是**左上角**还是**中心**，用看图的办法**判不出来**：两种解释只差
`(w/2, h/2)`，在 2782×1683 的图上约 **20px / 0.7%**，总览图上「红框都框住器件」两边都成立。
（我第一版就是靠裁图目视下了「是左上角」，**下一版被推翻**。）

定量判据（`work\newbbox_final.py`，58 张 / **2516 个 ≥2 引脚的元件**）——**用引脚点关于器件本体对称**：

| 假设 | 「引脚质心 − 框心」/对角 中位 | 引脚质心落在框内 |
|---|---|---|
| `[x,y,w,h]`（左上角） | **0.5000**（**恰好半个对角 = 把中心当左上角的指纹**） | 63.2%（逐根引脚则只有 26.5%） |
| **`[cx,cy,w,h]`（中心）** | **0.0000** | **91.9%** |

更直接的一条：`bbox[0], bbox[1]` 与「该元件所有引脚点的质心」的距离中位 = **0.00000**（n=2516）
⇒ **`bbox[:2]` 就是元件中心**。判据要选**与假设强相关、与分辨率无关**的量（这里是「偏移/对角」的 0.5 指纹），
而不是「看起来对不对」。

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
4. **`fusion.run` 在 `--workers > 1` 时不可复现 ⇒ 所有对比臂一律 `--workers 1`。**（2026-10，子代理
   6be262af 在阅读顺序先验的 4 臂 A/B 里撞上）同一份代码、同一份输入，`--workers 2` 的两次 9 张批跑
   在 **case 8** 上给出不同的 `pins` / `nets` / `__pin_index`（MD5 `3D87…` vs `239C…`），而 `--workers 1`
   与历史上所有跑法都是 `8B26…`（case 8 单独跑 3 次 hash 全同）。9 张均值因此出现 **±0.09 的假差异**
   ——一次「带闸 C 臂 = 62.2699」的假读数就是这么来的，`--workers 1` 复跑回到 62.1763。
   fusion 本身很快（9 张 25~66 s），**不差这点时间**：凡是要进报告的数字，都用 `--workers 1`；
   用了 `--workers 2+` 的批量跑必须在报告里标注这个噪声底。

5. **`--no-llm` ≠「不碰 LM Studio」。** `fusion.run --no-llm` 只关掉 `fusion/run.py:417` 那个「LLM 轮」
   （`do_llm=(not a.no_llm) and DO_LLM`）。fusion 里还有几条**按自己开关走、默认开着**的 VLM pass，
   它们照样 `vlm.client.ask` → `127.0.0.1:1234`：`fusion/cropfix.py:61 ON = env_on("FUSION_CROPFIX","1")`
   （在 `run.py:219` 跑）、`fusion/iterate.py:58 FUSION_ITERATE`（默认开，`run.py:330`）、
   `fusion/audit.py:74/76 FUSION_AUDIT / FUSION_AUDIT_REREAD`（默认开）。
   **2026-10-08 实测后果**：一个 `fusion.run --no-llm --workers 2` 的 220 张扫描把 LM Studio 打成持续推理
   （`llama-server` 5 s 内烧 4.91 s CPU、GPU 47→78%，server log 里刷 `Running chat completion on
   conversation with 1 messages`，task id 已到 5 万+），**而它没有拿闸门锁** ⇒ 闸门在等一个永远等不到的
   显存：我自己的扫描把模型钉住，又让我自己的训练排不进去（用户先发现的：「为什么 LM Studio 一直能吃到请求」）。
   ⇒ 规矩：**只要进程可能打 LM Studio，就必须走 `gpu_gate`（哪怕它自己不吃显存）**；要做纯 CPU 的对照实验，
   显式加 `-vlm off` 或 `FUSION_CROPFIX=0 FUSION_ITERATE=0 FUSION_AUDIT=0`，并在报告里写明用了哪一组。

6. **「修完仍然是空框」是数据集自检的硬指标。** `build_trainset_v4.py` 的 `--fix-ink recenter`
   老版本把空框中心挪到**最近连通域的质心**（`cv2.connectedComponentsWithStats`）。对「一整条网络」那种
   大连通域，质心能落在空白处 ⇒ 挪完**仍是空框**；更糟的是多个空框会被挪到**同一个质心**，随后被
   `--drop-dup` 当成重复框删掉（真丢标注）。实测（2026-10-08，1920 张 train 抽样 12000+ 框）：
   v4（旧）有 **329 个空框 = 2.6%**（其中 278 个是类 `r`、全在新增语料、尺寸正好 20×8 / 8×20 的图章尺寸），
   空框邻域 1.5× 内**总能**找到墨迹、位移中位仅 **4.5 px** —— 根因是 `snap()` 只改尺寸不改中心。
   修法（现在的实现，`eda-pcb/tools/dev/build_trainset_v4.py` 的 `ink_recenter` + `ink_frac`）：在框周围开
   搜索窗，用**积分图**枚举所有候选位置，在「框内墨迹够多」的候选里取**位移最小**者；窗里连一个墨迹像素
   都没有时才退回最近墨迹像素。修后同一把尺子：空框 **290 → 12**（v3 基线 50），居中偏移不变（0.097），
   框总数 84760 → 84781（找回 21 个被 dedup 吃掉的标注）。
   ⇒ 构建脚本现在**每次都打**一行「空框自检：new X + base Y = Z   [v3 基线 27/13620 = 0.2%]」，
   残留非 0 会标 ✗；**重训前先看这一行**。量它的两把尺子：`eda-pcb/tools/diag/_inkcheck.py`
   （抽样量空框数 + 墨迹质心对框中心的偏移 + 框内墨迹占比，`--roots tag=path,...` 可随意比多个数据集）、
   `eda-pcb/tools/diag/_inktest.py`（4 条合成图硬测试，含上面那个质心陷阱的用例）。

7. **训练慢 20× 的元凶是「显存贴卡沿 + WDDM 溢出」，不是单一的 `--workers`。**（2026-10-08，v4 43 类重训；
   这条结论当天被推翻过一次，留着教训）现象 A：`--workers 6` 跑出 `1/60 28.7G … 290/480 5.2s/it`，
   而 v3 同一批超参是 **100.4 s/轮（0.22 s/it）**。现象 B：改成 `--workers 2` 后**同一套 args** 又慢
   （`1/60 16G→17.6G … 14/480 5.5s/it`，一个 epoch 42 min）；`Compare-Object` 比对
   `runs/detect/probe_w2/args.yaml` 与 `runs/detect/pcb43_v4_im1536-2/args.yaml` 证明两份参数**只差**
   `model` 路径形态（绝对/相对）、`epochs 1→60`、`name`/`save_dir` ⇒ **「`--workers 6` 是唯一元凶」不成立**。
   病态签名（两次完全一致）：`utilization.gpu 100%` 但 **`power.draw ≈ 84 W`**（好状态 199~220 W）、
   `utilization.memory 0%`、`clocks 2760/3105 MHz`、`clocks_throttle_reasons.active 0x0`、温度 48 ℃、
   系统 RAM 空 35 GB ⇒ **不是 CPU / RAM / 温度 / 降频**。`Get-Counter '\GPU Process Memory(*)\Dedicated Usage'`
   给出决定性证据：**我们自己的 python 独占 15.04 GB**（dwm 0.86 + msedge 0.13 GB，其余为 0）
   ⇒ 16 GB 卡被占满、WDDM 往系统内存溢出，并且**逐轮恶化**（1.3→1.7→2.1→2.9→3.9→5.5 s/it）。
   ⇒ 处置（定稿，`work/_gpu_queue.ps1` 已改）：**`--batch 4 → 2`**（imgsz 1536 不变）、`--workers 2` 保持、
   **删掉 `$env:PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:128'`**（抗碎片设置会把 PyTorch 的保留段撑大，
   实测 reserved 12.8G→17.4G，正好越过卡沿）。实测 `--batch 2`：**reserved 6.49~6.91G、199 W、
   6.4 it/s、960 步不退化（无该 env）**。⇒ 规矩：**进度条里的 `GPU_mem` 只要逼近/超过物理显存就是溢出，
   速度会塌 20~40×**；本机 YOLO 训练的 batch·imgsz² 上限就按 **batch 2 @1536** 定；`--workers` 仍 ≤ 2
   （它只决定溢出多少：reserved 12.8G→28.7G）。**判定阈值/参数要先看「离物理上限还有多远」，再谈算法**。

### PowerShell 5.1 与中文日志（本机踩过两次）

* 本机**没有 `pwsh`（PowerShell 7）**，`pwsh -File x.ps1` 会报 `CommandNotFoundException`；
  只能用 `powershell -NoProfile -ExecutionPolicy Bypass -File x.ps1`（默认执行策略禁止直接 `& x.ps1`，
  会报 `cannot be loaded. The file ... is not digitally signed`）。
* `.ps1` 存成 **UTF-8 无 BOM** 时，5.1 会按 ANSI 解析 ⇒ **脚本里的中文字面量全成乱码**（Python 侧写的
  日志文件不受影响，只有脚本自己的 `Write-Host` 会乱）。要中文可读就把脚本存成 **UTF-8 with BOM**，
  并加上 `[Console]::OutputEncoding = [Text.Encoding]::UTF8`（后者修的是「PS 解码子进程 stdout」这一半）。

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

### 「写进了产物」不等于「有人读」：**同一份产物跑两条口径，差值就是「没人读」的量**

**症状**（2026-10，引脚通道）：`vlm/enrich.py` 的引脚通道把读到的号/名写进了补强文档的
`pins[...]["pinname"]`，坐实了「读得对」——**同一份文档** `stage2_eval` 打 **60.7776**
（pin F1 0.5174），可把它喂给融合层跑交付口径，总分 **58.6304，与「通道关着」的基线
逐位相同**（pin 0.3920、`pin.name_mismatch` 16、`net.pred_pins` 238）。
**逐位相同**就是铁证：这条通道的产物在下游**一次都没被用到**。

**根因**：融合层 `fusion/claims.py:450 _add_vlm()` 只吃 `vlm["components"]`（键 / `Name` /
`value`），而引脚是 `stage2/assemble.py:588 build_doc()` 从 `stage1`（点表）+ `stage2/_assigned`
（号/名的文本）**重新装配**的 —— 补强文档里的 `pins` 没有任何人读。

**尺子**（两口径对账，成本 30 s，不用重跑 OCR）：

```powershell
# ① 文档级打分：把补强文档目录当预测目录直接评
python work\tools\eval\stage2_eval.py work\_vlm_pinfix_full --quiet
# ② 交付级（融合层）打分：同一份文档喂给 fusion.run，只关 LLM 轮
cd eda-pcb
$env:FUSION_RUN_DIR='Y:\workspace\EDA\work\_fast_out_reg9\runs'
$env:FUSION_VLM_DIR='Y:\workspace\EDA\work\_vlm_pinfix_full'
$env:FUSION_OUT_DIR='Y:\workspace\EDA\work\_f9_B_pin'
$env:FUSION_LLM='0'; python -m fusion.run 1 2 3 4 5 7 8 9 10 --no-llm --workers 3 --quiet
python ..\work\tools\eval\stage2_eval.py $env:FUSION_OUT_DIR --quiet --json ..\work\_f9_B_pin.json
```

⇒ **规矩：任何「把东西写进某个中间产物」的改动，验收时必须把该产物走完下游再打分。**
文档级分数只证明「读到了」，不证明「用上了」。同理，`_vlmab2.py` 的臂要叠在
**交付口径自己那份**补强文档上（`--in work\_fast_out_reg9\runs\stage3_vlm`），否则值/名
通道没开，比出来的是 confound（`work\_vlm_pinfix` 53.73 就是这么误读出来的）。

**补一刀（2026-10）：还要在真交付路径上复核一次，别只信融合-only 的表。**
上面 ② 是融合层口径；容器里还隔着一层 `submission/run.py` 的编排（CLI → VLM opts、
`FUSION_VLM_PIN_NAME` 这类环境变量必须在 `from fusion import run` **之前** set，
否则模块级常量已经求值完了）。**查接线只要单张**：`fast.py` 跑 9 张要 26 min，
而 `python -m submission.run -input work\_fast_in -output work\_smoke_pin_c2 -cases 2` 只要 **31 s**：

| case 2 | 总分 | comp | pin | netH | netL | `pin.name_mismatch` |
|---|---:|---:|---:|---:|---:|---:|
| 旧交付（`work\_fast_out_reg9\final_results`，通道关） | 53.5038 | 1.0000 | 0.2727 | 0.4516 | 0.0879 | 3 |
| 新交付（单张重跑，通道开） | **58.0492** | 1.0000 | **0.4545** | 0.4516 | 0.0879 | **1** |

⇒ 与融合-only 臂里 case 2 的 **+4.55 逐位一致**（都是 +4.545），其余三项**一位没动** ——
只有「单张真交付」对上了这个数，才能说 CLI → opts → `_pin_channel` → 融合层 (d) 段这条线通了。

---

### 报「均分」前先按 split 拆开：**平均分会把方向相反的档抵消掉**

**症状**（2026-10，同一个引脚通道的 A/B，全量 220 张）：全量均分 **29.4819 → 29.4099
= −0.072**，看上去「这条通道在非官方数据上是负的」。按 split 拆开（`work\_f220_A.json` /
`work\_f220_B.json` 逐张）：

| 子集 | n | A | B | Δ 均分 |
|---|---:|---:|---:|---:|
| 官方 `10GT` | 10 | 57.8383 | 60.4310 | **+2.5927** |
| `10_GTcase` | 10 | 32.7960 | 33.5577 | **+0.7616** |
| `200_train_cases` | 200 | 27.8984 | 27.6515 | **−0.2469** |

⇒ **三个集子里两个在赚**，负号只来自最大那一档；而那一档的 GT 是自动命名派生的、
引脚号先验又正是在它上面调的（**自证**）。

**同一个通道再把「号」「名」两半拆开（四臂 × 220 张）**，负号的来源就唯一了：

| 子集 | n | A 通道关 | B 全开 | C 只名 | D 只号 |
|---|---:|---:|---:|---:|---:|
| 官方 `10GT` | 10 | 57.8383 | 60.4310 | 59.0231 | 59.5944 |
| `10_GTcase` | 10 | 32.7960 | 33.5577 | 33.1807 | 33.2005 |
| `200_train_cases` | 200 | 27.8984 | 27.6515 | 27.5801 | 27.8006 |
| **均分** | 220 | 29.4819 | 29.4099 | **29.2639** | **29.4912** |

- **B − C =「号」那一半**：`10GT` **+1.41** / `10_GTcase` **+0.38** / `200_train` **+0.071**
  ⇒ **三个 split 全正**（只补号，不会去覆盖已经读对的名）。
- **C − A =「名」那一半**：+1.18 / +0.38 / **−0.32** ⇒ **220 上的负号完全来自「名」**。
- 只号臂在两个官方档 split 上**零回归**（3 变好/0 变差、1/0）⇒ 「要稳就不要名」是 220 档的结论；
  但官方 9 张上「名」值 **+0.93**（61.5112 vs 60.5816）⇒ **默认全开、`-vlm-pin-name off` 作为一行退守**。
- 用法：`python eda-pcb\tools\diag\_cmp9.py work\_f220_A.json work\_f220_n1m0.json`（自带按 split 汇总）。

**规矩：任何跨数据集的 A/B，报均分前先按 split（或任何已知的异质子集）拆一次。**
200 张的权重足以让 20 张上的 +2.6 在均分里看不见 —— 别让加权平均替你做结论。
**如果一个通道有两个子开关，就再拆一层**：合并的净效应会把两个方向相反的分量抵消成
「什么都看不出来」（这里 −0.07），拆开才发现负号只来自其中一半。

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

### 「只重跑融合层」**不等于**「不碰 VLM」：`FUSION_LLM=0` 关不掉引脚名回传

2026-10-09 我按 `FUSION_LLM=0` + `fusion.run.run_case(case, do_llm=False)` 跑 220 张的
默认值 A/B，结果是 **GPU 一直被 LM Studio 占着（9.7 GB / 82% util）、训练在 `gpu_gate`
上等了 40 min**。查证过程（可复用）：

```powershell
# 谁在调 LM Studio？先找客户端 PID，再看它的 cmdline
Get-NetTCPConnection -RemotePort 1234 | Where-Object State -eq Established |
  Select-Object LocalPort,OwningProcess
Get-CimInstance Win32_Process -Filter "ProcessId=<PID>" | Select-Object CommandLine
```

⇒ 客户端就是那个「纯 CPU 融合」进程。原因在 `fusion/run.py:330`：

```python
rep_it = IT.pin_fields(assigned, doc, pool.bgr, log=log)   # ⓪.9 定向回传：问一次 VLM 的引脚名
```

这一句**不受 `do_llm` / `FUSION_LLM` 控制**：它只受 `fusion\iterate.py:58` 的
`FUSION_ITERATE`（默认 `1`）控制，`do_llm=False` 只关掉后面的 LLM 仲裁轮。
后果有两层：

1. **GPU 占用**：任何「融合层 A/B」都会真调 VLM ⇒ 与训练抢卡（16 GB 卡上
   llama-server 9.7 GB + YOLO 6.5 GB 必炸）⇒ 要么等训练跑完，要么显式 `FUSION_ITERATE=0`。
2. **结论噪声**：两臂都吃一次真 VLM 读数，逐张 Δ 里混进 VLM 抖动。`FUSION_ITERATE=0`
   能得到确定性，但**两臂同时偏离交付配置**，只适合做「机制定位」，不适合改默认值。

纪律：跑「离线/纯 CPU」对照实验前，先 `Get-NetTCPConnection -RemotePort 1234` 确认没有
Established 连接；改默认值的 A/B 一律接受「会调 VLM」，并在报告里写明这一点。

### 产物里必须有「这条结论是谁下的」字段，否则归因只能靠猜

官方 10GT 第 1 张交付出去 6 个位号是 `C1513/C1514/C1515/C1517/C1518/C1519`，
而图上印的和 GT 都是 `C153/C154/C155/C159/C160/C161`。它们不是读错的：

```powershell
# 每颗元件都带 _by = 这条 designator 是谁定的
python -X utf8 work\tools\diag\_c1src.py          # 逐条打印 (元件, 读数, 终值, _by)
```

⇒ 6 个全带 `_by = "solve:prior:run_altwindow"`。而 `run_altwindow` **只存在于 `fusion/priors.py`**，
所以错号是融合层编的、`stage2` 原来是对的（两份 `_assigned\1.json` 的 md5 不同：
`work\_f9_B_pin\_assigned\1.json` = `44f0f1d7e9a9`，
`work\_fast_out_reg9\runs\stage2\_assigned\1.json` = `3a6bee232875`）。

机制在 `fusion/priors.py:528 repair_variants()`：它把数字段每个位置插回一个 `1`
（`C153`→`C1153`/**`C1513`**/`C1531`），这些修复变体被 `windows_for()` 当成锚点开窗，
于是多出一个**偏移窗口 `[1511,1522]`**；`annotate()` 又把**并列窗口**（`ws[1:]`）的号一起塞进候选池，
于是 `C53→C1513`、`C15→C1515`、`C101→C1519`…（`_fusion_report.json`：这张图 181 个槽位里
**95 个是 `prior` 派的**，`score_final.sources={"ocr":18,"vlm":100,"stage2":14,"prior":95,"geo":6}`）。

三条要记住的：
1. **先验只能细化读数，不能发明身份**（`metrics.is_unkeyed` 只认 `^∅\d+$`/`^p\d{3,4}$`，
   `fusion/priors.py:619-629` 已用 `0001` 的教训写过一遍）。
2. **并列窗口不能靠分差选，要靠阅读顺序选**：官方 GT 坐标系 y 越大越靠上，
   case 1 的 12 个电容按「y 降序、同 y 按 x 升序」正好是 `C151…C162`（12/12 全对）。
   只把并列窗口关掉（`FUSION_RUN_ALTWINDOW=0`）是 61.5112→61.5798、case 1 的 compF1
   反而 0.5909→0.5455 —— **只是打乱，不是修好**；**必须再开 `FUSION_RUN_ORDER=1`
   才算修好**（2026-10-09 定案：两个默认值一起翻，官方 9 张 +0.6291，case 1 的 12 颗
   电容终于恰好 `C151…C162`，其余 8 张逐字节不动，见 `docs\RESULTS.md` §2.7）。
3. 归因顺序：**先看出处字段（`_by` / `source` / `sources`），再看分数**。
   只按分数拆，你会以为是 OCR；有出处就能一步定位到哪一层、哪个先验。

### 杀进程前先看 cmdline：**按 pid 猜归属 = 杀掉正在干活的人**

2026-10-09 07:47，42 类合并集训练在 24 min 处猝死，日志尾部是：

```
RuntimeError: DataLoader worker (pid(s) 27632, 5412) exited unexpectedly
[gate] 结束：退出码 1，用时 1438.7s，峰值整卡占用 15868 MiB（锁已释放）
```

原因不是数据、不是显存：是我上一步清 `python -X utf8 -m fusion.run ...` 的卡死残留时
**按 pid 列表杀 python**，而 `27632/5412` 正是这次训练的 dataloader worker（`--workers 2`；
它们启动的时间 == ②b 开始的时间，我把「CPU 累计 388 s、看着像孤儿」当成了残留）。
正确做法：

1. **任何 `Stop-Process` 之前先看 cmdline**：
   `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId,ParentProcessId,CommandLine`
   —— 按「CPU 高 / 活得久 / 看着像孤儿」判断归属，一定会杀错。
2. 「孤儿 worker」的可判据是**父进程已经不在了**（拿 `ParentProcessId` 去查活进程），不是 CPU 时间。
3. 真要清残留，只清**自己刚起的那个进程树**：按启动时间 + cmdline 里的 `--out` / `--tag` 匹配，
   动手前先确认没有别的长任务在跑（`gpu_gate` 的锁文件只保护 GPU，不保护「谁在跑 dataloader」）。
4. 被杀的长任务**目录必须移走**（本例移到 `work\_dead\pcb43_v4m42_im1536_dead5ep`），
   否则 ultralytics 会另建 `...-2` 目录，而队列第 ③ 步拷的 `results.csv` 还是旧目录那份（10-08 已踩过同款）。

---

### 交付入口的失效面：**5 个整进程死点 + 3 类静默降级**（2026-10-14，用户 m42543「考虑各种灾难性现场」）

判题机**一次性给全部样例、允许并发**，所以「只挂一张图」和「整批 0 分」都要当成必须处理的现场。
只读审计（`submission/run.py` + `stage1/pipeline.py`）查出的失效面：

| 现场 | 旧行为 | 现在 |
| --- | --- | --- |
| 阶段一进程池崩（worker 段错误 / OOM-kill） | `stage1/pipeline.py:479 r = f.result()` 与 `run.py:233 run_cases(...)` 都**没有 try** ⇒ `BrokenProcessPool` 冒泡，**整进程死、退出码 0、`final_results/` 空** | `run_cases` 每个 future 单独 try + 池级 except ⇒ 余下 case 在**本进程串行补跑**；`run.py:main` 外层再套 `_safety_net` |
| 阶段二/融合/导出抛异常 | 导出与 `decision_log.json` 还没写 ⇒ **整批 0 分**（`:296-302` 记的历史事故） | `main` = `parse_args → _fix_device → try _run except _safety_net`；`_safety_net` 对**输入目录每张图**补空壳 + 写 `decision_log.json`（`summary.fatal`）+ traceback 进 `run.log`，返回 **3** |
| 无 CUDA 但默认 `-device 0` | ultralytics `torch_utils.py:326-346` 直接 `raise ValueError("Invalid CUDA 'device=0'")`，被 `_worker` 吞成 `ok:False` ⇒ 0 产物、退出码 0 | `_fix_device(a)`：`device` 是数字/`cuda*` 且 `torch.cuda.is_available()=False` ⇒ 打印警告并回退 `cpu`（与 `textrec/infer.py:78-101` 的既有语义对齐） |
| 某张图根本没产物 | `run.py:471-479` 只 `_log("没有产物")` 后 continue ⇒ **该 case 一份 `_Pred.json` 都没有** | 导出循环写**空壳** `{"components":{},"pins":{},"nets":{}}`（+`case`/`note`）+ `n_shell` 计数；`metrics.py:309/:455/:552` 是直接下标，空壳三键必须齐全 |
| 同名 case（`a.png` + `a.jpg`、多目录同名） | `discover_inputs` 只 `sorted` ⇒ 阶段一互相覆盖（workers>1 时并发写同一文件）、只出 1 份产物且 `:483` 的计数察觉不到 | `_dedupe_items()` 保序去重 + ⚠️ 日志报出被丢的项 |
| 非 ASCII / 带空格路径 | `stage1/pipeline.py:154 cv2.imread(str(path))` 在 Windows 上**静默返回 None** ⇒ `RuntimeError` ⇒ 该图零产物（阶段二早就用 `stage2/text.py:181-187 imread_unicode`） | `_imread_bgr()`：`np.fromfile` + `cv2.imdecode`（PIL 只读文件头） |
| 巨图 | 无像素上限，`imread` 直接吃 9 亿像素（硬 OOM-kill 不是异常，`except` 救不了） | `STAGE1_MAX_PIXELS_HARD`（默认 **8e7 = 80 MP**，只看文件头）⇒ 超过就按 1/2、1/4、1/8 解码（`IMREAD_REDUCED_COLOR_*`）。本地最大图 4491×3173=14 MP ⇒ **正常路径逐像素不变** |
| 资产缺失（权重/库） | 权重缺失会联网下载或抛异常；**CRNN 静默降级**（`stage2/text.py:343 allow_missing=True`）⇒ 位号/值全空但「产物齐全、退出码 0」 | `run.py` 把 `assets.describe()` 逐项 `exists` 打进行首日志，缺失项打 ⚠️「该通道会静默降级」 |
| 并发写盘 | `runs/fusion/_fusion_report.json` 读-改-写**无锁**；决策日志直接覆写 | 决策日志与 `_fusion_report.json` 改 `_atomic_write_text`（`.tmp` + `os.replace`）；阶段二主循环**渐进导出**（每张成功即拷贝一份 `_Pred.json`） |

**验证**（`tools\dev\_hardtest.py`，五个现场 **21 项断言全过**）：非 ASCII 路径 `cv2.imread → None` 而
`_imread_bgr` 与 ASCII 路径**逐像素相同**；坏图 ⇒ 退出码 0 + 空壳 + `decision_log.json`；同名 `same.png/same.jpg`
⇒ 只导出 1 份；`_fix_device` 三种输入行为正确；注入 `run_cases` 抛异常 ⇒ `main` 返回 **3**、两张图各补空壳、
`summary.fatal` 记录异常串。**回归**：`fast.py --cases 1 --no-vlm` 前后都是 **62.62**（逐位不变）。

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

**「GT 有、我方没有」的 IC 引脚到底丢在哪**（2026-10，`work/tools/diag/_pinmiss.py`）：

```powershell
# ⚠️ 半径必须 8 px：引脚列间距 ~20 px，30 px 会串到隔壁脚
python -X utf8 work\tools\diag\_pinmiss.py --pred work\_fnew --assigned work\_fnew\_assigned `
    --stage1 work\_fast_out_reg9\runs\stage1 --cases 1,2,3,4,5,7,8,9,10 --radius 8 --verbose 1
```

它把每个 missing 引脚追回原始读数（`_assigned/<case>.json` 的 `comps[cid].pins[*].name/number` +
`stage1.pins[cid]` 的桩点列表）再分桶。交付级 `work/_fnew` 的 55 个 missing（**半径 8 px**）：

| 桶 | 数 | 占比 | 含义 | 怎么修 |
|---|---|---|---|---|
| **A3** 名读到、桩在、号没读到 | **22** | 40.0% | 缺「这一个桩的号」 | 号识别 / **用官方库补号** |
| **C** 附近没有名字读数 | 14 | 25.5% | 检测真没拿到 | 检测/裁剪 |
| **A2** 名+号都读到、但该处无桩 | 9 | 16.4% | stage1 没给出这个引脚点 | 引脚检出 |
| **A1** 名+号+桩都在、号读错 | 7 | 12.7% | 读数齐了却读错号 | 号识别 |
| D 同位同名、号不同 | 2 | 3.6% | 号读错 | 号识别 |
| A4 只有名 | 1 | 1.8% | | |

- **教训（本轮自己犯的）**：半径 30 px 时出现的 `E-swap 13`（「同位发了、名是隔壁脚的」）**是伪影** ——
  8 px 下 E 桶**整桶为 0**。判定阈值必须小于结构周期（这里是同列相邻脚间距 ~20 px）。
- **A1 的回头复核**：那 9 个引脚读到的号文本里**没有一个含 GT 号**（样例 `num='2 / X2'`、`num='0'` ——
  我方把两处读数拼进了一个字段）；同理 E 桶我方键的号 **0/15** 等于 GT 号。
  ⇒ 「A1+E = 22 个纯绑定、不动模型就能拿」的说法**不成立**：它们都是**号识别错**。
- attach 交叉列（8 px）：A3 no-attach 21 / attach 1、A2 no-attach 9、C no-attach 11 / attach 3
  ⇒ **A3 的桩大多是孤脚/悬空脚**，这解释了它们为什么只有名没有号。
- 修正后的结论：最大矛盾是「**号的召回**」（A3 22 + A1 7 + D 2 = **31/55 = 56%**），
  第二是检测（A2 9 + C 14 = 23/55 = 42%）；**不存在「号键撞车」**（`_libopp` 的 `extra 0` 也支持）。
- **号的来源**（几何插值已判负：`_pinlat.py` 里 ≤5 px 只 1/52）⇒ 唯一外部来源是官方库：
  见 `eda-pcb\docs\COMPETITION_RULES.md` §6 ④（`_libnum.py` 量上界、`_libsim.py` 量交付级收益）：
  规则 = 库里 `(型号, 名)` 唯一映射一个号 ⇒ 补号；`(型号, 号)` 唯一对应一个名 ⇒ 改名。
  官方 9 张实测 **+2.4635 分**（pin F1 0.4951→0.5936，tp 44→54，fp 21→17），**纯 CPU、不动模型**。

同一批的对照（`_libopp.py`）：文档级臂 `pinname ok 43 / missing 61`、号通道 `extra 6` →
融合臂 `work/_fnew` `ok 58 / missing 55`、`extra 0` ⇒ **融合的 VLM 轮把 6 个错号修回来了**。

⚠️ 陷阱：`stage1.<case>.json` 的 `attachments[cid].hits` 是**截断过的**（一颗大 IC 只记 3 条），
所以本工具那个 `attach` 交叉列**不可当判据**；判「桩在不在」要用 `stage1.pins[cid]`（点列表）。
⚠️ 另一个陷阱：`_assigned` 的引脚索引与 `stage1.pins[cid]` 的桩点**同序**，但**名字读数框不在引脚点上**
（在符号体内部）——用「名字框中心离 GT 引脚 ≤8 px」判「名读到」会得到 **0/55** 的假结论，
要用「最近的桩索引 j 的 `raw_pins[j].name`」。

#### 官方库补号/改名已落进管线（`STAGE2_LIB_PINS`，**2026-10-09 默认已改为开**）

上面那条「唯一外部来源是官方库」的建议**已经实现**，不再是模拟：

| | |
|---|---|
| 实现 | `eda-pcb\stage2\libpins.py`（新）+ `eda-pcb\stage2\assemble.py` `build_doc` 的 box 引脚路径 |
| 总闸 | `STAGE2_LIB_PINS`，**默认 `"1"`（开）** —— 2026-10-09 依据下表 220 张拆分回归从 `0` 翻过来；`STAGE2_LIB_PINS=0` 时 `libpins` 根本不会被 import，产物逐字节不变 |
| 库路径 | `STAGE2_BOXIC_LIB`（新）/ `STAGE2_CPNT_LIB`（兼容旧名）；默认走 `assets.libraries()`：**包内 `eda-pcb\data\BoxICLibrary_200cases.json`**（1,826,150 B / 360 键，**2026-10-14 起为默认库**，用户 m42932）→ 包内 `eda-pcb\data\cpntLibrary.json`（兜底）→ 本机 `work\data\` → `赛题资料\` → 数据集目录（2026-10-12 前默认写死本机 `work\data\cpntLibrary.json`，容器里必然缺） |
| 模糊阈值 | `STAGE2_LIB_PINS_FUZZY`，默认 `0.85`（**只用来找库键**；某条引脚仍走「唯一」判定） |
| 诊断 | `STAGE2_LIB_PINS_DBG=1` 打印补/改了哪几处；`python -X utf8 stage2\libpins.py` 跑自检 |

两条规则（都不看 GT，都**只在唯一映射时动手**）：**A 补号** = 桩读到了引脚名、但号是空的
或者是**位置补位号**（`fill`）⇒ 用库里 `(型号, 名)` **唯一**对应的号；**B 改名** = 号读到了、
但 `pinname` 空或不在库里该型号的名字集合里 ⇒ 用库里 `(型号, 号)` **唯一**对应的名字。

**验证（纯 CPU；不跑 `fusion.run` / `fast.py`）**——两件事都做了：

1. **开关关掉时与「不挂接」逐字节一致**：`work\_libpins_bak\mk_noedit.py` 从当前文件里把
   libpins 挂接**逐字挖掉**重建出「编辑前」的 `assemble.py`，`work\_libpins_bak\cmp_off.py`
   让两者对同一批输入各跑一次 `build_doc` ⇒ **9/9 张 `sort_keys` JSON 的 md5 相同**。
2. **`stage2` 直出这条路上两条规则几乎不触发**（关/开两次跑的 9 份产物 MD5 全 SAME）——
   stage2 的 `Name` 是 OCR 原文，17 个 box 里只有 1 个命中库（收益靠型号名被读对，
   那是**融合层**的事）。真正能触发的是「融合层的 `_assigned` + `stage1` 直接喂 `build_doc`」：

```powershell
# 交付级等价 A/B，纯 CPU（融合层也是这么调 build_doc 的，见 fusion\run.py:128）
python -X utf8 work\tools\diag\_libpins_cpu.py `
    --stage1 work\_fast_out_reg9\runs\stage1 --assigned work\_fnew\_assigned `
    --out work\_libpins_cpu --cases 1,2,3,4,5,7,8,9,10
python -X utf8 work\tools\eval\stage2_eval.py work\_libpins_cpu\off --quiet
python -X utf8 work\tools\eval\stage2_eval.py work\_libpins_cpu\on  --quiet
```

官方 9 张实测（2026-10）**67.3015 → 68.7995（+1.4980）**：pin F1 0.5350→0.5715
（tp 47→51、fp 18→21、**fn 73→69**）、net_hypergraph F1 0.6800→**0.6972**（tp 245→255、fn 174→164）。
补号 7 处、改名 2 处、受影响网络 37 条，**逐条对 GT 核过：7/7 号对、2/2 名对**
（`assign_components + rewrite_keys` 对齐后 `{'add_ok':7,'add_bad':0,'ren_ok':2,'ren_bad':0}`）。
产物目录 md5 合并值：off `6d57f48961f4388e` / on `316345d07513e8cf`（逐张 md5 见 `_diff.json`）。
⚠️ 这与 `_libsim.py` 的模拟 +2.4635 **不是同一口径**（模拟基线的分母不同，且模拟补进去的
引脚**不进网络**）：本实现补的号**进** net（netH 因此多赚一份），而模拟那份的净收益只在 pin 通道。

⚠️ 打分口径坑：`stage2_eval.py <dir>` 会把 `<case>.pinindex.json` 当预测、造成重复打分
（18 份预测配 9 张 GT，分数虚低）。打分前把那两个文件挪走。

**220 张拆分回归（2026-10-09，这就是改默认值的依据）**：输入换成全量 220 的
`work\_full220\runs\stage1` + `work\_f220_A\_assigned`（case 名单 `work\_cases220.txt`，
220 个逗号连接；32 s 跑完，纯 CPU）：

```powershell
$cases = (Get-Content work\_cases220.txt -Raw).Trim()
python -X utf8 work\tools\diag\_libpins_cpu.py --stage1 work\_full220\runs\stage1 `
    --assigned work\_f220_A\_assigned --out work\_libpins220 --cases $cases
python -X utf8 work\tools\eval\stage2_eval.py work\_libpins220\off --quiet --json work\_libpins220\_score_off.json
python -X utf8 work\tools\eval\stage2_eval.py work\_libpins220\on  --quiet --json work\_libpins220\_score_on.json
python -X utf8 work\tools\diag\_split220.py work\_libpins220\_score_off.json work\_libpins220\_score_on.json
```

| suite | n | off → on | Δ | 逐张 |
|---|---:|---|---:|---|
| `10GT`（官方） | 10 | 62.5743 → **65.1834** | **+2.6091** | 7 涨 0 跌（pin tp 38→47 / fp 42→38 / fn 98→89） |
| `10_GTcase`（官方） | 10 | 36.8815 → 36.8815 | 0.0000 | 全不变（型号名不在库里） |
| **官方 20** | 20 | 49.7279 → **51.0324** | **+1.3045** | — |
| `train200`（派生） | 200 | 31.3691 → 31.4300 | +0.0609 | 24 涨 25 跌（最差 `0167` −2.924） |

改动的量：补号 125 / 改名 367 / 受影响网络 917。门槛「官方为正 且 派生不为负」已过 ⇒ 默认改开。
`STAGE2_LIB_PINS_FUZZY=1.0`（只用精确键）与默认 0.85 几乎相同（派生 +0.0616 vs +0.0609、改动更少
116/305/720）⇒ 派生档的回退不是模糊键造成的，阈值保持 0.85。另：**交付口径官方 9 张 +1.60
九张零回归**（`RESULTS.md` §2.6 表；那才是真正决定分数的口径）。

⚠️ **交付包必须带上库文件**：默认已开，它是运行时依赖。
**2026-10-12 修**：库已随仓库提交在 `eda-pcb\data\cpntLibrary.json`，`libpins.lib_path()` 与
`assets.cpnt_library()` 都优先解析它。旧默认写死 `WORKSPACE\work\data\` ⇒ 官方镜像里解析成
`/work/data/cpntLibrary.json`，**必然不存在**；缺文件只打印一行 `[libpins] 库不可用` 并自动
跳过（不炸管线、能跑出分，但**白丢实测 +1.60**）—— 这是「测出来有分、交付里没有」的典型陷阱。
**2026-10-14**：默认库再换成包内 `eda-pcb\data\BoxICLibrary_200cases.json`（360 键，用户 m42932）；
`lib_path()` 的顺序是 `STAGE2_BOXIC_LIB` → `STAGE2_CPNT_LIB` → **包内 BoxIC** → 包内 cpnt → 本机兜底，
**BoxIC 缺席时自动退到 cpnt**（只带旧库的镜像照样开着）。两个库文件都在 `submission\run.py` 的
`assets.describe()` 里打进行首日志 ⇒ 缺哪个会直接现形。只换库文件的 A/B：官方 9 张
68.7995 → **69.0883**、派生 220 张 33.2120 → **33.4364**（`COMPETITION_RULES.md` §6 尾）。

### 引脚坐标超差：**「发的是 base 不是 tip」被判负**（2026-10，`_pintip.py`）

官方附录三说元件有 outerbbox/bodybbox、**pin tip 在外侧框边缘、base 连内侧框**，我们
`_pinpos.py` 又量到超差引脚里 20~50 px 桶的误差中位数是 `dx +2.0, dy +29.0` —— 于是有个
很自然的假设：**我们发的是符号里侧的 base 点**。用 `work\tools\diag\_pintip.py` 逐条量过（两臂都量）：

```
python -X utf8 work\tools\diag\_pintip.py --pred work\_fnew                       # 融合层
python -X utf8 work\tools\diag\_pintip.py --pred work\_libpins_cpu_verify\off     # stage2 直出
```

| 量 | 结果（`work\_fnew`，58 对「号同+名同」的引脚） | 含义 |
|---|---|---|
| `sG`（GT 点到 **GT bbox** 的有符号距离） | 中位 **−13.0**，`|sG|≤3` 占 **0/58** | GT 引脚**根本不在 GT bbox 边缘**，而在框外 ~13 px ⇒ 我们手里的 GT bbox 是**内侧体框**，不是外框 |
| `sP`（我们的点到同一 bbox） | 中位 **−12.3** | 与 GT 只差 **1.2 px** |
| `sP − sG` | 中位 **+1.2**，>0 占 35/58 | **没有系统性的里/外偏移** ⇒ 假设 (H1) 判负 |
| 内法向证据（位移 ≥70 % 落在内法向） | 5/14 | 全是 case1 的 5 个 GND（`|d|=8.0`、`tan≈0`），是**局部**现象 |
| 邻脚证据（我们的点落在同元件另一个 GT 脚 5 px 内） | **3/14**，切向偏移 **28.5~29.5 px** | ≈ 引脚列间距 ⇒ 这是**读号串扰**的下游后果，不是几何估计问题 |

**结论**：位置超差（14 个，≈ −1.6 分）**不能**靠「统一往外挪 N px」修 —— 需要的位移方向逐条相反
（case1 的 GND 要往外 +8，case4/case5 要往内 −6/−5），中位数就是 0。真正要修的是**号↔桩的绑定**
（见 A1 归因：R1 一框一桩、R2 列内保序匹配、R3 元件框仲裁 + 停用序数伪造号）。

### 元件（component，权重 0.30）

```powershell
python tools\diag\_keytype.py     # 按元件类型量键命中率
python tools\diag\_des_ceiling.py # 位号链的天花板：认不出 vs 归错元件，分开量
python tools\diag\_desigring.py   # GT 位号在周围 R 像素的文字框里到底有没有
```

**★ 2026-10-10 定案：派生档（`200_train_cases`）的 `component.missing/extra` 是「位号键对不上」，不是漏检、也不主要是文字。**

- 工具：`work\tools\dev\_compgap.py`（按 `∅NNN` 伪元件 / 普通位号分桶量键命中率）、
  `work\tools\dev\_compdump.py`（单 case 并排印 GT / keymap / 两边没配上的）、
  `work\tools\diag\_suite_diag.py`（按 `case_id` 拆 `10GT` / `10_GTcase` / `train200` 的 F1 与 `diagnostic`）。
- 读数（`--dir work\_ro220b --limit 12`）：**普通位号 249 个里 123 个（49.4%）没配上**，
  而 `∅NNN` 伪元件 253 个里只有 14 个（5.5%）没配上；预测元件数 482 ≈ GT 502 ⇒
  **数量和坐标都在，只是键不同**。case `0001` 实例：GT `R5@[280,280,300,288]` ↔ pred
  `R9@[280.02,280.25,300.02,288.25]`（同一位置、键不同），另有「整排错位一个」的挂载错。
- 机制（`schematic_parser\metrics.py:299 assign_components`）：① 先按**键**成对
  （`set(gc)&set(pc)`），配不上的记 `component.missing`/`extra`；② 坐标匹配只在
  `cfg.unkeyed_match == 'gt'`（**注释写明「官方口径」**）时放行，且**只放行 GT 侧
  `is_unkeyed`（`^∅\d+$`/`^p\d{3,4}$`）的键**（`:338`），配上后撤销第①步的 fn/fp。
  ⇒ **位号读错 = 必丢一对 tp（fn+fp 各一），并级联到该元件的 pin 与 net。**
  与旧结论一致：**上帝键**能涨 **+5.66**（`docs\RESULTS.md:175`，`tools\diag\_keyoracle.py`）。
- 反面（别把预算全押在 CRNN）：train200 的 `component.text_mismatch` 只有 **583** 对；
  域内语料里 designator 读到位的行与 GT 一致率约 86%（`_offtext_eval.py` 留出集 74%）。
  ⇒ 杠杆排序应是 **位号挂载/识别 > 文字字形**。
- 字体实测（为文字域适配定方向，工具 `work\tools\dev\_fontfit.py` / `_fontsheet.py`）：
  **`jlc` → arial 0.875**、**`Datasheet` → arial 0.888**、**`Altium Designer` → times 0.866**，
  10GT 那批目视是**衬线等宽**（≈ `courbd.ttf`）；而自建合成语料把位号/值/引脚号都画成
  `mono` = `consola.ttf`（`synth\style.py:23-27`）⇒ **字体类别不匹配**。
  对应新增官方字体合成器 `work\tools\dev\mk_fonttext.py`（按变体频次 jlc 3355/Datasheet 836/
  Altium 313/10GT 307 采样字体组，字号照官方墨迹高度分布采样）。

**★ 2026-10-10 位号挂载体检（`work\tools\dev\_desacc.py`）：派生档 60 张 / 2026 个 GT 元件**

```
伪键（没挂位号） 47.6%   键对 21.0%   挂了别的位号 20.9%   位置没配上 10.4%
r: 键对 42% 伪键 19% 挂错 35% | c: 45/20/35 | box: 26/31/41 | switch: 19/24/57
gnd 496 个、v 211 个是**按设计**用 ∅NNN 伪键（靠坐标匹配），要从分母里剔掉
```
- 三张现场图（`work\_desview_0001.png` 等）证明：**文字本身又大又清楚**（`C6` 印得明明白白），
  我们却输出 `C1`/`C9→Y1`/`R1→R1210`/`R5→R9` ⇒ **不是字形/字体问题，是「框↔元件」的配对问题**。
  `R1210`/`L3215`/`R110` 这种是**把两个位号（`R12`+`10`）读成了一个**。
- 已知的一条线索（`vlm\enrich.py:550-553` 注释）：伪键元件 220 张里 **2601 个，其中 92% 在 10px
  内就有文字框**，只是 `assign.py` 的「同一个框只能给一个人」全局去重没分给它 ⇒ **兜底一轮就能救**。
- 纪律：量位号要用**自己的配对指标**（IoU 配对 + 键比较，`_desacc.py`），不要只看 end-to-end F1
  —— F1 被 netH/引脚稀释，位号错 60% 时 F1 只掉一点点。

### 域内微调权重**不能直接换进 stage2**：语料裁片 ≠ 运行时裁片（2026-10-11）
- 现象：域内语料微调出的 `work\ocr_official_v1\best.pt`，在**语料留出集**上 ALL **66.05% → 87.44%**
  （value 37.59% → 90.78%，`work\tools\dev\_offtext_eval.py --split val`，23 个官方 case / 645 行）；
  但把它**直接换进 stage2**（`python -X utf8 -u -m stage2.run --stage1 work\_full220\runs\stage1
  --out work\_s2_ft --ckpt work\ocr_official_v1\best.pt --cases 1,2,3,4,5,7,8,9,10 --device cpu`）后，
  官方 9 张由 **62.7992 掉到 58.3532**，主因是 **case 1：61.81 → 38.60**（compF1 0.7727 → 0.5909、
  pinF1 0.5581 → 0.0000）；其余 8 张只在 ±1.5 内摆动。
- 定位到「**同一个框、读法变差**」：`work\_s2_base\_assigned\1.json` 与 `work\_s2_ft\_assigned\1.json`
  逐元件比位号槽，7 处不同而 **box_i 完全相同** —— base `C152 / C1S4 / C153 / C1S5 / C159 / R164`
  （多数正是 GT）→ ft `C12 / C14 / C3 / C155 / C153 / R14`；value 也从 `10uF` 变 `10F`。
- **★ 修正（同日，2×2 矩阵跑完）**：掉分的真因是**外扩量是给旧权重调的**，不是「微调没用」。
  语料裁片是**未外扩**的原始框（`image.crop(bbox)`），而 `stage2/text.py:384 TextEngine.read`
  会先 `pad_box`（水平 `PAD_MIN=2.0`、垂直 `PAD_FRAC_Y=0.20/PAD_MIN_Y=3.0`）再喂模型 ——
  把外扩关掉（`STAGE2_PAD_MIN=0 STAGE2_PAD_FRAC=0 STAGE2_PAD_FRAC_Y=0 STAGE2_PAD_MIN_Y=0`，
  环境变量在 import 时读）后：

  | 臂 | 官方 9 张（10GT） | 派生 30 张（train200 前 30） |
  |---|---|---|
  | 旧权重 + 外扩 2.0/0.20/3.0（**现行交付**） | 62.7992 | 25.4934 |
  | 旧权重 + 关外扩 | 58.7400 | — |
  | 微调权重 + 外扩 2.0/0.20/3.0 | 58.3532 | 25.4518 |
  | **微调权重 + 关外扩** | **63.2776** | **26.2658** |
  | 微调权重 + 小外扩 1.0/0.10/1.5 | 61.9761 | **26.5236** |

  ⇒ **换权重必须同时把外扩改成与训练口径一致**：这样在**两个套系上都是净赚**（官方 9 张 +0.48、
  派生 30 张 +0.77），component F1 与 tp/fp/fn 回到旧权的水平（192/27/24），pin F1 还涨
  （官方 9 张 0.4226 → **0.4555**）。小外扩（1.0/0.10/1.5）在派生档更高（26.52）但官方 9 张掉到
  61.98 —— **9 张的样本太小**，定默认前要用大样本（≥80 张派生）复核。
  复现用的对照脚本：`tools/dev/_engchk.py`（同一条框走「语料口径（裸框）」与「stage2 口径（`pad_box`）」
  各读一遍，权重可分别指定），先跑它就能看出「模型漂移」还是「口径不同」。
- ⇒ 纪律：**微调权重的 A/B 必须走 `stage2.run`（运行时真实的框/角色/裁片），并按 case 拆开看**；
  「语料留出集准确率」**不能**当上线依据 —— 语料裁片取的是 `work\_f220_A\_assigned` 的槽位框，
  与 stage2 运行时喂给模型的裁片不是同一口径。要看单条裁片的读数差异用
  `tools/dev/_deswhy.py --dir <_assigned 目录>` 或直接比 `_assigned` 的槽位 `text`/`box_i`。
- 附带（同一轮的两个坑）：
  1. **位号重解 pass 不值**：`work\tools\dev\_desres.py`（空闲框补充 + 类型前缀 + 唯一键）在 20 张派生
     case 上只有 **+0.08**（27.9172 → 27.9971）；去掉唯一键/距离门的宽松版看着有 +0.48，其实是
     **重名塌缩**的假收益 —— `components` 是 dict，两个元件改成同一个键会**静默吞掉一个元件**。
     任何时候改元件键都必须保证键唯一。
  2. **读 case 名单要用 `utf-8-sig`**：PowerShell `Set-Content -Encoding utf8` 会写 BOM，
     第一个 token 被 `^[0-9A-Za-z_]+$` 过滤掉 ⇒ **静默少一张 case**（本轮 `cases=19` 就是这么来的）。

### 裁片口径 = 「紧框」还是「外扩」：交付权重偏向**外扩**，紧框语料微调只是**在管线不用的那种裁片上更好**（2026-10-11，`tools/dev/_padchk.py`）
- 探针 `work\tools\dev\_padchk.py`：取紧框语料 `work\official_text\manifest.jsonl` 的行（`bbox` = 紧框、
  `text` = GT），对同一行做两份裁片 —— 紧框、以及 `pad_box(2.0/0.0/3.0/0.20)`（= `stage2/text.py:402-403`
  对**非** `TIGHT_ROLES` 角色的口径）—— 再用两个权重各读一遍，比 exact / CER。
- 留出集必须按**被评权重**的训练切分来选（否则测的是记忆）：

  | 权重 | 紧框 exact | 外扩 exact | 差（外扩−紧框） |
  |---|---|---|---|
  | `work\ocr_mix\runs\best.pt`（现行交付 e57） | 63.49% | **78.84%** | **+15.35** |
  | `work\ocr_official_v1\best.pt`（紧框语料微调） | **83.90%** | 78.69% | **−5.21** |

  样本 671 行 = 官方 `1,5` + 「每 10 个四位 case 取一个」（正是 `ocr_official_v1` 的训练留出集，
  两个权重都没在这批行上训过；若换成官方 `1..10` 全集，`ocr_official_v1` 的 87% 行是它自己的训练集，
  数字会虚高到 87.8%）。复现：`python -X utf8 work\tools\dev\_padchk.py --val-cases "1,5" --val-every 10`。
- 三条结论：
  1. **交付权重是「外扩口径」的模型**：在管线真正喂的裁片上比紧框高 15 个点 ⇒ 管线默认外扩对它是
     **对齐**的（这也解释了 `mix+pad 62.7992 > mix+nopad 58.7400`）。按角色最夸张的是 value：
     紧框 33.12% → 外扩 76.25%（designator 72.86 → 78.00、pin_name 70.92 → 82.27）。
  2. **紧框语料微调那次「离线大胜」是口径假象**：`_offtext_eval.py` 报 66.05% → 87.44%，但在它自己的
     留出集上换成**管线口径**后是 **78.69% vs 78.84% = 平**（`pin_number` 还从 90.0% 掉到 75.0%）。
     所以 b926 那个 −4.45 不是「微调没用」，而是「微调只学好了管线不用紧框时的那种裁片」。
  3. 修法是**让语料对齐管线口径**，不是改管线外扩：`tools/dev/mk_official_textset.py` 现在默认按
     `pad_box` 写 `bbox`（`--no-pad` 退回紧框），每行同时保留 `bbox_tight`（原框，`verify_rows`
     的复读校验仍用它，保证「框↔读数」判据不变）；并对 `pin_name` / `pin_number` 复刻
     `stage2/text.py:133 tight_ink_box`（`--no-pipe-crops` 可关）—— 这两个角色在管线里是
     「先切掉彩色描边、再外扩」的（`stage2/text.py:404-415`，`pin_number` 的 `pad_frac_y` 取
     `PINNUM_PAD_FRAC_Y=0.0`）。产出目录 `work\official_text_pad`（只外扩）与
     `work\official_text_pipe`（外扩 + 引脚切描边，**与管线逐位同口径**）。
- 纪律：**评一个权重之前先问「它是在哪种裁片上训的」** —— 跨口径比准确率会得到两个方向相反、看着都
  「显著」的结论。语料里同时留 `bbox`（管线口径）与 `bbox_tight`（原框）两列，就是为了让这件事可查。

### 交付 CRNN 权重陈旧：`weights/textrec_best.pt` == `work/ocr_real`(e51)，而 `work/ocr_mix`(e57) 在**两把尺子**上都更好（2026-10-11）
- 事实链：
  1. `assets.py:29 CRNN_ENV="STAGE2_CKPT"`，按 `stage2/text.py:38` 的兜底与 `assets.py:39 REPO/weights/textrec_best.pt`
     解析交付权重；两者**都是** `work/ocr_real/runs/best.pt`（`epoch 51`、`val_cer 0.1871`）。
     `work/ocr_mix/runs/best.pt`（`epoch 57`、`val_cer 0.1698`）只是 `TRAINING.md:180` 里告诉你要
     `$env:STAGE2_CKPT` 指过去的东西 —— **没有任何一步真的把它用上**。
  2. 复核缓存产物的来源：`work\_full220\runs\stage2\_assigned\1.json` 的槽位文本与
     `ocr_real` 跑的 `work\_s2_real_pad\_assigned\1.json` **27/27 完全相同**，与 `ocr_mix` 跑的只同 12/27
     ⇒ **当前交付缓存（也是历次 A/B 用的那一份）是 ocr_real 产的**。
  3. 官方 9 张 stage2 端到端（同一份 stage1 缓存、同一套外扩 2.0/0.20/3.0、纯 CPU）：
     `ocr_real` **57.2569** → `ocr_mix` **62.7992**（+5.54，9 张里 8 张变好，只有 case 10 −3.28）。
  4. 「唯一算数的尺子」（`tools/diag/_numread.py`，引脚号绑定准确率，官方 9 张）：
     `ocr_real` **0.7090**（n=134）→ `ocr_mix` **0.7615**（n=130）。
  5. 干净子集（只取**没进过任何训练语料**的官方 case 1、5）加权均分：`ocr_real` 46.32 → `ocr_mix` 60.49。
- ⇒ 结论：**换 `weights/textrec_best.pt` 为 `ocr_mix` 是一个现成的、干净的交付增益**（两个尺子、两个套系同号）。
- ✅ **已执行（2026-10-11，用户 m38289 批准）**：
  1. `eda-pcb/weights/textrec_best.pt` ← `work/ocr_mix/runs/best.pt`（md5 由 `0CD23402952D` 变
     **`7559096C9979`**）；旧权重留在 `work/weights_backup/textrec_best.real_e51.pt`（与
     `work/ocr_real/runs/best.pt` 同 md5）。
  2. 兜底路径一起改：`assets.py:41 WORKSPACE/work/ocr_mix/runs/best.pt`（`ocr_real` 降为第二兜底）
     + `stage2/text.py:38` 的 ImportError 分支；`docs/KNOBS.md:126` 的 `STAGE2_CKPT` 默认值同步。
  3. 补测（都在换之前跑完）：
     * **在管线内、全 220 张、确定性**（`python -X utf8 work\tools\dev\_textrate.py --assigned <缓存> --cases-file work\_cases220.txt`）：
       exact ALL `ocr_real` **38.2%** → `ocr_mix` **40.9%**；10GT 71.7→**79.3**、10_GTcase 40.7→**49.6**、
       train200 37.2→**39.7**（三个套系同号；注意这是**整个 220 张**，不是小样本）。
     * **fusion 层 220 张、无 VLM**（`FUSION_LLM=0 FUSION_CROPFIX=0 FUSION_ITERATE=0`，
       `work\_cnofix_mix` vs `work\_cnofix_real`，`_ro220score.py --a … --b …`）：
       全部 220 **33.1950 vs 33.0042（+0.19）**；10GT +0.33、10_GTcase +1.10、train200 +0.14、
       官方 20 +0.71；逐张 A 好 88 / 差 77 / 平 55 ⇒ **方向一致但幅度小**（VLM 层会重读文字，
       把 CRNN 的一部分增益吃掉 ⇒ 交付口径（VLM 开）的 A/B 仍要补，见下）。
  4. ⏳ 待补：**交付口径（VLM 开）220 张 A/B** —— 必须在 ②b 训练跑完、GPU 空出来之后跑
     （两臂都要新跑，VLM 有抖动，不能拿旧产物当对照）。
- ⚠️ **同时记下的方法论坑**：域内微调那条线的离线收益**被泄漏污染了**。`mk_official_textset.py` 的语料是从
  **全部 220 张官方图**用 GT 文本框+GT 文本造的，而 `mk_textrec_manifest.py` 只按「`--val-cases 1,5` +
  每 10 个四位 case 取一个」划 val ⇒ 官方 2/3/4/7/8/9/10 与派生 0001–0080 里的绝大多数**都在微调训练集里**。
  所以「语料留出集 66%→87%」和官方 9 张上的 `ft` 读数**都不可用**；只在干净子集 {1,5} 上复算，
  `ft+nopad` 58.45 / `ft+p1y15` 58.01 都**低于** `ocr_mix` 的 60.49（引脚号尺子上 ft 反而更高：case1 1.0000 vs
  0.8889、case5 0.9474/1.0000 vs 0.9474 —— 样本只有 18~24 个，噪声大）。
  **教训：任何「用 GT 造语料再评自己」的实验，必须先按 case 切干净，再看数字**（切分要用**案例级**，
  不能用「同一张图的不同框」）。

### 「无 VLM 的融合层 A/B」要**三个**开关一起关（2026-10-11，比 455 行那条更细）
`FUSION_LLM=0` 只关哲学仲裁轮；实测**还有两处会真调 LM Studio**：

| 调用点 | 开关（默认） | 关掉后 |
|---|---|---|
| `fusion/run.py:265 LJ.judge` / `:285 AU.run` / `RS.run` | `FUSION_LLM`（`1`） | 关掉仲裁轮 |
| `fusion/cropfix.py:256 VC.ask` | `FUSION_CROPFIX`（`1`） | 关掉「裁切方向反馈」 |
| `fusion/run.py:330 IT.pin_fields` | `FUSION_ITERATE`（`1`） | 关掉引脚名定向回传 |

- 复现（别用 CPU 占用猜，直接看客户端）：`Get-NetTCPConnection -RemotePort 1234 -State Established`
  的条数 + `Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'"` 的 **CreationDate**
  是不是**本次运行开始之后**；`& "$env:USERPROFILE\.lmstudio\bin\lms.exe" ps` 能看到模型与 TTL，
  `lms unload --all` 立刻把显存还回去（本轮 llama-server 9.7 GB 与 ②b 训练同抢 16 GB 卡）。
- 代价对比（同一台机、220 张）：三开关全关 ⇒ **0.4 min / 220 张（~0.1 s 每张）、显存只剩训练**；
  只关 `FUSION_LLM` ⇒ 每张 30~60 s（真在等 VLM）⇒ **「慢」本身就是「偷偷调了 VLM」的症状**。
- 用途边界：三开关全关**偏离交付配置**，只能做「机制定位/权重替换」的对照；改默认值仍要按纪律跑
  VLM 开的真交付口径 A/B，并在报告里写明。
- 附带坑：`work\tools\diag\_ro220arm.py:26` 会 `os.chdir(eda-pcb)`，所以 `--cases-file` 传**相对路径**
  会在 chdir 之后解析失败（本轮 `FileNotFoundError: 'work\\_cases220.txt'`）——已改成相对**仓库根**解析；
  `--out` 一直是相对仓库根。
- ⚠️ **同时记下的方法论坑**：域内微调那条线的离线收益**被泄漏污染了**。`mk_official_textset.py` 的语料是从
  **全部 220 张官方图**用 GT 文本框+GT 文本造的，而 `mk_textrec_manifest.py` 只按「`--val-cases 1,5` +
  每 10 个四位 case 取一个」划 val ⇒ 官方 2/3/4/7/8/9/10 与派生 0001–0080 里的绝大多数**都在微调训练集里**。
  所以「语料留出集 66%→87%」和官方 9 张上的 `ft` 读数**都不可用**；只在干净子集 {1,5} 上复算，
  `ft+nopad` 58.45 / `ft+p1y15` 58.01 都**低于** `ocr_mix` 的 60.49（引脚号尺子上 ft 反而更高：case1 1.0000 vs
  0.8889、case5 0.9474/1.0000 vs 0.9474 —— 样本只有 18~24 个，噪声大）。
  **教训：任何「用 GT 造语料再评自己」的实验，必须先按 case 切干净，再看数字**（切分要用**案例级**，
  不能用「同一张图的不同框」）。

### 换 stage2 读数的 A/B 必须**重建下游模型缓存**（`stage3_vlm`），否则测的是「缓存失配」

- 事实：交付口径 A/B 一直用**固定**的 `FUSION_VLM_DIR=work\_full220\runs\stage3_vlm`（220 张），
  而那份缓存是 `vlm.enrich` 从**旧** stage2 臂（`work\_full220\runs\stage2`，即 `ocr_real`）生成的。
  ⇒ 每次换 CRNN 权重，都是「缓存已对齐的旧臂」比「缓存不对齐的新臂」：VLM 补强写的是
  **旧权重的读数**，新权重更好的读数被顶掉。
- 症状：**同一批改动出现唯一一个反向读数**。
  `ocr_pipe_v1` vs `ocr_mix`（同 20 张、同 `_assigned`）：VLM 开 **−0.4178**（46.9383 vs 47.3561，
  官方 10 张 −0.5365）；**VLM 关**（`FUSION_VLM_DIR` 指向空目录，`work\_novlm_ab.ps1`）
  **+1.5318**（45.9800 vs 44.4482，`train200` 派生 10 张 **+2.9949**、`10GT` +0.0686；
  逐张 A 好 11 / 差 9）；stage2 侧 +1.94 ⇒ 三个同向，只有一个反向。
- 规则：**任何改 stage2 读数的 A/B（权重 / 裁片 / 绑定额）**，要么重建下游缓存，要么把下游层关掉
  并写明「偏离交付配置」；**反向读数出现在装置不对称的地方，先怀疑装置，再怀疑结论**。
- 工具：`work\tools\dev\_mkcache.py --stage2 <臂目录> --out <缓存目录> --cases-file …`
  （包 `vlm.enrich.enrich_dir`；图源 `work\_submit_in220\<case>.png`），
  之后 `FUSION_VLM_DIR=<该臂缓存>` + `FUSION_ASSIGNED_DIR=<该臂 _assigned>` 再融合；
  三臂公平驱动见 `work\_fairab.ps1`。
- 交付路径本身没有这个问题：`submission\run.py -vlm` 写的是 `runs/stage3_vlm`，对**当前** stage2
  **现算** ⇒ 「比赛口径 = 每臂自己的缓存」。固定缓存复用的装置**不模拟交付**。

### `net_line`（权重 0.10，长期爆 0）

```powershell
python tools\diag\_linewhy.py     # 逐条看边的端点距离
python tools\diag\_lineweld.py    # 「共线且首尾相接」焊成整条直线值不值
python tools\diag\_nlfork.py      # ★ 逐 case 把 tp/fp/fn 拆到「网没配上 / 边数不对 / 端点超差」
python tools\diag\_nlfork2.py     # ★ 区分「双描边」与「过分割」两种机制（856 对 vs 236 片）
```

**★ 2026-10-08 根因定案：我们发的是笔画的「两条外沿」，GT 是一条中心线。**

- 病灶在 `pipeline\pskel.py:118` 的 `thin_stroke_mask`：
  `thin = (ink>0) & (dt <= max_half_width)`，而 `max_half_width` 默认 **1.6**。
  一根 **4px 宽**的导线横截面上 `dt = 1,2,2,1` ⇒ `dt <= 1.6` **只留下两条外沿、把芯挖掉**，
  掩膜变成**空壳**；Zhang-Suen（`pskel.py:31`）于是在空壳的两条壁上各给一条骨架 ——
  我们镜像出**两条平行边**（相差 2~3px）。
- 为什么这就是失分：`net_line_f1` 是**一对一派**（`metrics.py:566-578` 网内边匈牙利），
  每条 GT 边最多命中两条里的一条，**另一条必然是 fp**。官方 9 张：配对网内我们 **993** 条
  边 / GT **523**（1.90×），其中 **856 对**是「平行且两端都近 ≤4px」的双胞胎。
- 像素铁证：case 9 图 `(1292, 386..403)` 处竖线是**实心 4px 绿**（`GGGG`）—— 空壳是
  **我们掩膜造的**，不是图里画的。（`work\_wirepix.py` 一次性取证脚本。）
- 反面教训（同一个发现、但当时走错了路）：`_nldup.py` 早就在头注释里写了这句话，但它的
  离线臂 `par2/par3/par5/par3col` 是**在产物上事后把双描边并成中心线**，全部判负
  （`par3` 把 case 2 的导线打成 0）⇒ **不要在产物上合并，要在提取处让骨架回到中心线。**

三个旋钮（`stage1\wires.py`，2026-10-08 接线、**2026-10-09 改默认**；`close_thin` / `dilate_to`
是 `WireExtractor` 早就有、但构造时从没传进去的参数）：

| 旋钮 | 含义 |
|---|---|
| `STAGE1_WIRE_THIN_W` | 覆盖半宽阈值（**默认已改 2.6**，`wires.py:320`；1.6 = 历史值，会把 4px 描边掏空） |
| `STAGE1_WIRE_CLOSE_THIN` | 1 = 3×3 形态学闭运算把两条外沿之间 2px 的缝焊上（`pskel.py:401-404`，默认 0） |
| `STAGE1_WIRE_DILATE` | N>0 = 把「离细笔画 N px 以内的墨」补回来（`pskel.py:405-409`，N=2 可填满 4~5px 描边，默认 0） |

**官方 9 张横扫**（`fast.py --no-vlm --cases 1,2,3,4,5,7,8,9,10`，`STAGE1_WIRE_COLOUR=off`，
纯 CPU、0 次 LM Studio）：

| 臂 | 总分 | 元件 | 引脚 | netH | **导线** | 双描边对 | 网内我们的边 / GT |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 50.43 | 0.7682 | 0.3502 | 0.5044 | 0.0978 | 1212 | 1262 / 518 (2.44×) |
| `DILATE=1` | 50.98 | 0.7682 | 0.3606 | 0.5003 | 0.1407 | — | — |
| `DILATE=2` | 51.11 | 0.7682 | 0.3606 | 0.5026 | 0.1460 | **348** | **800 / 514 (1.56×)** |
| `DILATE=3` | 51.11 | 0.7682 | 0.3606 | 0.5026 | 0.1460 | — | — |
| `DILATE=2 + CLOSE=1` | 51.11 | 0.7682 | 0.3606 | 0.5026 | 0.1460 | 348 | 800 / 514 |
| `THIN_W=2.2` | 51.23 | 0.7682 | 0.3606 | 0.5086 | 0.1368 | 339 | 798 / 522 (1.53×) |
| **`THIN_W=2.6`** | **51.30** | 0.7682 | 0.3606 | 0.5078 | **0.1469** | — | — |
| `THIN_W=3.0` | 51.18 | 0.7682 | 0.3606 | 0.5053 | 0.1434 | — | — |
| `THIN_W=2.2 + D=2` | 51.11 | 0.7682 | 0.3606 | 0.5026 | 0.1460 | — | — |
| `DILATE=2 + EDGE_MERGE=1` | 50.64 | 0.7682 | 0.3606 | 0.5026 | **0.0987** | — | — |
| `chroma` 基线 | 50.41 | 0.7682 | 0.3502 | 0.5009 | 0.1076 | — | — |
| `chroma + THIN_W=2.2` | 51.29 | 0.7682 | 0.3595 | **0.5124** | 0.1324 | — | — |

结论与待办：

1. **`THIN_W=2.6`（或 `DILATE=2`）把导线 F1 从 0.0978 抬到 0.1469（+50%）、总分 +0.87**，
   引脚 +0.0104 是连带（`pin_f1` 按网配对计数）；`DILATE=3` 与 `DILATE=2+CLOSE` 不再加收益。
2. **`STAGE2_EDGE_MERGE=1` 在修正后的几何上依然判负**（netL 0.1460→0.0987）——旧的
   「共线焊接」结论在新基线上复现，别再试。
3. ⚠️ 上面这张表全是 `--no-vlm` + `off`/`chroma` 的**纯 CPU 口径**。`dilate_to` 的回填写作
   `near & ink`（`pskel.py:409`）而不是 `& cm`，色相掩膜开着时会**把紧邻导线的非导线墨也补回来**。
4. **真交付口径已验（2026-10-09）**：`fast.py` 不加 `--no-vlm`、`STAGE1_WIRE_COLOUR=hint` 默认，
   每臂各两次重复 —— 导线 F1 **0.1929 → 0.2317**（tp 152→141）、总分 66.33/66.52 → 66.79/66.60；
   **同一臂两次的 netL 分项与逐 case 导线 F1 逐位一致** ⇒ 增量确定性，总分里的 ±0.19 是 VLM 噪声。
   逐 case **8 涨 1 跌**（case 3 0.1560→0.1308）。表在 `RESULTS.md` §4.8(b)。
   ⇒ **默认值已改为 2.6**（这一条推翻了下面第 5 条）。
5. ~~旋钮默认值一律没改~~（2026-10-08 的状态；2026-10-09 已按第 4 条改默认）。

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

- **相邻信息尺子** `tools\diag\adjmeas.py`（`--assigned/_assigned 目录 --cases-file --radius 60 --poolscan`）：
  量「位号的邻座有没有信号」。默认模式按同前缀的**行/列 run**分桶（run 内 GT 位号 2.5%，
  邻座插值能补的只有 **0.3%** ⇒ 生成式相邻信息 ≈ 0）；`--poolscan` 换成
  「正确读法去哪了」：`chosen_ok 39.7%` / `near_not_chosen 12.2%`（★ 被**邻座的号**顶掉，
  典型 `R1→R2`、`C151→C152`）/ `not_in_pool 37.4%` / `no_slot 10.0%`。
  结论：相邻信息是**归属（一对一）**问题，不是生成先验问题。读数见 `RESULTS.md` §2.9。

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
| ~~**官方档的引脚键形状**（“10GT 全是纯数字”）~~ → **改用下一行** | ~~`tools/diag/_gtpin10.py --show 4`~~（结论写反过，见 `_pinconv.py`） |
| **缺号三分账**（策略 / OCR / 检测） | `tools/diag/_pinledger.py --run --split --show` |
| **按位置对账引脚号与名的读法**（完全对 / 读错 / 空 / 截断） | `tools/diag/_ocrstat.py --run --split --n --show` |
| **读法全对的写实上界**（上帝视角改写再打分） | `tools/diag/_ocroracle.py --out <产物> --cases …` |
| **装配开关的离线单变量 A/B**（不占 GPU、不重跑 OCR） | `tools/diag/_gateab.py --run <产物> [--all]` → `tools/eval/stage2_eval.py work\_gate_ab\{on,off}` |
| **引脚邻域肉眼核对**（十字位置、号在框外上方、名在框内） | `tools/diag/_pincropgt.py` |
| **GT 引脚键的约定普查**（物理号式 vs 1..N 计数器式，按 split / 只算 IC） | `tools/diag/_pinconv.py` |
| **逐槽位三方对照**（GT / 我们 / VLM 的号与名 + 点对齐） | `tools/diag/_pincase.py 0105 U1`（官方加 `--official`） |
| **VLM 通道的臂 A/B**（引脚替换/合并/各种开关，产出补强文档） | `tools/diag/_vlmab2.py --in <stage3_vlm> --out <新目录> --arms pinrepl --workers 3` |
| **两份 `stage2_eval --json` 逐张对比**（含按 split 汇总） | `tools/diag/_cmp9.py A.json B.json` |

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
