# 工具索引

> 由 `python tools/dev/gen_toolindex.py --write` **自动生成**，不要手改。

`tools/` 是本项目「怎么量」的全部家当。每个回答一个具体问题的脚本都在这里，
配套的用法说明见 `docs/DIAGNOSTICS.md`。
共 **630** 个脚本。


<a id="data"></a>

## `tools/data/`（24 个脚本）

数据集构建与体检（合成数据、标签审计、官方口径评测）。

| 脚本 | 行 | 说明 |
|---|---:|---|
| `alipan_download` | 215 | 通过阿里云盘匿名分享接口，列出/下载 PCBnet Dataset300 全量数据。 |
| `axis_profile` | 66 | Decisive y-axis test: correlate the image ink row/col profile with the |
| `build_pcb_trainset` | 323 | Build Y:\workspace\EDA\PCB训练集 -- a drop-in YOLO dataset for the team's component detector. |
| `build_trainset_v2` | 225 | Assemble `训练集_v2`: real sheets + 3000 synthetic sheets, with an honest held-out test. |
| `build_trainset_v3` | 276 | Assemble `训练集_v3`: the real sheets + a long-tail-preserving slice of the synthetic ones, |
| `check_axis` | 60 | Decide the JSON y-axis direction by comparing the ink bbox of the PNG with the |
| `check_leak` | 95 | Two leak questions behind "why is mAP50 this high after 1 epoch?" |
| `check_synth_json` | 172 | Validate a synthetic set against the corpus schema and the project's own scorer. |
| `corpus_stats` | 132 | Measure the statistics a synthetic sheet must reproduce, from the 220 real cases. |
| `corpus_stats2` | 99 | Second probe: the structural details the generator must match exactly. |
| `fix_boxes` | 160 | 把 GT 的固定尺寸图章扩成符号真实画出来的范围。 |
| `gen_synth_batch` | 248 | Generate a large synthetic schematic set (default 3000) into `合成原理图数据集`. |
| `gt_look` | 114 | 查一个 case 的 GT：某个类的每个框长什么样、框里到底有没有东西。 |
| `inspect_data` | 75 | Inspect the schematic dataset structure: components / pins / nets. |
| `label_stats` | 52 | 统计 YOLO 数据集中每个 split 的每类实例数。 |
| `make_gallery` | 308 | 把这次排查过程中产生的所有问题图整理成一个自包含的 HTML 图集。 |
| `make_sheet_profiles` | 79 | Regenerate `eda-pcb/synth/profiles.py` from the real 213-sheet corpus. |
| `md5_10gt_check` | 146 | Is the 10GT hold-out really a hold-out? |
| `ocr_manifest` | 138 | 把 Dataset300 的 CVJsonStd 变成 textrec 能吃的真实语料清单。 |
| `ocr_real_build` | 192 | 从官方 220 张 GT 里把「真实文字的图 + 真值串」挖出来（弱监督标注）。 |
| `official_eval` | 222 | 用官方评测口径（schematic_parser/metrics.py）评估 YOLO，并给出归因。 |
| `survey_all` | 83 | Global survey of the whole dataset: file layout, type vocabulary, pin key patterns. |
| `survey_images` | 49 | Check image properties across the dataset: size, mode, colour palette, uniqueness. |
| `verify_trainset` | 127 | Verify Y:\workspace\EDA\PCB训练集 before it is handed to the team's trainer. |

<a id="debug"></a>

## `tools/debug/`（3 个脚本）

单样例对账：把官方评测器的打分过程逐条摊开。

| 脚本 | 行 | 说明 |
|---|---:|---|
| `dbg` | 274 | 单样例调试器（IDE debug 模式）。 |
| `ledger` | 544 | 逐条对账：把官方 `metrics.py` 的打分过程**一条一条摊开**。 |
| `report` | 577 | 把 `ledger.build_ledger` 的结果渲染成**单文件可交互 HTML**。 |

<a id="dev"></a>

## `tools/dev/`（30 个脚本）

开发辅助（本目录的生成器）。

| 脚本 | 行 | 说明 |
|---|---:|---|
| `_compdump` | 70 | 把单个 case 的 GT 元件 / 预测元件 / keymap 并排打印，用来定位「为什么普通位号配不上」。 |
| `_compgap` | 107 | component 失分溯源：GT 里哪些元件一个都没配上（连坐标都没配上），按「伪元件 ∅NNN / 普通位号」分开。 |
| `_cropchk` | 95 | **裁片口径核对**：同一条语料裁片，分别用两个权重读一遍。 |
| `_ctlib` | 131 | 容器式**库路径解析**回归测试（用户 m42543「各种灾难性现场」）。 |
| `_desacc` | 128 | **位号挂载体检**：把 GT 元件与产物元件按 JSON 坐标 IoU 配对，统计 |
| `_desceil` | 203 | **位号重解的天花板探针**：对每个「没挂上位号 / 挂了错位号」的 GT 元件，看它附近 |
| `_desceiling220` | 16 | 把 `eda-pcb/tools/diag/_des_ceiling.py` 指到 220 张的 stage1 产物上跑（原工具写死 work\stage1_all14）。 |
| `_desres` | 259 | **位号重解 pass（原型）**：把「没挂上位号」的元件，用附近**空闲**文字框里读出的位号补上。 |
| `_deswhy` | 196 | **位号为什么错**：把「认不出来（字形/字体）」和「挂到别的元件上」分开量。 |
| `_engchk` | 101 | **口径对照**：同一条框，走「语料口径（裸框裁片）」与「stage2 运行口径（`pad_box` 外扩）」 |
| `_fantest` | 141 | case 级扇出（`-case-workers`）的回归测试：坏图 / 看门狗 / 关掉 / 自动。 |
| `_fontfit` | 174 | 官方渲染用的是哪一款字体？—— 拿真实裁片（GT 文本已知）去套候选字体的字形。 |
| `_fontsheet` | 80 | 把真实裁片和候选字体的渲染并排画出来，肉眼看哪个像（`_fontfit.py` 的数字靠这个兜底）。 |
| `_hardtest` | 200 | 灾难现场加固的**故障注入**测试（用户 m42543）。 |
| `_libmerge` | 51 | 把官方两份元件库合成本地开发用的第三份（**只做测量，不改交付**）。 |
| `_offtext_eval` | 109 | 在**官方渲染图的域内语料**上评一个 CRNN 权重：只走生产路径（`textrec.infer.Recognizer`）。 |
| `_padchk` | 179 | 2x2 探针：{交付权重, 域内微调权重} x {紧框裁片, 管线外扩裁片}。 |
| `_pindiag` | 191 | **引脚失分归因**：官方口径 `pin_f1` 只统计 GT 里 type 为 box 的元件 |
| `_pinnum` | 190 | **引脚号能不能靠几何推**：官方口径的引脚配对键是 `(元件键, 引脚号)` |
| `_sheet_from_manifest` | 62 | 从已落盘的文字语料 manifest 里挑一批出来拼对照图（核标签真假 / 看还没修好的）。 |
| `_textrate` | 175 | 量「CRNN 在官方渲染图上读文字的错读率」——按套（10GT / 10_GTcase / train200）× role 分开。 |
| `audit_labels` | 277 | Training-label quality audit for a YOLO dataset — "错误的标号" triage. |
| `build_trainset_v4` | 599 | Assemble `work/trainset_v4` = `trainset_v3` (optionally label-repaired) + `新补充训练集`. |
| `gen_knobs` | 97 | 扫描源码里所有环境开关，生成 `docs/KNOBS.md`。 |
| `gen_toolindex` | 96 | 生成 `docs/TOOLS.md` —— `tools/` 下每个脚本的索引（含首行 docstring）。 |
| `gpu_gate` | 363 | GPU 串行闸门：**同一时刻只允许一个自己起的吃卡任务**，且起之前先确认显存够。 |
| `mix_crnn_manifest` | 180 | 把「真实语料」和「合成语料里挑出来的那几类」混成一份 CRNN 训练清单。 |
| `mk_fonttext` | 280 | **官方字体合成语料**：用与官方渲染器同款/同类的字体直接合成文字裁片，补上合成语料的字体域缺口。 |
| `mk_official_textset` | 523 | 从**官方公开数据**（10GT + 200_train_cases）造一份「域内」文字识别训练集。 |
| `mk_textrec_manifest` | 152 | 把「官方渲染图上的域内文字语料」（`mk_official_textset.py` 的产物）转成 `textrec` 能直接 |

<a id="diag"></a>

## `tools/diag/`（488 个脚本）

**诊断与归因**：把某一项失分拆到可行动的最小单元，再逐一证伪对策。

| 脚本 | 行 | 说明 |
|---|---:|---|
| `_asgflds` | 27 |  |
| `_asgkeys` | 53 | `_assigned/<case>.json` 的结构探针：有没有「没被任何元件认领的文字」。 |
| `_asgpins` | 22 |  |
| `_asgstruct` | 27 |  |
| `_asgstruct2` | 17 |  |
| `_axdbg` | 31 | 调试 axis 全局解码：打印每条轴的候选窗口与分。 |
| `_axisdbg` | 36 | 诊断 axis 两层解码在 case 1 三条真实轴上的行为（读数逐条抄自 _assigned/1.json）。 |
| `_axisdiff` | 152 | 轴层的**决策级**差分：轴开/关时，最终定下来的值到底哪里不一样、谁对谁错。 |
| `_axiseval` | 171 | 量「轴 + 全局铺号」这一层到底能救回多少位号 —— 纯 CPU，不碰流水线。 |
| `_axisfix` | 63 | 把 `fusion/axis.py::_selftest` 里的真实夹具拿出来单独看：新长度律下 |
| `_axistest` | 109 | 判定 GT 框的 y 方向：对每个 case 分别按「y 向上（翻转）」与「y 向下（不翻转）」 |
| `_axiswire` | 53 | 查「轴层到底有没有出声」：照 `fusion/run.py:run_case` 的方式建池，再直调 `AX.inject`。 |
| `_basecmp` | 15 |  |
| `_basemean` | 67 | 汇总 work/_fast_base/*.json 的均分与四个分项（fast.py --save 存的 per-case 列表）。 |
| `_bbgate` | 107 | 量「差一点点没过 IoU 0.5 闸门」的元件有多少。 |
| `_bgstat` | 94 | 背景估计体检：解释 bglab 在 0037-Datasheet 上把导线通道打崩的原因。 |
| `_box_pin_diag` | 110 | `box`（IC）引脚为什么只 recall 0.47 —— 一次性诊断。 |
| `_boxcrop` | 108 | 把 GT 某个元件的框（含四周留白）裁出来放大，看**位号到底画没画在图上**。 |
| `_boxdesig` | 289 | 位号归因：GT 的大 IC（`box`）为什么**检出了却键对不上**。 |
| `_boxgap` | 122 | 量「围绕元件 box 的重扫」到底缺什么：按 cid 统计位号/引脚槽位与取景框。 |
| `_boxgeom` | 162 | 检出框 vs 真值框的几何差：宽高比、中心偏移。看清「框到底差在哪」。 |
| `_boxkey` | 146 | GT 的 `box`（大 IC）为什么键（位号）对不上：**没检出** 还是 **检出但位号错**？ |
| `_boxmiss` | 97 | 量一个事：**GT 的 `box`（大 IC）有多少是我们整颗没输出的。** |
| `_boxquality` | 123 | `cvparts.textbox` 的框到底有多准？—— 决定「修框」还是「重写切分」。 |
| `_boxread` | 171 | GT box(IC) 元件的位号到底有没有印在图上、印在环带哪个位置。 |
| `_boxspace` | 89 | 确认 `_assigned` 里各种 box 的坐标空间（图像空间 vs JSON 空间）。 |
| `_boxwhy` | 97 | box(IC) 位号为什么挂不上 —— 官方口径下唯一的 pinF1 来源。 |
| `_c1` | 83 | 看 case `1`（10GT）的 `U1` 到底为什么没配上键：GT 的 IC 框 vs 我们预测里的同一个框。 |
| `_c151row` | 190 | Part 1 probe: the C151..C156 row in 10GT case 1, end to end. |
| `_c1src` | 89 | `_assigned/<case>.json` 的**出处归因**：每个字段到底是哪一层定的。 |
| `_caserows` | 54 | 一次性：把某个 case 的元件对账里某一批键打出来。 |
| `_cfrep` | 29 |  |
| `_chainmeas` | 223 | `_chainmeas.py` —— 「导线 ↔ 元件 ↔ 引脚」链式单点失效的**量化**（用户 m42009 第二问）。 |
| `_charfit` | 150 | 从 38.8 万条真实「框 + 文本」标注里**拟合逐字符的宽度**，替换 axis.py 里那个 |
| `_cmp.cmp` | 16 |  |
| `_cmp9` | 78 | 逐张对比两个 stage2_eval --json 产物（用法：python work\_cmp9.py <A.json> <B.json>）。 |
| `_cmpab` | 93 | 多臂基线对比：`work\_fast_base\<arm>.json`（`fast.py --save <arm>` 的产物）。 |
| `_cmpcfg` | 13 |  |
| `_comp_diag` | 51 | 连通分量到底有多大？在「墨迹掩膜」上量，还是「骨架」上量？ |
| `_compextra` | 185 | 把 `component.extra`（我们多出来的元件 = 纯 fp）按**类别 / 尺寸 / 与已匹配框的关系**分桶。 |
| `_comptext` | 127 | 逐对拆 `component` 的失分：**text_mismatch 到底是 IC 的 `Name` 还是 r/c 的 `value`**。 |
| `_conn_check` | 96 | 定位「消融里的连通 R」和「stage1_conn_eval 里的连通 R」为什么差 2~9 倍。 |
| `_conn_comp` | 97 | 连通分量的结构诊断：我的段图到底是不是「一坨」。 |
| `_conn_reconcile` | 96 | 对账：为什么 60 张消融说 all12 的连通 F1 是 0.5421，而端到端评测器说 0.1457？ |
| `_cpntlib` | 202 | 官方元件库 `赛题资料/cpntLibrary` 到底能救多少 **box(IC) 引脚名**。 |
| `_crnneval` | 138 | CRNN 验证集快速评测——训练用的那份 manifest（work\ocr_real\manifest.jsonl， |
| `_crop` | 12 |  |
| `_cropfixtest` | 82 | 裁切方向反馈的现场试验：对可疑槽位问方向、按方向做外扩阶梯、看能不能读回完整串。 |
| `_cropvote` | 232 | 同一检出框多裁片投票：量「TTA 投票」能不能超过**任何**单一裁片。 |
| `_cvjson_texts` | 38 | 看 Dataset300 的 CVJsonStd 里 texts 是否可信（精确框 + 内容）。 |
| `_des_ceiling` | 175 | 位号到底读不读得出来？—— 量这条链的**天花板**，把「认不出」和「归错元件」分开。 |
| `_des_probe` | 131 | 位号挂载探针：直接看某颗元件的**邻居文字框**和它们的 5 种读法。 |
| `_des_why` | 159 | 位号挂载归因：把「GT 名字可读但我们没挂上位号」的元件拆成三类。 |
| `_des_why2` | 284 | 「位号候选没产生」到底属于哪一种？—— 决定性归因，跑全量。 |
| `_descover` | 81 | 图上到底印了哪些位号——把 stage1 的**全部**文字框过一次 CRNN（designator 角色）， |
| `_desigabsent` | 163 | 决定性一问：那些「我们读不出位号」的 GT box，**图上到底有没有位号**？ |
| `_desigassoc` | 207 | 位号「关联错」还是「识别错」—— 决定「全局重归属」这条路的**天花板**。 |
| `_desigbox` | 242 | 位号新机制试验：不再让 VLM 在整块区域里自己找位号，而是**把 stage1 已经切出来的 |
| `_desigcand` | 126 | 「GT 真位号读不出」的那 579 颗 IC：**正确位号到底在不在候选表里？** |
| `_desigcand2` | 236 | 「正确位号到底在不在候选表里」—— 决定「修位号选择」这条路的**天花板**。 |
| `_desigdetect` | 192 | 位号缺口的**最上游**归因：那个文字框**到底被检出来没有**。 |
| `_desigexp` | 250 | IC 位号读取：prompt / 裁片策略的多种写法对照。 |
| `_desigmatch` | 209 | 没配上的 GT IC：正确位号离候选表**有多远**（决定「模糊/家族修复」值不值得做）。 |
| `_desigorder` | 204 | GT 的位号是不是**按画的顺序编的**？—— 决定「猜位号」能不能打赢「读位号」。 |
| `_desigprobe` | 75 | 对单张图里**每一颗 IC/box 元件**直接问一次位号，和管线当时拿到的读数并排看。 |
| `_desigread` | 126 | 直接审问位号读取链路：对一个 `box` 元件，`DESIG_PROMPT` 到底回了什么？ |
| `_desigring` | 93 | 位号的可达性上限：GT box 元件的位号，在它周围 R 像素内的 stage1 文字框里， |
| `_desigsrc` | 165 | 未对上的 GT `box` 元件，按**失效模式 × 画图工具**分类： |
| `_desigsweep` | 183 | 位号（器件键）识别精度的**直接**基准 + 裁剪外扩网格扫描。 |
| `_desigwhy` | 135 | GT 的 box(IC) 元件为什么还是配不上键 —— 把 636 个 miss 逐条归因。 |
| `_det_why` | 188 | 元件检测（第一阶段）的漏检 / 多检归因（用户 m07122）。 |
| `_diag_chain` | 62 | 找到包含那条斜线的 polyline（RDP 之前的 chain），看跳变发生在哪一步。 |
| `_diag_chain2` | 63 | 在 chain 里精确定位斜向跳变，并回溯是哪个结点的配对造成的。 |
| `_diag_check` | 96 | 查 0101 里那些「莫名其妙的大尺度斜线」。 |
| `_diag_segs` | 36 | 列出 stage1 产物里某个 case 的长斜线段，用来确认大尺度斜线是否消失。 |
| `_diag_src` | 54 | 沿那条 45 度斜线逐级采样：ink -> thin -> skel，看它在哪一级出现。 |
| `_docdbg` | 55 | 临时调试 doc.apply。 |
| `_docdiff` | 81 | 对比两个融合产物目录里同一张图的交付文档，说清「到底改了什么」。 |
| `_docshape` | 45 | 看最终文档里 `pins` / `nets` 的真实字段形状 —— 融合层要能改它们，得先知道改什么。 |
| `_docshape2` | 21 |  |
| `_dumpstruct` | 32 | 把 GT / pred 的 components 与 pins 两个 section 的真实形状打出来。 |
| `_dupscore` | 134 | 量化「重复检测」对本方提交文档的影响 —— 纯 CPU，不动流水线。 |
| `_edgeclr` | 59 |  |
| `_envdump` | 21 |  |
| `_envflagprobe` | 41 | 验证「假值词表统一」：把每个手写短词表换成 `envflag.env_on` 之后， |
| `_errbudget` | 86 | 按 GT 类型拆开 component 的错误预算。 |
| `_evidence_sheet` | 66 | 把 owner 7 条判断对应的证据图拼成一张总览（每格带标题）。 |
| `_fastdiff` | 15 |  |
| `_fbceil` | 129 | 给「兜底键」那 2601 个元件量一个上限：它们的 GT 位号，在图上**附近**读得出来吗？ |
| `_fbview` | 108 | 把「GT 有真名、我们没配上」的元件连同它的 GT 键一起画出来，人眼核对： |
| `_fix1` | 18 |  |
| `_flags` | 82 | 开关普查：把 eda-pcb 里所有「环境开关」连默认值一起列出来，判定默认开/关。 |
| `_flowaudit` | 56 | 按 id 比对 workflow.html 与备份（而不是按下标），并扫描残留的旧数字。 |
| `_flowcheck` | 50 | Sanity-check work/flow/workflow.html after patching: JSON validity, link/port integrity, |
| `_flowchk4` | 18 |  |
| `_flowdiff` | 67 | 对比 `work\flow\workflow.html*` 各自内嵌的 graph（看重新构建有没有弄丢内容）。 |
| `_flowdump` | 13 |  |
| `_flowdump2` | 12 |  |
| `_flowfix` | 47 | 第三批小修：把外扩参数从 ev_ocr 收敛到 ③ 节点（避免两处并存讲同一件事）， |
| `_flowlink` | 18 | 打印 link schema、groups 与关键节点坐标。 |
| `_flowlist` | 23 | 列出 workflow.html 里的节点 id / title / 进度行。 |
| `_flowmeta` | 18 |  |
| `_flownode` | 21 | 打印若干节点的完整 JSON 与一条 link，用来照抄 schema。 |
| `_flownodes` | 16 |  |
| `_flownodes2` | 21 |  |
| `_flownotes` | 14 |  |
| `_flowpatch` | 105 | Patch work/flow/workflow.html's graphdata block with this round's measured findings. |
| `_flowpatch2` | 181 | 第二批 workflow.html 更新（用户要求「完全更新」）： |
| `_flowpatch3` | 100 | 第三批补丁：把 fusion 层已落地的节点从「⚠ 未落地」改成实际情况。 |
| `_flowpatch4` | 89 | 第四批补丁：把融合层的 **220 张口径**实测与「已接进生产链」写进 workflow.html。 |
| `_flowpatch5` | 439 | 第五批 workflow.html 补丁：把 24 个节点的状态、file 徽章、算法与落地事实全部更新。 |
| `_flowpatch6` | 127 | 第六批 workflow.html 补丁：把「定点档」实测结论写进 cropfix 节点，并撤掉 boxscan 里那条错判。 |
| `_flowports` | 14 |  |
| `_fncrops` | 125 | 把 FN/FP 的具体框从原图裁出来拼成证据图，用于肉眼确认「标注错 vs 模型错」。 |
| `_frame_ab` | 114 | A/B：把「占据整张图的大连通分量」当成图纸边框/标题栏表格线删掉，值不值？ |
| `_free_end_diag` | 183 | 自由端的来源分解 —— 决定导线图该怎么修。 |
| `_frontrecall` | 179 | **前端召回**：在完全不看 LLM 的前提下，stage1 产物里到底「有没有」GT 要的东西。 |
| `_fusdiff` | 77 | 逐张对比两个 `stage2_eval.py --json` 结果，找出融合层掉分/加分的 case。 |
| `_fusiondiff` | 54 | 对比 VLM 补强文档与融合层产物：哪些键被改了、改成什么。 |
| `_fuslanded` | 27 |  |
| `_fz1` | 56 | fusion 层第一轮实测：拿真实产物建池 + 跑先验，看位号候选变成什么样。 |
| `_fzenrich` | 61 | 在指定的一批 case 上跑 VLM 补强，产出干净的 `<case>.json`（供融合层消费）。 |
| `_fzrep` | 32 | 一页看融合层的报告：逐 case 打印 cropfix / judge / audit / rescan / 分数。 |
| `_gateab` | 86 | `STAGE2_BOX_PIN_NEED_NAME` 的**离线单变量 A/B**（纯 CPU，不需要 GPU / OCR）。 |
| `_gndname` | 29 |  |
| `_gndnet` | 135 | _gndnet.py —— GND 互证先验（用户 m12359）到底有没有头寸？ |
| `_gndpot` | 49 |  |
| `_gpuprobe` | 34 | 判断「GPU 是不是被 VLM 占住导致 CUDA 变慢」：量一次小矩阵乘 + 一次 YOLO 推理的墙钟。 |
| `_gtdup` | 82 | GT 标注里有没有「同一颗器件被标了两遍」—— 重复检测是不是被**训练出来的**。 |
| `_gtdup2` | 65 | 重复框对到底是「同一个框写了两遍」还是「两个挨得很近的符号」。 |
| `_gtdup3` | 91 | 同类近邻 GT 框（corner_dev<=20px）到底是「一颗符号被标了两遍」还是「两颗真符号」。 |
| `_gtfaith` | 870 | GT 可信度审计：val77 上把每个 FN / FP 重新归类，回答「是模型错还是标注错」。 |
| `_gtmatch` | 19 |  |
| `_gtoverlay` | 91 | 把某个 case 的 GT 框全部画回原图（可缩放、可圈定区域），用来判断 |
| `_gtpeek` | 18 |  |
| `_gtpin10` | 109 | 官方 10GT 的 **IC(box) 引脚键**到底是什么形状 —— 决定「引脚号对不上」能不能赢。 |
| `_gtpinname_stat` | 26 |  |
| `_gtpinplace` | 36 |  |
| `_gtshape` | 38 | 看 GT 数据里**一个元件**到底有哪些字段（尤其有没有位号文字框）。 |
| `_gtypes` | 43 | 看各 split 的 GT `components[].type` 取值分布，以及 `canon_type` 之后落在 IC_TYPES 的比例。 |
| `_huetime` | 87 | 量 `stage1.wires._hint_wire_hue` 的真实代价：整图 vs 缩到 1024，以及有多少张图会触发它。 |
| `_icbox` | 159 | For every GT `box` (IC) component, decide WHY it failed the key match. |
| `_ickey` | 142 | IC 的**钥匙**为什么配不上 —— 逐条拆开。 |
| `_ickeyprobe` | 87 | 逐颗 IC 直接调 `enrich.read_designator`，看位号通道**到底读出了什么**。 |
| `_icmatch` | 144 | Use the OFFICIAL matcher to see how GT IC boxes are matched. |
| `_icmisswhy` | 127 | 没配上的 GT IC 到底是哪一类 —— 决定「补 IC」这条路值不值得走。 |
| `_icread` | 147 | 量「GT 的 IC 元件，位号在图上到底读不读得出来」。 |
| `_icread2` | 185 | `_icread.py` 的 220 张版：把「GT 那颗 IC 的位号，图上到底读不读得出来」 |
| `_icsource` | 153 | Bucket the A1/A2/A3/A4 IC-box accounting of `_icmatch.py` by SOURCE TYPE. |
| `_icview` | 102 | 把「GT 是 IC、键没配上」的元件在原图上画出来，人眼核对**图上到底有没有印那个位号**。 |
| `_imgdbg` | 23 | 临时探针：看 stage1 doc 的 image.path 能不能读出来。 |
| `_imgdbg2` | 24 | 临时探针 2：看 from_file 之后 pool.bgr 到底有没有。 |
| `_inboxsel` | 187 | _inboxsel.py -- 被框涂白吃掉的导线边，能不能用「独立证据」挑回来？ |
| `_inkband` | 89 | 量「墨迹高度 g / 标注框高度 H」的稳定性。 |
| `_inkcheck` | 114 | v3 vs v4：框是不是真的套在墨迹上（recenter 修复的定量体检）。 |
| `_inkdiag` | 143 | 归一的逐条诊断：把 val 拆成「归一几乎没动框」和「归一大幅改了框」两组， |
| `_inktest` | 71 | `ink_recenter` 的三条硬测试（合成图，不碰真实数据）。 |
| `_inkvis` | 117 | 对照：同一个框，左＝现行裁剪（框 + PAD_X/Y_FRAC），右＝墨迹归一后的裁剪。 |
| `_jrep` | 95 | 把 `_fusion_report.json` 里 LLM 轮（`rounds[].judge`）的**开关与计数**整行打出来。 |
| `_jsdump` | 9 |  |
| `_judgedbg` | 79 | LLM 裁决的取景/提示词 A/B：同一个槽位，看模型到底吃到了什么、答出了什么。 |
| `_judgeprobe` | 46 | 跑一次融合层并把它内部的 log 全打出来（看 LLM 轮到底做了什么）。 |
| `_kdegs` | 73 | 量出 pskel._trace 里「每个节点的分支数 k」的分布 —— 决定 k^2 双循环代价。 |
| `_keydbg` | 62 | 看融合层把哪些位号改掉了，以及每个改动槽位的候选与来源。 |
| `_keydiff` | 54 | 对比两份文档的元件键集合，看融合层到底改了什么。 |
| `_keyfix` | 154 | _keyfix.py —— 「检测对了、类型对了、中心 <=20px，只是 key 对不上」的那批元件， |
| `_keyoracle` | 139 | ★ 关键实验：**位号（元件键）错误到底值多少分？** |
| `_keyorigin` | 44 | `U580` 是哪来的：把 r36 的最终文档键 与 `_assigned` 的 cid/位号 摆在一起。 |
| `_keyprint` | 142 | Is the GT key of a mis-named IC box *readable anywhere in the image*? |
| `_keytype` | 131 | 按**元件类型**量「键配上了没有」——决定 component / pin / net 三块权重的天花板。 |
| `_label_audit` | 245 | 训练集标签体检：回答用户 m07249 的七条质疑。 |
| `_labelaudit` | 104 | 标注审计：当前训练集（trainset_v3）里的标签**本身**有多可疑。 |
| `_labelaudit_gt` | 458 | 全类别 GT 错标体检 —— 扫描 + 逐框指标 + CSV。 |
| `_labelaudit_gt_report` | 363 | GT 错标体检 —— 汇总与报告生成。 |
| `_labelaudit_gt_visual` | 141 | 肉眼复核工具：把可疑 GT 框渲染成 ASCII 字符网格（或放大裁图）。 |
| `_leadmiss` | 187 | **两脚元件引脚候选审计**：为什么 r/c/l/d/led 的引脚总是挂不上网（`_netdiff.py` 的 |
| `_leadrule` | 293 | **远场引线规则**（`STAGE1_PIN_LEAD` 的设计验证，只读、不跑流水线）。 |
| `_leakpair` | 79 | 对照 train 侧的 `synth_<id>` 与 val 侧的 `200_train_cases-<id>__full` 的标注直方图。 |
| `_lencal` | 142 | 长度律的**域校准**：`fusion/charwidth.py` 是从**标注框**（manifest）拟合的， |
| `_libcov` | 195 | 元器件库（官方 `cpntLibrary` / BoxIC 库）对 GT 的覆盖率体检（`--lib <file>` 可换库）。 |
| `_libnum` | 240 | 官方元器件库当「引脚号来源」的机会 —— 只针对**名读到了、号没有**的引脚。 |
| `_libopp` | 187 | 量化「官方元器件库能给我方当前产物补多少分」的机会。 |
| `_libpins_cpu` | 107 | 交付级等价 A/B（**纯 CPU**，不碰 GPU / LM Studio）；`--lib` 换库文件（换默认库就是用它量的）。 |
| `_libsim` | 294 | **不改管线**的模拟：把「官方库给出的引脚号」直接补进交付 JSON，再打分。 |
| `_lineprobe` | 163 | 为什么 NetLineF1 只有 0.019？ |
| `_lineweld` | 170 | `net_line`（权重 0.10）爆 0 的对策实验：把「共线且首尾相接」的导线片段焊成整条直线。 |
| `_linewhy` | 125 | `net_line` 为什么爆 0 —— 逐条看边的端点距离。 |
| `_llmab` | 76 | 阶段三 A/B：`mode=off` vs `mode=llm`，同一份 stage2 产物、同一批 case。 |
| `_llmprobe` | 95 | 探 LM Studio 的能力：关思维链、JSON 模式、批量多槽位。 |
| `_llmprobe2` | 115 | 第二批探测：批量槽位、以及各模型的耗时对比。 |
| `_llmprobe3` | 116 | 第三批探测：批量大小 vs 耗时、以及并发能力。 |
| `_metricsfix` | 16 |  |
| `_misspair` | 98 | missing 的 GT 元件 → 没被配上的预测元件，到底是什么关系。 |
| `_misswhy` | 131 | 有名字的 GT 元件里，我们没配上键的那些到底为什么丢 —— 分四类。 |
| `_mkpinidx` | 74 | 为已有的 stage2 产物补出 <case>.pinindex.json（纯 CPU，不跑任何模型）。 |
| `_nameacc` | 76 | IC 框的 `Name`（器件型号）到底读对没有 —— `component_text_ok` 对 IC_TYPES 比的就是它， |
| `_nameboxes` | 90 | 把 `_namefix.py` 找出的「名错」引脚连同它在 `_assigned` 里的来源文字框一起打出来。 |
| `_namecat` | 194 | 引脚名错配的**成因分类**：截断 / 扩读 / 字符混淆 / 完全无关。 |
| `_namefix` | 204 | 引脚名（`pinname`）的字符级混淆普查 —— `pin_f1` 的 tp 要求 pinname **逐字**相等。 |
| `_namerescue` | 556 | **引脚名救援**可行性：IC 引脚「号读对了、名字没读到」的那些桩，名字到底还在不在图上？ |
| `_namerule` | 120 | 引脚名救援：在 _namerescue.py 的明细上离线比较「发不发、发哪个」的规则。 |
| `_netconv` | 179 | 量「净 hyperGraph」的两种口径，看差多少分。 |
| `_netcover` | 200 | 网络分（`net_hypergraph_f1`，权重 **0.35** —— 四项里最重）到底卡在哪。 |
| `_netdiff` | 125 | **网络失分账本**：把 `net_hypergraph_f1`（权重 0.35，micro 口径）的 fn/fp 逐条拆到 |
| `_netdump` | 32 |  |
| `_netfrag` | 69 | 把某个 case 的「GT 网 <-> 我们网」的碎裂结构打印出来。 |
| `_netlab` | 62 | 看我们的产物里网络标签到底有没有被用上。 |
| `_netloss` | 142 | 网络（hypergraph）失分的**三层分解** —— 把 netH 的损失拆成可行动的三块。 |
| `_netmerge` | 191 | 网络合并的离线实验：不动管线，只改预测文档的 `nets`，看 `net_hypergraph` / `net_line` 涨不涨。 |
| `_netoracle` | 69 | `net_hypergraph` 的 tp 上限分解：我们的引脚引用里有多少「内容正确」的？ |
| `_netrail` | 231 | 按「电源/地轨引脚名」合并网络的离线实验（`netH` 权重 0.35）。 |
| `_netside` | 90 | 把一张图上「GT 的网络」与「我们的网络」并排打出来，用来定位拓扑失分。 |
| `_netsplit` | 111 | 量 netH（权重 0.35）的结构性杠杆：把预测的每张网**拆成单引脚网**会怎样。 |
| `_netweld` | 185 | `netH` 的 `D_在网里但配错`（10GT 上 96 个 / 0.229）对策实验：**按几何把网络焊起来**。 |
| `_netwhy` | 120 | 网络（`netH`，权重 0.35）的丢分逐条拆开：GT 的网络引脚里，我们缺的那些**为什么缺**。 |
| `_newbbox` | 49 | 新补充训练集 bbox 约定终审：把候选约定放到「引脚对称性」下比较。 |
| `_nldiag` | 145 | _nldiag.py —— `net_line_f1` 为什么只有 0.06？ |
| `_nldup` | 325 | _nldup.py —— `net_line` 的真凶：我们把**笔画的两条边**都当成导线输出。 |
| `_nledge` | 97 | _nledge.py —— 把某一对配对 net 的 GT 边/我方边**逐条并排打出来**，肉眼看差在哪。 |
| `_nlfork` | 155 | net_line 失分分解：把 tp/fp/fn 拆成「网没配上」「网配上了但边数不对」「边数对但端点超差」。 |
| `_nlfork2` | 137 | net_line 结构归因：把「我们边太多」拆成 ①双描边（平行近重复）②过分割（GT 长边被切成几段）。 |
| `_nlpair` | 83 | _nlpair.py —— `net_line` 是被**边**卡住，还是被**网络配对**卡住？ |
| `_nobom` | 9 |  |
| `_nodeclust` | 71 | 量 pskel 的结点聚簇尺寸 —— 验证「长斜线 = 跨大聚簇的隐式直连」假设。 |
| `_nothink` | 129 | 同一个视觉任务上试各种「关思维链」的写法，看哪个能真的把 thinking 压掉。 |
| `_numcrop` | 117 | 把 stage1 检出的「引脚号码框」裁出来堆成接触表，用来判断 CRNN 读错是 |
| `_numprobe` | 200 | 引脚**号**读不出来的原因：是「字没检出」还是「检出没认领」？ |
| `_numread` | 340 | 引脚号（pin_number）识别精度的直接基准 + **按框宽分桶** + 裁剪外扩网格扫描。 |
| `_numsweep` | 146 | 引脚号码识别：真值取自 GT 的 `pin_<n>`（按 y 配对，权威），扫「裁剪几何」看 exact。 |
| `_numtruth` | 90 | 把 stage1 的「引脚号码框」按 y 位置对到 GT 的 `pin_<n>` 上，得到**权威真值**， |
| `_occpol` | 139 | _occpol.py -- 「我们自己的框该不该在提线前涂白」的口径实验（用户 m42009 的互相等待）。 |
| `_ocr_boxcmp` | 120 | CRNN 到底行不行？把「在 GT 的紧框上读」与「在检测框上读」分开量。 |
| `_ocr_boxtune` | 126 | 检出框该怎么修，才能让 CRNN 读对？ |
| `_ocr_conf` | 141 | 量「CRNN 置信度能不能把**文字框**与**符号误检框**分开」。 |
| `_ocr_pad` | 194 | 检出框系统性偏小，怎么修？—— 定长外扩 vs 按高度比例外扩 vs 生长到墨迹边界。 |
| `_ocr_recon` | 99 | 侦察：文字定位器 API + 官方 GT 的文字真值锚点。 |
| `_ocr_verify` | 92 | 检查弱监督标注的质量：随机抽 manifest 里的行，裁图 + 显示「声称的真值」。 |
| `_ocrerr` | 278 | CRNN 真实错误典例采集器（只读，不改任何产物）。 |
| `_ocroracle` | 183 | ★★ **文字识别一共值多少分？**（写实上界，不改管线、不占显存） |
| `_ocrstat` | 136 | **文字识别到底抽不抽风**：把 GT IC 引脚的号/名，与「位置上最近的那个我方槽位」逐条对账。 |
| `_overcheck` | 56 | 规则 (5)「穿越式微越过」的自检。 |
| `_ownasgn` | 361 | 量「一块文字框被两颗元件同时当成位号」的规模，并离线试 `own.assign` 全局一对一的收益。 |
| `_owndiff` | 60 | 对比同一张图、两个 `STAGE1_OWN` 设定下的 stage1 产物，看**归属闸到底并掉了谁**。 |
| `_owneval` | 119 | 同类相邻框：把「判据阈值」和「并了会不会误并真符号」放在同一张表上。 |
| `_padab` | 105 | 对照图：同一个框、同一个模型，在**两条预处理路径**下的读数。 |
| `_padsweep` | 173 | 外扩参数扫描：在**同一批检出框**上比不同 (pad_frac, pad_min, 纵横比) 的 OCR exact。 |
| `_paramcheck` | 38 | 核对 workflow.html 各节点 params 栏里的常量是否与当前代码一致。 |
| `_paramcheck2` | 28 | 第二批参数核对：detect.py 的 conf/imgsz、metrics.py 的口径、enrich.py 的引脚名开关。 |
| `_patch10` | 93 | 把「第十批」的结论按字面替换写进 _wfbullets.py（work\flow\ 与 eda-pcb\tools\diag\ 两份）。 |
| `_patch10b` | 62 | 第十批·补：把「闸门 A/B 判负」这条 bullet 插到 _wfbullets.py 的 ev_pin 里。 |
| `_patch8` | 88 | 第八批：把本轮实测结论写进 `_wfbullets.py`（徽标已改，bullet 要跟上）。 |
| `_patch9` | 122 | 把「第九批」的实测结论写进两份 `_wfbullets.py`（字面替换，可重复执行）。 |
| `_peek220` | 33 | 看一眼 work/_fast_base/*.json 的顶层结构，判断是「行列表」还是别的形状。 |
| `_peek_stage1` | 23 |  |
| `_percase` | 73 | 每张图的「位号可读性」分布 —— 用来判断哪些 case 压根没标注位号。 |
| `_phibox` | 20 |  |
| `_phipins` | 18 |  |
| `_phisym` | 32 |  |
| `_pidxcheck` | 24 |  |
| `_pin200` | 422 | 200 训练集 **引脚** 为什么大面积缺失 —— 看清 GT/预测里 `pins` 的真实结构与总量。 |
| `_pin_stamp` | 190 | 引脚「图章点位」可行性核算。 |
| `_pin_why` | 174 | 引脚失配归因：把 `pin_f1` 的损失拆成「键丢 / 号丢 / 名错 / 点错」。 |
| `_pinab` | 110 | 引脚窗口 A/B：`pin_pitch_frac=0`（旧，垂直半宽 = cross）vs `0.45`（新，按引脚间距收窄）。 |
| `_pinaxis` | 120 | **引脚轴向账本**：非 IC（c/r/l/d/…) 元件的两个引脚应该长在**符号主轴**两端 |
| `_pinband` | 82 | 量「某颗 IC 框外的引脚号文字框」到底有没有被检出、又有没有被读成号。 |
| `_pinbox` | 65 | 核对 boxscan 的第 ③ 条：引脚取景框到底是不是 None。 |
| `_pinboxdiff` | 134 | 逐 case 的 IC 引脚对账：GT vs 我方 `final_results/<case>_Pred.json`。 |
| `_pincase` | 139 | 逐槽位对照表：**GT / 我们（stage2 装配）/ VLM 引脚通道** 三方的 (号, 名)。 |
| `_pinchain` | 113 | **引脚在哪一段被丢掉**：GT → stage1 产物 → 最终提交文档，逐跳数引脚。 |
| `_pincheck` | 66 | 自检：新 `crop_dir` 在 `cross_perp=None` 时**逐位等于**旧实现（A/B 对照臂的正确性前提）， |
| `_pincolmatch` | 84 | 逐元件对账：stage1 产物的 `pincol` 枚举点 vs GT box 元件的引脚点。 |
| `_pincolprobe` | 134 | 探针：把「框外号码窗」里到底有什么量化出来。 |
| `_pincolread` | 98 | `stage1/pincol.py` 的号码窗直接送进 CRNN，读出来到底对不对？ |
| `_pincoltest` | 167 | 验证 `stage1.pincol` 的引脚列枚举：拿真实 case 的 stage1 产物 + 原图跑一遍， |
| `_pinconv` | 118 | GT 引脚键的**约定**普查：键是「物理引脚号」还是「1..N 计数器」？ |
| `_pincov` | 143 | Pin coverage on *matched* ICs: is the loss in the POINT or in the NUMBER? |
| `_pincover` | 122 | 量「引脚几何」的可用性：GT box 元件的引脚、我们预测里的引脚、stage1 检出的引脚点。 |
| `_pincrop` | 132 | 把 GT 引脚点周围的邻域裁出来，看「一个引脚」在图上到底长什么样。 |
| `_pincropab` | 85 | 引脚取景「旧(cross 对称) vs 新(按引脚间距收窄垂直半宽)」对照图 —— 纯 CPU，不叫 LLM。 |
| `_pincropgt` | 92 | 把指定 case 的若干 GT 引脚邻域裁成一张拼图（带十字与标签），用来看「图上到底画了什么」。 |
| `_pindbg2` | 114 | `pin_f1`（权重 0.25，现在只有 0.0474）到底卡在哪：把「键已匹配上的 box 元件」单独拎出来， |
| `_pindiff` | 45 |  |
| `_pindist` | 121 | 引脚**位置**与**名字**的失分归因：GT 的 `point` 到底指哪里？我们给的又是什么？ |
| `_pinerr` | 180 | Where exactly do our pin POINTS miss the GT pin points? |
| `_pinextra` | 119 | `pin_f1` 的失配按**元件**归因：extra / missing / name_mismatch / position_mismatch 各是谁贡献的。 |
| `_pinfunnel` | 218 | 引脚 tp 到底丢在哪一关：**键 → 号 → 名 → 位置** 四关漏斗。 |
| `_pingrid` | 286 | 逐引脚问 VLM：把「一个引脚」的邻域裁片拼成网格，一次调用读回「编号:名称」。 |
| `_pinjunk` | 105 | 量「IC 的预测引脚里有多少是垃圾」—— 用来判断 `pin_f1` 的 fp 有多少是白送的。 |
| `_pinkey` | 16 |  |
| `_pinkey2` | 60 | stage2 文档里 `pins` 是按 cid 还是按位号存的？以及 `assigned['comps'][cid]['pins']` 的规模。 |
| `_pinlabel` | 130 | GT 的 IC 引脚名**在图上到底有没有被我们检出文字**，以及它落在元件框内还是框外。 |
| `_pinlands` | 22 |  |
| `_pinlat` | 151 | GT 的 IC 引脚排布有多「规则」？—— 决定「按库补缺失引脚」这条路值不值得走。 |
| `_pinledger` | 266 | **引脚失分账本**：把 `pin_f1` 的 fn/fp 逐条拆到「缺号 / 多号 / 名字错 / 位置错」， |
| `_pinloss` | 221 | 引脚失分分解：把 220 张的 `pin.missing`（8335）拆成 |
| `_pinmap` | 53 |  |
| `_pinmatch1` | 58 | 逐脚对账：我们的某个元件 vs GT —— 号对不对、名对不对。只读。 |
| `_pinmiss` | 281 | 把「GT 有、我方没有」的 IC 引脚按**成因**分桶（取代 2026-10 之前的占位版）。 |
| `_pinname` | 153 | Why does `pin.name_mismatch` fire?  (`p.get('pinname') != g.get('pinname')`) |
| `_pinname_stat` | 23 |  |
| `_pinnamepair` | 117 | **引脚名字账本**：官方档 IC 引脚的 GT `pinname` vs 我方 `_assigned` 槽位里读到的 |
| `_pinoff` | 122 | 引脚**位置**误差普查：同号引脚 GT 点 vs 我们的点到底差多少、往哪个方向差。 |
| `_pinoracle` | 129 | 引脚分的**结构上界**：如果「我们产出的引脚」能一一对上 GT 的引脚，能拿多少？ |
| `_pinpos` | 115 | IC 引脚「点距」误差体检：PinF1 里 tp 要求 pinname 相等 **且** 点距 ≤ pin_tol(5 px)。 |
| `_pinprobe` | 153 | PinF1 为什么只有 0.039？（权重 25，是四个指标里最大的窟窿） |
| `_pinprune` | 140 | 引脚剪枝的离线实验：不动管线，只删预测里的引脚，再按官方口径重打分。 |
| `_pinrotnum` | 136 | case 6/7/10 竖直脚的「号」与「名」在 4 朝向下的读数，并对上 GT 参考。 |
| `_pinrotprobe` | 50 | 直接调 stage2.pinrot.fix()（跳过整条管线，省掉每轮几分钟），把每个候选读串打出来。 |
| `_pinrotview` | 37 | 看 `_pinrotprobe.py` 存下的逐帧读串：按 (cid, pin, field) 归组打印 0°/cw/ccw 三帧。 |
| `_pinshift` | 128 | 量我方 box 引脚点相对 GT 的偏移（按元件边的方位分组），看有没有可用的系统性修正。 |
| `_pinslot` | 52 | 看 `_assigned/<case>.json` 里**每个引脚槽位**到底带了什么（尤其：名字那段文字的框在不在）。 |
| `_pinsnap` | 256 | 引脚第④关（位置 ≤ `cfg.pin_tol` = 5.0 px）到底能不能靠**纯几何**救回来。 |
| `_pinsweep` | 187 | 引脚窗口扫描：**在 GT 的引脚点上**读号/名，直接量「窗口几何」这一层的能力上限。 |
| `_pintext` | 151 | **GT 的 `point` 是「框边接点」还是「引脚名/号那行文字的中心」？** |
| `_pintip` | 210 | _pintip.py — 引脚点几何体检：**tip vs base** 与 **邻脚错位** 两个假设的纯 CPU 判据。 |
| `_pinwhy` | 95 | **`pin`（权重 0.25）为什么爆 0** —— 在快测集上按官方判定口径逐条拆。 |
| `_pinwin` | 111 | 不依赖 LLM 的几何判据：引脚窗口里**混进了几条别的引脚行**。 |
| `_pinwin_u` | 104 | 三臂几何对照：窗口里混进几条**别的引脚行**（不依赖 LLM）。 |
| `_pinzero` | 137 | `pin.missing` 的来源分解：是「整颗 IC 没配上」还是「配上了但引脚没有」。 |
| `_plausible` | 131 | `plausible` 节点（21 节点目标架构里那一格）的第一版落地：**不需要 LLM 的结构性矛盾体检**。 |
| `_pncheck` | 195 | 核查引脚名救援发出去的名字到底对不对（对比 GT），并导出可用于离线挑门限的特征。 |
| `_pnull` | 121 | 兜底键元件「名下明明有文字框，却没读出位号」——把那些框的真实 CRNN 读法打出来。 |
| `_prdbg` | 66 | 临时调试：看 priors.window_for 在 case 1 那组读数上算出什么。 |
| `_probe0077` | 13 |  |
| `_probe0077b` | 15 |  |
| `_probe0077c` | 17 |  |
| `_probe0095` | 42 | 0095 为什么全军覆没？ |
| `_probe2` | 50 | 0095：把 GT 框和新 textseg 框并排打出来。 |
| `_probe3` | 89 | 某个真值文字框在 textseg 里的命运：属于哪个墨迹分量、为什么被丢。 |
| `_probe_mods` | 40 | Quick module reconnaissance + workflow node geometry dump. |
| `_probe_score` | 11 |  |
| `_prof1` | 68 | 给单张图的 stage1 做 cProfile，找出 total_s 之外的「黑洞」。 |
| `_prof_wires` | 33 | 对最慢的 case 做 cProfile，定位 pskel 到底慢在哪。用法：python _prof_wires.py 0107 |
| `_prpeek` | 11 |  |
| `_prunepol` | 159 | _prunepol.py -- 判别「框内 GT 导线边到底死在哪一道过滤上」。 |
| `_psympin` | 12 |  |
| `_pwhy` | 137 | 兜底键（pNNN）的 2191 个元件到底缺什么——逐条查 `_assigned`。 |
| `_pwrfilter` | 195 | 电源节点「怎么放」的稳健性实验：把电源符号写进净图有多少种放法，各种放法在 |
| `_recover` | 23 |  |
| `_region` | 39 | 渲染任意图的任意区域，用于肉眼判定图形到底是什么。 |
| `_repsum` | 17 |  |
| `_rescore` | 160 | 把已存档的 A/B 臂**用新口径重打一遍**，看有没有结论会翻。 |
| `_rlook` | 123 | 看 GT 里 r / l 到底长什么样：在原始分辨率上裁 GT 框，按类型拼接触表。 |
| `_ro220arm` | 87 | 220 张融合层 A/B 的**单进程**驱动器。 |
| `_ro220score` | 129 | 把两个「220 张融合臂」目录按官方口径打分，并按 suite 拆分（官方 20 / 派生 200）。 |
| `_rotdiff` | 168 | 对比两种旋转裁片口径（定位 `pinrot.fix()` 与 `_c6num.py` 探针读数不一致的原因）。 |
| `_rotread` | 127 | **旋转文字体检**：竖直（上下边）引脚的引脚名/号在图上常是**旋转 90°** 的， |
| `_row1` | 92 | 把 case 1 电容排（GT `C155`..`C164`）整排打出来： |
| `_rowfull` | 63 | 把某个 case 的**全部**位号候选框按「前缀 + y 带」排成轴打印出来， |
| `_run_recovered` | 277 |  |
| `_rzoom` | 86 | 宽视野看某个 GT 元件的上下文（含值/位号文字），用于判断符号到底是什么。 |
| `_s1pins` | 11 |  |
| `_s1split` | 90 | 在 stage1 产物上「就地」重做 T 形分叉切段，并同步重建 pins / nets。 |
| `_s1sym` | 19 |  |
| `_s1text` | 96 | stage1 文档里有哪些文字框？以及「desig_slots 空的元件」附近有没有**没被认领**的文字？ |
| `_schema` | 53 | 打印 GT 与预测 JSON 的结构，供后续 dedup 实验用。 |
| `_schemadump` | 70 | 把官方 GT 与我们的预测各打印一小段，确认 schema 完全对齐。 |
| `_score1` | 50 | 给**单个 case** 的预测文档打分（`fast.py --score-only` 只认 `_fast_out`，这张表要单点）。 |
| `_score220` | 116 | 把任意产物目录 (`<out>/final_results/*_Pred.json`) 打成 `_220_*` 那种形状的分数文件， |
| `_scorelist` | 9 |  |
| `_seg_miss` | 99 | 新 textseg 漏掉了哪些真值文字框？（best IoU < 阈值 的真值框） |
| `_segap` | 99 | 量「碎片网到别的线段」的缺口，用来判断并网容差该怎么定。 |
| `_segdump` | 31 |  |
| `_shapechk` | 56 | 形状闸落地前后的键对照（只读）。 |
| `_shapekeys` | 52 | 形状闸诊断：0001 的 comp.*.key 候选表 + solve 结果，闸开/闸关各一遍。 |
| `_shapekeys2` | 47 | 形状闸诊断 v2：把 comp.c0003.key / comp.c0048.key 的**全部候选**连证据打全。 |
| `_shortchk` | 159 | **自短接体检**：某颗两脚元件（r/c/l/d/led/fuse/crystal_2pin）的**两个端脚出现在同 |
| `_skeleton` | 11 |  |
| `_slotcount` | 40 | 数一数 stage2 产物里到底有多少个真歧义槽位（决定 LLM 预算）。 |
| `_slotdbg` | 46 | 把一个位号槽位的原始证据全部摊开：desig_slots 原文、stage1 的文字框、本体框。 |
| `_slotdiff` | 60 | 用真实的 stage3.build 量「真歧义槽位」数，并统计 MARGIN 敏感性。 |
| `_slotwhy` | 106 | 追一张图里「位号被改成错号」的现场：把该槽位的全部候选（值/分/来源）打出来。 |
| `_split220` | 71 | 把两个 stage2_eval --json（220 张）按 suite 拆分汇总。 |
| `_splitaudit` | 87 | split 审计：val 的 77 张到底是不是 77 张**独立**的图？train/val 有没有同源泄漏？ |
| `_st1_nets` | 48 | stage1 的 nets / wires 结构摸底（第二阶段组装要用）。 |
| `_st1_probe` | 72 | stage1 产物 schema 摸底 + 官方 GT 的 Name/位号约定。 |
| `_stage220` | 152 | 把一次 `submission\run.py` 全量跑的**关键阶段**逐个抽出来看（用户 m31144 的要求）。 |
| `_stages` | 205 | 全量 220 张的**阶段漏斗**分析 —— 不跑管线，只读已经打好的分。 |
| `_stale` | 12 |  |
| `_stampchk` | 114 | **贴章元件引脚槽位审计**：`stage2/assemble.py:stamp_pins` 把「stage1 的 4 个 |
| `_stampshape` | 152 | 图章形状勘察：官方 GT 的每类 bbox 到底是「正方形」还是「长方形」。 |
| `_stubfill` | 139 | Could geometric *pin stubs* fill the pins that have no wire attached? |
| `_stubmode` | 221 | `STAGE1_PIN_STUB` 各模式对 **stage1 引脚点集**的影响（按侧对齐量 P/R vs GT）。 |
| `_stubreach` | 155 | 引脚桩「远场判据」(`STAGE1_PINC_STUB_REACH`) 的离线取证。 |
| `_subst` | 48 | 按「字面子串」替换文件内容（避开 PowerShell 引号地狱）。 |
| `_suite_diag` | 60 | 按套看某一臂的 per-case 诊断分布（pin/component 的 missing/extra/mismatch 计数）。 |
| `_swapceil` | 243 | 位号**归属**可修性测量（GT 锚定版）：`_adjmeas --poolscan` 的 78 个 |
| `_swapmeas` | 353 | 一对一归属的**天花板**测量：把「元件 ↔ 位号实例」从独立最近邻改成邻域内双射， |
| `_symcmp` | 12 |  |
| `_symdbg` | 13 |  |
| `_symdbg2` | 19 |  |
| `_symdbg3` | 25 |  |
| `_symshare` | 33 |  |
| `_text_geom` | 151 | 量清「文字框 ↔ 元件框」的几何关系。 |
| `_textmis_why` | 123 | 把 `text_mismatch`（元件框对上了、type 也对，但 Name/value 不同）拆成可行动的几类。 |
| `_textmiss` | 96 | 配对上、类型对、bbox 对，但**文字不对**的那些，到底是什么类型、错在哪。 |
| `_texts_ab` | 27 |  |
| `_textseg_common` | 176 | 文字框切分的严格对照：所有臂都在**同一批真值框**上比。 |
| `_textseg_eval` | 164 | 旧 `cvparts.textbox` vs 新 `cvparts.textseg`：检测质量（IoU）+ OCR exact。 |
| `_textseg_group` | 132 | 文字行分组参数扫描。**关键指标是「原样裁（不 pad）的 exact」** —— 真值紧框原样读是 |
| `_textseg_sweep` | 163 | 扫描 `cvparts.textseg` 的关键开关：墨迹阈值模式 / 字形短边下限 / 最小面积。 |
| `_tile_look` | 55 | 把某个切片的标签画出来，人眼核对 (用户 m07249「是不是训练集本身就有问题」)。 |
| `_tipprobe` | 120 | 量「引脚列的号画在离框边多远」以及 `TIP_PAD` 能救回多少。 |
| `_tmp_pc` | 22 |  |
| `_tmp_pinkey` | 45 |  |
| `_toollist` | 21 |  |
| `_trace_jump` | 173 | 复刻 pskel._trace 的并链循环并加日志：找出哪一次 append 产生了远距离跳变。 |
| `_trace_prof` | 206 | 把 pskel._trace 逐段计时，定位真正的瓶颈。用法：python _trace_prof.py 0107 |
| `_u1dump` | 15 |  |
| `_u1probe` | 73 | 探针：case 0001 里 GT 的 U1 对应我们哪颗元件、那颗元件的位号候选表长什么样。 |
| `_val_ab` | 38 |  |
| `_val_rule` | 122 | 候选「输出规范」规则的离线 A/B —— 直接跑官方 metrics。 |
| `_val_why` | 158 | 官方的 `component.text_mismatch` 到底是什么？—— 按 GT 类型拆开「文字错」。 |
| `_valab` | 124 | `valbase` vs `valonly` 的完整对账：在**键匹配上的 r/c** 上，把 (GT 值, 管线值, VLM 值) |
| `_valbatch` | 142 | **请求合并（batch 节点）**的隔离 A/B：值通道「一格一次」vs「N 格拼一张图」。 |
| `_valbeam` | 157 | r/c `value` 到底能不能靠「后处理/词表」救：把 CRNN 每个字的**前 3 名**打出来。 |
| `_valdiff` | 130 | 对比两版 stage2 预测的 component.value 改动：每一处改动是变好还是变坏。 |
| `_valhead` | 102 | 把交付产物的**失分拆到「哪一类、哪一种错」**上——只读已有产物，不跑管线。 |
| `_valiou` | 82 | 按 IoU 配对（不看键）比较两版预测的 component.value 正确率。 |
| `_valnone` | 105 | 「GT 有值、我们写 None」的 351 条：到底是**没读到**还是**读到了被丢**？ |
| `_valpol` | 83 | 离线比较「VLM 值覆盖」的几种采纳策略：不重新调 VLM，只改 v18 的 value 再打分。 |
| `_valprobe` | 95 | 把某张图某个 GT 元件附近的 stage1 文字框全部列出来，并当场 OCR， |
| `_valread` | 138 | 「GT 有值 / 我们 None」的那 477 条，正确的值串到底**读没读出来**？ |
| `_visionprobe` | 127 | 探 LM Studio 的视觉能力：逐个模型试 OpenAI 风格的 image_url 部件。 |
| `_vizprobe` | 68 | 可视化前的探针：确认 GT 图 / 预测图 / stage1 图像尺寸三者一致，且 PIL 可用。 |
| `_vlmab2` | 128 | 隔离 A/B：对同一份 stage2 预测跑 VLM 补强的几种开关组合，直接打分对比。 |
| `_vlmdelta` | 81 | 量 VLM 节点到底改了什么：逐 case 对比 runs/stage2 与 runs/stage3_vlm 的产物。 |
| `_vlmdesig` | 153 | 量 VLM 读「元器件位号」（IC 外框外的 U1 / U5 / IC2 / J3 …）的准确率。 |
| `_vlmdoc` | 56 | 看一份 VLM 补强后的文档结构，并与 stage2 原文逐字段比较。 |
| `_vlmgrid` | 173 | VLM 网格批量读文字：把 N 个裁片排成带编号的网格，一次调用读回全部。 |
| `_vlmname` | 149 | VLM 读 IC 的**器件型号**（GT 的 `Name` 字段）—— 用来中和 box 那 276 个 text_mismatch。 |
| `_vlmnamewin` | 173 | 把 pincol 的**名字窗**（`in_win`）/ 号码窗（`out_win`）单独送 VLM，看它答得出来几个。 |
| `_vlmpin` | 190 | VLM 读 IC 引脚号/引脚名 —— 现在管线上「从来没从图上读」的那一项。 |
| `_vlmpin_dbg` | 75 | 单颗 IC 的 VLM 引脚读取调试：看裁片尺寸、思维链、finish_reason。 |
| `_vlmpins_apply` | 269 | 用 VLM 给 stage2 的预测补上两件从没读对过的东西：**IC 位号** 与 **IC 引脚号/名**。 |
| `_vlmprobe` | 41 | 探一次：这个端点上的模型到底能不能读图（用 vlm.client 的同一条通道）。 |
| `_vlmprov` | 22 |  |
| `_vlmread` | 249 | VLM（LM Studio 视觉）读文字 vs CRNN 的对照实验。 |
| `_vlmval` | 187 | VLM 读元件「值」（0.1uF / 10K / 15pF …）的可行性 —— 目前 `component.text_mismatch` |
| `_vlmval2` | 181 | VLM 读元件「值」——**只裁 stage2 已分配的那个 value 文字框**（加 pad），而不是整块元件区域。 |
| `_vlmval3` | 166 | 只量**键已经匹配上**的 r/c 元件（也就是真正会被 `component_text_ok` 比较的那批） |
| `_weld_selftest` | 110 | 方向性桥接（stage1.labels._weld_unions / wire_groups 规则 3）的合成自检。 |
| `_weldprobe` | 151 | 量「方向性桥接（规则 3 `weld`）」被哪一道闸门卡住。 |
| `_wfbuild` | 533 | 重建 workflow.html：要点清单 + 按数据流分带布局 + CSS/JS 补丁（阅读面板/分层 tidy）。 |
| `_wfbullets` | 418 | workflow.html 的第二轮内容：每个节点的「要点清单」（正面可见）。 |
| `_wfdump` | 19 |  |
| `_wfextract` | 19 |  |
| `_wfhead` | 179 | 改写 workflow.html 内嵌 graphdata 的 `name` 与 `note` 开头（第十批）。 |
| `_wfkind` | 93 | 改 `workflow.html` 里节点的 `kind`（徽标），用 `json` 模块安全改写内嵌 graphdata。 |
| `_wfmd` | 20 |  |
| `_wfnew` | 76 | 把 workflow.html 里 kind='new'（★ 要新写）的节点的 note + bullets 全文倒出来， |
| `_wfnodes` | 38 | 看 workflow.html 里指定节点的 file / note / params（用法：python _wfnodes.py [id ...]）。 |
| `_wfpeek` | 20 |  |
| `_wfsize` | 12 |  |
| `_wfupdate` | 335 | 把本会话（2026-10-07）的改动与实测写进 work/flow/workflow.html 的 graphdata。 |
| `_widebox` | 93 | 把「宽框引脚号」的裁片摊开看（含 35px 邻域），用来判「框里到底有几个字形」。 |
| `_wire_ab` | 128 | 导线提取的 A/B：无遮挡 / 遮挡元件 / 遮挡+边框与笔画过滤。 |
| `_wirecand` | 260 | _wirecand.py —— 端点精度的三个最小候选，官方 9 张离线重算（不动管线/不改产物）。 |
| `_wirecount` | 54 | stage1 提出来的线 vs 最终进网络的边：差在哪一步？ |
| `_wirecover` | 91 | _wirecover.py —— 我们的一条 GT 边被几条预测边覆盖？预测边比 GT 边短多少？ |
| `_wireedge` | 200 | `_wireedge.py` —— 导线掩膜的**彩色边缘证据通道** A/B（用户 m40034 的第一项）。 |
| `_wireeval19` | 49 | _wireeval19.py —— 用官方评测器口径对派生 19 张汇总（stage2_eval 的 --cases 按 stem 过滤，这里自己配）。 |
| `_wirefix` | 99 |  |
| `_wirefunnel` | 125 | _wirefunnel.py —— 逐 case 漏斗：stage1 线段 → stage1 网 → stage2 → fusion → 输出 net_line。 |
| `_wiregeo` | 84 | _wiregeo.py —— 几何失配到底长什么样（抽样打点 + 长度分布）。 |
| `_wirejson` | 30 | _wirejson.py —— 打印预测/GT 的 nets 段原样，确认 edges 里点的形状。 |
| `_wirekey` | 21 | _wirekey.py —— 页面关键数字在位自检。 |
| `_wiremeet` | 116 | _wiremeet.py —— 候选 (a)/(b) 端点吸附到**支撑直线交点**：穷举参数看有没有正收益。 |
| `_wiremiss` | 172 | _wiremiss.py —— 逐 case 追「边到底在哪一步消失」+ 6 张爆 0 拆解。 |
| `_wirenew` | 95 | 找出「把 `WIRE_TOL` 从 1.5 抬到 N 之后**新产生**的段并」是哪几对。 |
| `_wirepage` | 361 | _wirepage.py —— 生成独立的「导线 workflow.html」单文件页面（不依赖任何外部资源）。 |
| `_wirepagecheck` | 56 | _wirepagecheck.py —— 静态自检生成的导线页（不依赖浏览器）。 |
| `_wirepairs` | 75 | _wirepairs.py —— 「如果网全部配对正确，导线还能得几分？」 |
| `_wireprobe` | 66 | _wireprobe.py —— 抽查预测边与 GT 边的坐标系/端点对齐情况。 |
| `_wireskel` | 130 | _wireskel.py —— 「骨架到底有没有画出这根线」的逐 case 上限测量。 |
| `_wiresnap` | 106 | _wiresnap.py —— 5px 到底是「差一点」还是「结构性错位」： |
| `_wiresnapaudit` | 141 | _wiresnapaudit.py —— 追 EDGE_SNAP 为什么没吃掉那 5~10px。 |
| `_wirestage` | 265 | _wirestage.py —— 导线失分按阶段拆解（只读现成产物，不跑 stage1）。 |
| `_wiresweep` | 18 |  |
| `_wiretime` | 42 | 把 stage1 产物的 wires 耗时与图片规模/骨架复杂度关联起来。 |
| `_wiretj` | 163 | Can a pure-CV *T-junction* count pick the wire colour? |
| `_wiretok` | 19 | _wiretok.py —— 定位内嵌 JSON 的断点（一次性诊断）。 |
| `_wireview` | 125 | 把 stage1 的导线段按「连通分量」上色画在原图上，看分组到底碎在哪。 |
| `_wirezero` | 71 | _wirezero.py —— 为什么「网配上了但预测侧零边」：把那些网对挖出来看。 |
| `_wlen` | 96 | 量「位号框宽度 / 字高」能不能预测位号字符串长度。 |
| `_wlen2` | 98 | 位号框「宽/高」能不能预测真实位号的字符数 —— 决定连号窗口能不能靠几何定案。 |
| `_zerocount` | 42 | 数一数 220 张里每个分项**爆 0 / 近 0** 的图有多少（用当前最佳产物）。 |
| `adjmeas` | 433 | **相邻信息度量**：位号候选的「行 / 列 run」结构里，邻座的号能不能救回读不出来的号。 |

<a id="eval"></a>

## `tools/eval/`（29 个脚本）

**打分尺子**：端到端 A/B、批量评测、结果归档与对比。

| 脚本 | 行 | 说明 |
|---|---:|---|
| `baseline_v2` | 83 | Score the previous model on the NEW yardstick, so a fine-tuned run can still be compared. |
| `cmp220` | 146 | 两个 `final_results` 目录的逐项对比 —— 专为「全量 220 张跑完看净效果」写。 |
| `confusion` | 173 | 在真实 val 上算「真类 → 预测类」的混淆统计（IoU 贪心匹配）。 |
| `epoch_budget` | 40 | 从 results.csv 读 time 列，给出每 epoch 真实耗时与 2 小时能跑多少 epoch。 |
| `fast` | 233 | 小规模快速回归 —— **跑全量 220 张要 78 分钟，改一次代码等不起。** |
| `fn_fp` | 250 | 把漏检(FN)与误检(FP)样本抠出来，做成可肉眼审阅的接触表。 |
| `fn_fp_official` | 226 | 按【官方口径】重做 FN/FP 接触表。 |
| `gt_ink_check` | 64 | GT edge 压墨率卡在 ~48%：是坐标偏了，还是线宽不对？ |
| `gt_pin_geom` | 151 | 量 GT 的「引脚点相对元件框」几何 —— 为「没有导线触到的引脚」找一条可用的规则。 |
| `per_class2` | 99 | 按类别打印检测器在某个 split 上的 P/R/mAP50/mAP50-95/实例数，并按「漏检数」排序。 |
| `pintrace_eval` | 221 | 贪心射线 vs 全局骨架 vs 两者取并 —— 同一把尺子的对照。 |
| `snap_eval` | 223 | 把「框尺寸错了」和「框位置错了」分开量。 |
| `stage1_attach_eval` | 141 | 第一阶段第三个交付物的验收：元件 ↔ 导线的几何附着关系。 |
| `stage1_conn_eval` | 429 | stage1 连通性评测（用户 m04360 第三条）。 |
| `stage1_eval` | 216 | stage1 产出 vs GT 的验收评测。 |
| `stage1_filter_ablation` | 267 | 导线过滤器消融：每一道过滤对「覆盖 F1」和「连通性」各值多少。 |
| `stage1_label_eval` | 248 | 第一阶段：网络标签**候选**的验收（不涉及文字内容）。 |
| `stage1_pin_eval` | 219 | 量 stage1 的引脚点：位置精度 / 召回（**只比位置，不比引脚号**）。 |
| `stage1_reconcile_eval` | 151 | 评测「悬空交叉校验」的收益：把端点全悬空的导线分量删掉，看导线指标怎么变。 |
| `stage1_stats` | 62 | 汇总 stage1 在全部 220 个官方 case 上的产物：元件数/类型对照 + 耗时分布。 |
| `stage1_wire_attrib` | 144 | 给 stage1 提取出的导线线段做「假阳性归因」：没贴住 GT 的那些，究竟是哪来的？ |
| `stage1_wire_colour` | 167 | 提取线段按「是否真的压在 GT 网表线段上」分组，比较两组的图像颜色。 |
| `stage1_wire_filter` | 179 | stage1 导线过滤扫描：到底有多少「提取线段」是文字/符号/边框？ |
| `stage1_wires_eval` | 201 | stage1 导线产出 vs GT 的验收评测。 |
| `stage2_eval` | 236 | 第二阶段的评测：把 `<pred_dir>\<case>.json` 与官方 GT 对起来。 |
| `tp_diag` | 119 | 诊断 CV testpoint 检测器的误报：TP 与 FP 的环特征分布差在哪里。 |
| `tp_probe` | 119 | testpoint 的 CV 可行性探针。 |
| `tp_probe2` | 86 | 按 case 解剖 testpoint 的三种画法：每个 GT 附近都有哪些圆、半径多少、有没有孔。 |
| `tp_sweep` | 98 | 为 CV testpoint 找判别门槛。 |

<a id="flow"></a>

## `tools/flow/`（4 个脚本）

`docs/workflow.html` 的构建与校验。

| 脚本 | 行 | 说明 |
|---|---:|---|
| `_merge_own` | 96 | 把 `own`（独家归属）节点并进**内容更全的那一版** graph，再写回 `graph.json`。 |
| `_recover` | 70 | 从旧 `workflow.html` 里把内嵌的 graph 抠出来，和 `graph.json` 逐节点对比。 |
| `build` | 78 | 把 `graph.json` 灌进 `template.html`，产出一个自包含的工作流编辑器。 |
| `graphcheck` | 342 | `graphcheck.py` —— 工作流图的**接口注册表 + 统一 + 校验器**（用户 m40034 第二项）。 |

<a id="gnd"></a>

## `tools/gnd/`（17 个脚本）

电源/地（`∅NNN`）符号专项。

| 脚本 | 行 | 说明 |
|---|---:|---|
| `_gnd_modes_p3` | 195 | Part 3: use the Part-1 shape modes as a VALIDATOR for YOLO's existing gnd boxes. |
| `_gnd_modes_p3b` | 91 | Part 3b: per-arm comparison of the mode-validator's threshold sweep. |
| `_gndbox` | 139 | Are the k-means 'gnd modes' really modes -- or degrees of GT box misalignment? |
| `_gndmodes3` | 172 | Follow-up to work/_gndbox.py -- are the k-means 'gnd modes' real, or are they |
| `_modesctl` | 124 | Control for work/_gndmodes3.py -- the same k-means/silhouette sweep on other |
| `cv_gnd_val` | 113 | 把「纯 CV 找 GND」和「YOLO 找 GND」放在同一把尺子、同一批图上比。 |
| `gnd_anchor` | 209 | Can a ground-named PIN anchor the ground SYMBOL on the same net? |
| `gnd_census` | 200 | Census of GND glyph FORMS in the real corpus. |
| `gnd_ctx` | 188 | Add geometric context to the per-mode GND matched filter and see if the |
| `gnd_filter` | 172 | GND matched-filter feasibility probe. |
| `gnd_grammar` | 216 | Structural GND grammar detector: "a bar, a shorter bar below, a shorter one |
| `gnd_margin` | 195 | GT bbox 是不是把符号切掉了？—— 对 gnd / v / r / c / testpoint 量「框外还有多少墨」。 |
| `gnd_modes` | 289 | Is `gnd` one shape or several?  And does splitting it help a detector? |
| `gnd_modes2` | 147 | GT 框把符号切掉了 —— 那把裁剪放大再聚类一次，看那 10 个模式是不是「切出来的假象」。 |
| `gnd_seed` | 163 | Two follow-ups on the "seed one GND, then traverse the sheet" plan. |
| `gnd_sheet` | 93 | Contact sheet of real gnd crops, per source family -- is gnd ONE shape? |
| `gnd_slide` | 194 | Does a SLIDING GND template (matched filter) find gnd symbols? |

<a id="recover"></a>

## `tools/recover/`（3 个脚本）

中断恢复工具。

| 脚本 | 行 | 说明 |
|---|---:|---|
| `find_snapshots` | 112 | 从 DSH 会话日志里回收被误改的源文件。 |
| `replay` | 166 | 从 DSH 会话日志里回收被误改的源文件。 |
| `restore` | 64 | 把 `work\_recover\recovered_*.py` 写回源文件，并**正确地**重做「假值词表统一」。 |

<a id="viz"></a>

## `tools/viz/`（32 个脚本）

可视化：把错例裁片、对照图渲染成人眼可核对的图。

| 脚本 | 行 | 说明 |
|---|---:|---|
| `_boxsheet` | 88 | 把一张图里所有 GT box(IC) 元件周围的「环带」裁出来拼成一张对照图， |
| `_crop_cmp` | 166 | 裁片对照表：同一个真值框，三种裁法放大后并排 + 各自读出来的串。 |
| `_crop_one` | 91 | 把某个 stage1 元件周围裁出来看一眼（判断位号到底印没印、印在哪）。 |
| `_cvjson_align` | 62 | 决定性检查：CVJsonStd 的 texts[].location 到底对不对得上图像像素。 |
| `_jstest` | 73 | 把 `case_*.html` 里内嵌的 `<script>` 抽出来，配一套最小 DOM 桩，用 node 真跑一遍。 |
| `_look_stage1` | 23 |  |
| `_net_break` | 197 | 把一个「同一 GT net 的引脚在我的图里被拆到多个分量」的网络画出来。 |
| `_ocr_contact` | 140 | 看得见的第一步：把官方 GT 的文字真值与检测到的文字框对上，裁出来放大看。 |
| `_peek` | 34 | 把某个 case 的几个图层拍平到一张图上看效果（给开发期自检用）。 |
| `_peek_stage1` | 43 |  |
| `_probe_align` | 21 |  |
| `_probe_align2` | 14 |  |
| `_probe_b` | 9 |  |
| `_probe_c` | 13 |  |
| `_probe_ids` | 61 | 探查两件事： |
| `_probe_idx` | 10 |  |
| `_probe_ph` | 6 |  |
| `_probe_ph2` | 8 |  |
| `_probe_split` | 8 |  |
| `_probe_struct` | 21 |  |
| `_seg_cmp` | 139 | 文字框切分对照图：真值（CVJsonStd）绿 / 旧 textbox 红 / 新 textseg 蓝。 |
| `_stage2_diag` | 105 | stage2 诊断：把 GT 的位号/参数/引脚名对照到 stage2 实际读到的文字上。 |
| `_valcrop` | 102 | 把「位号读对了、但值没写出来」的元件裁出来拼成对照图，肉眼看值到底在不在图上。 |
| `_vcheck` | 30 | 看 mkviz 产出的 <case>.json 里各数组的数量与样例（绘图自检用）。 |
| `_wire_look` | 89 | 看一眼导线的真相：原图 + 提取骨架段（绿） vs 原图 + GT 网表线段（红）。 |
| `analyze_run` | 41 | 分析一次 ultralytics 训练的 results.csv 曲线。 |
| `boost_report` | 42 | For each rare class, how many real training images carry it, and how concentrated is it? |
| `comp_cases` | 283 | 第一阶段**元件级**错例的典型示例集（对应 `viz/wire_cases.py` 的导线级那一份）。 |
| `mkview` | 582 | 把 `mkviz.py` 产出的 `work/viz/cases/*.json` 变成可点的网页。 |
| `mkviz` | 567 | 逐批次识别可视化：把一个 case 的 GT / 各阶段产物 / 候选池 / 取景框画成可叠加的图层。 |
| `stage1_draw` | 78 | 把 stage1 的产物画回原图，供人眼核验。用法：python stage1_draw.py 0050 0011 ... |
| `wire_cases` | 320 | 把 stage1 导线提取的错例挑成「典型示例集」，用于人工分析到底哪里出了问题。 |
