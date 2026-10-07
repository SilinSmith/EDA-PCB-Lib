#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第五批 workflow.html 补丁：把 24 个节点的状态、file 徽章、算法与落地事实全部更新。

用法：
    python work\\tools\\diag\\_flowpatch5.py           # 打补丁
    python work\\tools\\diag\\_flowpatch5.py --dry     # 只看会改哪些节点

幂等：靠 MARK="【第五批·审计】" 判重；进度行用 `**进度：…**` 就地替换。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HTML = Path(__file__).resolve().parents[2] / "flow" / "workflow.html"
BAK = HTML.with_suffix(".html.bak5")
MARK = "【第五批·审计】"
RX_PROG = re.compile(r"\*\*进度：[^*]*\*\*")

# ---------------------------------------------------------------- 进度 + file
# (进度行, 新的 file 徽章)
STATUS = {
    "input":                ("✅ 已落地",
        "submission/run.py"),
    "ev_sym":               ("✅ 已落地",
        "stage1/detect.py + stage1/bigic.py"),
    "ev_text":              ("✅ 已落地", "cvparts/textbox.py"),
    "ev_crop":              ("✅ 已落地（外扩常数已越过最优，见下）",
        "stage2/text.py pad_box + textrec/dataset.py"),
    "ev_ocr":               ("✅ 已落地", "textrec/infer.py"),
    "ev_vlm":               ("✅ 已落地 —— 「围绕 box 扫描」真正落地的地方",
        "vlm/enrich.py（753 行）"),
    "ev_wire":              ("✅ 已落地", "stage1/wires.py"),
    "ev_pin":               ("✅ 已落地", "stage1/pintrace.py"),
    "ev_net":               ("✅ 已落地", "stage1/nets.py"),
    "know":                 ("✅ 已落地（原先写着「未落地」，已过期）",
        "fusion/priors.py（911 行）"),
    "pool":                 ("✅ 已落地", "fusion/claims.py（744 行）"),
    "solve":                ("✅ 已落地", "fusion/solve.py（470 行）"),
    "plausible":            ("✅ 已落地", "fusion/plausible.py"),
    "fixstr":               ("⛔ 已证否（保持）", "（已证否）fixstr.py"),
    "conflict":             ("✅ 已落地（并进 claims/solve）", "fusion/solve.py + fusion/plausible.py"),
    "llm_judge":            ("✅ 已落地", "fusion/llm_judge.py（484 行）"),
    "rescan":               ("✅ 已落地", "fusion/rescan.py（336 行）"),
    "iterate":              ("✅ 已落地", "fusion/run.py:145-181"),
    "batch":                ("⚠ 部分落地（值通道拼图已写、默认关）",
        "vlm/enrich.py"),
    "assemble":             ("✅ 已落地", "stage2/assemble.py（960 行）"),
    "export":               ("✅ 已落地", "submission/run.py:380-400"),
    "metrics":              ("✅ 已落地", "schematic_parser/metrics.py"),
    "cropfix":              ("✅ 已落地，但生产默认关",
        "fusion/cropfix.py（456 行）"),
    "audit":                ("✅ 已落地，但生产默认关",
        "fusion/audit.py（588 行）"),
}

# ------------------------------------------------------------------ 审计附注
A_INPUT = """
【第五批·审计】**接线核对（全库 54 个 .py 逐一 grep 导入点）**
- `input` = `eda-pcb/submission/run.py`（471 行）。导出优先序写死在 `run.py:389`：
  `for _d in (fuse_dir, s3v_dir, s3_dir, s2_dir)` ⇒ **融合 > VLM 补强 > 阶段三 > 阶段二**，取最靠后存在的那一层。
- ★ **生产默认 `-fusion on` 会把融合层的 LLM 轮整个关掉**：
  ```python
  run.py:356  os.environ["FUSION_CROPFIX"] = "1" if a.fusion == "llm" else "0"
  run.py:357  os.environ["FUSION_LLM"]     = "1" if a.fusion == "llm" else "0"
  ```
  只有显式 `-fusion llm` 才开。⇒ 交上去的那一份里，`cropfix`/`llm_judge`/`rescan`/`audit` **一次都没跑**。
- `vlm\enrich.py` 的 `DEFAULTS` 是 `name=False` / `desig_fallback=False`，但 `run.py:127-128`（`-vlm-name default="on"`）
  与 `:136`（`-vlm-desig-fallback default="on"`）把两者都设成 `on`，并在 `:307` 显式传进 `opts`
  ⇒ **比赛提交时 name 通道是开的**；自己调 `enrich_dir` 跑实验时**必须显式传 `{"name": True}`**，否则少 2.4 分。
- 实测：`_submit_out220r36\runs\stage2` == `stage3` == `final_results` **逐字节相同**（MD5 `28F99796`/`D4BD46E9`/`3FDBABFA`）
  ⇒ 那条生产链的 VLM 阶段一个字节都没写进去（`run.log` 里连「VLM 补强」段都没有，是用 `-vlm off` 跑的），交的就是 **stage2 = 25.9861**。
"""

A_EV_SYM = """
【第五批·审计】**检出层接线与实测**
- `stage1/detect.py` 是 YOLO 检测器；`stage1/bigic.py` 用几何把 YOLO 检不出的大矩形找回来，
  **已接线**：`stage1/pipeline.py:149  from stage1 import bigic as bigic_mod`。
- `bigic` 的 220 张实测：有 GT box 的图 213 张｜GT box 元件 996 个｜**键对上 425（0.427）｜整颗没输出 571（0.573）**
  ｜带走 **4598 个 GT 引脚**｜至少漏一颗的图 181/213 = 0.850。
- 降 conf 0.25 → 0.01 时 `box` 类输出**仍然是 0 个** ⇒ 这是**能力缺口，不是阈值问题**。
"""

A_EV_CROP = """
【第五批·审计】**尺度归一的三次失败（已写入 `work/reports/尺度归一_诊断.md`）**
- 同一份 796 条真实 val：直调 `Recognizer.read_many`（喂真值框）exact **0.677**（designator 0.885）；
  走线上 `TextEngine.read`（喂 stage1 紧框）只有 **0.197**（designator 0.239）⇒ 差在预处理。
- 墨迹尺度归一**三版全败**（`textrec/dataset.py:_ink_band` + `config.py:INK_*`，`TEXTREC_INK_YNORM` 门控，**默认关**）：
  ① 行投影 `dark.any(axis=1)` → **0.146**（比全 0 的 0.197 还差，导线经过的每一行都「有墨」，归一等于没做）；
  ② 行内最长暗游程 ≤0.5×框宽 → designator **0.84 → 0.36**（门限把短标签自己的笔画判成线）；
  ③ 形态学开运算抠长直线 + 中心行扩张（+`INK_FRAME` 定取景）→ 最好 **0.378**（1.5/2.0/2.5/3.0/4.0 → 0.213/0.319/0.378/0.355/0.228）。
- 更根本的障碍：归一化要「复刻训练裁剪」得从墨迹高 `g` 反推**标注框高 H**，实测 `g/H` 在 val 上 p5..p95 = **0.47..1.36**（中位 0.60）
  —— **不是常数** ⇒ 反推不成立。
- ⇒ **结论：墨迹归一除非连同识别器一起重训（训练/推理同用这一个函数），否则做不「可靠」。可靠的是「定长外扩」。**
- 代码保留（默认关）作后续重训的备选；新增旋钮 `INK_LINE_FRAC`(0.85) / `INK_PROBE_PAD_FRAC`(0.25) / `INK_FRAME`(2.5)。
- 教训：`work\\tools\\diag\\_crnneval.py --shrink` 把「裁掉字形」与「多加白底」混在一起，**不能用来判定外扩优劣**；要判外扩必须走 `TextEngine.read` 或 `fast.py`。
"""

A_EV_VLM = """
【第五批·审计】**这一层就是「围绕元件 box 展开扫描」的落地处 —— 而且它确实在跑**
- `read_designator(bgr, ib, opts, stat, ctype)`（`vlm/enrich.py:213`）：`ib` 是元件 `bbox` 换成图像坐标
  （`:453  ib = (bb[0], H-bb[3], bb[2], H-bb[1])`），按 `desig_margin`（默认 60px）外扩裁片 → `client.ask`；
  读不出（`NONE`/`U?`/乱码）就把外扩 **×2.2** 再试一次 ⇒ **这就是「围绕 box 扫位号」**。
- `read_pins` 每 8 格拼图一问；`read_value` 外扩 90px（45/55/65/80/100/140/200 扫过，逐字命中率 0.836/0.875/0.923/**0.942**/0.933/0.894/0.827，25px 只有 0.212）。
- 220 张实测：**25.9861 → 28.4133（+2.43）**，component F1 0.5416 → 0.6182，pin tp 178 → 206。
- ⚠ **四个通道里只有 `read_name` 挡 `ERR:` 前缀**（`:380`）；位号通道（`:227-232`）没有挡板 ——
  `vlm/client.py:177-178` 把所有 HTTP/超时/JSON 错误压成 `"ERR:%s"` 返回、**永不抛**，
  而 `:230 t = re.sub(r"[^0-9A-Za-z?]", "", lines[0])` 会把 `ERR:...` 洗成连续串，
  只要落成「1-3 字母 + 1-4 数字」形状（如 `ERR400`）就被当位号返回，进而走 `:481-489` 的全局改名
  （改 `comps`/`pins`/`nets[*].hyperGraph`）；失败时还会 ×2.2 **再烧一次请求**（`:221`），`stat["desig"]` 照样自增（`:228`）。
- ⚠ `enrich.py:730-738` 单张异常 ⇒ **整张回退为原样拷贝**，产物与输入逐位相同，导出阶段无法察觉（只有 `failed[:20]`）。
- ⚠ **无任何磁盘/内存缓存**，重跑重复全部 HTTP（220 张约 9658 次请求：位号 4094 / 值 3492 / 引脚网格 1051 / 型号 1021）。
- ★ **位号冲突消解只认「无人争抢」**：`:464 solo = {g: ks[0] for g, ks in claims.items() if len(ks) == 1 and ks[0] != g}`
  ⇒ 两颗元件读出同一位号时**两颗一起放弃**。这是保守设计，但也是位号缺口的一个未量化来源。
"""

A_KNOW = """
【第五批·审计】**这一块已经落地了（原先的「⚠ 未落地」是过期状态）**
- `eda-pcb/fusion/priors.py` = **911 行**（819-911 是 `_demo` / `check`）。
- 四类知识、四条 `pool.add`：位号连号 `_cap(pool, sid, cand, own+0.02)`（`:627`）｜
  引脚号连号（1..N 空位，`score=0.5`，`:713`）｜词表（`GROUND_WORDS`/`POWER_WORDS`/`PIN_NAME_VOCAB`）｜类型约束。
  **不调 LLM**，只造候选；入口 `Priors.annotate(pool, run_only_gaps=True)`（`:570`，⚠ 该参数函数体内**未引用**，是死参数）。
- ⚠ 死代码：`priors.py:458 run_keys_for`（无调用者）｜`:93-95 slot_prefix`（只 `return ""`）｜`:176 DESIG_DIGITS`｜`:85 VAL_RE`｜
  `:424-425 Priors.load()`（无调用者，且 `fusion/priors.json` 不存在 ⇒ `self.extra` 恒空，**词表扩展从未启用**）。
- ⚠ `priors.py:741-745` 的「形状惩罚」是**静默 no-op**：整段只写 `ev["形状分"]` 与 `st["shape_pen"]`，**没有任何 `a.score` 赋值**。
- ⚠ `priors.py:680` 的并列窗口路径漏了 `_cap`：`score=(_f(b.score) if b else 0.5) * 0.90` ⇒ `b.score < 0` 时 `0.9*b > b`，
  先验候选直接变 `best()`；而同族 `:627` 包了 `_cap`（`:201`：`min(sc, b.score - 0.01)`）。
- ⚠ `priors.py:711-713` 引脚先验硬编码 `score=0.5` 且不过 `_cap`；`zip(targets, free)` 在 `len(targets) > len(free)` 时**静默丢目标**。
"""

A_POOL = """
【第五批·审计】**这一块的真实结构、落地缺口与写入规范问题**
- `eda-pcb/fusion/claims.py` = **744 行**。真实结构：`Region`（`:131-142`，字段 `box/focus/note/src`）｜
  `ClaimPool`（`:188`，字段 `case/doc/assigned/hy/regions/meta/notes/bgr`）｜`add(...)`（`:203`，**唯一**加候选入口，
  `keep=False` 只登记 meta/取景框）｜`set_region`（`:231`，**覆盖**语义）｜`grow_region`（`:242`，只撑窗口不动 focus）｜
  `apply(assigned=None)`（`:311`）｜`_add_vlm`（`:398`）｜`from_stage2`（`:503`）｜`load_image`（`:653`）。
- 槽位 id 形态全集：`comp.<cid>.key`｜`comp.<cid>.value`｜`comp.<cid>.partname`｜`comp.<cid>.type`（**仅被消费，全库无处产生**）｜
  `pin.<cid>.<k>.number` / `.name`｜`netlabel.<i>`。文档字符串声明的 `net.<net_key>.label`（`claims.py:31`）**本包不产生**。
- ★★ **`claims.py:616-629` 的 `keep=False` 让 stage2 的 OCR 引脚读数被整条丢弃**：
  ```python
  claims.py:629          keep=False)          # 单候选不进候选表，等 priors 补先验
  ```
  `keep=False` 时两个分支都不进 ⇒ `priors.py:700 b = pool.best(sid)` 永远 `None` ⇒
  `:708 targets = [s for s in slots if s not in read]` 命中**全部**引脚 ⇒ 每颗引脚被**凭空派 1..N、`score=0.5`**。
  `claims.py:626-628` 的 `meta` 只存 `cid/k/type/pt`、**不存文本**，救不回来。
  ⇒ 这个 `keep=False` 把「先验补空位」的原始设计打成了「先验全覆盖」。
- ★★ **`doc.apply` 只处理 `comp.<cid>.key` / `.value` / `.partname`**（`doc.py:156`、`:184-205`）
  ⇒ **`pin.*` 与 `netlabel.*` 的裁决根本进不了交付文档**：`claims.py:341-353` 写进 `assigned` 的引脚裁决
  只落到旁路产物 `OUT_DIR/_assigned/<case>.json`（`run.py:192-193`），**不影响最终文档**。
  ⇒ **融合层的整条引脚通道，对交付文档的贡献是 0。**
- ⚠ `comp.<cid>.type` 是**死分支**：`claims.py:339-340 rec["_type_override"] = val` / `rec["_type_by"] = by`
  全库**无任何读取方**，也无人产生该槽位。
- ⚠ `claims.py:352-353 pmap[k] = slot` 与 `pmap[str(k)] = slot` 指向**同一对象**、**int 键盖 str 键**；
  且让 `:611 len(pmap)` 把一颗引脚数成两个。
- ⚠ `claims.py:590 score=_f(d.get("score"), 1.0)` —— 缺分数**给满分 1.0**；而同族 `:551 score=_f(cd.get("score"))` 给 **0.0**。
  同一个「未知置信度」在两处得到最高分与最低分。
- ⚠ `claims.py:544-550` 的形状闸在两种模式下**语义相反**：`PR.SHAPE_SCORE` 默认 `""` ⇒ `if` 为假 ⇒ **整条删除候选**；
  一旦设了这个 env 又变成「保留并压分」。一个环境变量把「降权」悄悄变成「删证据」。
- ⚠ `claims.py:565-570` 取 `boxed[0]` 当「分数最高的候选框」，实际是 `desig_slots[cid]` 的**原始列表顺序**，没有任何排序。
"""

A_SOLVE = """
【第五批·审计】**算法原文核对**
- `fusion/solve.py` = **470 行**。`solve(pool, do_gap=True)`（`:277`）→ 分槽位（`:279-283`）→ `gather`（`:286`）
  → 唯一性槽位留 `match`、非唯一直接取 `rows[0]`（`:288-295`）→ `match`（`:153-154`）→ `gap_fill`（`:204`）有产出则重跑 `match`（`:302-309`）。
- ★ `match` 是**贪心 + 增广路**（核心 `:167-178`，增广递归 `:135-150`），**不是 Hungarian**。
- 常量：`OPEN_MARGIN=0.10`（`:47`）｜`MAX_OPEN=24`（`:49`）｜`GAP_MIN_RUN=4`（`:51`）｜`GAP_MIN_COVER=0.45`（`:53`）｜
  `GAP_SCORE=0.90`（`:55`）｜`GAP_FILL` 默认 True（`:56`）｜`MAX_CANDS=8`（`:58`）。
- **全程不写池**：`write_back(pool, sol, by="solve")`（`:368`）单独回写。
- ⚠ `write_back`（`:376 pool.hy.resolve(sid, v, "%s:%s" % (by, src or "?"))`）**无条件覆盖既有裁决**，
  且 `hypo.py:156-159` 没有 `resolved()` 守卫、`solve()` 也不读 `chosen`/`decided_by`
  ⇒ **最后一次 `write_back` 赢**，上一轮 `llm:`/`vlm:judge` 的来源标记被改写成 `solve:...`，**归因失真**。
- ⚠ `solve.py:258-260 gap_fill` 在**负分槽位整条静默失效**：`sc = min(base, own - 0.01) if own_read else base` / `if sc <= 0.0: continue`。
- ★ `open_questions`（`:322-364`）**要求槽位至少两个不同值的候选**：
  ```python
  solve.py:337          if not alts: continue
  solve.py:347          if second is None: continue
  ```
  ⇒ **一个「零候选」的槽位永远不会被 `llm_judge`/`rescan`/`audit` 问到** —— 这是「box 引导重扫」缺的那一环（见图上 `boxscan` 节点）。
"""

A_CONFLICT = """
【第五批·审计】`conflict` 与 `plausible` 的疑点来源（`priors.check`：座号撞号/断号/前缀与类型不符…）
经 `solve.bad` → `conflict.bad` → `llm_judge.q` / `audit.susp` 分发。
⚠ `solve.py:351 reg = pool.region(sid)` 取的仍是 `claims` 建好的那份取景框；**没有任何模块会为「还没有候选」的槽位新建框**。
"""

A_LLM_JUDGE = """
【第五批·审计】**它到底在做什么（行号已校正）**
- `eda-pcb/fusion/llm_judge.py` = **484 行**。它**不产生新框**：只把切法在 `tight`（`:55 FUSION_JUDGE_CROP`）之间切，
  裁决结果以候选身份回池：`:370-374 pool.add(..., source="vlm:judge", score=LLM_SCORE=1.5)`。
- `_admit`（`:202-240`）给 `mult`：形状不合 → 拒（0.0）；前缀与类型不符 → `MISMATCH_W=0.30`（`:57`）；否则 1.0。
- 三级实测（9 张 GT 子集）：自由读数 **22.2291（−0.765）** → 候选二选一 **22.4965** →
  `+agree_prior` **22.4965** → `+跳过占位符` **22.9943**（＝与 no-LLM 完全一致，耗时 70.7s → 6.2s）。
- ⇒ **它现在是「安全的中性」：不再掉分，但也不加分。**
- ★ 真正的限制是**它问不到该问的东西**：`solve.open_questions` 要求「两个不同值的候选」（`solve.py:347`），
  零候选槽位被挡在门外；`MAX_OPEN=24` 又让引脚槽位（即使修好也进不了文档）挤占 `comp.*` 的名额。
"""

A_RESCAN = """
【第五批·审计】
- `eda-pcb/fusion/rescan.py` = **336 行**。`plan(pool, sus)`（`:86`）把疑点变成任务，`run(pool, tasks)`（`:195`）执行。
- ⚠ 它**不产生新框**：`:219 mode=t.get("crop") or RESCAN_CROP` —— 只在 `tight`/`window`（`:42 FUSION_RESCAN_CROP`）之间切换。
- 裁决以候选身份回池：`:247-254 pool.add(..., source="vlm:rescan", score=1.2)`。
- ★ `run(pool, tasks)` 依赖 `pool.region(sid)`（`:211`）—— **没有取景框的槽位直接跳过**，而取景框只在 `claims.from_stage2` 里建。
"""

A_ASSEMBLE = """
【第五批·审计】**`pin_<N>` 的 N 到底是什么**
- `stage2/assemble.py` = **960 行**。`_g["pin_%s" % num]`（`:556` 贴章路径）与 `pk = "pin_%s" % num`（`:660` 一般路径）
  ⇒ **`pin_<N>` 的 N 就是引脚号**，`pinname` 是显示名（`:664 "pinname": nm`）。
- `num` 由 `raw[k]["num"]`（OCR 读数）或**结构期望**（`:632-647` 把整颗重排成 `1..N`）决定；
  `:650 num = r["num"] or str(k + 1)`。
- ★★ **`num_of: Dict[Tuple[str, int], str]`（`:512` 声明、`:560`/`:667` 填充）只在内部用来建 `nets[*].hyperGraph`
  （`:702`/`:712`），`build_doc` 的返回（`:746`）里没有它** ⇒ 融合层**没有任何办法**把「第 k 脚现在是几号」映射回 `pin_<N>` 键。
  这是「引脚裁决进不了文档」的**根本技术障碍**（不是不想写，是没有映射）。
- 实测三种映射的覆盖率（case 0001-0005，840 个引脚位）：**插入序 0.560｜按点匹配 0.462｜按 OCR 原文 0.318**
  ⇒ **都不能用来安全地改键**。要修必须先让 `build_doc` 把 `num_of` 导出成 `doc["pin_num"]`（一行），
  代价是 stage2 重跑一遍（纯 CPU、无 VLM）。
"""

A_EXPORT = """
【第五批·审计】
- `submission/run.py:380-400`：`for _d in (fuse_dir, s3v_dir, s3_dir, s2_dir)` → `shutil.copyfile(src, final_dir / "<case>_Pred.json")`。
- ⚠ **这一层是「静默降级」的**：某一层缺产物就自动往下一层退，**导出成功的张数不会区分来自哪一层**
  —— `_submit_out220r36` 就是这样交了纯 stage2（25.9861）而看不出问题（三份 MD5 逐字节相同）。
"""

A_CROPFIX = """
【第五批·审计】**这是全项目唯一一处「LLM 引导重新框选 + 重新识别」**
- `eda-pcb/fusion/cropfix.py` = **456 行**，自检 16 项通过。
- 完整链（行号已校正）：`run.py:125-127`（`if CF.ON and pool.bgr is not None`）→ `CF.run`（`:375`）→
  `looks_cut` 纯文本筛（`:357-372`）→ `fix_slot`（`:269`）→ **LLM 定方向** `_ask(PROMPT_DIR)`（`:305/312`）→
  `parse_dir(txt)`（`:313`）→ **规则定幅度** `h = max(1.0, focus[3]-focus[1])`（`:320`）→
  `cand_box = expand(focus, d, k*h, shape=(W,H))`（`:323`，`LADDER` 默认 `0.5,1.0,2.0,4.0` @`:54-55`；
  `expand` @`:162-177`）→ **新框裁片** `_crop`（`:180-183`/`:326`）→ **LLM 重读** `PROMPT_READ`（`:329`）→
  `LJ.clean_answer`（`:330`）→ `_score`（`:331`，`:334-337` 只在 `s > best_s` 时收）→
  `pool.add(source="vlm:cropfix", score=FIX_SCORE=1.25)`（`:391-397`）。
- ★ **重新识别一律是「再发回 VLM 看图」，不是 CRNN** —— 四个融合 LLM 文件里没有任何 `textrec`/CRNN 调用；
  图片走 `vlm/client.py:133-137` 的 `{"type":"image_url","image_url":{"url":"data:image/png;base64,"+enc_png(img_bgr)}}`。
- ★ **LLM 能输出自己的读数**：全部以候选身份进 `ClaimPool`（`vlm:judge 1.5` / `vlm:cropfix 1.25` / `vlm:audit 1.30` /
  `vlm:fix 1.25` / `vlm:rescan 1.2` / `vlm:enrich 1.15`），**没有任何一条直接改文档**。
- ★ 但 `submission/run.py:356` 在默认 `-fusion on` 时把 `FUSION_CROPFIX` 置 `0`
  ⇒ **交上去的那一份里它一次都没跑**；报告里的 `rep["cropfix"]` 只是全 0 字典（`run.py:141`）。
"""

A_AUDIT = """
【第五批·审计】
- `eda-pcb/fusion/audit.py` = **588 行**。裁决回池：`:413-417 source="vlm:audit" score=1.30`｜`:430-434 source="vlm:fix" score=CF.FIX_SCORE`。
- ⚠ 它同样依赖 `pool.region(sid)`（`:270`）—— **没有取景框的槽位直接跳过**。
- ⚠ `run.py:150 j = LJ.judge(...)` **不在任何 try 里**（对照 `:164-171 AU.run` 包了 try），
  抛异常则该 case **一份融合产物都没有**（写盘在 `:188-193`）。
- ⚠ `run.py:179-180 if not tasks and not q_first: break` 检查的是**上一轮**的 `q_first`（只在 `:181` 更新）。
- ⚠ `run.py:112-116` VLM 底稿读取失败**静默降级 `None`**，报告里只有 `"vlm_doc": bool(vlm_doc)`（`:139`），
  损坏与「本来就没有」不可区分；退回 `build_doc` 重装配的代价（9 张）：22.5020 → 22.3306。
- ⚠ `run.py:71 VLM_DIR` 默认 `work\\_fusion_vlm`，**与 submission 用的 `runs\\stage3_vlm` 不是同一目录**，
  且**无同链校验**（只看 `vp.exists()`，`:111-112`）⇒ 直接 `python -m fusion.run` 会读**另一条运行链**的 VLM 文档。
- ⚠ `run.py:273-283` 报告读取失败的 except 把「合并」退化成「覆盖」（`old = []`），不留痕迹。
"""

A_METRICS = """
【第五批·审计】官方口径三份产物**逐字节相同**是这一轮最硬的一条证据：
`runs\\stage2\\0001.json` = `runs\\stage3\\0001.json` = `final_results\\0001_Pred.json` = MD5 `28F99796`
（`0003`=`D4BD46E9`、`0005`=`3FDBABFA`），时间戳 `stage2 13:02:10 → stage3 13:05:16 → final_results 13:05:21`。
打分：`stage3` = **25.9861**、`final_results` = **25.9861**，逐项完全相同。
而带 VLM 补强的 `work\\_fusion_vlm` = **28.4133**（配对 220/220）⇒ **同一份 stage2，加了 VLM 补强就 +2.43 分，生产链没把它交上去。**
"""

A_BATCH = """
【第五批·审计】请求量实测（220 张）：位号 4094 / 值 3492 / 引脚网格 1051 / 型号 1021 ≈ **9658 次**。
值通道拼图（`VALUE_PROMPT_GRID` 每格 `#i: C12=100nF`）已实现但**默认关**（`value_batch=1`）——
因为拼图会让**每格分辨率下降**，而值通道对裁片尺寸极敏感（margin 25 → 0.212 vs margin 80 → 0.942）
⇒ 必须先 A/B 证明每格精度不掉再打开。
⚠ `enrich.py` **没有任何磁盘/内存缓存**，重跑重复全部 HTTP。
"""

ADDENDA = {
    "input": A_INPUT,
    "ev_sym": A_EV_SYM,
    "ev_crop": A_EV_CROP,
    "ev_vlm": A_EV_VLM,
    "know": A_KNOW,
    "pool": A_POOL,
    "solve": A_SOLVE,
    "conflict": A_CONFLICT,
    "llm_judge": A_LLM_JUDGE,
    "rescan": A_RESCAN,
    "assemble": A_ASSEMBLE,
    "export": A_EXPORT,
    "cropfix": A_CROPFIX,
    "audit": A_AUDIT,
    "metrics": A_METRICS,
    "batch": A_BATCH,
}

# ------------------------------------------------------------------ 新节点
NEW_NODE = {
    "id": "boxscan",
    "group": "fz",
    "kind": "new",
    "title": "★★ 围绕元件 box 的重扫（融合层缺的那一环）",
    "sub": "每个元件框都该建取景框再问一遍；现在只有「已经有文字候选」的元件才有取景框",
    "x": 420, "y": 880, "w": 300,
    "file": "（要补）fusion/boxscan.py 或 claims.from_stage2 第 5 步",
    "ins": [{"id": "claims", "label": "证据集"}, {"id": "det", "label": "元件框"}],
    "outs": [{"id": "region", "label": "新取景框"}, {"id": "reads", "label": "重扫读数"}],
    "params": [
        ["来源", "用户 m23186：『探测到了 box 元件，那么肯定要围绕他的 box 展开 pin 和文字的扫描』"],
        ["现状", "set_region/grow_region 只被 claims.from_stage2 自己调用，池子建好之后没人再建框"],
        ["卡在哪", "solve.open_questions 要求「两个不同值的候选」，零候选槽位永远问不到"],
        ["实测缺口", "629 颗没配上的 GT IC 中 418 颗（56.0%）的重叠元件 desig_slots 是空的"],
        ["实测缺口", "_desigassoc：91.3% 的缺失引脚在全部 cid 的候选表倒排里都查不到"],
        ["顺带要修", "claims.py:611 用 pinpts.get(cid)，而 doc['pins'] 的键是位号 ⇒ 引脚取景框恒为 None"],
    ],
    "note": """**进度：⚠ 未落地（融合层这一半没做；`vlm/enrich.py` 那一半已做）**

**用户 m23186 原话**：『我探测到了 box 元件，那么肯定要围绕他的 box 展开 pin 和文字的扫描。这也是为什么需要融合层：统筹各种资源引导其他的子调度。』

**已经做了的那一半**：`eda-pcb/vlm/enrich.py:213 read_designator` —— 把元件 `bbox` 换成图像坐标、外扩 `desig_margin`（60px，失败 ×2.2）裁片、问模型；`read_pins` 每 8 格拼图。这是**证据源**层的围绕 box 扫描。

**没做的那一半（本节点）**：融合层**建完池子之后，不会再围绕任何元件框新建取景框**。
实测接线（全 fusion 目录 grep `set_region|grow_region|\\.region\\(`）：
- `set_region` / `grow_region` 的**全部**调用点都在 `claims.from_stage2` 里（`claims.py:497 / 570 / 575 / 597 / 630 / 647`）；
- 其余只是**读**：`audit.py:270`、`cropfix.py:292`、`llm_judge.py:285/413`、`rescan.py:211`、`solve.py:351`。

⇒ 一个 `desig_slots` 为空的元件**没有 `comp.<cid>.key` 槽位**，于是：
1. 没有取景框 ⇒ `llm_judge`/`rescan`/`cropfix`/`audit` **四个模块全部跳过它**；
2. 就算给它建了槽位，`solve.open_questions`（`solve.py:347 if second is None: continue`）**也问不到它**（零候选）。

**这条缺口有多大（220 张实测）**
| 层面 | 可回收 | 脚本 |
|---|---|---|
| 融合层「候选择优」（正确键在该元件**自己**的候选表里） | **6.0%** 引脚 | `_desigcand2.py` / `_desigassoc.py` |
| 融合层「跨元件重归属」（纯交换 **0** 个） | **2.7%** 引脚 | `_desigassoc.py` |
| 重新读邻域**全部** stage1 文字框能读出正确键 | **6.0%**（38/629） | `_icread2.py` |
| ⇒ 剩下 **94%**：位号文字根本没进文字框集合 | 融合层无解 | `_icread2.py` + `_desigdetect.py` |

拆开看 629 颗 / 5210 个没配上的 GT 引脚：
**50 颗 / 623 引脚**（R0，连重叠元件都没有）｜**418 颗 / 2915 引脚（56.0%）**：重叠元件的 `desig_slots` **是空的**｜
**152 颗 / 1484 引脚（28.5%）**：有候选但**全错**｜**9 颗 / 188 引脚（3.6%）**：正确位号**就在候选表里**却输了选择｜
**4 颗 / 82 引脚（1.6%）**：键只差大小写（GT `u2` vs 我方 `U2`，`metrics.py:311 for key in set(gc) & set(pc)` 是精确字符串匹配）。

**该补的三件事**
1. 对**每个**元件（尤其 `type == "box"`）无条件建 `comp.<cid>.key` 槽位 + 取景框（本体外扩，`REGION_MIN=120` 兜底）；
2. 给「零候选」槽位一条**去问一遍**的路径 —— 现在 `open_questions` 的 `if second is None: continue` 把它挡死了，
   要么给一条「没有读数」的哨兵候选，要么给 `open_questions` 加一条 `not alts` 的分支；
3. 顺带修 `claims.py:611-614`：`pinpts.get(cid)` 用的是 `cid`，而 `doc["pins"]` 的键是**位号**（`U3`）不是 `cid`（`c0003`）
   ⇒ 永远 `None` ⇒ **每个引脚取景框都是 `box=None`**（`claims.py:630-633`），引脚槽位拿不到图。

⚠ 与 `pin` 通道的关系：即使修好取景框，**引脚裁决依然进不了交付文档**（`doc.apply` 只认 `comp.*`，见 `pool` 节点），
而 `build_doc` 没有导出 `num_of` ⇒ 映射回 `pin_<N>` 键也做不了。**所以 boxscan 的第一优先级是位号，不是引脚。**""",
}

NEW_LINKS = [
    {"from": "pool.claims", "to": "boxscan.claims", "kind": "data"},
    {"from": "ev_sym.det", "to": "boxscan.det", "kind": "data"},
    {"from": "boxscan.region", "to": "ev_vlm.engine", "kind": "ctrl"},
    {"from": "boxscan.reads", "to": "pool.reads", "kind": "data"},
]

TOP_NOTE = """**用户 m23186 的架构批评（本轮任务的直接来源）**：『你要坚持把整个大局做出来才能得到显著的性能增益……你要核对一下你现在用了什么算法什么思路，以及总的流程，所有的业务逻辑到底是怎么样的。』

**★ 第五批审计的三条硬结论**
1. **LLM 确实在引导重新框选与重新识别 —— 但它被默认配置关掉了。**
   唯一一处是 `fusion/cropfix.py`（LLM 定方向 → 阶梯扩框 → 裁新图 → LLM 重读 → 自洽性采纳）。
   而 `submission/run.py:356-357` 在默认 `-fusion on` 时把 `FUSION_CROPFIX` 与 `FUSION_LLM` 都置 `0`
   ⇒ **交上去的那一份里融合层的 LLM 轮一次都没跑**。220 张上 `-fusion on`（无 LLM 轮）= 28.4200，
   与不融合的 VLM 补强 28.4133 只差 **+0.007** —— 这就是「略微上升所以看不出价值」的原因。
2. **LLM 能输出自己的读数**：`vlm:judge 1.5` / `vlm:cropfix 1.25` / `vlm:audit 1.30` / `vlm:fix 1.25` /
   `vlm:rescan 1.2` / `vlm:enrich 1.15`，**全部以候选身份回池，没有一条直接改文档**（这是对的，别改）。
3. **有两块「写了但不通」**：`doc.apply` 只认 `comp.*.key/value/partname` ⇒ **`pin.*` 与 `netlabel.*` 的裁决进不了交付文档**；
   `stage2/assemble.py` 的 `num_of` 没导出 ⇒ 引脚号连映射回去都做不了。两块都画在下面对应节点的【第五批·审计】里。

**★ 生产链的实测事实**：`_submit_out220r36` 的 `stage2` = `stage3` = `final_results` **逐字节相同**（MD5 `28F99796`）= **25.9861**；
带 VLM 补强的 `work\\_fusion_vlm` = **28.4133**。**同一份 stage2，补强 +2.43 分，生产链没交。**
"""


def main() -> int:
    dry = "--dry" in sys.argv
    s = HTML.read_text(encoding="utf-8")
    m = re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', s, re.S)
    if not m:
        print("找不到 graphdata")
        return 2
    g = json.loads(m.group(2))
    by = {n["id"]: n for n in g["nodes"]}

    for nid, (st, f) in STATUS.items():
        n = by.get(nid)
        if n is None:
            print("  !! 图上没有节点 %s" % nid)
            continue
        line = "**进度：%s**" % st
        old = n.get("note") or ""
        if RX_PROG.search(old):
            new = RX_PROG.sub(line, old, count=1)
        else:
            new = line + "\n" + old
        n["note"] = new
        n["file"] = f
        print("  %-12s 进度=%-30s file=%s" % (nid, st, f))

    for nid, add in ADDENDA.items():
        n = by.get(nid)
        if n is None:
            continue
        if MARK in (n.get("note") or ""):
            print("  %-12s 审计附注已在，跳过" % nid)
            continue
        n["note"] = (n.get("note") or "").rstrip() + "\n\n" + add.strip() + "\n"

    if "boxscan" not in by:
        g["nodes"].append(NEW_NODE)
        g["links"].extend(NEW_LINKS)
        print("  + 新节点 boxscan（4 条连线）")
    else:
        print("  boxscan 已存在，跳过")

    g["note"] = TOP_NOTE

    if dry:
        print("\n--dry：不写盘")
        return 0
    if not BAK.exists():
        BAK.write_text(s, encoding="utf-8")
        print("备份 -> %s" % BAK)
    out = s[:m.start(2)] + json.dumps(g, ensure_ascii=False, indent=1) + s[m.end(2):]
    HTML.write_text(out, encoding="utf-8")
    print("已写入 %s（%d 字节 -> %d 字节）" % (HTML, len(s), len(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
