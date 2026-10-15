# -*- coding: utf-8 -*-
"""第八批：把本轮实测结论写进 `_wfbullets.py`（徽标已改，bullet 要跟上）。

跑： python work/tools/diag/_patch8.py [--dry]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _subst import subst  # noqa: E402

# ★ 两份都要改：`_wfbuild.py` 是 `sys.path.insert(0, FLOW)` + `import _wfbullets`，
#   所以**构建读的是 `work/flow/_wfbullets.py`**；`eda-pcb/tools/diag/` 那份是仓库快照。
ROOT = HERE.parents[2]
TGT = [ROOT / "work" / "flow" / "_wfbullets.py",
       ROOT / "eda-pcb" / "tools" / "diag" / "_wfbullets.py"]

PAIRS = [
    # ---- llm_judge：上下文块 + 整表判负 + 交付口径三臂 ----
    (""" "★ 真正限制是**问不到该问的**：`solve.open_questions` 要求两个不同值 ⇒ 零候选被挡；`MAX_OPEN=24` 又让引脚槽位挤占 `comp.*` 名额",
 "⚠ 它**不产生新框**，只把切法在 `tight`(`FUSION_JUDGE_CROP`) 之间切",
],""",
     """ "★ 真正限制是**问不到该问的**：`solve.open_questions` 要求两个不同值 ⇒ 零候选被挡；`MAX_OPEN=24` 又让引脚槽位挤占 `comp.*` 名额",
 "⚠ 它**不产生新框**，只把切法在 `tight`(`FUSION_JUDGE_CROP`) 之间切",
 "★★ **上下文块（本会话新增，默认开 `FUSION_JUDGE_CTX=1`）**：`judge` 原来只发「一张小图 + A/B 两个选项」，**没有上下文**。现按规格五样拼齐：**元件类型**（`comp.<cid>.type` 的读数）/ **同一颗元件上别的字段读数** / **邻近元件**（按 `bbox_img` 中心距取前 `FUSION_JUDGE_CTX_NB=6`，提示词写明「一张图上位号唯一，邻近的号都已被别人用了」）/ **已定位号**（`_known_designators`）/ **⚠已被判可疑**（`plausible.check(pool)` 里 `severity ≥ FUSION_JUDGE_CTX_SEV=0.5` 的）",
 "★ 并要求模型回一句 `依据=<…>`（`FUSION_JUDGE_WHY=1`）—— 这句落进 `evidence['依据']` 与报告的 `judge_rows[].why`，是「上下文到底进没进提示词」的**直接证据**",
 "★★ **实测（官方 9 张，`--no-vlm` + `FUSION_LLM=1`，只切 `FUSION_JUDGE_CTX`）：开 53.27 / 关 52.48 ⇒ +0.79**，无一张变差；case 5 +1.65、case 7 +2.50、case 8 +3.00；元件 tp 167→172、导线 tp 135→139。归档 `work/_fast_base/{ctxA,ctxB}.json`",
 "★ 生效证据（case 1）：`ctx=1` 时 **12/12 行都有 `依据=…`**，`ctx=0` 时 **0/12**",
 "⚠ 代价 **+10 s/张**（case 1 墙钟 37.3 s vs 27.3 s，judge 自身 11.1 vs 7.2）⇒ 220 张约 **+37 min**",
 "⚠ **`n_ctx_slot: 0`（实测）**：`judge` 跑在 `pool.apply()` / `plausible.check(pool, pred)` **之前**，而 `solve` 已经把重号解掉了 ⇒ round 1 的「冲突内容」基本是空的；真正吃到的是类型 / 同件别的字段 / 邻近元件 / 已定位号这四样",
 "⛔ **整表裁决（contact sheet）判负，默认关**：`judge` 的入口原来是 `if mode == "sheet" and SHEET_ON:`，而 `MODE` 默认 `"auto"` ⇒ **`judge_sheet`（`SHEET_ON` 默认 `1`、`_sheet_image`/`_known_designators` 全是为它写的）一次都没跑过**。修好路由后 4 张 A/B：**sheet 55.32 vs 逐题 59.62（−1.07）**，且更慢（160.5 s vs 83.6 s）⇒ 新增 `FUSION_JUDGE_SHEET_MIN`（**默认 `9999` = 关**），要试用 `FUSION_JUDGE_MODE=sheet`",
 "★★ **交付口径下整个 LLM 轮值 0.00**：`tools/eval/fast.py` 官方 9 张，**LLM 关 / LLM 开 / LLM 开+上下文 三臂全是 58.63 且逐位相同**（逐张 `1=47.24 2=53.50 3=68.25 4=66.60 5=52.20 7=62.12 8=62.71 9=57.13 10=57.93`；四项 comp 0.9403 / pin 0.3920 / netH 0.5355 / netL 0.1876）。已用 `_jrep.py` 证明三臂**确实是三套配置**（`reg9` 有 judge 块、无 `ctx` 键；`reg9ctx` 有 `ctx: True`；`reg9nollm` **没有 judge 块**）⇒ **LLM 轮是「没有 VLM 时的补强」，不是交付口径的增量，而它要 +1.8 h/220 张**",
],"""),

    # ---- iterate：两条 ⚠ 已修 ----
    (""" "⚠ `run.py:179-180 if not tasks and not q_first: break` 检查的是**上一轮**的 `q_first`（只在 `:181` 更新）",
 "⚠ `run.py:150 j = LJ.judge(...)` **不在任何 try 里**（对照 `:164-171 AU.run` 包了 try）⇒ 抛异常则该 case **一份融合产物都没有**",
],""",
     """ "✅ `run.py:179-180 if not tasks and not q_first: break` 检查的是**上一轮**的 `q_first`（只在 `:181` 更新）⇒ **已修**：新增 `FUSION_CONV_EXACT=1`，改用轮末重算的 `q_next = SV.open_questions(pool, sol['assign'])`，并把 `n_q_next` 写进报告（case 1 实测 `n_q_next 15`）",
 "✅ `run.py:150 j = LJ.judge(...)` **不在任何 try 里**（对照 `:164-171 AU.run` 包了 try）⇒ 抛异常则该 case **一份融合产物都没有** —— **已修**：`judge` 与 `RS.run` 各自包 try，异常落 `r['judge']={'err':…}` 并打 `[judge] 跳过：…`",
],"""),

    # ---- audit：同一条 ⚠ 已修 ----
    (""" "⚠ `run.py:150 j = LJ.judge(...)` **不在任何 try 里** ⇒ 抛异常则该 case **一份融合产物都没有**",""",
     """ "✅ `run.py:150 j = LJ.judge(...)` **不在任何 try 里** ⇒ 抛异常则该 case **一份融合产物都没有** —— **已修**（见 `iterate` 条目）","""),

    # ---- pool：两条旧 ⚠ 复核为「不是缺口 / 已修」 ----
    (""" "⚠ `claims.py:616-629` 的 `keep=False` 让 stage2 的 OCR 引脚读数被**整条丢弃**（`priors.py:700 b = pool.best(sid)` 永远 `None`）⇒ 「先验补空位」被打成「**先验全覆盖**」",
 "⚠ `:590 score=_f(d.get('score'),1.0)` 缺分数给**满分**，同族 `:551` 给 0.0",""",
     """ "✅ `claims.py:616-629` 的 `keep=False` 让 stage2 的 OCR 引脚读数被**整条丢弃**（`priors.py:700 b = pool.best(sid)` 永远 `None`）⇒ 「先验补空位」被打成「**先验全覆盖**」—— **复核：这不是缺口**，`keep=True` 会让 220 张 28.4973→**27.6871**（pin 0.0821→0.0783、netH 0.2163→**0.1966**），注释在 `:739-746`",
 "✅ `:590 score=_f(d.get('score'),1.0)` 缺分数给**满分**，同族 `:551` 给 0.0 —— **已修**（注释在 `:694-699`）","""),

    # ---- solve ----
    (""" "⚠ `write_back`(`:376`) **无条件覆盖**既有裁决（`hypo.py:156-159` 没有 `resolved()` 守卫）⇒ **最后一次 write_back 赢**，来源标记被改写成 `solve:...` ⇒ **归因失真**",
 "⚠ `gap_fill`(`:258-260`) 在**负分槽位**整条静默失效（`if sc <= 0.0: continue`）",""",
     """ "⚠ `write_back`(`:376`) **无条件覆盖**既有裁决（`hypo.py:156-159` 没有 `resolved()` 守卫）⇒ **最后一次 write_back 赢**，来源标记被改写成 `solve:...` ⇒ **归因失真**（**未改**：只影响诊断归因、不影响分数；要改得动 `hypo.py` 的 `resolved()` 守卫）",
 "⚠ `gap_fill`(`:258-260`) 在**负分槽位**整条静默失效（`if sc <= 0.0: continue`）—— **已加计数** `st['gap_skip_negative']`，保守行为保留但不再静默","""),

    # ---- know：词表扩展复核 ----
    (""" "⚠ `Priors.load()` 无调用者、`fusion/priors.json` **不存在** ⇒ 词表扩展从未启用",""",
     """ "⚠ `Priors.load()` 无调用者、`fusion/priors.json` **不存在** ⇒ 词表扩展从未启用（**已复核**：`grep -r Priors.load eda-pcb` 零调用点；且 `pin_name_shape`/`value_shape` 用的是**模块级** `PIN_NAME_SET`，而 `Priors.load()` 填的是**实例字段** `self.pin_names` ⇒ 两者不通，接了也是 no-op。**不做**：可赢空间只有官方 9 张约 8 个引脚）","""),

    # ---- fixstr / batch / boxscan：徽标已改「停用」 ----
    (""" "⛔ **已证否（保持）** —— `fixstr.py` **不存在**",""",
     """ "⛔ **徽标已改「停用」** ｜ **已证否（保持）** —— `fixstr.py` **不存在**","""),

    (""" "`vlm/enrich.py`；请求账：位号 4094 + **引脚网格 1051** + 值 3492 + 型号 1021 = **9658 次 / 220 张 ≈ 44 次/张**",""",
     """ "⛔ **徽标已改「停用」**（本会话 19 个「★要新写」逐个对账）：值通道拼图已实测判负（见下），位号 / 型号的合并**还没量** ⇒ 不实现。用户口径是「如果这样没有带来额外的开销就算了」",
 "`vlm/enrich.py`；请求账：位号 4094 + **引脚网格 1051** + 值 3492 + 型号 1021 = **9658 次 / 220 张 ≈ 44 次/张**","""),

    (""" "★★ **2026-10：①②两条都实现了，实测「打平偏负」⇒ `FUSION_BOXSCAN` 默认 `0`**（代码保留，`=1` 可复现每一臂）",""",
     """ "⛔ **徽标已改「停用」** ｜ ★★ **2026-10：①②两条都实现了，实测「打平偏负」⇒ `FUSION_BOXSCAN` 默认 `0`**（代码保留，`=1` 可复现每一臂）",""")
]


def main():
    dry = "--dry" in sys.argv
    for t in TGT:
        h, n, _ = subst(t, PAIRS, write=not dry)
        print("命中 %d/%d  ->  %s" % (h, n, t))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
