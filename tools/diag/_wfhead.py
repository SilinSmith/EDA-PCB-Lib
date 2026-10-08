# -*- coding: utf-8 -*-
"""改写 workflow.html 内嵌 graphdata 的 `name` 与 `note` 开头（第八批）。

用法：
    python work\\tools\\diag\\_wfhead.py            # 预演
    python work\\tools\\diag\\_wfhead.py --write    # 落盘（先备份 .bakhead）

为什么单独一个脚本：`name` / `note` 是 graphdata JSON 里的长字符串，
用编辑器手改要处理一堆 `\\n` 转义；这里走 json 模块，改完再 `json.dumps`。
注意 `_wfbuild.py` 只重写 bullets/params，不碰 `name`/`note`，所以本脚本
必须在 `_wfbuild.py --write` **之前**跑，让 SRC 先带上新标题。
"""
import io
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _find(*rels):
    for r in rels:
        p = os.path.normpath(os.path.join(HERE, r))
        if os.path.exists(p):
            return p
    raise SystemExit("找不到：%s" % (rels,))


SRC = _find("../../flow/workflow.html", "workflow.html")
PAT = re.compile(r'(<script type="application/json" id="graphdata">\s*)(.*?)(\s*</script>)', re.S)

NAME = ("目标架构 · 证据层 → 约束求解 → 迭代重扫"
        "（实测：官方 10GT 9 张 58.63 / 派生 19 张 36.35 · 交付口径 VLM auto + fusion on · "
        "**关掉 LLM 轮还是 58.63，逐位相同**）")

BLOCK = """**★ 第八批：把 19 个「★要新写」逐个对账，并给 LLM 补上下文**

19 个 `kind='new'` 徽标逐个读代码对账（不是信进度叙述）。结论：
**16 个是「早就落地、只是徽标没改」**（`know` `pool` `solve` `plausible` `conflict` `llm_judge`
`rescan` `iterate` `cropfix` `audit` `own` `ev_bridge` `pincol` `ownnum` `boxpinprior` `gtcheck`），
**3 个是「量过、判负/证否」**（`fixstr` `batch` `boxscan`）—— 已全部去掉「要新写」徽标。

| # | 改了什么 | 文件 | 量出来的结果 | 结论 |
|---|---|---|---|---|
| 1 | **LLM 上下文块**（类型 / 同件别的字段 / 邻近元件 / 已定位号 / 已判可疑，+ 要求回一句 `依据=`） | `fusion/llm_judge.py` | VLM 关时官方 9 张 **52.48 → 53.27（+0.79）** | 采纳，默认开 |
| 2 | **整表裁决（contact sheet）真的被执行** | `fusion/llm_judge.py` | 4 张 **59.62 → 55.32（−1.07）**，且更慢（160.5 s vs 83.6 s） | **判负**，`SHEET_MIN=9999` 默认关 |
| 3 | **交付口径三臂**：LLM 关 / LLM 开 / LLM 开+上下文 | `fusion/run.py` | 三个臂**全是 58.63 且逐位相同** | LLM 轮在交付口径下值 **0.00**，却要 **+1.8 h/220 张** |
| 4 | `judge` / `rescan` 各自包 `try` | `fusion/run.py` | `judge` 一抛异常该 case 原本**一份融合产物都没有** | 采纳 |
| 5 | 收敛判据用**本轮结束**的题面（`FUSION_CONV_EXACT=1`） | `fusion/run.py` | case 1 实为 `n_q_next 15`，旧判据看的是上一轮 | 采纳 |
| 6 | 位号取景框取**最高分**那个（`FUSION_REGION_TOPK=1`） | `fusion/claims.py` | `stage2` 从不保证 `desig_slots` 按分降序 | 采纳 |
| 7 | `solve.gap_fill` 的负分跳过**留计数** | `fusion/solve.py` | 原为静默 `continue` | 采纳（只加计数） |

**★ 第八批的方法学（比上表更值钱）**

1. **「开关开着」不等于「这条路被执行过」。** `SHEET_ON` 默认 `1`、`_sheet_image` /
   `_known_designators` / `judge_sheet` 全是为整表写的，**但入口是
   `if mode == "sheet" and SHEET_ON:`，而 `MODE` 默认 `"auto"`** ⇒ 那条路一次都没跑过。
   「没有上下文环境」这句话因此有了代码级坐标。**默认值 ≠ 生效值**：双开关（允许 + 选择）
   两个都要看。尺子：`python tools/diag/_jrep.py --out <产物目录> [--rows]`。
2. **提示词改动必须回读「模型到底收到了什么」。** 分不变也可能是真的生效了。
   `ctx=1` 时 **12/12 行都有 `依据=…`**，`ctx=0` 时 **0/12**。只报分数说不清为什么变，
   只报 `why` 说不清值不值 —— **两样都要**。
3. **「LLM 有没有用」必须先给它上下文，再问。** 本轮第一次给出有前提的答案：
   在 VLM `auto` 的交付口径下，整个 LLM 轮 **+0.00**（三臂逐位相同）；在 VLM 关时 **+0.79**。
   ⇒ 它是在**没有 VLM** 时的补强，不是交付口径的增量。
4. **审计报告里的「这里有个 bug」四个字不等于这里真有 bug。** 一份审计说
   `claims.py` 的 `pinpts.get(cid)` 与 `doc["pins"]` 的键对不上；实测
   `_pinbox.py`：融合读的是 **stage1** 文档，`pins` 按 `cid` 记（case 1 = 22/22/22 全中），
   按**位号**记键的是 **stage2** 文档 —— 审计把两份文档搞混了，
   照它改**反而会引入 bug**。**复现不出来就不是 bug。**
5. **一个「零效果」的改动，先问它有没有可赢空间。** `own.assign` 端到端逐位相同，
   因为一对一不变式**改造前就成立**；官方 9 张 124 颗里失配 20 颗，而其中只有
   **1 颗**的候选池里有正确读法。先量天花板，再决定动不动手。

---

"""


def main():
    write = "--write" in sys.argv
    t = io.open(SRC, encoding="utf-8").read()
    m = PAT.search(t)
    if not m:
        raise SystemExit("找不到 graphdata 块")
    d = json.loads(m.group(2))
    old_name = d.get("name") or ""
    d["name"] = NAME
    note = d.get("note") or ""
    if "★ 第八批" in note:
        print("note 里已经有第八批，跳过插入（只改 name）")
    else:
        d["note"] = BLOCK + note
    print("name:")
    print("  -", old_name[:80])
    print("  +", NAME[:80])
    print("note 长度：%d → %d" % (len(note), len(d["note"])))
    if not write:
        print("[预演] 加 --write 才落盘")
        return
    shutil.copy2(SRC, SRC + ".bakhead")
    body = json.dumps(d, ensure_ascii=False, indent=1)
    io.open(SRC, "w", encoding="utf-8", newline="").write(
        t[:m.start(2)] + body + t[m.end(2):])
    print("已写入", SRC, os.path.getsize(SRC), "B（备份 .bakhead）")


if __name__ == "__main__":
    main()
