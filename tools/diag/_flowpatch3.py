# -*- coding: utf-8 -*-
"""第三批补丁：把 fusion 层已落地的节点从「⚠ 未落地」改成实际情况。

按 **node id** 定位（不按下标），幂等（靠 MARK 判断）。

用法：python work\tools\diag\_flowpatch3.py [--dry]
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
HTML = ROOT / "work" / "flow" / "workflow.html"
MARK = "【第三批】"

PROG = {
    "priors": "**进度：✅ 已落地** —— `eda-pcb/fusion/priors.py`（含 `read_score` 读数模型、"
              "`fit_numbers`、连号滑窗与 `no_read` 闸）。",
    "pool": "**进度：✅ 已落地** —— `eda-pcb/fusion/claims.py`（`ClaimPool` / `Slot` / "
            "`Region` / `from_stage2` / `_add_vlm`）。",
    "solve": "**进度：✅ 已落地** —— `eda-pcb/fusion/solve.py`（`unique_family` / `match` "
             "贪心+增广路 / `gap_fill` / `open_questions` / `write_back`）。",
    "plausible": "**进度：✅ 已落地** —— `eda-pcb/fusion/plausible.py`（七类疑点；"
                 "`value_outlier` 按 `(ctype, unit)` 分组）。",
    "llm_judge": "**进度：✅ 已落地** —— `eda-pcb/fusion/llm_judge.py`。"
                 "实测：自由读数模式净 **−0.765**（22.9943→22.2291），改成"
                 "**候选二选一 + 拒收 OTHER/拒给占位符出题/不许给先验加冕**之后回到 **22.9943**（中性）。",
    "rescan": "**进度：✅ 已落地** —— `eda-pcb/fusion/rescan.py`（带上下文的二次扫描，"
              "`value_outlier` 默认关，`FUSION_RESCAN_VALUE=1` 才排）。",
    "iterate": "**进度：✅ 已落地** —— `eda-pcb/fusion/run.py` 的 `MAX_ROUNDS`（默认 2 轮）。",
    "conflict": "**进度：✅ 已落地（并进 `claims`）** —— 冲突就是 `ClaimPool` 里同一槽位的多来源候选；"
                "没有单独的 `conflict.py`。",
    "fixstr": "**进度：⛔ 已证否** —— 前提被测量否掉，见下。",
}

ADDENDUM = (
    "\n\n"
    + MARK
    + " **融合层落地实测（9 张 GT 子集，`work\\tools\\eval\\stage2_eval.py`）**\n"
    "- `stage2`（无 VLM）**19.9804**｜VLM 补强 **22.5020**｜融合（重装配）22.3306｜"
    "**融合（补丁式）22.5136**。\n"
    "- 加三条闸后逐级抬升：补洞不许盖过读数 → **22.7299**；"
    "先验不许盖过读数 → **22.9943**；LLM 裁决加闸 → **22.9943**（不再掉分）。\n"
    "- 结论：**融合层现在是「净 +0.49 over VLM、9 张零回归」**。\n"
    "- ★ 三条闸是同一条原理的三种写法：**读数是证据，先验/连号/裁决只是形状** —— "
    "形状可以帮**读不出号**的槽位补号，不能把已经读出来的号改掉，"
    "也不能借 VLM 的嘴给先验造的数加冕。\n"
    "- ★ 另一条：**位号在「键」上不在 `Name` 里**（`assemble.build_doc:534-538`，"
    "阻容的 `Name` 就是键），照着 `Name` 读位号会把 VLM 的改名整个丢掉。\n"
    "- 文件：`eda-pcb/fusion/` 下 `claims.py`/`priors.py`/`solve.py`/`llm_judge.py`/"
    "`plausible.py`/`rescan.py`/`run.py`/`doc.py`。**最终文档不再 `build_doc` 重装配，"
    "而是以 VLM 补强文档为底打补丁**（`doc.apply`）—— 重建一次就丢 `enrich` 对 "
    "`pins` 的补号/补名与它新增的元件。\n"
)


def main(argv):
    dry = "--dry" in argv
    txt = HTML.read_text(encoding="utf-8")
    # 图数据是 `<script type="application/json" id="graphdata">…</script>`
    m = re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)',
                  txt, re.S)
    if not m:
        print("找不到 graphdata")
        return 1
    data = json.loads(m.group(2))
    nodes = data["nodes"]
    n = 0
    for nd in nodes:
        nid = nd.get("id")
        if nid in PROG:
            note = nd.get("note") or ""
            if MARK in note:
                continue
            if PROG[nid].startswith("**进度：⛔"):
                nd["note"] = re.sub(r"\*\*进度：[^*]*\*\*", PROG[nid], note, count=1) + ADDENDUM
            else:
                nd["note"] = re.sub(r"\*\*进度：[^*]*\*\*（[^）]*）", PROG[nid], note, count=1) + ADDENDUM
            n += 1
    if not n:
        print("没有需要改的节点（已打过第三批）")
        return 0
    if dry:
        print("会改 %d 个节点：%s" % (n, ", ".join(
            nd["id"] for nd in nodes if MARK in (nd.get("note") or ""))))
        return 0
    bak = HTML.with_suffix(".html.bak3")
    if not bak.exists():
        bak.write_text(txt, encoding="utf-8")
    new = txt[:m.start(2)] + json.dumps(data, ensure_ascii=False, indent=1) + txt[m.end(2):]
    HTML.write_text(new, encoding="utf-8")
    print("改了 %d 个节点，写入 %s（备份 %s）" % (n, HTML, bak))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
