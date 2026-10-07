# -*- coding: utf-8 -*-
"""第四批补丁：把融合层的 **220 张口径**实测与「已接进生产链」写进 workflow.html。

按 **node id** 定位（不按下标），幂等（靠 MARK 判断）。

用法：python work\tools\diag\_flowpatch4.py [--dry]
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
HTML = ROOT / "work" / "flow" / "workflow.html"
MARK = "【第四批】"

NODE_IDS = ("priors", "pool", "solve", "plausible", "llm_judge", "rescan",
            "iterate", "conflict", "fixstr")

ADDENDUM = (
    "\n\n"
    + MARK
    + " **融合层 220 张全量实测 + 接线（当前状态）**\n"
    "- 分数阶梯（同一把尺，220 张，`work\\tools\\eval\\stage2_eval.py`）："
    "纯 `stage2` **25.9861** → VLM 补强 **28.4133** → 融合·首测 27.8939 → "
    "**融合·修 `Name` 顶位号后 28.4200** ✅（+0.0067，不降分）。\n"
    "- 融合层总耗时 **220 张 620 s**（`--workers 8` + `FUSION_LLM=0 FUSION_CROPFIX=0`）；"
    "带 LLM 轮每张约 50 s（LM Studio 串行化）⇒ 220 张约 4 小时，**已放弃全量 LLM 轮**。\n"
    "- ★★ **已接进生产链** `eda-pcb/submission/run.py`：新增 `-fusion/--fusion {off,on,llm}`，"
    "**默认 `on`**；导出段的优先序改成 **`fuse_dir > s3v_dir > s3_dir > s2_dir`**。"
    "端到端冒烟 3 张墙钟 22.8 s，`final_results/*_Pred.json` 与 `runs/fusion/*.json` **MD5 逐字节相同**、"
    "与 `runs/stage3` 不同 ⇒ 接线生效。\n"
    "- 两个**真 bug**（都已修）：① `eda-pcb/cvparts/hypo.py` 的 `Slot.add` 合并同值候选后"
    "**不重排**，而 `best()` 就是 `alts[0]` ⇒ 分数涨了名次没动，正确读法被埋；"
    "② `eda-pcb/fusion/claims.py` 的 `_add_vlm` 拿 `Name` 顶位号，而 `priors.DESIG_RE`"
    "**器件型号与位号同形**（`MRF521`/`MAX232`/`BCM2835` 全命中）⇒ case 0036 把带 7 个引脚的 `U1` "
    "改名成 `MRF521`，一张掉 16.7 分。改判据为「**只有当前键是占位符 `pNNN` 或无键时才允许 `Name` 顶位号**」。\n"
    "- 失败的消融（已回退，不要再试）：把 `priors` 分支 (b) `prior:run_free` 的 "
    "`score = best * 0.95` 改成「正分打折 / 非正分再低 0.01 / 过 `_cap`」⇒ **28.4200 → 28.3137（−0.11）**。"
    "该先验是**净帮忙**的：它造的「图上没有的号」在 `solve` 的连号竞争里替别的槽位守住了位置。\n"
    "- ★ 三条天花板（**融合层已到顶**）：① 融合层候选择优可回收 **6.0%** 缺失引脚；"
    "② 跨元件重归属 **2.7%**（纯交换 0）；③ 重读邻域全部 stage1 文字框只读出 **6.0%**（38/629）。"
    "⇒ **位号缺口 94% 是「位号文字根本没进文字框集合」，要动的是 `stage1` 的文字检测/分割。**\n"
    "- ★ 另一条独立发现：**`_submit_out220r36` 那条链交上去的就是纯 stage2（25.9861）** —— "
    "`runs/stage2` = `runs/stage3` = `final_results/*_Pred.json` 三份 MD5 逐字节相同，"
    "`run.log` 里完全没有 VLM 段（是用 `-vlm off` 跑的）。**单独把 `-vlm-name` 打开就值 +2.43 分。**\n"
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
        if nid in NODE_IDS:
            note = nd.get("note") or ""
            if MARK in note:
                continue
            nd["note"] = note + ADDENDUM
            n += 1
    if not n:
        print("没有需要改的节点（已打过第四批）")
        return 0
    if dry:
        print("会改 %d 个节点：%s" % (n, ", ".join(
            nd["id"] for nd in nodes if MARK in (nd.get("note") or ""))))
        return 0
    bak = HTML.with_suffix(".html.bak4")
    if not bak.exists():
        bak.write_text(txt, encoding="utf-8")
    new = txt[:m.start(2)] + json.dumps(data, ensure_ascii=False, indent=1) + txt[m.end(2):]
    HTML.write_text(new, encoding="utf-8")
    print("改了 %d 个节点，写入 %s（备份 %s）" % (n, HTML, bak))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
