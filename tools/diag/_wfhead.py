# -*- coding: utf-8 -*-
"""改写 workflow.html 内嵌 graphdata 的 `name` 与 `note` 开头（第七批）。

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
        "同一批官方 9 张在本会话起点是 44.16）")

BLOCK = """**★ 第七批（本会话后续）**

**交付口径成绩：官方 `10GT` 9 张 44.16 → 58.63、派生 19 张 34.69 → 36.35**
（`tools/eval/fast.py`，不加 `--no-vlm`、不设任何 `$env:` ⇒ VLM `auto` + `-fusion on` + `-mode off`）。

| # | 改了什么 | 文件 | 官方 9 | 派生 19 | 结论 |
|---|---|---|---|---|---|
| 5 | 引脚号裁片**描边剔除**（绿色 no-connect 叉与数字同框被读成 `K4`/`L14`） | `stage2/text.py` | 56.71 → **57.54** | 35.39 → **35.75** | 采纳，只对 `pin_number,pin_name` |
| 6 | 导线端点**同网吸附**（`STAGE2_EDGE_SNAP=20`，只吸近似垂直的邻居） | `stage2/assemble.py` | 57.54 → **58.63** | 35.75 → **36.35** | 采纳，增量全在 netL |
| 7 | 形状惩罚真的改分（`FUSION_SHAPE_W`） | `fusion/priors.py` | 0 | −0.57 | **判负**，默认回 0.0 |
| 8 | boxscan（给零候选元件建框去问一遍） | `fusion/claims.py` | −0.01 | +0.05 | **判负**，默认回 0 |
| 9 | `own.assign` 全局一对一归属（`STAGE2_OWN_ASSIGN`） | `stage2/assign.py` | 0 | 0 | **零效果**，默认 0 |
| 10 | GND 系引脚并成一条网 | — | — | — | **证伪**（官方 GT 不这么编码） |

**★ 第七批的方法学（比上表更值钱）**
1. **「A/B 有差别」不等于「你改的那条分支生效了」。** boxscan 第一版实测有差，
   但把 `_fusion_report.json` 的 `notes` 打出来才看到每张只建了 1 个槽位 ——
   分差是那 1 颗元件的读数对错，不是 boxscan 整体的效果。改完先证明分支执行了、
   执行了几次，再解释分差。
2. **对照臂必须配对口径。** boxscan 在 VLM 开着时基本是 no-op
   （`_add_vlm` 已把 `comp.<cid>.key` 覆盖到 82/83、12/12、22/22），只能在
   `--no-vlm` + `FUSION_LLM=1` 这个口径量；交付口径的 58.63 / 36.35 是另一张表。
3. **归属类改动可能先天没有空间。** `own.assign` 端到端两档**逐位相同**，
   因为 `owned` 本来就是一份划分 —— 一对一不变式**改造前就成立**。
   先量天花板（`tools/diag/_ownasgn.py --headroom --gt`：官方 9 张 124 颗里
   读对 104 / 空 11 / 错 9，而这 20 颗失配里只有 **1 颗**的候选池里已有正确读法），
   再决定要不要改。
4. **一个数字可以在两种跑法下不同。** 曾出现同参数下 9 张单跑 51.06 vs
   19 张合跑的子集 51.74（差 <1 分）。**小于 ~1 分的差值采信前必须重复测量**
   （本轮 `reg9` 与 `reg19` 印出的 10GT 子集已逐位一致）。
5. **改默认值之前先看它有没有被执行。** `fusion/run.py` 的 `_fusion_report.json`
   原先只在它自己的 `main()` 里落盘，而真实提交路径直接调 `run_case` ⇒
   提交产物里**根本没有逐轮明细**。已在 `submission/run.py` 补上，并在
   `fusion/run.py` 的 `rep` 里加了 `notes`。

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
    if "★ 第七批" in note:
        print("note 里已经有第七批，跳过插入（只改 name）")
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
