# -*- coding: utf-8 -*-
"""第六批 workflow.html 补丁：把「定点档」实测结论写进 cropfix 节点，并撤掉 boxscan 里那条错判。

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_flowpatch6.py [--dry]

做法：只替换 `note` 这一个 JSON 字符串，文件其余部分逐字节不动。
幂等：带 `MARK`，重复跑不会重复追加；先备份 `workflow.html.bak6`。
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "work" / "flow" / "workflow.html"
BAK = ROOT / "work" / "flow" / "workflow.html.bak6"
MARK = "【第六批·定点档】"

CROPFIX_ADD = """
 |  | 【第六批·定点档】24 张定点档（`work\\tools\\eval\\tiers.json` 的 `quickset`）**只开 cropfix、LLM 轮关**：
得分 **25.4434**，与基线**逐点完全相同**（component 0.4652 / pin 0.1597 / netH 0.2026 / net_line 0.0404）。
但报告 `_fusion_report.json` 显示 **21/24 张 `n_sus > 0`、96 个可疑槽位全部真跑过**（每槽位都要发 LLM 请求，24 张共 28.7 s），
而 **`n_gain = 0`** —— 一次都没有被采纳。
⇒ 闭环**接线正确、零产出**：文本筛挑出来的槽位本来就「没被切」（模型也答 `none`，与 `0001/0005` 上「增益 0」一致）；
而真正的盲区（`C151 → C51` 这类漏字）**不在文本筛的射程内**（前缀对、没空格、结尾 `1` 前面是数字）。
★ 顺带修掉一个**误导性计数**：`cropfix.py:402` 原先把 `rep["n_fix"] += 1` 写在 `if` **之外**
⇒ `n_fix` 实际等于「跑过的槽位数」，旧报告里 `n_sus == n_fix`（`2/2`、`8/8`、`7/7`、`5/5`）正是这个 bug 的指纹，
看起来像「凡是可疑的都修好了」。现已改成：`n_proc` = 跑过的槽位数、`n_fix` = 重读给出**不同**答案的、`n_gain` = 被采纳的。
自检 `cd eda-pcb && python -m fusion.cropfix --selftest` → **16 项全部通过**。
"""

BOXSCAN_OLD_3 = """3. 顺带修 `claims.py:611-614`：`pinpts.get(cid)` 用的是 `cid`，而 `doc["pins"]` 的键是**位号**（`U3`）不是 `cid`（`c0003`）
   ⇒ 永远 `None` ⇒ **每个引脚取景框都是 `box=None`**（`claims.py:630-633`），引脚槽位拿不到图。"""

BOXSCAN_NEW_3 = """3. ~~顺带修 `claims.py:611-614`~~ —— **已撤：我先前判错了，那不是 bug。**
   `from_stage2(case, doc, assigned, ...)` 收到的 `doc` 是 **stage1 文档**（`run.py:107 doc = _load(s1d / "%s.stage1.json")`），
   而 **stage1 的 `pins` 就是按 `cid` 键的**（实测 `work\\tools\\diag\\_s1pins.py`：顶级键 `c0000…`，
   样例 `c0000 -> [[1158.2, 528.0], [1158.2, 542.0], [1158.2, 557.0]]`，83 个元件全部有点集）
   ⇒ `pinpts.get(cid)` 正常命中，引脚槽位**拿得到**取景框。"""


def json_str_span(s, start):
    """s[start] == '"'；返回 (开引号下标, 闭引号下标+1)，正确跳过转义。"""
    i = start + 1
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return start, i + 1
        i += 1
    raise ValueError("JSON 字符串没有闭合")


def note_span(s, nid):
    i = s.index('"id": "%s"' % nid)
    j = s.index('"note": ', i) + len('"note": ')
    if s[j] != '"':
        raise ValueError("节点 %s 的 note 不是字符串" % nid)
    a, b = json_str_span(s, j)
    return a, b


def get_note(s, nid):
    a, b = note_span(s, nid)
    return json.loads(s[a:b])


def set_note(s, nid, new):
    a, b = note_span(s, nid)
    enc = json.dumps(new, ensure_ascii=False)
    return s[:a] + enc + s[b:]


def main():
    dry = "--dry" in sys.argv
    s = P.read_text(encoding="utf-8")
    before = len(s)

    # ---- ① cropfix：追加定点档结论 ----
    n = get_note(s, "cropfix")
    if MARK in n:
        print("cropfix 已带 %s，跳过" % MARK)
    else:
        s = set_note(s, "cropfix", n + CROPFIX_ADD)
        print("cropfix  note 追加 %d 字符" % len(CROPFIX_ADD))

    # ---- ② boxscan：撤掉那条错判 ----
    n = get_note(s, "boxscan")
    if BOXSCAN_OLD_3 in n:
        s = set_note(s, "boxscan", n.replace(BOXSCAN_OLD_3, BOXSCAN_NEW_3))
        print("boxscan  note 撤掉 claims.py:611 错判")
    else:
        print("boxscan 未找到旧文本（可能已改过），跳过")

    # ---- ③ boxscan 的 params 里也有一条同样的错判 ----
    bad = "claims.py:611 用 pinpts.get(cid)，而 doc['pins'] 的键是位号 ⇒ 引脚取景框恒为 None"
    if bad in s:
        s = s.replace(bad, "（已撤）claims.py:611 不是 bug：stage1 的 pins 本来就按 cid 键")
        print("boxscan  params 里的错判已撤")
    else:
        print("boxscan  params 里的错判未找到（可能已改过），跳过")

    # ---- 复验 ----
    m = re.search(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', s, re.S)
    d = json.loads(m.group(2))
    print("复验：节点 %d 个、连线 %d 条、JSON 可解析" % (len(d["nodes"]), len(d.get("links") or [])))
    n = get_note(s, "boxscan")
    assert "已撤" in n and "box=None" not in n, "boxscan 里还残留错判"
    print("复验：boxscan 已无 `box=None` 错判")

    if dry:
        print("[--dry] 不写盘")
        return
    if not BAK.exists():
        shutil.copy2(P, BAK)
        print("备份 → %s" % BAK)
    P.write_text(s, encoding="utf-8", newline="")
    print("写入 %s：%d → %d 字节" % (P, before, len(s)))


if __name__ == "__main__":
    main()
