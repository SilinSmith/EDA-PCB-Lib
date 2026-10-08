# -*- coding: utf-8 -*-
"""改写 workflow.html 内嵌 graphdata 的 `name` 与 `note` 开头（第九批）。

用法：
    python work\\tools\\diag\\_wfhead.py            # 预演
    python work\\tools\\diag\\_wfhead.py --write    # 落盘（先备份 .bakhead）

为什么单独一个脚本：`name` / `note` 是 graphdata JSON 里的长字符串，
用编辑器手改要处理一堆 `\\n` 转义；这里走 json 模块，改完再 `json.dumps`。
注意 `_wfbuild.py` 只重写 bullets/params，不碰 `name`/`note`，所以本脚本
必须在 `_wfbuild.py --write` **之前**跑，让 SRC 先带上新标题。

★ 本脚本只有从 `work\\tools\\diag\\` 下跑才解析得到 `work\\flow\\workflow.html`；
  从 `eda-pcb\\tools\\diag\\` 跑会 SystemExit（`_find` 找不到）。改完记得
  `Copy-Item work\\tools\\diag\\_wfhead.py eda-pcb\\tools\\diag\\_wfhead.py -Force`。
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
        "**★ 全量 220 stage1 从「单张黑洞 315 s」修到 220 张 416.5 s（1.89 s/张）**）")

BLOCK = """**★ 第九批：全量 220 的阶段级归因 —— 分到底丢在哪一段**

第一次把 220 张按**阶段**摊开看，而不是只看总分。结论：**引脚 + 网络占 60% 权重、
共用同一个根因（IC 引脚我们没有）**，而位号（元件分）根本不是最大的洞。

| # | 改了什么 / 查了什么 | 文件 · 工具 | 量出来的结果 | 结论 |
|---|---|---|---|---|
| 1 | **stage1 色相询问有界化**：先缩到最大边 1024，独立超时 30 s，连续失败 3 次熔断 | `stage1/wires.py` | `0003` 单张 **315.73 s → 8.3 s**；220 张 stage1 **416.5 s** | 采纳（默认仍是 `hint`） |
| 2 | 按**官方权重**算可回收空间 | `schematic_parser/metrics.py:11` | 网络 **+27.9** ＞ 引脚 **+23.1** ＞ 元件 +11.4 ＞ 导线 +9.7 | 先修引脚/网络 |
| 3 | **上帝键实验**（位号按中心距强行配对正确） | `_keyoracle.py --n 60` | 21.53→**27.19**，但**引脚只 +0.0112**、元件 +0.1310（那是强行配对的作弊分） | **否掉「是位号/键的问题」** |
| 4 | 放宽 `pin_name` 到框外（第一版，仍在引脚号**之前**跑） | `stage2/assign.py` | 19 张 **30.81 → 30.09**（引脚 0.1987→0.1756、tp 44→42） | **判负** |
| 5 | 改成 **late pass**：引脚号那一轮跑完再补空名字槽 | `stage2/assign.py` | 合成自检 **15 项**两种开关状态都过；真实 A/B 待显存 | 采纳（`STAGE2_PINNAME_OUTSIDE` 默认 **0** = 关） |
| 6 | `_patch9`/`_wfhead`/`_wfkind` 等工装的**同名两份**坑 | `docs/DIAGNOSTICS.md` | 改一份、重建后输出字节数一模一样 | 已写成纪律 |

**★ 第九批的根因链（硬数据，全在 10/6 的 CV-only 产物上量）**

- GT 的 IC 引脚名 **86.5% 是真信号名**（`CS / D1/DO / CLK / IOVDD / XTAL2`），纯数字只 13.5%。
- 我们产出的引脚里 **78.2% 的 `pinname` 就是引脚号本身**（回退点 `assemble.py:761 nm = r['nm'] or num`）。
- GT 引脚 `point` **100% 在元件 bbox 之外**（中位 9 px / p90 35 px / p99 67 px）。
- 2220 根 GT IC 引脚里 **64.9% 在配对上的元件键下一根都没有**；能比的 780 根里名字逐字符相等只 **3.1%**。
- ⇒ **位号全修对也救不回引脚**（上帝键只值 +0.0112）。要动的是**引脚检出率 + 名字读出**。

**★ 第九批的方法学（比上表更值钱）**

1. **「慢」先量余额，再上 profiler，最后才怀疑硬件。** `total_s` 减去 `timing` 里各段之和，
   差额只可能来自**进程外**。这一条免费；一次 cProfile（`work/tools/diag/_prof1.py 0003`）
   就落到 `wires.py:230 → vlm/client.py:161 → urlopen → socket.recv_into` 的 303.4 s。
2. **任何发到进程外的调用都要自带超时、上界与熔断，失败必须计数。**
   `ask()` 把一切异常吞成 `"ERR:..."` **字符串正常返回** ⇒ 失败不会以报错的形式出现，
   只会以「这里有段时间不见了」的形式存在。全局 `VLM_TIMEOUT=600` + 重试 4 次 + **整张原图**
   三处叠加，把一次「问个颜色」变成单张 315 s。
3. **先用上帝视角量天花板，再决定动不动手。** 上帝键给出 +5.66，但拆开看引脚只占 +0.0112
   ⇒ 花在位号绑定上的每一分钟都是在修一个**不是根因**的东西。
4. **顺序就是优先级。** `pin_name` 排在 `pin_number` **之前**，而 `taken_box` 规定同一块文字
   只能用一次 ⇒ 一旦放宽 `pin_name` 的取景范围，它先抢走文字、**引脚号被饿死**。
   要放宽，只能改成**事后补空槽**（late pass），不能只改距离阈值。
5. **负分候选上不能用除法做归一。** `score / (1 + d/20)` 在 `score < 0`（框外候选的常态）时
   **越远的分越高**，实测把 `VDDIO` 与 `VSSA` 两个名字对调；换成 `score - 0.035*d`。

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
    if "★ 第九批" in note:
        print("note 里已经有第九批，跳过插入（只改 name）")
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
