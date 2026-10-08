# -*- coding: utf-8 -*-
"""改写 workflow.html 内嵌 graphdata 的 `name` 与 `note` 开头（第十批）。

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
        "**★ 引脚通道三分账：GT 的 120 根 IC 引脚我们只发 75；「读法全对」的上界只有 +4.60**）")

BLOCK = """**★ 第十批：引脚通道逐条审计 ——「不产出 / 点错 / 读错」三分账**

第九批定了「引脚 + 网络 = 60% 权重、同一个根因」。这一批把这个根因**拆到根数**，
结论是：**一半的缺口跟识别模型无关**（读对了号却因为没名字整条不发），
而**就算识别全对，也只值 +4.60 分**。

| # | 查了什么 | 文件 · 工具 | 量出来的结果 | 结论 |
|---|---|---|---|---|
| 1 | 官方档的引脚键形状 | `_gtpin10.py` | 10GT 的 **136 根键 100% 是纯数字**、`pinname` 全是真信号名 | 官方档的「引脚号」**可赢**（派生档不是） |
| 2 | 引脚账本 | `_pinledger.py` | GT 的 IC(box) 引脚 **120 根，我们只发 75**；缺的 45 = **号读对了却被丢 23（51%）** + 号读错 22 + 附近没槽位 6 | **一半的缺口是策略，不是 OCR** |
| 3 | 按**位置**绑定的对账 | `_ocrstat.py` | 45px 内有槽位 **135/136 = 99%**；号 完全对 86 / 整串读错 26 / 空 14 / 截断 7；名 完全对 53 / 空 40 / 整串读错 25 / 截断 15 | 几何几乎不缺，分丢在**字符串**上 |
| 4 | 截断是不是框切了字 | `_ocrstat.py` + `_pincropgt.py` | `39→3` 框宽 **24px**、`113→13` **34px**、`115→15` **35px**（都够装） | **是 CRNN 少读一个字符**，不是检测 |
| 5 | **读法全对的写实上界** | `_ocroracle.py` | 官方 9 张 **58.6304 → 61.2888（名全对）→ 63.2307（号+名全对）= +4.6003**；tp 引脚 37→52 | **动权重之前先看这个数** |
| 6 | 模型现状 | `vlm/client.py:4`、`_flowpatch.py:20` | CRNN 真值紧框 **0.843** / stage1 紧框 **0.197** / 外扩 +3 → **0.654** | 瓶颈在**裁剪**，不在权重 |
| 7 | ⛔ 把引脚点吸附回框边 | `_pinsnap.py --split 10GT` | T0 原样 **28** → T1/T2/T4 = **0**、T3 外扩 8 才 15、T5 最优 (1,1) = 29 | **官方档也判负**，不用实现 |
| 8 | ⛔「GT 点 = 文字框心」「点在框边线上」 | `_pintext.py` | 号框心 7.4% / 名框心 **0.0%** / 中点 1.6%；落在 bbox 边线 **0%**（到边线 p50 12px） | 两个假说都**证伪**；真实形态是「框边往外每边一个固定桩距」 |

**★ 第十批的根因链（官方 9 张，交付产物 `work\\_fast_out_reg9`）**

- 引脚分项 `tp 37 / fp 38 / fn 83`；`pin.missing 51`、`name_mismatch 16`、`position_mismatch 16`。
- **GT 的 IC(box) 引脚 120 根，我们只发出 75 根**（缺 45）：
  - **23 根是「号已经读对了，但名字没读到 ⇒ `stage2/assemble.py:759` 的 `BOX_PIN_NEED_NAME`（默认开）整条不发」**；
  - 22 根是号读错（`113→13`、`39→3`、`48→46`、`68→88`）；
  - 6 根是附近 45px 内没有任何号槽位。
- 「读法全对」的上界：**+4.60**（名 +2.66、号再 +1.94），前提是 IC 的 75 根预测引脚里
  **点已对 54 根（72%）**、点没对 21 根 —— **点没对的那些，识别再准也拿不到**。
- 训练集 `textrec\\data\\manifest.jsonl` **388518 条 = 合成 381876 + 真实 6642**，
  而真实那份 `split="real"` **只验证、绝不训练**（`textrec\\manifest.py:32`）；
  真实 val（785 条、batch 256 + 截断解码）exact **0.6866 / 位号 0.8850**；混训要显式 `--include-real`。

**★ 第十批的方法学**

1. **动权重之前先算天花板。** `_ocroracle.py` 给的是「这条路**全部**能值多少」（+4.60），
   先有它，才谈得上「重训值不值」。上界不是承诺，但它是**投入的上限**。
2. **按位置查，不要按号查。** 同一批缺号：按号查只能找到 94/136（读错的被算成「没槽位」），
   按位置查是 135/136 ⇒ 几何不缺。**绑定方式换了，结论就换了。**
3. **把「读错 / 没产出 / 没检测到」分成三笔账再决定投哪。** 本轮缺的 45 根里
   23 根（51%）只要**改一行开关**，去重训识别器等于白跑。
4. **A/B 的臂要先证明它和基线不一样。** `_vlmab2.py` 的 `"pins"` 臂 `pin=True` 根本没设
   ⇒ 等于「拿基线比基线」。凡是量新通道，先让脚本把每臂的开关 `print` 出来。

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
    if "★ 第十批" in note:
        print("note 里已经有第十批，跳过插入（只改 name）")
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
