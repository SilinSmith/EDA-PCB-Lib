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
        "**★ 引脚通道：写进文档却没人读的真 bug，修掉后官方 9 张 +2.88（派生 220 −0.07、"
        "「只补号」零回归）**）")

BLOCK = """**★ 第十批：引脚通道逐条审计 ——「不产出 / 点错 / 读错」三分账**

第九批定了「引脚 + 网络 = 60% 权重、同一个根因」。这一批把这个根因**拆到根数**，
结论是：**一半的缺口跟识别模型无关**（读对了号却因为没名字整条不发），
而**就算识别全对，也只值 +4.60 分**；但要小心：**「缺口在这一类」不等于「关掉那个开关就能拿回来」**
（第 9 行那个 A/B 就是反例）。

| # | 查了什么 | 文件 · 工具 | 量出来的结果 | 结论 |
|---|---|---|---|---|
| 1 | 官方档的引脚键形状 | ~~`_gtpin10.py`~~ → **`_pinconv.py`（第十一批第 6 行已更正）** | 10GT 的 21 个 IC 里 **20 个（95.2%）的键就是 1..N**、`pinname` 纯数字 **0.0%**；只有 `10GT/1/U1`（n=26）是物理号 | 官方档的号是**小顺序号**、缺口在**名字**上（派生档相反） |
| 2 | 引脚账本 | `_pinledger.py` | GT 的 IC(box) 引脚 **120 根，我们只发 75**；缺的 45 = **号读对了却被丢 23（51%）** + 号读错 22 + 附近没槽位 6 | 缺口**分成三笔账**，其中一半不在 OCR 上 |
| 3 | 按**位置**绑定的对账 | `_ocrstat.py` | 45px 内有槽位 **135/136 = 99%**；号 完全对 86 / 整串读错 26 / 空 14 / 截断 7；名 完全对 53 / 空 40 / 整串读错 25 / 截断 15 | 几何几乎不缺，分丢在**字符串**上 |
| 4 | 截断是不是框切了字 | `_ocrstat.py` + `_pincropgt.py` | `39→3` 框宽 **24px**、`113→13` **34px**、`115→15` **35px**（都够装） | **是 CRNN 少读一个字符**，不是检测 |
| 5 | **读法全对的写实上界** | `_ocroracle.py` | 官方 9 张 **58.6304 → 61.2888（名全对）→ 63.2307（号+名全对）= +4.6003**；tp 引脚 37→52 | **动权重之前先看这个数** |
| 6 | 模型现状 | `vlm/client.py:4`、`_flowpatch.py:20` | CRNN 真值紧框 **0.843** / stage1 紧框 **0.197** / 外扩 +3 → **0.654** | 瓶颈在**裁剪**，不在权重 |
| 7 | ⛔ 把引脚点吸附回框边 | `_pinsnap.py --split 10GT` | T0 原样 **28** → T1/T2/T4 = **0**、T3 外扩 8 才 15、T5 最优 (1,1) = 29 | **官方档也判负**，不用实现 |
| 8 | ⛔「GT 点 = 文字框心」「点在框边线上」 | `_pintext.py` | 号框心 7.4% / 名框心 **0.0%** / 中点 1.6%；落在 bbox 边线 **0%**（到边线 p50 12px） | 两个假说都**证伪**；真实形态是「框边往外每边一个固定桩距」 |
| 9 | ⛔ **关掉 `BOX_PIN_NEED_NAME` 去收那 23 根** | `_gateab.py` + `stage2_eval.py` | **官方 9 张 52.6510 → 52.4987（−0.15）**、`name_mismatch` 16→41、tp **没多**；**全量 220 反而 +0.10**（26.0962 → 26.1978） | **闸门保护的是精度**，效果还**分档相关**；要补的是**名字** |

**★ 第十批的根因链（官方 9 张，交付产物 `work\\_fast_out_reg9`）**

- 引脚分项 `tp 37 / fp 38 / fn 83`；`pin.missing 51`、`name_mismatch 16`、`position_mismatch 16`。
- **GT 的 IC(box) 引脚 120 根，我们只发出 75 根**（缺 45）：
  - **23 根是「号已经读对了，但名字没读到 ⇒ `stage2/assemble.py:759` 的 `BOX_PIN_NEED_NAME`（默认开）整条不发」**
    —— 但**关掉它并不赚**（见上表第 9 行）：名字只能回退成引脚号，官方档的 GT 引脚名
    86.5% 是真信号名 ⇒ 多发出来的引脚全变 fp；
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
3. **把「读错 / 没产出 / 没检测到」分成三笔账再决定投哪** —— 但**分完账不要顺手宣布
   「改一行开关就能拿回来」**：那 23 根就是反例（第 9 行）。**「归到哪一类」和
   「关掉哪个开关」是两个命题，后一个必须单独 A/B。**
4. **A/B 的臂要先证明它和基线不一样。** `_vlmab2.py` 的 `"pins"` 臂 `pin=True` 根本没设
   ⇒ 等于「拿基线比基线」。凡是量新通道，先让脚本把每臂的开关 `print` 出来。
5. **能离线做的 A/B 就别占卡。** `assemble.build_doc(stage1, assigned, case)` 的两个输入
   都在盘上、装配开关是模块级全局 ⇒ `_gateab.py` 改一行全局就能切，
   **纯 CPU、秒级、可以和长跑并行**（本轮就是在全量 220 的 VLM 阶段同时跑完的）。
6. **开关的效果可能分档相关。** 同一个 `BOX_PIN_NEED_NAME=0`：官方 9 张 **−0.15**、
   全量 220 **+0.10**（派生档的引脚名常是数字）。**只在一档上量会得出相反的结论。**

**★ 第十一批：引脚通道端到端 ——「写进产物」不等于「有人读」**

第十批把缺口拆到根数、并量出「读法全对只值 +4.60」。这一批去拿那 23 根
（号读对、名字没读到的），结论是：**通道早就写对了，是融合层根本没读** ——
把「没人读」这个真 bug 修掉，官方 9 张 **58.6304 → 61.5112（+2.88）**。

| # | 查了什么 | 文件 · 工具 | 量出来的结果 | 结论 |
|---|---|---|---|---|
| 1 | VLM 引脚通道到底写出了什么 | `vlm/enrich.py`、`work\\_vlm_pinfix_full\\1.json` | 补强文档里 `pins[cid][slot]["number"]` **恒为 `None`**，号**只活在键名** `pin_<号>` 里、名在 `pinname`；`VLM_PIN_CHANNEL=1` 把号/名写进 `pins`（case 1 的 `U1` 21 个槽位，`pin_17 GND / pin_115 VDDC / pin_122 VDD`） | 数据**确实写进去了** |
| 2 | 写进文档 = 进得了交付吗 | doc 级 `stage2_eval` vs 融合级 `fusion.run`，**同一份产物** | doc 级 **60.7776** vs 融合级 **58.6304 = 与基线逐位相同**；根因：`fusion/claims.py:450 _add_vlm()` **只吃 `vlm["components"]`**，引脚是 `stage2/assemble.py:588 build_doc()` 用 `stage1`+`stage2/_assigned` **重新装配**的 | **「写进了产物」≠「有人读」** —— 这是真 bug，不是通道无效 |
| 3 | （中途 confound）拿错文档比 | `FUSION_VLM_DIR=work\\_vlm_pinfix` 得 **53.7317** | 那是 `_vlmab2` 的臂**没开 value/name 通道**（comp F1 掉回 0.8031） | **A/B 必须叠在同一份「交付自己的」文档上** |
| 4 | 修掉之后值多少 | `claims.py` (d) 段：点最近邻 **3px** 定槽位、`FUSION_VLM_PIN_SCORE=1.15`、`_pin_channel` 出处标记 | 官方 9 张 **58.6304 → 61.5112 = +2.88**；pin F1 0.3920→**0.4858**（tp 37→**45**、fp 38→31、fn 83→75）、netH 0.5355→**0.5493**、netL 0.1876→0.1930；失配 `pin.extra` **6→1**、`name_mismatch` 16→11、`missing` 51→45 | 引脚通道**真能赚**；`FUSION_VLM_PIN_SCORE` 在 0.90/1.30/1.60/2.50 四档**逐位相同**（槽位没有别的候选）⇒ 不用调 |
| 5 | 号与名各值多少 | `FUSION_VLM_PIN_NUM` / `_NAME` 四臂 | 只补号 **60.5816（+1.951，九张没有一张变差）** / 只补名 **59.9468（+1.316）** / 全开 **61.5112（+2.881）** / 空跑 **58.6304（0.000，逐位相同）** | **两半之和 ≠ 全开**：号赢 case 1/2/9、名赢 3/5/10，**case 9 全开只 +4.84 而只号 +9.39（名倒扣 4.55）** |
| 6 | GT 引脚键约定到底怎么定（**更正第十批第 1 行**） | `_pinconv.py` 全 split 普查 | 官方 10GT **21 个 IC 里 20 个（95.2%）的键就是 1..N**、IC `pinname` 纯数字 **0.0%**（唯一例外 `10GT/1/U1` n=26 物理号）；派生 `200_train_cases` 946 个 IC 只有 **22.6%** 顺序键、**50.9% 的引脚键 >8**、`pinname` 纯数字 **20.1%** | **官方档号小、缺名字；派生档号大、名字常是号** ⇒ 通道该不该开**按档定** |
| 7 | 全量 220 张上开不开 | 同口径 A/B（各 220 张、3 worker、~29 min） | **29.4819 → 29.4099（−0.072）**；按 split 拆开是**三档**：官方 `10GT` 10 张 **+2.59**、`10_GTcase` 10 张 **+0.76**、`200_train_cases` 200 张 **−0.25**（全量 42 张变好 / 63 张变差；最差 `0105 −12.11`、最好 `0036` +7.62） | **只亏在自动命名的那 200 张**（自证），两个 `10GT*` 集子都赚 |

**★ 第十一批的方法学**

7. **「写进了产物」不等于「有人读」。** 同一份产物跑两条口径（doc 级 / 交付级），
   差值就是「没人读」的量。凡是新写一个产物字段，**先问谁读它**。
8. **一条通道的两半要拆开各量一次。** 全开 +2.88 里号值 +1.95、名值 +1.32，
   但**两半之和 ≠ 全开**（名会覆盖「号纠正后本该跟着变」的名）⇒ 只有 2×2 四臂
   才看得出「只补号零回归」这条路；**空跑臂是必要的对照**（证明旋钮真的在起作用）。
9. **默认值按档定，不按均分定。** 同一个通道官方 `10GT` **+2.88**、`10_GTcase` **+0.76**、
   `200_train_cases` **−0.25**（220 张合计才 −0.07）⇒ 「谁开」比「开不开」更重要；
   而且 `200_train_cases` 自建的 GT 是**自证**的，别拿它当判据。
10. **平均分会把方向不同的档抵消掉。** 220 张合计 −0.07 里同时藏着 +2.59（10 张）与
   −0.25（200 张）。**报均分前先按 split 拆一次**，否则结论会被最大那一档的权重决定。

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
        # 已经插过一版第十批：切掉旧块再重新插，否则改过的 BLOCK 永远进不去。
        i = note.find("**★ 第十批：")
        j = note.find("\n---\n\n", i)
        if i < 0 or j < 0:
            raise SystemExit("note 里有「★ 第十批」但定位不到块的结尾，手动检查")
        j += len("\n---\n\n")
        print("替换已有第十批块：note[%d:%d]（%d 字）被换掉"
              % (i, j, j - i))
        note = note[:i] + note[j:]
        d["note"] = BLOCK + note
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
