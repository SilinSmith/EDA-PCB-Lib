# -*- coding: utf-8 -*-
"""把「第九批」的实测结论写进两份 `_wfbullets.py`（字面替换，可重复执行）。

用法：
    python work/tools/diag/_patch9.py            # dry-run，只报每条命中与否
    python work/tools/diag/_patch9.py --write

两份文件：
    work\\flow\\_wfbullets.py            ← `_wfbuild.py` 真正 import 的那份
    eda-pcb\\tools\\diag\\_wfbullets.py   ← 仓库快照
改完把 work 那份覆盖到 eda-pcb（见 --sync）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EDA = os.path.abspath(os.path.join(HERE, "..", "..", "..", "eda-pcb"))
CANON = os.path.abspath(os.path.join(HERE, "..", "..", "flow", "_wfbullets.py"))
SNAP = os.path.join(EDA, "tools", "diag", "_wfbullets.py")

# (名字, old, new)
PATCHES = [
    # ---------------------------------------------------------------- ev_wire
    ("ev_wire 默认值纠错（默认已是 hint，不是 dominant）",
     "⇒ 默认仍 `dominant`；`STAGE1_WIRE_COLOUR=dominant|chroma|off`",
     "⇒ **不走 chroma**；`STAGE1_WIRE_COLOUR=hint|dominant|chroma|off`（**默认 `hint`**）"),
    ("ev_wire + 色相询问黑洞（3 条）",
     ' "★ 官方评分里 `pin 0.25 + netH 0.35 + netLine 0.10 = 0.70` **全部以引脚点为最小单位**",\n],\n"ev_net": [',
     ' "★ 官方评分里 `pin 0.25 + netH 0.35 + netLine 0.10 = 0.70` **全部以引脚点为最小单位**",\n'
     ' "★ **按官方权重（`metrics.py:11`：元件 0.30 / 引脚 0.25 / 网络 0.35 / 导线 0.10），'
     '引脚 + 网络 = 60% 的权重共用同一个根因：IC 引脚我们没有。** 全量 CV-only 基线上算可回收空间：'
     '网络 **+27.9** ＞ 引脚 **+23.1** ＞ 元件 +11.4 ＞ 导线 +9.7",\n'
     ' "★ 硬数据（`_pinoracle.py` / `_gtpinplace.py` / `_pinname_stat.py` / `_gtpinname_stat.py`）：'
     '2220 根 GT IC 引脚里 **64.9% 在配对上的元件键下一根都没有**；有可比的 780 根里名字逐字符相等只 '
     '**3.1%（68 根）**；GT 引脚 `point` **100% 落在元件 bbox 之外**（中位 9px / p90 35px / p99 67px）；'
     '我们产出的引脚里 **78.2% 的 `pinname` 就是引脚号本身**（回退点 `assemble.py:761 nm = r[\'nm\'] or num`），'
     '而 GT 的引脚名 **86.5% 是真信号名**（纯数字名只 13.5%）",\n'
     ' "⛔ **上帝键实验否掉了「这是键的问题」**：把位号按中心距强行配对正确后 60 张 21.53→27.19（+5.66），'
     '但**引脚只 +0.0112**、元件 +0.1310（那 +0.1310 是强行配对的作弊分）⇒ 缺口在**引脚检出率 + 名字读出**，'
     '不在绑定/位号上（`_keyoracle.py --n 60`）",\n'
     ' "⛔ 第一版「放宽 `pin_name` 到框外、仍在引脚号之前跑」**判负**：19 张 30.81→30.09'
     '（引脚 0.1987→**0.1756**、tp 引脚 44→**42**）。机制：`pin_name` 排在 `pin_number` **之前**，'
     '抢走框外那块文字后 `taken_box` 禁止复用 ⇒ **引脚号被饿死**，丢的 tp 比补来的名字更贵",\n'
     ' "✅ 改成 late pass（`STAGE2_PINNAME_OUTSIDE`，默认 0 = 关）：引脚号那一轮**跑完之后**只补还空着的名字槽；'
     '闸门只压「适配度 × 置信度」**不压距离**（`_score` 的 `-0.035*d` 会让 30px 外的合法名字算成负分），'
     '且**拒绝纯数字文本**（GT 引脚名有 13.5% 本来就是数字，`nm = r[\'nm\'] or num` 的兜底已经填对）。'
     '合成自检 15 项在开关两种状态下都过",\n'
     '],\n"ev_net": ['),
    # ---------------------------------------------------------------- ev_net
    ("ev_net + 阶段账（3 条）",
     ' "⚠ 但全局改 chroma 会把彩色 Altium 打下去（**9 −6.88 / 2 −3.20**）⇒ **不走 chroma**；'
     '`STAGE1_WIRE_COLOUR=hint|dominant|chroma|off`（**默认 `hint`**）",\n],\n"know": [',
     ' "⚠ 但全局改 chroma 会把彩色 Altium 打下去（**9 −6.88 / 2 −3.20**）⇒ **不走 chroma**；'
     '`STAGE1_WIRE_COLOUR=hint|dominant|chroma|off`（**默认 `hint`**）",\n'
     ' "★ `hint` = 先问一次 VLM「这张图的导线是什么颜色」，写进 `pskel.HINT_HUE`，'
     '**只在主色相是红系时**才覆盖 dominant。账面收益很小（19 张 24.87 vs dominant 24.81、'
     '导线 F1 0.0146→0.0201、官方 10GT 40.13→40.14），但**代价曾经是无界的**：",\n'
     ' "⛔ **这一次询问曾经是整个管线的黑洞**：对**整张原图**发、用全局 `VLM_TIMEOUT=600`、重试 4 次，'
     '而 `vlm/client.py:161` 又把一切异常吞成 `\\"ERR:...\\"` **字符串正常返回** ⇒ 失败不报错、'
     '只表现为「这张图莫名多了 5 分钟」。`0003`（4491×3173）单张 **315.73 s**，而同一张 `timing` 里'
     'yolo+cv+wires+rays+text+bigic+pins 加起来只有 **4.31 s**；cProfile 定位到 '
     '`wires.py:230 _hint_wire_hue → vlm/client.py:161 ask → urlopen → socket.recv_into`（303.4 s）",\n'
     ' "✅ 已修：先缩到最大边 `STAGE1_WIRE_HUE_MAXSIDE=1024` + 独立超时 `STAGE1_WIRE_HUE_TIMEOUT=30`'
     '（不再用全局 600）+ 连续失败 `STAGE1_WIRE_HUE_MAXFAIL=3` 次**熔断** ⇒ `0003` **8.3 s**；'
     '220 张 stage1 一共 **416.5 s**（1.89 s/张）。220 张里 **168 张**会触发（52 张灰度图跳过），'
     '>2000px 的 **23 张**，最大 **17.4 MPx**；整图首发实测 **121.98 s 换来 `HTTP 400`**，'
     '缩到 1024 后 **0.55 s**",\n'
     ' "★ 教训：**任何发到进程外的调用都要自带超时、上界与熔断，失败必须计数。** '
     '失败一旦被吞成字符串，它就不会以报错的形式出现，只会以「这里有段时间不见了」的形式存在",\n'
     '],\n"know": ['),
]

TAIL_PATCH = (
    "全量 220 阶段账（写进 SUB 摘要）",
    '"llm_judge": "llm_judge.py：候选二选一 + **上下文块（官方 9 张 +0.79）**；整表判负",',
    '"llm_judge": "llm_judge.py：候选二选一 + **上下文块（官方 9 张 +0.79）**；整表判负",\n'
    '"ev_wire": "wires.py：线过碎片化 → 规则 5；★ 色相询问曾是无界黑洞（315s→8.3s）",\n'
    '"ev_pin": "引脚：pin+netH = 60% 权重同一根因；上帝键证明不是键的问题 → 修名字与产出率",',
)


def apply(path, write):
    if not os.path.isfile(path):
        print("!! 找不到 %s" % path)
        return 0
    src = open(path, encoding="utf-8").read()
    hits = 0
    for name, old, new in PATCHES + [TAIL_PATCH]:
        n = src.count(old)
        if n == 0:
            print("  MISS  %s" % name)
            continue
        if n > 1:
            print("  !! 命中 %d 次，拒绝改：%s" % (n, name))
            continue
        src = src.replace(old, new, 1)
        hits += 1
        print("  ok    %s" % name)
    if write and hits:
        open(path + ".bak9", "w", encoding="utf-8").write(
            open(path, encoding="utf-8").read())
        open(path, "w", encoding="utf-8").write(src)
    return hits


def main(argv):
    write = "--write" in argv
    print("=== work\\flow\\_wfbullets.py ===")
    h1 = apply(CANON, write)
    print("=== eda-pcb\\tools\\diag\\_wfbullets.py ===")
    h2 = apply(SNAP, write)
    print("命中 %d + %d ；%s" % (h1, h2, "已写入" if write else "dry-run（加 --write 落盘）"))
    if write and "--sync" in argv:
        import shutil
        shutil.copyfile(CANON, SNAP)
        print("已把 work 那份覆盖到 eda-pcb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
