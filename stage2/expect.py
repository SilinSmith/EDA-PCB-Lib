# -*- coding: utf-8 -*-
"""第二阶段 · 「期望」：由读到的前缀反推「这个名字该有多长」，并用它决定「该抓长名字还是短名字」。

**动机（用户 m06470 的原话）**：「R 默认是电阻，我期望后面跟一位数或者两位数，而如
『RP』这种，就像是『STC12C5A60S2』这样子的长名字，『STC』出来的时候所有人都知道他的
名字是很明显不可能短的，所以应该有一个『期望』，争取框一个『长的』名字。或者反过来
如果不行，那可以期望框一个短的。」

这条先验之所以关键：`work\\reports\\错例集_导线与元件.md` 里 **7645 条 text_mismatch**
（当前最大的一个错）全是这一类 —— GT `RP2350B` 读成 `GP/2`、`APS6404L-3SQR-ZR` 读成
`C1H+`、`W25Q128JVPM` 读成 `1.0KF`。共同点是**长器件名被切碎**，而当前的
`stage2\\assign.py` 只会「挑一个框里读得最好的那个」，**没有任何机制去把碎块重新拼起来**。

本模块只做两件事，都不依赖 OCR：

1. `expect(text)` —— 由前导字母给出「这个串期望多长、属于哪一类」（`designator` /
   `partname` / `free`）与 `score_expect(text)`。
2. `merge_runs(boxes, texts, ...)` —— 把同一行里**相邻的碎片框**拼成更长的框，交给
   OCR 重读。碎片判据是「读出来很短、没有小写、没有分隔符」——正是 `GP`、`/2`、`C1H+`、
   `1.0KF` 的样子。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

# 一字母就足以定性的「位号首字母」（官方 9609 个元件的统计里，这些占绝大多数）
REF1 = frozenset("RCLDQJKXYFSTBEMWAGHPNVZ")
# 2~3 字母的位号前缀（读全了也还是位号，不该当成长名字）
REFN = frozenset({"SW", "TP", "LED", "RV", "VR", "RT", "FB", "CN", "JP", "BT",
                  "IC", "XT", "ANT", "FID", "MH", "MOV", "TVS", "LBL", "USB",
                  "TEST", "CON", "SEN", "REL", "SCR", "DIP"})
# 器件名里常见的公司/系列前缀：**看到就知道后面不可能短**
LONG_HINT = frozenset({"STC", "STM", "STM32", "APS", "W25", "W25Q", "RP", "ESP",
                       "ATM", "AT", "PIC", "MSP", "NRF", "LM", "NE", "TDA", "MAX",
                       "TPS", "AMS", "CH", "CP", "HT", "SN", "TL", "IR", "BCM"})

_LEAD_RE = re.compile(r"^([A-Za-z]+)")
_DESIG_RE = re.compile(r"^([A-Za-z]{1,3})(\d{1,4})([A-Za-z])?$")


def lead_letters(text: str) -> str:
    m = _LEAD_RE.match((text or "").strip())
    return m.group(1).upper() if m else ""


def expect(text: str) -> Tuple[int, int, str]:
    """返回 `(最短期望长度, 最长期望长度, 类别)`。

    类别：`designator`（位号）/ `partname`（器件名）/ `free`（说不准）。
    规则按用户 m06470 的表述：**前导字母越多、越不像常见位号前缀，就越该是个长名字**。

    **用户 m06734 的补充**：「不可分就默认是长的」—— 只有当这个串**本身就长得像
    一个干净位号**（`字母1~3 + 数字1~4 + 可选字母`，见 `_DESIG_RE`）**且**它的前缀
    确实在位号前缀表里时，才判 `designator`。只要有一点不像（`P1C1`、`GP/2`、
    `1.0KF`），一律按 `partname` 处理 —— 宁可让它去抓长名字，也不要它把一个碎片
    当成完整位号收工。
    """
    s = (text or "").strip()
    u = s.upper()
    lead = lead_letters(u)
    if not lead:
        return 1, 4, "free"                      # 纯数字：引脚号/参数
    if lead in LONG_HINT:
        return max(6, len(lead) + 3), 32, "partname"
    clean = bool(_DESIG_RE.match(s))
    if clean and (lead in REFN or (len(lead) == 1 and lead in REF1)):
        return 2, 6, "designator"
    if len(lead) == 1 and clean:
        return 2, 6, "designator"                # 生僻首字母 + 干净形态，仍按位号处理
    # 到这里就是「不可分」：默认按长的算。
    return 5, 32, "partname"


def score_expect(text: str) -> float:
    """这个串「像不像它自己前缀所期望的样子」，1.0 = 正好落在期望区间内。

    落在区间外按距离线性衰减到 0（区间外 4 个字符）。另外：读成 `partname` 却连一个
    数字都没有、或读成 `designator` 却没有数字，都要扣分。
    """
    s = (text or "").strip()
    if not s:
        return 0.0
    lo, hi, kind = expect(s)
    n = len(s)
    if n < lo:
        base = max(0.0, 1.0 - (lo - n) / 4.0)
    elif n > hi:
        base = max(0.0, 1.0 - (n - hi) / 4.0)
    else:
        base = 1.0
    has_digit = any(ch.isdigit() for ch in s)
    if kind in ("designator", "partname") and not has_digit:
        base *= 0.4
    if kind == "partname" and len(s) < 4:
        base *= 0.3
    return base


def should_grow(text: str) -> bool:
    """按用户 m06734 的「不可分就默认是长的」：这个读法**该不该再去图里多框几个字**。

    只有被判为 `designator`（干净位号形态 + 已知位号前缀）时才不去动它；
    其余一律认为「可能是一长串名字的某一段」，值得往阅读方向再框一点。
    """
    return expect(text or "")[2] != "designator"


def fragment_score(text: str) -> float:
    """这个读法有多像「一长串名字被切下来的一块」，1.0 = 极像。

    特征：短（<=5）、没有小写、带 `/ - . +` 分隔符、或者干脆只是个数字/单字母。
    `GP/2`、`C1H+`、`1.0KF`、`P1C1` 都该拿高分。
    """
    s = (text or "").strip()
    if not s:
        return 1.0
    sc = 0.0
    if len(s) <= 5:
        sc += 0.45
    if len(s) <= 2:
        sc += 0.2
    if not any(ch.islower() for ch in s):
        sc += 0.15
    sep = any(ch in s for ch in "/-._+")
    if sep:
        sc += 0.15          # 分隔符出现在碎片边界上（`GP/2`）
    if s.isdigit():
        sc += 0.15
    # 已经是「两个字母 + 数字」的样子，更像**完整位号**而不是碎片；但带分隔符的例外
    # （`GP/2` 明显是个长名字的前半截，不是位号）。
    if len(lead_letters(s)) >= 2 and any(ch.isdigit() for ch in s) and not sep:
        sc -= 0.25
    return max(0.0, min(1.0, sc))


def _row(box: Sequence[float], other: Sequence[float], min_ov: float = 0.5) -> bool:
    """两个框是否在同一行（竖直重叠 >= 较矮那个的 min_ov）。"""
    h1 = max(1.0, box[3] - box[1])
    h2 = max(1.0, other[3] - other[1])
    ov = min(box[3], other[3]) - max(box[1], other[1])
    return ov >= min_ov * min(h1, h2)


def merge_runs(boxes: Sequence[Sequence[float]], texts: Sequence[str],
               char_h: float = 0.0, gap_mult: float = 1.6,
               max_run: int = 6) -> List[Dict]:
    """把同一行里相邻的**碎片框**拼成更长的框，返回要补读的候选。

    每个返回项：`{"bbox": [x1,y1,x2,y2], "src": [框下标, ...], "text": "拼起来的读法"}`。
    只拼满足这些条件的连续段：同行、水平间隙 <= `gap_mult × char_h`、**首尾都不像完整
    位号**、且拼起来的串更符合自己的期望（`score_expect(拼起来) > 单块最好`）。
    """
    n = len(boxes)
    if n < 2:
        return []
    hs = [max(1.0, b[3] - b[1]) for b in boxes]
    ch = char_h if char_h > 0 else (sorted(hs)[len(hs) // 2])
    gap = gap_mult * ch
    order = sorted(range(n), key=lambda i: (boxes[i][1], boxes[i][0]))
    used: set = set()
    out: List[Dict] = []
    for a in range(n):
        i = order[a]
        if i in used:
            continue
        run = [i]
        for b in range(a + 1, n):
            j = order[b]
            if j in used:
                continue
            k = run[-1]
            if not _row(boxes[k], boxes[j]):
                continue
            gx = boxes[j][0] - boxes[k][2]
            if gx > gap or gx < -gap:
                continue
            run.append(j)
            if len(run) >= max_run:
                break
        if len(run) < 2:
            continue
        seg = run
        # 只保留「首尾都是碎片」的极大段：整串拼起来要更长、更像一个名字
        joined = "".join(texts[i2] or "" for i2 in seg)
        # 拼起来必须**是个器件名**才有意义 —— 否则会去拼 `R1`+`R2`、`D1`+`N` 这种。
        if expect(joined)[2] != "partname":
            continue
        if all(score_expect(texts[i2] or "") >= 0.9 for i2 in seg):
            continue
        best_single = max((score_expect(texts[i2] or "") for i2 in seg), default=0.0)
        frag = sum(fragment_score(texts[i2] or "") for i2 in seg) / len(seg)
        if frag < 0.5:
            continue
        if score_expect(joined) + 0.15 * (len(seg) - 1) <= best_single:
            continue
        xs1 = min(boxes[i2][0] for i2 in seg)
        ys1 = min(boxes[i2][1] for i2 in seg)
        xs2 = max(boxes[i2][2] for i2 in seg)
        ys2 = max(boxes[i2][3] for i2 in seg)
        used.update(seg)
        out.append({"bbox": [xs1, ys1, xs2, ys2], "src": list(seg), "text": joined})
    return out


# ---------------------------------------------------------------------------
def _selftest() -> int:
    fails: List[str] = []

    def chk(c, m):
        if not c:
            fails.append(m)

    # --- 期望：用户 m06470 举的例子 ---
    chk(expect("R14")[2] == "designator", "R14 是位号")
    chk(expect("R14")[1] <= 6, "R14 期望是短的")
    chk(expect("RP2350B")[2] == "partname", "RP2350B 是长名字")
    chk(expect("RP2350B")[0] >= 6, "RP2350B 期望长度 >= 6")
    chk(expect("STC12C5A60S2")[2] == "partname", "STC 开头必然是长名字")
    chk(expect("STC12C5A60S2")[1] >= 12, "STC12C5A60S2 期望可以长到 12+")
    chk(expect("SW2")[2] == "designator", "SW2 是位号（SW 在已知前缀表里）")
    chk(expect("C1")[2] == "designator", "C1 是位号")
    chk(expect("24")[2] == "free", "纯数字说不准")
    chk(expect("APS6404L-3SQR-ZR")[2] == "partname", "APS 开头是长名字")
    # --- 用户 m06734：「不可分就默认是长的」 ---
    for t in ("P1C1", "C1H+", "GP/2", "RM2", "APS6404", "STM32F103"):
        chk(expect(t)[2] == "partname", "%s 不可分，应默认按长的算" % t)
    for t in ("R14", "C1", "SW2", "TP3", "LED1", "Q7"):
        chk(expect(t)[2] == "designator", "%s 是干净位号，不该被当成短碎片" % t)
    chk(should_grow("GP/2") and should_grow("P1C1"), "碎片该去再框字")
    chk(not should_grow("R14"), "干净位号不该去再框字")
    chk(not should_grow("SW2"), "SW2 不该去再框字")

    # --- score_expect 与 fragment_score ---
    chk(score_expect("R14") > 0.9, "R14 完全符合期望")
    chk(score_expect("RP2350B") > 0.9, "RP2350B 完全符合期望")
    chk(score_expect("RP") < 0.7, "RP 太短，不符合它自己的长名字期望")
    chk(score_expect("RP2350B") > score_expect("RP"), "长名字比碎片更符合期望")
    for t in ("GP/2", "C1H+", "1.0KF", "P1C1"):
        chk(fragment_score(t) >= 0.6, "%s 应判为碎片: %.2f" % (t, fragment_score(t)))
    chk(fragment_score("RP2350B") < 0.5, "完整长名字不该判成碎片: %.2f"
        % fragment_score("RP2350B"))

    # --- merge_runs：`RP` + `2350B` 两块应拼成一块 ---
    boxes = [[0, 0, 20, 12], [22, 0, 70, 12]]
    txt = ["RP", "2350B"]
    got = merge_runs(boxes, txt, char_h=12.0)
    chk(len(got) == 1 and got[0]["bbox"] == [0, 0, 70, 12],
        "两块应拼成一个框: %r" % (got,))
    # 间隙太大不拼
    chk(merge_runs([[0, 0, 20, 12], [200, 0, 250, 12]], ["RP", "2350B"],
                   char_h=12.0) == [], "间隙 180px 不该拼")
    # 不在同一行不拼
    chk(merge_runs([[0, 0, 20, 12], [22, 40, 70, 52]], ["RP", "2350B"],
                   char_h=12.0) == [], "不同行不该拼")
    # 已经是完整位号的两块不拼
    chk(merge_runs([[0, 0, 20, 12], [22, 0, 42, 12]], ["R1", "R2"],
                   char_h=12.0) == [], "R1/R2 两块不该被拼")
    # 三块碎片拼成一块
    got3 = merge_runs([[0, 0, 18, 12], [20, 0, 34, 12], [36, 0, 96, 12]],
                      ["APS", "6404", "L-3SQR"], char_h=12.0)
    chk(len(got3) == 1 and got3[0]["bbox"] == [0, 0, 96, 12],
        "三块碎片应拼成一块: %r" % (got3,))
    # 单块不返回
    chk(merge_runs([[0, 0, 20, 12]], ["RP"], char_h=12.0) == [], "单块不拼")

    print("自检 %d 项：%s" % (24, "全部通过" if not fails else "失败"))
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
