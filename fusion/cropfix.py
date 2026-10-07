# -*- coding: utf-8 -*-
"""裁切方向反馈：让模型指出「该往哪边扩」，然后真的扩了再读一遍。

**用户 m21963 原话**：『如果喂给它一张残缺的图片，他能够判断一个重新裁切的方向，
比如"C151"读出"151"，判断需要向左扩。』

这个模块就是这件事的落点。它的存在理由，是裁剪外扩那条线上一路量出来的结论：

- `stage1` 的文字框是**按阈值墨迹**取的，系统性比真值框小（中位 IoU 0.36）；
- 而 CRNN 对**定长**外扩的 exact 曲线是 `+0 0.1915 · +2 0.6345 · +3 0.6544 ·
  +4 0.6105 · +5 0.5076 · +6 0.2554`（`work/_boxgeom.py`，10 张 CVJsonStd、
  1103 个配对）—— **最优是 +3px，且超过 +5 就崩**；
- 按比例外扩（`max(2.0, 0.30*h)`）在 `h≈19` 时给 5.7px，**已经越过峰**；
- 端到端网格证实「水平要 +2、垂直要 +3」而**垂直给 2 就掉 4.5 分**
  （`pad32` 44.12 vs `pad23y20` 49.35，10GT(9)）。

⇒ 一个**全局常数**无论如何定都会在某些框上过大、在另一些上过小，因为被切掉多少
是**每个框各自**的事，而它取决于「阈值取墨迹」这一刻的运气。真正需要的是
**逐框判断**：这个读数是不是被切了、缺的在哪边。

所以本模块不问「该扩多少像素」，而是问一个模型能可靠回答的问题：
**「这个读数看起来被切掉了吗？缺的那部分在哪一边？」** ——
方向是离散的、可以从字形本身推出来的（`151` 不是合法位号，字母前缀只可能在左边）；
具体像素数则**不由模型负责**，改成按方向做**外扩阶梯**（0.5h / 1h / 2h / 4h）
逐级重读，用「读数自洽性」挑一个（见 `_score`）。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve()
if str(_HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent))

try:
    from fusion import claims as CL
    from fusion import priors as PR
except ImportError:                                          # pragma: no cover
    sys.path.insert(0, str(_HERE.parent))
    from fusion import claims as CL                          # type: ignore
    from fusion import priors as PR                          # type: ignore


# ------------------------------------------------------------------ 常量
SCALE = int(os.environ.get("FUSION_CROPFIX_SCALE", os.environ.get("VLM_SCALE", "4")))
TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "600"))
# 外扩阶梯：以**字高 h** 为单位。第一档很小（只是把紧框补回 0.5h），
# 最后一档很大（把邻域整段都拿进来看）。
LADDER = tuple(float(x) for x in os.environ.get(
    "FUSION_CROPFIX_LADDER", "0.5,1.0,2.0,4.0").split(","))
ROUNDS = int(os.environ.get("FUSION_CROPFIX_ROUNDS", "2"))
MAX_FIX = int(os.environ.get("FUSION_CROPFIX_MAX", "24"))
FIX_SCORE = float(os.environ.get("FUSION_CROPFIX_SCORE", "1.25"))
from envflag import env_on                       # noqa: E402  (唯一权威假值词表)
# ★ 同 `fusion\run.py`：手写短词表 ⇒ `FUSION_CROPFIX=off` 被当成开。统一。
ON = env_on("FUSION_CROPFIX", "1")

DIRS = ("left", "right", "up", "down")

# 什么算「看起来被切了」。这三条都是**纯字形**的判断，不需要看图：
#   ① 位号没有字母前缀（`151`、`.151`）—— 字母前缀只可能在读数之外；
#   ② 读数以 `-`/`.`/`:` 这类分隔符开头或结尾（切到了边界符）；
#   ③ 读数里有「半个字符」常见的孤立窄笔画残留（`1`、`l`、`|` 单独出现在首尾）。
EDGE_JUNK = re.compile(r"^[\s\-_.,:;|]+|[\s\-_.,:;|]+$")
LONE_STROKE = re.compile(r"^(?:[1lI|])(?=[A-Za-z])|(?<=[A-Za-z])(?:[1lI|])$")
# 读数里带空格 —— 大概率是把两段文字一起圈进来了（`R8 33`、`R11 100`），
# 这正是「框比字大」的另一种切坏，值得走一次方向反馈。
HAS_SPACE = re.compile(r"\s")


def looks_cut(text: str, kind: str = "key", ctype: str = "") -> bool:
    """**纯文本**的「这个读数可能被切了」判据（不花 token 的第一道筛子）。"""
    t = (text or "").strip()
    if not t:
        return False
    if EDGE_JUNK.search(t):
        return True
    if kind in ("key", "value", "partname", "net", "netlabel") and HAS_SPACE.search(t):
        return True
    if kind == "key":
        pre, num = PR.split_desig(t)
        if num is None:
            return True                 # 完全不成位号形
        want = [p.upper() for p in PR.TYPE_PREFIX.get((ctype or "").lower(), ())]
        if want and pre.upper() not in want:
            return True                 # 前缀不对（`151` 的 pre 是空/数字）
    # 孤立窄笔画只在位号/网络名这类**短 token** 上有意义；型号（partname）里
    # 结尾是 `1` 太常见（`VD09A1`），一律当可疑会刷一堆假阳性。
    if kind in ("key", "net", "netlabel") and LONE_STROKE.search(t):
        return True
    return False


# ------------------------------------------------------------------ 提示词
PROMPT_DIR = (
    "这是一张电路图的局部截图。图上有一段文字，有人读成了「%s」。\n"
    "请判断这个读数是不是**被截断**了 —— 也就是完整的字符串有一部分落在画面之外。\n"
    "判断规则：\n"
    "- 若这个读数**本身已经是一个完整的标注**（一个完整成形的词 / 位号 / 数值），"
    "回答 DIR=none。\n"
    "- 只有当你能指出**缺的那一部分应该在哪个方向**时才给方向。"
    "例如位号 `C151` 只看到 `151` —— 字母前缀只可能在**左侧** ⇒ DIR=left。\n"
    "- **宁可回答 none，也不要猜。** 方向猜错会把旁边的字拉进来，比不扩更糟。\n"
    "请只回答一行，格式：\n"
    "DIR=<left|right|up|down|none> WHY=<不超过十个字>\n"
    "不要解释，不要输出别的行。"
)

PROMPT_HINT = (
    "这是一张电路图的局部截图。有人把图上的文字读成了「%s」。\n"
    "另一个读者认为正确的字符串很可能是「%s」（差异：%s）。\n"
    "请判断：**如果「%s」才是对的**，那么被读成「%s」的那一小段文字，"
    "缺掉的那部分应该落在画面的哪个方向？\n"
    "请只回答一行，格式：\n"
    "DIR=<left|right|up|down|none> WHY=<不超过十个字>\n"
    "如果两个读法其实指的是同一段完整的文字（画面根本没有被切），回答 DIR=none。\n"
    "不要解释，不要输出别的行。"
)

PROMPT_READ = (
    "这是一张电路图的局部截图，图**正中**是一个%s。\n"
    "请只回答**正中央**那个字符串，原样照抄，区分大小写与数字。\n"
    "不要解释，不要加标点，不要加空格。如果正中央没有文字就回答 NONE。"
)

KIND_HINT = {
    "key": "元件位号（形如 R12、C151、U3、L2）",
    "value": "元件参数值（形如 10k、0.1uF、4.7uF、16V）",
    "partname": "器件型号（形如 W25Q128JV、STM32F103）",
    "pin": "元件引脚的号或名（形如 12、8、VDD、SCLK）",
    "net": "网络标签（形如 GND、VCC、RST、ADC）",
    "netlabel": "网络标签（形如 GND、VCC、RST、ADC）",
}

_DIR_RE = re.compile(r"DIR\s*[=:：]\s*\**\s*(left|right|up|down|none|无|不用)",
                     re.I)
# 模型不照格式写时的词面兜底（中英都认）
_WORDS = (("left", ("left", "向左", "往左", "左边", "左侧")),
          ("right", ("right", "向右", "往右", "右边", "右侧")),
          ("up", ("up", "向上", "往上", "上面", "上方")),
          ("down", ("down", "向下", "往下", "下面", "下方")))
_NONE_WORDS = ("none", "无需", "不用", "已经完整", "完整")


def parse_dir(txt: str) -> str:
    """从模型输出里抠出方向；抠不到返回 `""`（不猜）。"""
    m = _DIR_RE.search(txt or "")
    if m:
        d = m.group(1).lower()
        return "" if d in ("none", "无", "不用") else d
    low = (txt or "").lower()
    if any(w in low for w in _NONE_WORDS):
        return ""
    hit = [d for d, ws in _WORDS if any(w in low for w in ws)]
    return hit[0] if len(hit) == 1 else ""


# ------------------------------------------------------------------ 几何
def expand(box, direction: str, amount: float, shape=None):
    """把框朝 `direction` 扩 `amount` 像素（其余三边不动）。"""
    x0, y0, x1, y1 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    if direction == "left":
        x0 -= amount
    elif direction == "right":
        x1 += amount
    elif direction == "up":
        y0 -= amount
    elif direction == "down":
        y1 += amount
    if shape is not None:
        W, H = int(shape[0]), int(shape[1])
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(W), x1), min(float(H), y1)
    return (x0, y0, x1, y1)


def _crop(reg: CL.Region, bgr, focus, scale=None):
    """按给定 `focus` 切一张紧裁片（复用 `Region.crop` 的紧切口径）。"""
    r = CL.Region(box=reg.box, focus=focus, note=reg.note, src=reg.src)
    return r.crop(bgr, scale=scale or SCALE, focus_rect=False, mode="tight")


# ------------------------------------------------------------------ 打分
def _score(orig: str, cand: str, kind: str, ctype: str) -> float:
    """新读数有多可信。**只用来在阶梯里挑一个**，不参与全局分数。

    三条加权：
      + 3.0  形状合法（位号前缀与元件类型相符）
      + 2.0  原读数被新读数**包含**（`151` ⊂ `C151`）—— 这是「补回被切掉的」，
             而不是「换了一个东西读」
      + 1.0  与候选表里的某个候选完全一致
      − 2.0  比原读数**短**（越扩越短说明扩错了方向 / 读到隔壁字）
      − 1.0  新读数与元件的类型前缀都不沾边
    """
    o = str(orig or "").strip()
    c = str(cand or "").strip()
    if not c:
        return -99.0
    s = 0.0
    pre, num = PR.split_desig(c)
    want = [p.upper() for p in PR.TYPE_PREFIX.get((ctype or "").lower(), ())]
    if kind == "key":
        if PR.DESIG_RE.match(c):
            s += 3.0 if not want else (3.0 if pre.upper() in want else -1.0)
    else:
        s += 1.0
    if o and o.lower() in c.lower():
        s += 2.0
    elif o:
        # 数字部分包含也算（`C51` 与 `C151`）
        _po, no = PR.split_desig(o)
        if no is not None and num is not None and str(no) in str(num):
            s += 1.5
    if o and len(c) < len(o):
        s -= 2.0
    return s


# ------------------------------------------------------------------ 主流程
def _diff_desc(a: str, b: str) -> str:
    """「`C51` vs `C151`：多了 1 个字符 `1`」这样说人话的差异描述。"""
    import difflib
    a, b = str(a or ""), str(b or "")
    if a == b:
        return "两者相同"
    sm = difflib.SequenceMatcher(None, a, b)
    bits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            bits.append("多出「%s」（在第 %d 位之后）" % (b[j1:j2], i1))
        elif tag == "delete":
            bits.append("少了「%s」（第 %d 位起）" % (a[i1:i2], i1))
        else:
            bits.append("「%s」应是「%s」" % (a[i1:i2], b[j1:j2]))
    return "；".join(bits) or "两者相同"


def _ask(client, img, prompt, timeout=TIMEOUT):
    """走 `vlm.client.ask`，返回 `(文本, 秒)`。

    ⚠ `vlm.client.ask` 的返回是 **`(文本, 耗时)` 二元组**（`client.py:171/176/178`），
    `preflight` 返回 `(bool, detail)` —— 不拆包就会把 `「('VDDBAT', 0.8)」`
    当成读到的字符串（现场实测踩过）。
    """
    import time
    t0 = time.time()
    if client is None:
        from vlm import client as VC
        client = VC.ask
    try:
        out = client(img, prompt, timeout=timeout)
    except Exception as exc:                                 # pragma: no cover
        out = "ERR:%s" % exc
    dt = time.time() - t0
    if isinstance(out, tuple):
        out, _extra = (list(out) + [None])[:2]
        dt = float(_extra) if isinstance(_extra, (int, float)) else dt
    out = str(out or "")
    if out.startswith("ERR:"):
        return "", dt
    return out.strip(), dt


def fix_slot(pool, sid: str, seen: str, client=None, log=None,
             hint: str = "") -> dict:
    """对一个槽位跑「问方向 → 沿方向做外扩阶梯 → 挑最自洽的读数」。

    `hint` 是**另一个读者提出的候选**（`audit` 的 `FIX`）。给了它，方向那一问就变成
    「如果 `hint` 才是对的，缺的那部分在哪个方向」—— 这是把「模型提议」和
    「图像裁决」接起来的关键：**候选由模型提，但对不对由图说了算**。不给 `hint` 时
    问的是「这个读数本身是不是被切了」，实测后者极难触发（模型对完整读数一律答 none）。

    返回 `{slot, seen, dir, why, rounds, ladder:[...], ans, gain, sec}`。
    """
    import time
    t0 = time.time()
    row = {"slot": sid, "seen": seen, "dir": "", "why": "", "ladder": [],
           "ans": seen, "gain": 0.0, "sec": 0.0, "mode": "", "hint": hint}
    if not ON:
        row["mode"] = "off"
        return row
    from fusion import llm_judge as LJ
    bgr = LJ.load_image(pool)
    if bgr is None:
        row["mode"] = "no-image"
        return row
    reg = pool.region(sid)
    if reg is None or not reg.focus:
        row["mode"] = "no-region"
        return row
    kind = LJ.kind_of(sid)
    ctype = (pool.meta.get(sid) or {}).get("type") or ""
    H, W = bgr.shape[:2]
    focus = tuple(float(v) for v in reg.focus)
    cur = seen
    # `hint` 与 `seen` 的差就是「缺/多的那一段」——直接告诉模型，比自己猜强得多。
    diff = _diff_desc(seen, hint) if hint else ""
    best, best_s = seen, _score(seen, seen, kind, ctype)
    for rd in range(max(1, ROUNDS)):
        img = _crop(reg, bgr, focus)
        if img is None:
            break
        if hint and rd == 0:
            prompt = PROMPT_HINT % (cur or "（空）", hint, diff, hint, cur or "（空）")
        else:
            prompt = PROMPT_DIR % (cur or "（空）")
        txt, _dt = _ask(client, img, prompt)
        d = parse_dir(txt)
        if rd == 0:
            row["dir"] = d
            row["why"] = (txt or "").strip().splitlines()[0][:40] if txt else ""
        if not d:
            break
        row["mode"] = "grew"
        h = max(1.0, focus[3] - focus[1])
        hit = None
        for k in LADDER:
            cand_box = expand(focus, d, k * h, shape=(W, H))
            if cand_box == focus:
                break
            img2 = _crop(reg, bgr, cand_box)
            if img2 is None:
                continue
            ans, dt = _ask(client, img2, PROMPT_READ % KIND_HINT.get(kind, KIND_HINT["key"]))
            ans = LJ.clean_answer(ans)
            s = _score(seen, ans, kind, ctype)
            row["ladder"].append({"k": k, "box": [round(v, 1) for v in cand_box],
                                  "ans": ans, "score": round(s, 2), "sec": round(dt, 1)})
            if s > best_s:
                best, best_s = ans, s
                hit = cand_box
                break                       # 第一档就自洽了就别再往外扩（越扩越容易被邻字带跑）
        if hit is None:
            # 这一轮没找到更好的：把取景**按最后一档**放宽再问一次方向
            focus = expand(focus, d, LADDER[-1] * h, shape=(W, H))
            if focus == reg.focus:
                break
            continue
        focus = hit
        cur = best
    row["ans"] = best
    row["gain"] = round((_score(seen, best, kind, ctype) -
                         _score(seen, seen, kind, ctype)), 2)
    row["sec"] = round(time.time() - t0, 1)
    if log is not None and best != seen:
        log("  %-18s 「%s」 -%s-> 「%s」 %s %.1fs" % (
            sid, seen, row["dir"] or "?", best,
            " ".join("k%g:%s" % (x["k"], x["ans"]) for x in row["ladder"]), row["sec"]))
    return row


def suspects(pool, slots: Optional[Sequence[str]] = None) -> List[str]:
    """挑出「可能被切了」的槽位：文本判据 + 当前最优是读来源。"""
    out = []
    for sid in (slots if slots is not None else pool.slot_ids()):
        b = pool.best(sid)
        if b is None:
            continue
        kind = sid.split(".")[2] if sid.startswith("comp.") and len(sid.split(".")) == 3 \
            else sid.split(".")[0]
        if kind not in ("key", "value", "partname", "net", "netlabel", "pin"):
            continue
        ctype = (pool.meta.get(sid) or {}).get("type") or ""
        k2 = "key" if kind == "key" else kind
        if looks_cut(str(b.value), k2, ctype):
            out.append(sid)
    return out


def run(pool, slots: Optional[Sequence[str]] = None, client=None,
        limit: int = MAX_FIX, log=print) -> dict:
    """对可疑槽位跑裁切修复，把修正后的读数作为 `vlm:cropfix` 候选写回池子。"""
    rep = {"n_sus": 0, "n_fix": 0, "n_gain": 0, "n_proc": 0, "rows": [], "sec": 0.0}
    if not ON:
        rep["mode"] = "off"
        return rep
    import time
    t0 = time.time()
    sus = suspects(pool, slots)
    rep["n_sus"] = len(sus)
    for sid in sus[:max(0, int(limit))]:
        b = pool.best(sid)
        seen = str(b.value) if b else ""
        row = fix_slot(pool, sid, seen, client=client, log=log)
        rep["n_proc"] += 1                       # 真跑过的槽位数（会发 LLM 请求）
        rep["rows"].append(row)
        if row["ans"] and row["ans"] != seen:
            rep["n_fix"] += 1                    # 重读给出了**不同**的答案
        if row["ans"] and row["ans"] != seen and row["gain"] > 0:
            rep["n_gain"] += 1                   # 且它的分**更高** ⇒ 真的被采纳
            pool.add(sid, row["ans"], score=FIX_SCORE, source="vlm:cropfix",
                     evidence={"原读数": seen, "方向": row["dir"], "理由": row["why"],
                               "阶梯": row["ladder"]},
                     question="元件 %s 的位号是哪一个？" % sid.split(".")[-1],
                     meta=dict(pool.meta.get(sid) or {}))
            # 新读数一定比旧的长/全，旧读数降级保留（它可能是对的子串）
            b2 = pool.best(sid)
            if b2 is not None and str(b2.value) == seen:
                pass
    rep["sec"] = round(time.time() - t0, 1)
    return rep


def to_dict(rep: dict) -> dict:
    return rep


# ------------------------------------------------------------------ 自检
def _demo() -> int:
    ok = 0
    bad = []

    def chk(cond, name):
        nonlocal ok
        if cond:
            ok += 1
        else:
            bad.append(name)

    # ① 文本判据：`151` 没有字母前缀 ⇒ 可疑
    chk(looks_cut("151", "key", "r"), "裸数字位号该被判可疑")
    # ② `C151` 合法 ⇒ 不可疑
    chk(not looks_cut("C151", "key", "c"), "合法位号不该被判可疑")
    # ③ 前缀与类型不符 ⇒ 可疑
    chk(looks_cut("R15", "key", "c"), "C 元件读出 R 前缀该被判可疑")
    # ④ 边缘分隔符 ⇒ 可疑
    chk(looks_cut("-GND", "net"), "边缘分隔符该被判可疑")
    chk(looks_cut("R8 33", "key", "r"), "含空格的两段该被判可疑")
    # ⑤ 方向解析
    chk(parse_dir("DIR=left WHY=缺字母") == "left", "解析 left")
    chk(parse_dir("dir : **RIGHT**") == "right", "解析 right（含星号）")
    chk(parse_dir("DIR=none") == "", "none 解析成空")
    chk(parse_dir("我猜是往左") == "left", "退一步的词面兜底")
    chk(parse_dir("看不懂") == "", "读不出方向就不猜")
    # ⑥ 几何
    chk(expand((100, 50, 140, 70), "left", 10) == (90, 50, 140, 70), "向左扩")
    chk(expand((100, 50, 140, 70), "right", 10) == (100, 50, 150, 70), "向右扩")
    chk(expand((-5, 50, 140, 70), "left", 10, shape=(200, 100))[0] == 0.0, "钳进图内")
    # ⑦ 打分：补回被切掉的 > 换一个东西
    s_fix = _score("151", "C151", "key", "c")
    s_swap = _score("151", "R8", "key", "c")
    chk(s_fix > s_swap, "补回被切掉的分应高于换一个")
    chk(_score("C151", "C15", "key", "c") < _score("C151", "C151", "key", "c"),
        "变短要扣分")
    # ⑧ 阶梯拼接
    chk(len(LADDER) >= 2 and all(x > 0 for x in LADDER), "阶梯至少两档且为正")
    print("cropfix 自检 %d 项：%s" % (ok + len(bad),
                                     "全部通过" if not bad else "失败 -> %s" % bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(_demo())
