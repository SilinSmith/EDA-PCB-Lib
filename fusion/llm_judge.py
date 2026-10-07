# -*- coding: utf-8 -*-
"""带图的 LLM 裁决：证据池里「判不出来」的那些槽位，切图去问视觉模型。

为什么必须存在
--------------
`stage3/arbitrate.py` 在 220 张上**逐位没有决定过任何东西**（与 stage2 产物完全相同），
根因不是「模型不行」——`qwen3.5-9b` 挂着视觉头，实测 0.2 s 就能从裁片里读出 `C151`。
根因是 **`stage1/llm.py` 从来没有把图发出去**：`_chat`/`ask_one` 构造的 messages 全是
纯文本 `content` 字符串，提示里只有候选的**分数**，于是纯文本模型必然重述
`assign.py` 的 argmax。

这一层把图真正接上：`claims.Region.crop()` 切出槽位的取景框（并在目标上画红框，
不给高亮时模型会去读窗口里**别的**文字），`vlm.client.ask()` 发出去。

证据如何回流
------------
`judge()` **不直接改结论**，它只是往池子里再加一条 `source="vlm:judge"` 的候选
（`LLM_SCORE` 默认 1.5，高于 CRNN 那条的 1.0 量级）。之后重新跑 `solve()` ——
唯一性与连号仍然生效，所以六颗电容即使被读成六个不同的号，也要和别的前缀竞争；
而如果模型把两颗电容都读成 `C151`，唯一性会拦下第二个。**「读」和「裁决」分开**
是这一层的全部要点。

⚠ 开工前必须 `preflight()`：VLM 链路可以「1 秒跑完、产物与输入逐位相同」地静默空转，
已栽过两次（空 KEY 的 401、端点不支持图片的 400），两者在统计里长得一模一样。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent                                  # eda-pcb
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fusion import claims as CL                        # noqa: E402
from fusion import priors as PR                        # noqa: E402
from fusion import solve as SV                         # noqa: E402

# LLM 读数的分数：必须高于 CRNN 那条（`desig_slots` 的 score 上限约 1.0）
LLM_SCORE = float(os.environ.get("FUSION_LLM_SCORE", "1.5"))
# 放大倍数（端点预算有限：放大 4 倍后 200x200 的窗口是 800x800）
SCALE = int(os.environ.get("FUSION_JUDGE_SCALE", os.environ.get("VLM_SCALE", "4")))
# 每张图最多问几问（每问 0.2~3 s）
MAX_ASK = int(os.environ.get("FUSION_JUDGE_MAX", "64"))
# 一次问几个问题（0=不限制，只做顺序请求；端点通常单线程更快）
TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "600"))
# 关掉裁决（做 A/B 用）
from envflag import env_on                       # noqa: E402  (唯一权威假值词表)
# ★ 手写短词表 ⇒ `FUSION_JUDGE=off` 被当成开（「关不掉」）。统一走 envflag。
JUDGE = env_on("FUSION_JUDGE", "1")
# 切图方式：`tight`=只切高亮框（默认，实测最准）｜`window`=切整个取景窗口
CROP_MODE = os.environ.get("FUSION_JUDGE_CROP", "tight").strip().lower()
# 答案前缀与槽位期望前缀不符时的分数折扣（不是拒收，只是不再一票通吃）
MISMATCH_W = float(os.environ.get("FUSION_JUDGE_MISMATCH", "0.30"))

# ⚠ 提示词里**必须**说明「红框」，但**不能真的画红框**：实测（`case 0001`，6/6）
# 画上 2px 红框后模型一致读错（`C5`→`R12`、`C12`→`J7`、`C11`→`R12`、`C4`→`Q4`、
# `C16`→`J16`、`R12`→`R17 100`），不画框 6/6 读对。原因是放大 4 倍后 2px 的框
# 变成 8px 宽的黑带，把 12x8 的小字本身盖住了。

# 提示：**先说要读什么形状的东西**，再要求只回字符串。给出「位号长这样」的例子，
# 但不给候选 —— 给了候选就变成确认偏差，模型会挑一个分数高的复述。
#
# ⚠ 键必须**按字段**分：`comp.*.key` 是位号、`comp.*.value` 是参数。
# 一开始只按 `comp` 分派，于是问参数时提示词说的是「图中一个元件位号」——
# 模型照办，把 `C13` 当成了「值」答回来（实测 `comp.c0019.value` → `C13`）。
KIND_HINT = {
    "key": "元件位号（形如 R12、C151、U3、L2）",
    "value": "元件参数值（形如 10k、0.1uF、4.7uF、16V、1.4A、100nF）",
    "partname": "器件型号（形如 W25Q128JV、STM32F103、AP56404L）",
    "type": "元件种类名",
    "pin": "元件引脚的号或名（形如 12、8、VDD、SCLK）",
    "net": "网络标签（形如 GND、VCC、RST、ADC）",
    "netlabel": "网络标签（形如 GND、VCC、RST、ADC）",
    "comp": "元件位号（形如 R12、C151、U3、L2）",
}

PROMPT = (
    "这是一张电路图的局部截图，图**正中**是一个%s，它左右可能还有别的标注。\n"
    "请只回答**正中央**那个字符串，原样照抄，区分大小写与数字。\n"
    "不要解释，不要加标点，不要加空格。如果正中央没有文字就回答 NONE。"
)

PROMPT_ALT = (
    "这是一张电路图的局部截图，图**正中**是一个%s。\n"
    "有人读成了 A = %s，也有人读成了 B = %s。\n"
    "请仔细看正中央那个字符串，只回答 A 或 B 或 OTHER（如果是别的字符串）。\n"
    "如果回答 OTHER，再另起一行原样写出你看到的字符串。不要解释。"
)

# ★ 裁决模式：**位号槽位默认让模型在候选之间二选一**，而不是自由读数。
#
# 实测（9 张 GT，`fusion/run.py … --no-llm` 对照）：
#   - 自由读数（`read`）的答案有大量**看似合理但不在任何候选里**的串
#     （`comp.c0018.key` 候选是 `C13/C12`，模型答 `C151`；`c0003.key` 候选
#     `U1/U580`，模型答 `USB_D-`）。它们以 `LLM_SCORE=1.5` 的分数盖过
#     `vlm:enrich`(1.15) 与 `ocr`(~1.0) 一票通吃，净效果 **−0.765**
#     （22.9943 → 22.2291，`0002 29.43→26.93`、`0009 31.29→26.81`）。
#   - 约束成 A/B/OTHER 之后，模型必须落在已知候选上（`OTHER` 且不在候选里
#     一律拒收，见 `REJECT_OTHER`），自由读数的幻觉通道被关掉。
MODE = os.environ.get("FUSION_JUDGE_MODE", "auto").strip().lower()  # auto|read|choose
REJECT_OTHER = env_on("FUSION_JUDGE_OTHER", "1")

_LETTER = re.compile(r"^\**\s*(?:答案\s*[:：])?\s*([AB])\b")


def _is_read(src) -> bool:
    """候选是不是**真的读出来的**（与 `fusion/solve.py`、`fusion/priors.py` 同判据）。"""
    return str(src or "").startswith(("ocr", "vlm"))


# 装配层给「没有位号」的元件发的占位键（与 `metrics.FKEY_RE` / `UKEY_RE` 一致）。
# 它们**不是读出来的**，只是键位占位符。
PLACEHOLDER = re.compile(r"^(?:p\d{3,4}|\u2205\d+)$", re.I)


def _from_choice(ans: str, raw: str, top, alt) -> str:
    """把 choose 模式的 `A`/`B`/`OTHER` 还原成字符串。"""
    t = (ans or "").strip()
    m = _LETTER.match(t.upper())
    if m:
        return str(top if m.group(1) == "A" else alt or "")
    body = [ln.strip().strip("*` ") for ln in (raw or "").splitlines() if ln.strip()]
    if body and body[0].strip().upper().startswith(("OTHER", "其他")) and len(body) > 1:
        return body[1]
    return t


# ------------------------------------------------------------------ 工具
def kind_of(slot_id: str) -> str:
    """槽位 → 提示词用的种类。**必须按字段分**（见 `KIND_HINT` 上方的说明）。"""
    parts = slot_id.split(".")
    head = parts[0]
    if head == "comp":
        if len(parts) == 3 and parts[2] in KIND_HINT:
            return parts[2]
        return "key"
    if head in KIND_HINT:
        return head
    return "key"


def load_image(pool):
    """把 `pool.doc["image"]["path"]` 读进来（相对路径按工作区根解析）。"""
    if pool.bgr is not None:
        return pool.bgr
    p = ((pool.doc.get("image") or {}).get("path") or "")
    if not p:
        return None
    cand = [Path(p), _ROOT.parent / p, Path(r"Y:\workspace\EDA") / p, _ROOT / p]
    for c in cand:
        try:
            if c.is_file():
                from stage2.text import imread_unicode
                pool.bgr = imread_unicode(str(c))
                if pool.bgr is not None:
                    return pool.bgr
        except Exception:                                  # noqa: BLE001
            continue
    return None


def clean_answer(t: str) -> str:
    """把模型的回答收敛成一个值：去围栏、去引号、只留一行、去空白。"""
    t = strip_fence(t)
    if not t:
        return ""
    t = t.splitlines()[0].strip()
    t = t.strip("`'\"“”‘’ \t:：=。，,;；")
    if t.upper() in ("NONE", "N/A", "NULL", "空", "EMPTY"):
        return ""
    return t


def strip_fence(t: str) -> str:
    t = (t or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rstrip("`").strip()
    return t


# ------------------------------------------------------------------ 问
def ask_one(reg: CL.Region, bgr, prompt: str, client=None,
            scale: int = SCALE, timeout: float = TIMEOUT):
    """切图 + 问一次。返回 `(答案, 原样文本, 秒)`；切不出图返回 `(None, "", 0.0)`。"""
    if reg is None or bgr is None:
        return None, "", 0.0
    img = reg.crop(bgr, scale=scale, focus_rect=False, mode=CROP_MODE)
    if img is None:
        return None, "", 0.0
    ask = client or _client_ask()
    txt, dt = ask(img, prompt, timeout=timeout)
    if isinstance(txt, tuple):                             # 兼容 (text, sec)
        txt, dt = txt
    return clean_answer(txt), str(txt or ""), float(dt or 0.0)


def _admit(pool, sid: str, ans: str) -> Tuple[bool, float]:
    """判断模型的答案能不能进候选表，返回 `(收不收, 分数倍数)`。

    ⚠ 这一步是**必须**的。实测 `case 0001` 的 `comp.c0003.key`（真值 `U3`，候选
    `U1/U580/U11`）—— 它那个候选框其实是 CRNN 把 `USB_D+` 读成 `U1` 的地方，于是
    模型照实答了 `USB_D+`。若不校验，这条 `LLM_SCORE=1.5` 的候选会**一票通吃**
    把整个槽位毁掉。判据：形状必须像该槽位的种类；前缀与元件类型不符只降分不收。
    """
    head = sid.split(".")[0]
    parts = sid.split(".")
    if head == "comp":
        field_name = parts[2] if len(parts) == 3 else "key"
        if field_name == "key":
            if not PR.DESIG_RE.match(ans):
                return False, 0.0
            ctype = (pool.meta.get(sid) or {}).get("type") or ""
            if ctype and not PR.key_fits_type(ans, ctype):
                return True, MISMATCH_W
            return True, 1.0
        if field_name == "value":
            # 值必须是「数字 + 单位」的形状；**不许把位号当值收下**
            # （官方是字符串严格匹配，`10k` ≠ `10K`，所以这里只判形状，不改写）。
            if PR.DESIG_RE.match(ans) or _si_shape(ans) is None:
                return False, 0.0
            return True, 1.0
        if field_name == "partname":
            if len(ans) < 4 or PR.DESIG_RE.match(ans):
                return False, 0.0
            return True, 1.0
        return True, 1.0
    if head == "pin":
        if not re.match(r"^[A-Za-z0-9_+\-./]{1,12}$", ans):
            return False, 0.0
        return True, 1.0
    if head in ("net", "netlabel"):
        if not re.match(r"^[A-Za-z0-9_+\-./:]{1,24}$", ans):
            return False, 0.0
        return True, 1.0
    return True, 1.0


def _si_shape(text: str):
    """`值` 的形状判据：数字 + 可选单位（`10k` / `0.1uF` / `1.5V` / `100nF` / `4.7`）。"""
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([A-Za-zµμΩ%/]{0,6})$", (text or "").strip())
    return m.groups() if m else None


def _client_ask():
    from vlm import client as VC
    return VC.ask


def preflight(strict: bool = True):
    from vlm import client as VC
    return VC.preflight(strict=strict)


# ------------------------------------------------------------------ 主流程
def judge(pool, questions: Optional[Sequence[dict]] = None,
          assign: Optional[dict] = None, mode: str = MODE,
          limit: int = MAX_ASK, client=None, log=print) -> dict:
    """对每个待裁决槽位切图问一次，把答案作为候选写回池子。

    `mode="read"`：自由读数（默认，不给候选，避免确认偏差）。
    `mode="choose"`：把 top/alt 摆给模型，让它选 A/B/OTHER（更省 token，但有锚定）。

    返回报告 dict（含每题的结果与统计）。**不重新求解**，调用方自己再跑 `solve()`。
    """
    if not JUDGE:
        return {"skipped": True, "reason": "FUSION_JUDGE=0"}
    qs = list(questions if questions is not None else
              SV.open_questions(pool, assign or {}))
    if mode == "sheet" and SHEET_ON:
        # 整表一次请求：给模型的是**整个待裁决集合的拼图 + 全局位号约束**。
        return judge_sheet(pool, qs, limit=min(limit, SHEET_MAX), client=client,
                           log=log)
    bgr = load_image(pool)
    rep = {"mode": mode, "n_q": len(qs), "n_asked": 0, "n_answered": 0,
           "n_new": 0, "n_agree_top": 0, "n_agree_alt": 0, "n_other": 0,
           "n_reject": 0, "crop": CROP_MODE,
           "rows": [], "sec": 0.0, "has_image": bool(bgr is not None)}
    if bgr is None:
        rep["error"] = "读不到图（pool.doc.image.path 解析失败）"
        return rep

    for q in qs[:max(0, int(limit))]:
        sid = q["slot"]
        reg = pool.region(sid)
        if reg is None:
            rep["rows"].append({"slot": sid, "skip": "无取景框"})
            continue
        use_choose = (mode == "choose" or
                      (mode == "auto" and kind_of(sid) in ("key", "pin", "net", "netlabel") and
                       q.get("top") and q.get("alt")))
        if use_choose:
            prompt = PROMPT_ALT % (KIND_HINT[kind_of(sid)],
                                   q.get("top", {}).get("value"),
                                   q.get("alt", {}).get("value"))
        else:
            prompt = PROMPT % KIND_HINT[kind_of(sid)]
        # ★ 不给**没有读数**的位号槽位出题。
        #
        # 这类槽位的 top 是装配层的占位键（`p002`）、alt 是 `prior:run_inwindow`
        # 编的号。模型若在二者间选，就是把「先验造的数」借 VLM 的嘴加冕 —— 实测
        # `0009` 的 `comp.c0002.key top=p002 alt=P12 -> vlm=C2`，`C2` 是**别人**
        # 已经拿着的号，于是连锁轮转 `C2→C12`，整张从 31.29 掉到 26.81。
        # 位号先验可以给无读位的元件**补号**（`priors`/`solve` 已经在做），但
        # 「给它命名」这件事不该由看图模型来背书。
        b0 = pool.best(sid)
        if kind_of(sid) == "key" and b0 is not None and PLACEHOLDER.match(str(b0.value)):
            row = {"slot": sid, "skip": "placeholder", "top": q.get("top", {}).get("value"),
                   "alt": q.get("alt", {}).get("value")}
            rep["n_skip_ph"] = rep.get("n_skip_ph", 0) + 1
            rep["rows"].append(row)
            continue
        ans, raw, dt = ask_one(reg, bgr, prompt, client=client)
        if use_choose:
            ans = _from_choice(ans, raw, q.get("top", {}).get("value"),
                               q.get("alt", {}).get("value"))
        rep["n_asked"] += 1
        rep["sec"] += dt
        row = {"slot": sid, "raw": raw.strip()[:60], "ans": ans, "sec": round(dt, 1),
               "top": q.get("top", {}).get("value"), "alt": q.get("alt", {}).get("value"),
               "mode": "choose" if use_choose else "read"}
        if not ans:
            row["res"] = "empty"
            rep["rows"].append(row)
            continue
        rep["n_answered"] += 1
        known0 = {str(a.value) for a in pool.alts(sid)}
        if use_choose and REJECT_OTHER and ans not in known0:
            # 选了 OTHER 又说了一个**不在候选里**的串 —— 正是自由读数的幻觉通道
            # （`comp.c0018.key` 候选 `C13/C12`，模型答 `C151`）。一律拒收。
            row["res"] = "reject_other"
            rep["n_reject"] = rep.get("n_reject", 0) + 1
            rep["rows"].append(row)
            if log is not None:
                log("  %-18s top=%-8s alt=%-8s -> vlm=%-8s [reject_other] %.1fs" % (
                    sid, row["top"], row["alt"], ans, dt))
            continue
        _settle(pool, q, ans, raw, dt, reg, row, rep, log=log)
    return rep


def _settle(pool, q, ans, raw, dt, reg, row, rep, log=None) -> dict:
    """`_admit` + 「不许给先验加冕」+ `pool.add` —— 单题路径与整表路径共用。

    这段原来内联在 `judge()` 的循环里；`judge_sheet()` 也要走一模一样的判据，
    抄一份必然发散，所以抽出来。
    """
    sid = q["slot"]
    ok, mult = _admit(pool, sid, ans)
    if not ok:
        row["res"] = "reject"
        rep["n_reject"] = rep.get("n_reject", 0) + 1
        rep["rows"].append(row)
        if log is not None:
            log("  %-18s top=%-8s alt=%-8s -> vlm=%-8s [reject]" % (
                sid, row.get("top"), row.get("alt"), ans))
        return row
    # ★ **不许用「同意一个先验造出来的候选」来给先验加冕。**
    #
    # choose 模式把 top/alt 摆给模型，其中 top 可能是**占位符 `p002`**、
    # alt 可能是 `prior:run_inwindow` 编出来的 `C2`。模型选 B 之后，我们在
    # 上面把答案还原成 `C2` 并以 `vlm:judge`(1.5) 写回 —— 于是「先验发明
    # 的号」借 VLM 的嘴拿到了最高分，`0009` 就是这么从 31.29 掉到 26.81 的
    # （`comp.c0002.key top=p002 alt=P12 -> vlm=C2`）。
    # 所以：**答案若与一个非读来源的既有候选相同，就不再加分**（它本来就在
    # 候选表里，模型同意它不构成新证据）；只有同意一个**读来源**的候选，
    # 才把它抬到 `LLM_SCORE`。
    prev = None
    for a in pool.alts(sid):
        if str(a.value) == str(ans):
            prev = a
            break
    if prev is not None and not _is_read(getattr(prev, "source", "")):
        row["res"] = "agree_prior"
        rep["n_agree_prior"] = rep.get("n_agree_prior", 0) + 1
        rep["rows"].append(row)
        if log is not None:
            log("  %-18s top=%-8s alt=%-8s -> vlm=%-8s [agree_prior]" % (
                sid, row.get("top"), row.get("alt"), ans))
        return row
    pool.add(sid, ans, score=LLM_SCORE * mult, source="vlm:judge",
             evidence={"原样": (raw or "").strip()[:120],
                       "取景框": (reg.to_dict()["box"] if reg is not None else None),
                       "耗时": round(dt, 1), "分数倍数": mult},
             question=pool.hy.slots[sid].question if sid in pool.hy.slots else "",
             meta=dict(pool.meta.get(sid) or {}))
    known = {str(a.value) for a in pool.alts(sid)}
    if ans == row.get("top"):
        row["res"] = "agree_top"
        rep["n_agree_top"] += 1
    elif ans == row.get("alt"):
        row["res"] = "agree_alt"
        rep["n_agree_alt"] += 1
    elif ans in known:
        row["res"] = "agree_other"
    else:
        row["res"] = "new"
        rep["n_new"] += 1
        rep["n_other"] += 1
    rep["rows"].append(row)
    if log is not None:
        log("  %-18s top=%-8s alt=%-8s -> vlm=%-8s [%s]" % (
            sid, row.get("top"), row.get("alt"), ans, row["res"]))
    return row


def resolve_again(pool, rep=None) -> dict:
    """把 LLM 证据并进去之后重新求解，并回写。"""
    sol = SV.solve(pool)
    SV.write_back(pool, sol)
    return sol


# ------------------------------------------------------------------ 整表上下文
# ★ 为什么要有这条路（owner m25661 的原话：「你根本就没有准备可以好好判断的
#   上下文环境」「本来最后的框架就是要让LLM参与的…必须放进去」）。
#
# 原来每道题**单独**裁一张小图、单独发一次请求（`MAX_ASK=64`，两轮最多 128 次）。
# 那样的「上下文」只有那一小块像素，模型看不见：
#   - 同一张图上别的元件的位号（于是不知道 `C2` 是否已经被占），
#   - 自己的上一次判断（同一轮里两题给不出互斥的答案），
#   - 候选是从哪来的（自由读数模式下它只能凭那一小块硬猜，实测幻觉率很高）。
#
# `judge_sheet` 改成：把所有待裁决槽位的取景框**排成一张拼图**，一格一道题，
# 格子上只印编号 `#k`（不印候选，免得模型直接抄标签），候选写进文字提示；再把
# 「图上已经确定的位号」一起给它。**一次请求**答完，模型于是能在整张图的语境里
# 判断（尤其是位号的唯一性）。
SHEET_ON = env_on("FUSION_JUDGE_SHEET", "1")
SHEET_CELL = int(os.environ.get("FUSION_JUDGE_CELL", "260"))
SHEET_COLS = int(os.environ.get("FUSION_JUDGE_COLS", "3"))
SHEET_MAX = int(os.environ.get("FUSION_JUDGE_SHEET_MAX", "36"))

SHEET_PROMPT = (
    "这是一张电路图上的 %d 个待确认标注，排成 %d 列的拼图，每格左上角有编号 `#k`。\n"
    "请只看每个格子**正中央**那个字符串，逐行回答，格式严格为 `k:A` 或 `k:B`；\n"
    "若两个候选都不对，回答 `k:OTHER 你看到的原文`。\n"
    "候选：\n%s\n"
    "%s"
    "不要解释，不要输出别的行列。"
)
SHEET_LINE = re.compile(r"^#?\s*(\d{1,3})\s*[:：.、]\s*(.*)$")


def _sheet_image(pool, items, cell: int = SHEET_CELL, cols: int = SHEET_COLS):
    """把待裁决槽位的取景框排成带 `#k` 编号的拼图。**不印候选。**"""
    import cv2
    import numpy as np
    bgr = load_image(pool)
    if bgr is None:
        return None
    tiles = []
    for k, (q, reg) in enumerate(items):
        img = reg.crop(bgr, scale=SCALE, focus_rect=False, mode=CROP_MODE)
        if img is None:
            tiles.append(None)
            continue
        h, w = img.shape[:2]
        f = min(cell / max(1, w), cell / max(1, h), 1.0)
        if f < 1.0:
            img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))),
                             interpolation=cv2.INTER_AREA)
        c = np.full((cell + 26, cell, 3), 255, np.uint8)
        ih, iw = img.shape[:2]
        c[26:26 + min(ih, cell), :min(iw, cell)] = img[:cell, :cell]
        cv2.putText(c, "#%d" % k, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 200), 2, cv2.LINE_AA)
        cv2.rectangle(c, (0, 0), (cell - 1, cell + 25), (210, 210, 210), 1)
        tiles.append(c)
    while len(tiles) % cols:
        tiles.append(None)
    rows = max(1, len(tiles) // cols)
    blank = np.full((cell + 26, cell, 3), 255, np.uint8)
    sheet = np.full((rows * (cell + 26), cols * cell, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * (cell + 26):(r + 1) * (cell + 26), c * cell:(c + 1) * cell] = \
            t if t is not None else blank
    return sheet


def _known_designators(pool, qs) -> str:
    """「图上已经确定的位号」—— 给整表请求当全局约束用。"""
    bad = {q["slot"] for q in qs}
    seen = []
    for sid in pool.slot_ids():
        if sid in bad or kind_of(sid) != "key":
            continue
        b = pool.best(sid)
        if b is None or PLACEHOLDER.match(str(b.value)):
            continue
        if _is_read(getattr(b, "source", "")):
            seen.append(str(b.value))
    if not seen:
        return ""
    return "这张图上已经确定（不要重复）的位号：%s\n" % " ".join(sorted(set(seen))[:60])


def judge_sheet(pool, questions=None, limit: int = SHEET_MAX, client=None,
                log=print) -> dict:
    """整表一次请求地回答所有待裁决槽位。返回与 `judge()` 同形的报告。"""
    import time as _t
    t0 = _t.time()
    qs = list(questions if questions is not None else SV.open_questions(pool, {}))
    rep = {"mode": "sheet", "n_q": len(qs), "n_asked": 0, "n_answered": 0,
           "n_new": 0, "n_agree_top": 0, "n_agree_alt": 0, "n_other": 0,
           "n_reject": 0, "crop": CROP_MODE, "rows": [], "sec": 0.0,
           "has_image": False}
    items = []
    for q in qs:
        sid = q["slot"]
        reg = pool.region(sid)
        if reg is None:
            rep["rows"].append({"slot": sid, "skip": "无取景框"})
            continue
        b0 = pool.best(sid)
        if kind_of(sid) == "key" and b0 is not None and PLACEHOLDER.match(str(b0.value)):
            rep["n_skip_ph"] = rep.get("n_skip_ph", 0) + 1
            rep["rows"].append({"slot": sid, "skip": "placeholder"})
            continue
        items.append((q, reg))
        if len(items) >= max(1, int(limit)):
            break
    if not items:
        rep["sec"] = round(_t.time() - t0, 2)
        return rep
    sheet = _sheet_image(pool, items)
    if sheet is None:
        rep["error"] = "读不到图（pool.doc.image.path 解析失败）"
        return rep
    rep["has_image"] = True
    try:                                                   # 给人眼复核用
        import cv2
        p = Path(os.environ.get("FUSION_SHEET_OUT",
                                r"Y:\workspace\EDA\work\preview\_judge_sheet.png"))
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), sheet)
        rep["sheet"] = str(p)
    except Exception:                                      # noqa: BLE001
        pass
    cand = "\n".join("%d: A=%s / B=%s" % (k, q.get("top", {}).get("value"),
                                          q.get("alt", {}).get("value"))
                     for k, (q, _r) in enumerate(items))
    prompt = SHEET_PROMPT % (len(items), SHEET_COLS, cand, _known_designators(pool, qs))
    ask = client or _client_ask()
    try:
        txt, dt = ask(sheet, prompt, timeout=float(os.environ.get("FUSION_SHEET_TIMEOUT",
                                                                  "300")))
    except Exception as exc:                               # noqa: BLE001
        rep["error"] = str(exc)
        return rep
    rep["n_asked"] = 1
    rep["sec"] = round(float(dt), 1)
    got: Dict[int, str] = {}
    for ln in (txt or "").splitlines():
        m = SHEET_LINE.match(ln.strip())
        if not m:
            continue
        k = int(m.group(1))
        if 0 <= k < len(items):
            got[k] = m.group(2).strip().strip("*` ")
    rep["n_raw"] = len(got)
    for k, ans in got.items():
        q, reg = items[k]
        sid = q["slot"]
        top, alt = q.get("top", {}).get("value"), q.get("alt", {}).get("value")
        row = {"slot": sid, "raw": ans[:60], "ans": ans, "sec": rep["sec"],
               "top": top, "alt": alt, "mode": "sheet"}
        up = ans.upper()
        if up.startswith("A") and len(up) <= 3:
            ans = str(top)
        elif up.startswith("B") and len(up) <= 3:
            ans = str(alt)
        elif up.startswith("OTHER"):
            ans = ans[5:].strip(" :：")
        ans = clean_answer(ans)
        row["ans"] = ans
        if not ans:
            row["res"] = "empty"
            rep["rows"].append(row)
            continue
        rep["n_answered"] += 1
        known0 = {str(a.value) for a in pool.alts(sid)}
        if REJECT_OTHER and ans not in known0 and up.startswith("OTHER"):
            row["res"] = "reject_other"
            rep["n_reject"] = rep.get("n_reject", 0) + 1
            rep["rows"].append(row)
            continue
        _settle(pool, q, ans, txt, rep["sec"], reg, row, rep, log=log)
    if log is not None:
        log("  [judge:sheet] %d 题一张表 · 回 %d 行 · 采信 %d · 拒 %d · %.1fs"
            % (len(items), rep["n_raw"], rep["n_answered"], rep["n_reject"],
               rep["sec"]))
    return rep


# ------------------------------------------------------------------ 对照图
def contact_sheet(pool, questions, path, scale: int = SCALE,
                  limit: int = 24, cell: int = 300):
    """把每个待裁决槽位的**模型实际看到的图**排成一张接触表，供人眼复核。"""
    import cv2
    import numpy as np
    bgr = load_image(pool)
    if bgr is None:
        return None
    tiles = []
    for q in list(questions)[:limit]:
        reg = pool.region(q["slot"])
        img = reg.crop(bgr, scale=scale, focus_rect=False, mode=CROP_MODE) if reg else None
        if img is None:
            continue
        h, w = img.shape[:2]
        k = min(cell / max(1, w), cell / max(1, h), 1.0)
        if k < 1.0:
            img = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))),
                             interpolation=cv2.INTER_AREA)
        c = np.full((cell + 22, cell, 3), 255, np.uint8)
        ih, iw = img.shape[:2]
        c[22:22 + min(ih, cell), :min(iw, cell)] = img[:cell, :cell]
        lab = "%s  T=%s A=%s" % (q["slot"], q.get("top", {}).get("value"),
                                 q.get("alt", {}).get("value"))
        cv2.putText(c, lab[:40], (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 200), 1, cv2.LINE_AA)
        tiles.append(c)
    if not tiles:
        return None
    cols = max(1, min(4, len(tiles)))
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * tiles[0].shape[0], cols * cell, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * t.shape[0]:(r + 1) * t.shape[0], c * cell:(c + 1) * cell] = t
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), sheet)
    return str(p)


# ------------------------------------------------------------------ CLI
def main(argv: List[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    run = Path(os.environ.get("FUSION_RUN_DIR",
                              r"Y:\workspace\EDA\work\_submit_out220r36\runs"))
    case = argv[0] if argv else "0001"
    do_ask = ("--dry" not in argv)
    pool = CL.from_file(case, run / "stage1" / ("%s.stage1.json" % case),
                        run / "stage2" / "_assigned" / ("%s.json" % case))
    PR.Priors().annotate(pool)
    sol0 = SV.solve(pool)
    print("case %s：槽位 %d，待裁决 %d 条" % (case, len(pool.slot_ids()),
                                             len(sol0["open"])))
    sheet = contact_sheet(pool, sol0["open"],
                          r"Y:\workspace\EDA\work\preview\_judge_%s_before.png" % case)
    print("  取景对照图：%s" % sheet)
    if not do_ask:
        for q in sol0["open"][:20]:
            print("  %-18s %s vs %s" % (q["slot"], q["top"]["value"], q["alt"]["value"]))
        return 0
    preflight(strict=True)
    rep = judge(pool, sol0["open"])
    print("  问 %d，答 %d，新值 %d，同意 top %d，同意 alt %d，共 %.1f s" % (
        rep["n_asked"], rep["n_answered"], rep["n_new"],
        rep["n_agree_top"], rep["n_agree_alt"], rep["sec"]))
    sol1 = resolve_again(pool)
    chg = [(s, sol0["assign"].get(s, ("",))[0], sol1["assign"].get(s, ("",))[0])
           for s in sorted(set(list(sol0["assign"]) + list(sol1["assign"])))
           if (sol0["assign"].get(s) or ("",))[0] != (sol1["assign"].get(s) or ("",))[0]]
    print("  解发生变化 %d 个槽位：" % len(chg))
    for s, a, b in chg[:30]:
        print("    %-18s %s -> %s" % (s, a, b))
    Path(r"Y:\workspace\EDA\work\preview\_judge_%s.json" % case).write_text(
        json.dumps({"case": case, "rep": rep, "before": SV.to_dict(sol0),
                    "after": SV.to_dict(sol1), "changed": chg},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
