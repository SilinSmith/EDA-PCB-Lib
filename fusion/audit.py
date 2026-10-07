"""`audit` —— 按**现实规律**判断电路是否合理（workflow.html 的 `plausible` → `conflict`）。

这个模块和 `llm_judge` / `rescan` / `cropfix` 是**三个不同的问题**：

| 模块 | 问模型什么 | 模型的角色 |
|---|---|---|
| `llm_judge` | 「红框里的字符串是什么」 | 第二个 OCR |
| `cropfix` | 「这张图是不是被切了，该往哪边扩」 | 裁切质检员 |
| `rescan` | 「这附近还有没有漏掉的参数」 | 搜索员 |
| **`audit`** | **「这样接，电学上成不成立」** | **硬件工程师** |

`plausible.check()` 用纯规则报疑点（电源地同网、两条电源轨并成一条、位号重号、
参数离群……），但它只会说「这看着不对」。**真正能判断「不对」的是领域知识** ——
`plausible` 报的 `net_short`（`GND` 和 `VCC` 挂在同一条网上）在规则层面只是一个
标签集合冲突，在工程上却是「这张图要么读错了、要么我理解错了」的确定结论。

### 为什么要给模型**两张图拼一起**
`vlm.client` 一次只发一张图。而判断合理性需要两个尺度：**全局**（这条网络走到哪、
连了哪些元件）和**局部**（那个标签到底印的是什么）。所以 `panel()` 把整张图缩到
`FULL_W` 宽放上面、把出问题的区域放大放下面，提示词里说明上下关系。只给局部，
模型看不出这条网络连了什么；只给全局，`C151` 和 `C151` 认不出。

### 输出契约
```
VERDICT=<ok|suspect|bad>
WHY=<一句话，中文，不超过三十字>
FIX=<要改成的值，没有就写 ->
```
`ok` = 规则报的疑点其实成立（比如真的是把 GND 和 VCC 短接了）；
`suspect` = 讲不通但不确定谁错；`bad` = 确定有错并给出 `FIX`。

### 这个模块**只提候选，不改结论**
`FIX` 只在 `key_duplicate` 这一种情形下作为 `vlm:audit` 候选写回池子（因为那时
「两个槽位共用同一个位号」这件事本身已经被证明是错的，模型的答案只需要通过
「形状合法 + 没被别人占」两道闸）；其余情形只记进 `pool.notes` 和报告。
**不这么做的话，`audit` 就会变成第四个幻觉来源。**
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent                                       # eda-pcb
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from fusion import claims as CL
    from fusion import plausible as PL
    from fusion import priors as PR
except ImportError:                                        # pragma: no cover
    sys.path.insert(0, str(_HERE))
    import claims as CL                                    # type: ignore
    import plausible as PL                                 # type: ignore
    import priors as PR                                    # type: ignore


# ------------------------------------------------------------------ 常量
AUDIT_SCORE = float(os.environ.get("FUSION_AUDIT_SCORE", "1.30"))
MAX_AUDIT = int(os.environ.get("FUSION_AUDIT_MAX", "16"))
#: `audit` 给出 `FIX` 后，**回去把那一块重新裁一次**再读。这是「让模型改正读数」的
#: 落地方式：**候选由模型提，对不对由图像裁决**（`cropfix.fix_slot(hint=…)`）。
from envflag import env_on                       # noqa: E402  (唯一权威假值词表)
# ★ 手写短词表 ⇒ `FUSION_AUDIT=off` / `FUSION_AUDIT_REREAD=none` 被当成开。
REREAD = env_on("FUSION_AUDIT_REREAD", "1")
TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "600"))
ON = env_on("FUSION_AUDIT", "1")
FULL_W = int(os.environ.get("FUSION_AUDIT_FULLW", "880"))
MAX_PANEL_H = int(os.environ.get("FUSION_AUDIT_MAXH", "1400"))

#: 只对这些疑点发问。`net_isolated` / `pins_unread` 是「我们的漏」而不是「图的错」，
#: 问模型「合不合理」它只会回答「信息不足」，白花 token（实测过）。
ASK_KINDS = tuple(x for x in os.environ.get(
    "FUSION_AUDIT_KINDS",
    "net_short,net_rail_merge,key_duplicate,key_type_mismatch,value_outlier"
).split(",") if x.strip())

#: 规则名 → 给模型的一句话背景（模型不知道我们的内部术语）。
RULE_HINT = {
    "net_short":
        "这条网络上同时出现了**地**（GND 之类）和**电源**（VCC/3.3V 之类）两种标签。"
        "在真实的电路图里，地和电源**不可能**是同一个节点。",
    "net_rail_merge":
        "这条网络上挂着两条**不同的电源轨**标签。不同的电源轨之间通常要经过"
        "稳压器/磁珠/0Ω 电阻，不应该直接是同一个节点。",
    "key_duplicate":
        "有两个元件被读成了**同一个位号**。同一张图里位号必须唯一，所以至少有一个读错了。",
    "key_type_mismatch":
        "这个元件的位号字母和它的元件类型对不上（比如位号以 R 开头、类型却是电容）。",
    "value_outlier":
        "同一排同类别的元件参数都很接近，这一个明显离群 —— 很可能是把旁边的"
        "耐压/精度标注（比如 16V、10%）错当成了参数。",
}

PROMPT = (
    "你是一名资深硬件工程师。下面这张图分两部分：\n"
    "- **上半部分**是整张电路图的缩略图（看清这条网络连到哪儿）；\n"
    "- **下半部分**是出问题的地方的放大图（看清印的字）。\n\n"
    "已知信息（我们的识别结果，可能有错）：\n%s\n\n"
    "判断规则：%s\n\n"
    "请判断「我们识别出来的这个样子，在电学上成不成立」，并只回答三行：\n"
    "VERDICT=<ok|suspect|bad>\n"
    "WHY=<一句话，中文，不超过三十字>\n"
    "FIX=<如果确定有错，写出应该改成的值；否则写 ->\n"
    "其中 VERDICT=ok 表示「虽然看着可疑，但确实可以这样接」；"
    "suspect 表示「讲不通，但我不能确定错在哪」；bad 表示「确定有错」。\n"
    "不要解释，不要输出别的行。"
)

_VERDICT_RE = re.compile(r"VERDICT\s*[=:：]\s*\**\s*(ok|suspect|bad)", re.I)
_WHY_RE = re.compile(r"WHY\s*[=:：]\s*\**\s*(.*)")
_FIX_RE = re.compile(r"FIX\s*[=:：]\s*\**\s*(.*)")


# ------------------------------------------------------------------ 小工具
def _f(v, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _ask(client, img, prompt, timeout: float = TIMEOUT):
    """走 `vlm.client.ask` → `(文本, 秒)`。

    ⚠ `vlm.client.ask` 返回的是 **`(文本, 耗时)` 二元组**（`vlm/client.py:171/176/178`），
    `preflight` 返回 `(bool, detail)`：不拆包会把 `「('VDDBAT', 0.8)」` 整串当成读数。
    """
    t0 = time.time()
    if client is None:
        from vlm import client as VC
        client = VC.ask
    try:
        out = client(img, prompt, timeout=timeout)
    except Exception as exc:                               # pragma: no cover
        return "", str(exc), time.time() - t0
    dt = time.time() - t0
    if isinstance(out, tuple):
        head, extra = (list(out) + [None])[:2]
        out = head
        if isinstance(extra, (int, float)):
            dt = float(extra)
    txt = str(out or "")
    if txt.startswith("ERR:"):
        return "", txt, dt
    return txt.strip(), "", dt


def parse(txt: str) -> Tuple[str, str, str]:
    """把模型的回答拆成 `(verdict, why, fix)`。抠不到就返回 `("", "", "")`。"""
    t = str(txt or "").strip()
    if not t:
        return "", "", ""
    m = _VERDICT_RE.search(t)
    verdict = m.group(1).lower() if m else ""
    why, fix = "", ""
    m2 = _WHY_RE.search(t)
    if m2:
        why = m2.group(1).strip().strip("*` ")
    m3 = _FIX_RE.search(t)
    if m3:
        fix = m3.group(1).strip().strip("*` ")
        if fix in ("-", "—", "–", "无", "NONE", "none", "N/A", "na", ""):
            fix = ""
    if not verdict:                                        # 兜底：词面
        low = t.lower()
        if "bad" in low or "有错" in t or "确定错" in t:
            verdict = "bad"
        elif "suspect" in low or "可疑" in t or "讲不通" in t:
            verdict = "suspect"
        elif "ok" in low:
            verdict = "ok"
    return verdict, why, fix


def _text_w(s: str) -> int:
    """粗略的显示宽度（中日韩算 2）。只用于排版判断，不追求精确。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in str(s))


# ------------------------------------------------------------------ 描述
def _comp_of(sid: str) -> str:
    return sid.split(".")[1] if sid.startswith("comp.") and sid.count(".") >= 2 else ""


def comp_desc(pool: CL.ClaimPool, cid: str) -> str:
    """`c0003：类型 box，位号 U1，参数 -` 这样的一行。"""
    if not cid:
        return "-"
    key = pool.chosen("comp.%s.key" % cid) or "（没读出位号）"
    val = pool.chosen("comp.%s.value" % cid) or "-"
    pn = pool.chosen("comp.%s.partname" % cid) or ""
    meta = pool.meta.get("comp.%s.key" % cid) or {}
    ctype = meta.get("type") or (pool.assigned.get("comps", {}).get(cid) or {}).get("type") or "?"
    tail = "，型号 %s" % pn if pn else ""
    return "%s：类型 %s，位号 %s，参数 %s%s" % (cid, ctype, key, val, tail)


def describe(pool: CL.ClaimPool, item: dict) -> str:
    """把一条疑点翻译成**模型看得懂的电学描述**。

    这一步是 `audit` 的核心：规则只知道「标签集合冲突」，模型需要知道
    「这条网络叫 net_3，上面挂着 GND 和 VCC 两个标签，连了 U3 的第 7 脚（引脚名 GND）、
    C13 的第 2 脚……」——否则它只能靠看图猜。
    """
    kind = str(item.get("kind") or "")
    slots = list(item.get("slots") or [])
    nets = list(item.get("nets") or [])
    lines: List[str] = []

    if nets and pool.doc:
        for nk in nets:
            n = (pool.doc.get("nets") or {}).get(nk) or {}
            labs = []
            for f in ("label", "labels", "net_label", "name"):
                v = n.get(f)
                if isinstance(v, str) and v.strip():
                    labs.append(v.strip())
                elif isinstance(v, (list, tuple)):
                    labs.extend(str(x).strip() for x in v if str(x).strip())
            refs = n.get("_refs") or []
            hg = str(n.get("hyperGraph") or "")
            parts = []
            for r in refs[:14]:
                cid = str(r).rsplit(".", 1)[0]
                pin = str(r).rsplit(".", 1)[1] if "." in str(r) else "?"
                nm = ""
                try:
                    pinrec = ((pool.doc.get("pins") or {}).get(
                        pool.chosen("comp.%s.key" % cid) or cid) or {}).get("pin_%s" % pin)
                    nm = (pinrec or {}).get("pinname") or ""
                except Exception:
                    nm = ""
                parts.append("%s.%s%s" % (cid, pin, ("(%s)" % nm) if nm else ""))
            lines.append("网络 %s：标签 %s%s；连了 %d 个引脚%s"
                         % (nk, labs or ["（没标签）"], ("｜" + hg) if hg and not refs else "",
                            len(refs), ("：" + "、".join(parts)) if parts else ""))

    for sid in slots:
        if sid.startswith("comp."):
            lines.append("元件 %s" % comp_desc(pool, _comp_of(sid)))
        elif sid.startswith("pin."):
            bits = sid.split(".")
            cid, k = (bits + ["", ""])[1], (bits + ["", ""])[2]
            b = pool.best(sid)
            lines.append("引脚 %s.%s 的读数「%s」（来自 %s）"
                         % (cid, k, getattr(b, "value", "-"), getattr(b, "source", "-")))
        elif sid.startswith("net"):
            b = pool.best(sid)
            lines.append("标签 %s 的读数「%s」" % (sid, getattr(b, "value", "-")))

    if not lines:
        lines.append(str(item.get("why") or ""))
    return "\n".join("- " + x for x in lines)


# ------------------------------------------------------------------ 取景
def _region_box(pool: CL.ClaimPool, item: dict, shape) -> Optional[Tuple[float, float, float, float]]:
    """把所有相关槽位的取景框并起来。"""
    box = None
    for sid in list(item.get("slots") or []):
        reg = pool.region(sid)
        if reg is None:
            continue
        box = reg.box if box is None else (CL.union_box(box, reg.box) or box)
    if box is None and item.get("nets") and pool.doc:
        for nk in item["nets"]:
            n = (pool.doc.get("nets") or {}).get(nk) or {}
            for seg in (n.get("_segs") or n.get("seg_index") or [])[:40]:
                try:
                    if isinstance(seg, (list, tuple)) and len(seg) >= 4:
                        b = [_f(x) for x in seg[:4]]
                    else:
                        continue
                    box = b if box is None else (CL.union_box(box, b) or box)
                except Exception:
                    continue
    if box is None:
        return None
    return CL.grow_box(box, 16.0, 0.0, shape=shape)


def panel(bgr, region_box, shape, full_w: int = FULL_W,
          max_h: int = MAX_PANEL_H):
    """上＝整图缩略，下＝问题区域放大。`vlm.client` 一次只发一张图，所以要拼。"""
    import cv2

    H, W = bgr.shape[:2]
    scale = float(full_w) / max(1.0, float(W))
    top = cv2.resize(bgr, (int(round(W * scale)), int(round(H * scale))),
                     interpolation=cv2.INTER_AREA) if scale < 1 else bgr.copy()
    if top.shape[1] != full_w:                             # 放大到正好 full_w
        top = cv2.resize(bgr, (full_w, int(round(H * full_w / max(1.0, W)))),
                         interpolation=cv2.INTER_CUBIC)
    bot = None
    if region_box:
        x0, y0, x1, y1 = [int(round(_f(v))) for v in region_box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, max(x0 + 4, x1)), min(H, max(y0 + 4, y1))
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            sub = bgr[y0:y1, x0:x1]
            sc = float(full_w) / max(1.0, float(sub.shape[1]))
            if sc > 1.0:
                sc = min(sc, 8.0)
                sub = cv2.resize(sub, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            elif sc < 1.0:
                sub = cv2.resize(sub, (full_w, max(4, int(round(sub.shape[0] * sc)))),
                                 interpolation=cv2.INTER_AREA)
            bot = sub
    if bot is None:
        return top
    sep = np.full((8, top.shape[1], 3), 255, dtype=top.dtype)
    if bot.shape[1] != top.shape[1]:                       # 补白对齐宽度
        pad = np.full((bot.shape[0], top.shape[1], 3), 255, dtype=bot.dtype)
        w = min(bot.shape[1], top.shape[1])
        pad[:, :w] = bot[:, :w]
        bot = pad
    out = np.vstack([top, sep, bot])
    if out.shape[0] > max_h:                               # 太高就整体缩一缩
        k = float(max_h) / float(out.shape[0])
        out = cv2.resize(out, (max(8, int(round(out.shape[1] * k))),
                               max(8, int(round(out.shape[0] * k)))),
                         interpolation=cv2.INTER_AREA)
    return out


# ------------------------------------------------------------------ 主流程
def _fix_ok(pool: CL.ClaimPool, item: dict, fix: str) -> Optional[str]:
    """`FIX` 能不能当成一条候选收下？只有 `key_duplicate` 允许。

    两道闸：①形状合法（`PR.DESIG_RE`）；②**不等于任何一个槽位当前的读数**。
    第二道的道理：「位号重复」这条疑点本身就是「这个值是错的」，模型把一个
    已经在用的号再报一遍等于没回答（`C5` 对 `C5`）；而撞到**别的**槽位头上
    会把冲突搬到别处，更糟。位号只由 `solve` 保证唯一，这里只负责不喂脏数据。
    """
    if str(item.get("kind")) != "key_duplicate":
        return None
    fix = str(fix or "").strip()
    if not PR.DESIG_RE.match(fix):
        return None
    taken = set()
    for s in pool.slot_ids():
        if not s.endswith(".key"):
            continue
        v = pool.chosen(s)
        if v:
            taken.add(str(v).upper())
    if fix.upper() in taken:
        return None
    slots = [s for s in (item.get("slots") or []) if str(pool.chosen(s) or "")]
    return slots[0] if slots else (list(item.get("slots") or []) or [None])[0]


def ask_one(pool: CL.ClaimPool, item: dict, client=None,
            scale: Optional[int] = None) -> dict:
    """对一个疑点发一次问。返回 `{kind, verdict, why, fix, sec, ok, raw}`。"""
    bgr = pool.bgr
    row = {"kind": item.get("kind"), "verdict": "", "why": "", "fix": "",
           "sec": 0.0, "ok": False, "raw": "", "err": ""}
    if bgr is None:
        row["err"] = "no-image"
        return row
    H, W = bgr.shape[:2]
    rbox = _region_box(pool, item, (W, H))
    img = panel(bgr, rbox, (W, H))
    desc = describe(pool, item)
    hint = RULE_HINT.get(str(item.get("kind")), str(item.get("why") or ""))
    prompt = PROMPT % (desc, hint)
    txt, err, dt = _ask(client, img, prompt)
    row["sec"] = round(float(dt), 1)
    row["raw"] = txt[:400]
    row["err"] = err
    if err or not txt:
        return row
    v, why, fix = parse(txt)
    row.update({"verdict": v, "why": why, "fix": fix, "ok": bool(v)})
    return row


def run(pool: CL.ClaimPool, sus: Optional[Sequence[dict]] = None,
        kinds: Sequence[str] = ASK_KINDS, limit: int = MAX_AUDIT,
        client=None, log=print) -> dict:
    """对疑点逐条发问；`key_duplicate` 的 `FIX` 会作为 `vlm:audit` 候选写回池子。"""
    if not ON:
        return {"n": 0, "rows": [], "sec": 0.0, "skipped": "off"}
    items = [x for x in (sus if sus is not None else PL.check(pool))
             if str(x.get("kind")) in set(kinds)]
    items = items[:max(0, int(limit))]
    t0 = time.time()
    rows: List[dict] = []
    n_bad = n_sus = n_ok = n_add = n_fix = 0
    for it in items:
        row = ask_one(pool, it, client=client)
        row["n_slots"] = len(it.get("slots") or [])
        if row["verdict"] == "bad":
            n_bad += 1
        elif row["verdict"] == "suspect":
            n_sus += 1
        elif row["verdict"] == "ok":
            n_ok += 1
        sid0 = (list(it.get("slots") or []) or [None])[0]
        if row["verdict"] in ("bad", "suspect") and row["fix"]:
            sid = _fix_ok(pool, it, row["fix"])
            if sid:
                pool.add(sid, row["fix"], score=AUDIT_SCORE, source="vlm:audit",
                         evidence={"裁决": row["verdict"], "理由": row["why"],
                                   "规则": it.get("kind")},
                         question="（位号重复）这一个应该是哪个位号？",
                         meta={"type": (pool.meta.get(sid) or {}).get("type")})
                row["added"] = sid
                n_add += 1
        # ---- 让模型改正读数：模型提候选 → 重新裁切 → 图像裁决 ------------------
        # 这一步才是用户要的「改正读数」：不直接信 `FIX`（那是第四个幻觉来源），
        # 而是拿它当**方向提示**去重新裁一次再读，只有图像给出更好的读数才收。
        if REREAD and sid0 and row["fix"]:
            try:
                from fusion import cropfix as CF
                seen = str(pool.chosen(sid0) or "")
                fr = CF.fix_slot(pool, sid0, seen, hint=row["fix"])
                row["reread"] = {k: v for k, v in fr.items() if k != "ladder"}
                if _f(fr.get("gain"), 0.0) > 0 and fr.get("ans"):
                    pool.add(sid0, fr["ans"], score=CF.FIX_SCORE, source="vlm:fix",
                             evidence={"来自": "audit:FIX", "提议": row["fix"],
                                       "裁决": row["verdict"], "理由": row["why"]},
                             question="（重裁后）这一段到底是什么？",
                             meta={"type": (pool.meta.get(sid0) or {}).get("type")})
                    n_fix += 1
            except Exception as exc:                       # pragma: no cover
                row["reread_err"] = str(exc)
        pool.note("audit", rule=it.get("kind"), verdict=row["verdict"],
                  why=row["why"], fix=row["fix"], slots=list(it.get("slots") or []))
        rows.append(row)
        if log:
            log("  [audit] %-18s %-8s %s%s" % (
                it.get("kind"), row["verdict"] or "-", row["why"] or row["err"] or "-",
                ("  FIX=%s" % row["fix"]) if row["fix"] else ""))
    return {"n": len(rows), "rows": rows, "sec": round(time.time() - t0, 1),
            "n_bad": n_bad, "n_suspect": n_sus, "n_ok": n_ok, "n_add": n_add,
            "n_fix": n_fix}


def report(rep: dict) -> str:
    """给人和 `fusion_report.json` 看的一行式摘要。"""
    if not rep or rep.get("skipped"):
        return "audit：跳过（%s）" % (rep or {}).get("skipped", "")
    return ("audit：问了 %d 条（确定错 %d / 讲不通 %d / 其实成立 %d），"
            "收下 %d 条位号候选 + %d 条重裁读数，用时 %.1fs"
            % (rep.get("n", 0), rep.get("n_bad", 0), rep.get("n_suspect", 0),
               rep.get("n_ok", 0), rep.get("n_add", 0), rep.get("n_fix", 0),
               rep.get("sec", 0.0)))


# ------------------------------------------------------------------ 自检
def _demo() -> int:
    ok: List[tuple] = []

    ok.append(("parse 全字段", parse("VERDICT=bad\nWHY=地和电源不可能同网\nFIX=VCC")
               == ("bad", "地和电源不可能同网", "VCC")))
    ok.append(("parse 无 FIX", parse("VERDICT=suspect\nWHY=讲不通\nFIX=-")
               == ("suspect", "讲不通", "")))
    ok.append(("parse 空", parse("") == ("", "", "")))
    ok.append(("parse 星号", parse("**VERDICT=ok**\n**WHY=确实可以**")[0] == "ok"))
    ok.append(("parse 词面兜底", parse("我觉得这里 bad")[0] == "bad"))
    ok.append(("parse NONE", parse("VERDICT=ok\nWHY=x\nFIX=NONE")[2] == ""))

    ok.append(("_comp_of", _comp_of("comp.c0003.key") == "c0003"))
    ok.append(("_comp_of 非法", _comp_of("net.foo") == ""))
    ok.append(("_text_w 中文算 2", _text_w("AB") == 2 and _text_w("中文") == 4))
    ok.append(("RULE_HINT 覆盖五类",
               set(ASK_KINDS) <= set(RULE_HINT)))

    # ---- 合成一个池子：两个元件共用位号 C5（真错），一条网络短接 GND/VCC -------
    doc = {
        "image": {"path": "x.png", "w": 400, "h": 300},
        "components": [
            {"id": "c0000", "type": "cap", "bbox_img": [40, 40, 80, 80]},
            {"id": "c0001", "type": "cap", "bbox_img": [140, 40, 180, 80]},
        ],
        "pins": {"c0000": [[50, 90], [70, 90]], "c0001": [[150, 90], [170, 90]]},
        "nets": {
            "net_1": {"label": "GND", "_refs": ["c0000.1", "c0001.1"],
                      "_segs": [[40, 90, 180, 90]], "hyperGraph": "(c0000.1,c0001.1)"},
        },
    }
    assigned = {"comps": {
        "c0000": {"type": "cap", "designator": None, "value": None, "pins": {}},
        "c0001": {"type": "cap", "designator": None, "value": None, "pins": {}},
    }, "desig_slots": {}, "net_labels": []}
    pool = CL.ClaimPool("demo", doc, assigned)
    pool.bgr = np.full((300, 400, 3), 255, dtype=np.uint8)
    pool.add("comp.c0000.key", "C5", 1.0, "ocr:designator")
    pool.add("comp.c0001.key", "C5", 0.9, "ocr:designator")
    pool.add("comp.c0000.key", "C6", 0.2, "prior:run_inwindow")
    pool.set_region("comp.c0000.key", [40, 40, 80, 80], [40, 40, 80, 80])
    pool.set_region("comp.c0001.key", [140, 40, 180, 80], [140, 40, 180, 80])

    sus = PL.check(pool, pred=doc)
    kinds = {s["kind"] for s in sus}
    ok.append(("plausible 报出 key_duplicate", "key_duplicate" in kinds))

    dup = [s for s in sus if s["kind"] == "key_duplicate"][0]
    d = describe(pool, dup)
    ok.append(("describe 含两个元件", d.count("元件") == 2 and "C5" in d))

    # ---- panel：上下拼接，宽度一致 --------------------------------------
    p = panel(pool.bgr, (40, 40, 180, 90), (400, 300), full_w=200)
    ok.append(("panel 高度 > 单张", p.shape[0] > 60 and p.shape[1] == 200))
    p2 = panel(pool.bgr, None, (400, 300), full_w=200)
    ok.append(("panel 无区域时只有上半", p2.shape[1] == 200))

    # ---- _fix_ok：C6 合法且没被占 ⇒ 收下；C5 撞号 ⇒ 拒收 -------------------
    sid = _fix_ok(pool, dup, "C6")
    ok.append(("_fix_ok 收下未占的号", sid in ("comp.c0000.key", "comp.c0001.key")))
    ok.append(("_fix_ok 拒收已占的号", _fix_ok(pool, dup, "C5") is None))
    ok.append(("_fix_ok 拒收非法形状", _fix_ok(pool, dup, "5C") is None))
    ok.append(("_fix_ok 只管 key_duplicate",
               _fix_ok(pool, {"kind": "value_outlier", "slots": ["comp.c0000.key"]}, "C6") is None))

    # ---- 离线跑一遍（client 假回答） ------------------------------------
    fake = lambda img, prompt, timeout=None: (                  # noqa: E731
        "VERDICT=bad\nWHY=地和电源不可能同网\nFIX=C7", 0.3)
    rep = run(pool, sus=sus, limit=3, client=fake, log=None)
    ok.append(("run 有行", rep["n"] >= 1))
    ok.append(("run 记了裁决", any(r["verdict"] == "bad" for r in rep["rows"])))
    ok.append(("run 收下候选", rep["n_add"] >= 1))
    ok.append(("run 池子里多了 vlm:audit",
               any(a.source == "vlm:audit" for a in pool.alts("comp.c0000.key"))
               or any(a.source == "vlm:audit" for a in pool.alts("comp.c0001.key"))))
    ok.append(("report 是一行", report(rep).startswith("audit：问了")))
    ok.append(("OFF 时跳过", report(run(pool, sus=sus, limit=0, client=fake, log=None)) != ""))

    bad = [n for n, c in ok if not c]
    print("audit 自检 %d 项：%s" % (len(ok), "全部通过" if not bad else "失败 %s" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        # 用法：python fusion/audit.py --live 0001 [0002 ...]
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cases", nargs="*", default=["0001"])
        ap.add_argument("--limit", type=int, default=MAX_AUDIT)
        a = ap.parse_args(sys.argv[2:])
        run_dir = Path(os.environ.get(
            "FUSION_RUN_DIR", r"Y:\workspace\EDA\work\_submit_out220r36\runs"))
        vlm_dir = Path(os.environ.get(
            "FUSION_VLM_DIR", r"Y:\workspace\EDA\work\_fusion_vlm"))
        total = {"n": 0, "n_bad": 0, "n_suspect": 0, "n_ok": 0, "n_add": 0, "sec": 0.0}
        for case in a.cases:
            pred = json.loads((run_dir / "stage2" / ("%s.json" % case)).read_text("utf-8"))
            asg = json.loads((run_dir / "stage2" / "_assigned" / ("%s.json" % case))
                             .read_text("utf-8"))
            vp = vlm_dir / ("%s.json" % case)
            vd = json.loads(vp.read_text("utf-8")) if vp.exists() else None
            s1 = json.loads((run_dir / "stage1" / ("%s.stage1.json" % case))
                            .read_text("utf-8"))
            pool = CL.from_file(case, run_dir / "stage1" / ("%s.stage1.json" % case),
                                run_dir / "stage2" / "_assigned" / ("%s.json" % case),
                                vlm=vd)
            # 先把先验与解跑上，让 `chosen` 是**真要送去装配的那个值** —— 不然
            # `plausible` 看到的是「没有先验时的读数」，报出来的疑点不是现场那批。
            try:
                from fusion import priors as _PR
                from fusion import solve as _SV
                _PR.Priors().annotate(pool)
                _sol = _SV.solve(pool)
                _SV.write_back(pool, _sol)
            except Exception as _exc:                      # pragma: no cover
                print("   [warn] 先验/解跳过：%s" % _exc)
            print("== %s  bgr=%s  槽位 %d"
                  % (case, None if pool.bgr is None else pool.bgr.shape,
                     len(pool.slot_ids())))
            rep = run(pool, limit=a.limit, log=print)
            print("   %s" % report(rep))
            for k in total:
                total[k] = total.get(k, 0) + rep.get(k, 0)
        print("合计：%s" % report(total))
        sys.exit(0)
    sys.exit(_demo())
