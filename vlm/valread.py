"""值文字框「紧框重读」通道：把 CRNN 读错的值（`10uF`↔`10nF` 的 n/u 混淆）用 VLM 复读补回来。

设计与全部读数见 `docs/RESULTS.md` §5.85（官方 10 张，`work/_valv10.py` / `work/_valv10.jsonl`）：

1. **槽位来自管线自己的账**：`runs/fusion/_assigned/<case>.json`（回退 `runs/stage2/_assigned/`）
   里每颗元件都记着**值取自哪个文字框**（`comps[*].value.{box,text,conf,_by}`）⇒ 不需要
   任何几何猜测，直接拿那个框（`+PAD` 像素）放大 `SCALE` 倍问模型。
2. **prompt 只问值**（不提位号）：`work/_valwhy.py` 的四变体对照——
   「整块元件裁片 + `位号=值`」（=线上 `vlm/enrich.py` 现在的问法）在 case 1 上 5/13 个槽位
   因为答案配错了位号被 `value_key_gate` 挡掉（其中 C154/C158 的值其实被读对了）；
   「值文字框 + 只问值」则 **13/13 全对**，且不误伤任何本来就对的槽位。
3. **触发面与收益**（10 张、117 个有 GT 值的槽位；当前值错 2 个 = C154/C158 的 `10uF`）：

      触发闸门                     调用   改对   改坏
      全问（117 个槽位）            117     2      0
      *F 形（默认 `f`）              73     2      0
      *F 形 且 conf<0.90             20     2      0
      uF 形（`uf`）                  13     2      0
      conf<0.80（`lowconf`）          7     1      0     ← 漏掉 conf 0.87 的 C154

   ⇒ **confidence 不能当闸门**（C154 错读的 conf 是 0.8726，比很多正确读还高）；
   真正可分的是**值的形状**（电容值 `*F`）。默认 `f`，`uf` 是省 token 的等价档。
4. **采信规则**：只采纳「答案与当前读**数字部分相同、只有单位/字母不同**」（`GUARD=1`）
   且答案形如值（非 `UNCERTAIN`、无 `=`、长度 ≤12）的槽位。
   这既是本样本的充要条件（两条真错都是 `10uF`→`10nF`），也是对 **pinzoom 式误采**
   （模型把邻居的值读进来）最直接的防线：邻居的值极少有完全相同的数字前缀。
   `FUSION_VALREAD_GUARD=0` 可放宽成「只要答案像值就采纳」。
5. **token**：单次 ≈130 token（§5.41 的账；本通道裁片更小），默认档 73 次 ≈9.5K token
   （交付档基数 152K）⇒ token 项最坏损失远小于值通道的 +0.27 分。

★★ **端到端 A/B（官方 10 张，`work/_fast_out_vr{on} / _ckey`，2026-10；`FUSION_VALREAD=1`）**：

    臂           总分     元件 F1   tp
    off(ckey)   81.14    0.9872   250
    on(vron)    **81.41** 0.9963   254     ← **+0.27 分**

    只有 case 1 动（81.46 → 84.19，comp 0.9091 → 1.0000），其余 9 张逐位不变；
    73 个槽位里**只采纳 2 个**（`C154`、`C158` 的 `10uF→10nF`），**零误伤**；
    墙钟 121.7 s → 132.9 s；退出码 0。⇒ 默认 **1（已上线）**。

旋钮：`FUSION_VALREAD`（**默认 1 = 开**）、`FUSION_VALREAD_TRIGGER`
（`f`/`uf`/`f+lowconf`/`lowconf`/`all`，默认 `f`）、`FUSION_VALREAD_CONF`（默认 0.90）、
`FUSION_VALREAD_SCALE`（默认 4）、`FUSION_VALREAD_PAD`（默认 3）、`FUSION_VALREAD_MAX`
（槽位上限，默认 40）、`FUSION_VALREAD_CALLS`（调用上限，默认 80）、
`FUSION_VALREAD_GUARD`（默认 1）、`FUSION_VALREAD_UF_ONLY`（旧开关，等价 `TRIGGER=uf`）。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import cv2

#: 值文字框周围的余量（像素，图像坐标系）
PAD = 3
#: prompt 与 `work/_valv10.py` 逐字一致（改动会让 §5.85 的表失效）
PROMPT = ("这是电路原理图上某颗元件旁边的**参数值**文字（例如 10K、4.7uF、100nF、10nF）。\n"
          "只输出你看到的字符串本身，不要解释、不要标点、不要规范化、不要补位号。\n"
          "看不清就只输出 UNCERTAIN。")
#: 电容值形状（默认触发闸门）
_F = re.compile(r"^\s*\d+(?:\.\d+)?\s*(?:p|n|u|μ|m)?F\s*$", re.IGNORECASE)
_UF = re.compile(r"^\s*\d+(?:\.\d+)?\s*(?:u|μ)F\s*$", re.IGNORECASE)
#: 「像值」的答案形状：数字 + 至多 3 个字母/符号单位（挡掉 `C153=10nF`、整句解释等）
_ANS = re.compile(r"^\s*\d+(?:\.\d+)?\s*[a-zA-ZμΩ%]{0,3}\s*$")
_NUM = re.compile(r"(\d+(?:\.\d+)?)")
_BAD = ("uncertain", "unclear", "unknown", "看不清", "不确定", "n/a", "none")


def _env_on(name: str, default: str) -> bool:
    v = (os.environ.get(name) or default).strip().lower()
    return v not in ("", "0", "false", "off", "no")


def _env_int(name: str, default: int) -> int:
    try:
        return int(float((os.environ.get(name) or "").strip() or default))
    except Exception:                                            # noqa: BLE001
        return default


ON = _env_on("FUSION_VALREAD", "1")
TRIGGER = ((os.environ.get("FUSION_VALREAD_TRIGGER") or "").strip().lower()
           or ("uf" if _env_on("FUSION_VALREAD_UF_ONLY", "0") else "f"))
CONF = float((os.environ.get("FUSION_VALREAD_CONF") or "").strip() or 0.90)
SCALE = max(1, _env_int("FUSION_VALREAD_SCALE", 4))
MAX_SLOTS = _env_int("FUSION_VALREAD_MAX", 40)
MAX_CALLS = _env_int("FUSION_VALREAD_CALLS", 80)
GUARD = _env_on("FUSION_VALREAD_GUARD", "1")


def _first(txt: str) -> str:
    t = (txt or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    return t.splitlines()[0].strip() if t else ""


def looks_value(txt: str) -> bool:
    """答案是否形如一个值（数字 + 可选短单位），且不是弃权/解释。"""
    t = _first(txt)
    if not t or len(t) > 12:
        return False
    low = t.lower()
    if any(b in low for b in _BAD) or "=" in t or ":" in t:
        return False
    return bool(_ANS.match(t))


def num_of(txt: str) -> str:
    m = _NUM.search(txt or "")
    return m.group(1) if m else ""


def why(cur: str, conf):
    """触发判据：返回理由字符串，或 None（不问）。"""
    c = (cur or "").strip()
    try:
        cf = float(conf)
    except Exception:                                            # noqa: BLE001
        cf = 0.0
    t = TRIGGER
    is_f = bool(_F.match(c))
    is_uf = bool(_UF.match(c))
    if t in ("all", "any", "1"):
        return "all"
    if t == "uf":
        return "uf" if is_uf else None
    if t == "lowconf":
        return "lowconf" if cf < CONF else None
    if t == "f+lowconf":
        return "f" if is_f else ("lowconf" if cf < CONF else None)
    return "f" if is_f else None            # 默认 "f"


def _box(val):
    b = (val or {}).get("box")
    if not b or len(b) < 4:
        return None
    try:
        return [float(v) for v in b[:4]]
    except Exception:                                            # noqa: BLE001
        return None


def slots(assigned, doc):
    """从 `_assigned` 账里挑出「有值文字框 + 触发成立」的槽位（不调用模型）。"""
    out = []
    comps = (doc or {}).get("components") or {}
    for _cid, comp in ((assigned or {}).get("comps") or {}).items():
        if not isinstance(comp, dict):
            continue
        des = comp.get("designator") or {}
        val = comp.get("value") or {}
        key = des.get("canon") or des.get("text")
        box = _box(val)
        if not key or not box or key not in comps:
            continue
        cur = (comps.get(key) or {}).get("value")
        if cur in (None, ""):
            cur = val.get("text")
        r = why(str(cur or ""), val.get("conf"))
        if not r:
            continue
        out.append({"key": key, "box": box, "cur": None if cur is None else str(cur),
                    "conf": val.get("conf"), "by": val.get("_by"),
                    "assigned": val.get("text"), "why": r})
    return out


def _ask(client, img, prompt):
    try:
        txt, _dt = client.ask(img, prompt)
    except Exception:                                            # noqa: BLE001
        return ""
    return txt or ""


def apply(img, doc, assigned, client=None, opts=None):
    """就地改写 doc 里的值。返回统计 dict（不抛异常）。"""
    opts = opts or {}
    st = {"slots": 0, "calls": 0, "adopted": 0, "unchanged": 0, "rejected": 0,
          "skipped": 0, "trigger": TRIGGER, "guard": int(bool(GUARD)), "trace": []}
    if client is None:
        try:
            from vlm import client as _c                          # noqa: PLC0415
            client = _c
        except Exception as exc:                                  # noqa: BLE001
            st["error"] = "no client: %s" % exc
            return st
    # ★ 与 pinzoom 同因：不要用 `client.available()` 当闸门（本机端点无 KEY 时返回 False，
    #   而它其实可用）。问不动就一个字都改不动，天然安全。
    if hasattr(client, "budget_exhausted") and client.budget_exhausted():
        st["error"] = "budget exhausted"
        return st
    H = img.shape[0]
    max_slots = int(opts.get("max_slots", _env_int("FUSION_VALREAD_MAX", MAX_SLOTS)))
    max_calls = int(opts.get("max_calls", _env_int("FUSION_VALREAD_CALLS", MAX_CALLS)))
    scale = int(opts.get("scale", _env_int("FUSION_VALREAD_SCALE", SCALE)))
    pad = int(opts.get("pad", _env_int("FUSION_VALREAD_PAD", PAD)))
    guard = bool(opts.get("guard", GUARD))
    todo = slots(assigned, doc)[:max_slots]
    comps = doc.get("components") or {}
    for t in todo:
        if st["calls"] >= max_calls:
            st["skipped"] += 1
            continue
        x0, y0, x1, y1 = t["box"]
        crop = img[max(0, int(y0) - pad):int(y1) + pad, max(0, int(x0) - pad):int(x1) + pad]
        if crop is None or not getattr(crop, "size", 0):
            st["skipped"] += 1
            continue
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ans = _first(_ask(client, big, PROMPT))
        st["calls"] += 1
        rec = {"key": t["key"], "cur": t["cur"], "ans": ans, "conf": t["conf"],
               "why": t["why"], "by": t["by"], "box": t["box"], "adopted": None}
        if not looks_value(ans):
            st["rejected"] += 1
            rec["adopted"] = False
            rec["note"] = "not_value"
        elif ans == t["cur"]:
            st["unchanged"] += 1
            rec["adopted"] = False
            rec["note"] = "same"
        elif guard and num_of(ans) != num_of(t["cur"] or ""):
            st["rejected"] += 1
            rec["adopted"] = False
            rec["note"] = "num_differs"
        else:
            comps[t["key"]]["value"] = ans
            st["adopted"] += 1
            rec["adopted"] = True
        st["trace"].append(rec)
    st["slots"] = len(todo)
    return st


def _write_json(path, doc):
    """原子落盘：先写同目录 `.tmp` 再 `os.replace`（同卷、原子）。

    ★ 加固（与 `submission/run.py:_atomic_write_text` 同一条理由）：本通道是**就地改写**
    已经落进 `final_results/` 的 `<case>_Pred.json`。判题机按墙钟硬杀是现实场景（见
    `submission/run.py` 渐进导出的注释），若被杀在 `write_text` 一半，这一张就是半截 JSON
    ⇒ 该 case 直接 0 分。先写 `.tmp` 再 `os.replace` 之后，任何时刻被读到的一定是完整的
    旧文件或完整的新文件。
    """
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    os.replace(str(tmp), str(p))


def apply_file(png_path, assigned_json, src_json, dst_json, opts=None):
    """读图/文档/账本，改写值，写回 dst_json。返回统计或 None（读不了图就 None）。"""
    img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if not Path(assigned_json).is_file():
        return {"slots": 0, "calls": 0, "adopted": 0, "error": "no assigned ledger"}
    with open(src_json, encoding="utf-8") as f:
        doc = json.load(f)
    with open(assigned_json, encoding="utf-8") as f:
        assigned = json.load(f)
    st = apply(img, doc, assigned, opts=opts)
    _write_json(dst_json, doc)
    return st
