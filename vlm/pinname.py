"""引脚名文字框「紧框复读」通道：把 CRNN 读错的芯片引脚名用 VLM 复读补回来。

★ 默认 **开**（`FUSION_PINNAME=1`）。设计与全部读数见 `docs/RESULTS.md` §5.86。

为什么是这条口径（§5.82 的反面证据 → §5.85 的正面证据）：
§5.82 的引脚名 VLM 通道用「沿引脚轴 90px 的带子 + 2AFC 菜单」，端到端 A/B **−0.97**（4 次采纳全坏，
模型把邻行的名字读进来）；§5.85 证明值上的杠杆是 **用 CRNN 自己那个文字框、只问「这段文字是什么」**。
`runs/fusion/_assigned/<case>.json` 里 `comps[*].pins[*].name` 就是 CRNN 读引脚名时用的那个框
（`{text, conf, box}`，与 `value` 同构）⇒ 不需要任何几何猜测。

**官方 10 张的全部读数**（`work/_pinv10.py` → `work/_pinv10.jsonl`，探针与模块同一 prompt/几何）：

1. 109 个能配上 GT 的引脚名槽位，当前错 6 个（其中 5 个真的扣分，`10/U2.3 Vin` 那条是
   大小写、尺子不计）；把 5 条全换成 GT = **+0.94 分**（`work/_pinoracle.py`：81.14 → 82.08）。
2. 换成「CRNN 自己的框 + 只问这段文字」后，模型**只改对 1 条**：`7/U1.4 CND → GND`（+0.33 分）。
   另外 4 条改不动，原因是**框根本不是那个名字**：
   * `6/U4.4 N`：GT `EN`，框 `[1646,208,1654,218]` 只有 8×10 —— 检测器把 `EN` 切成了碎片；
   * `6/U4.5 N`：GT `IN`，框 `[1645,207,1653,217]` 同样 8×10；
   * `5/U3.1 Q`：GT `ADJ(GND)`，框 `[559,723,567,731]` 8×8，模型答 `UNCERTAIN`；
   * `5/U3.2 ADIGND`：框 `[559,734,626,747]` 67×13 其实是 **pin 1 的 `ADJ(GND)`**（竖排串行）；
   ⇒ 这 4 条是**检测/归属**问题，不是读数问题。
3. **加宽裁片被否证**（`work/_pinv10w.py`，18 个槽位 ×3 种窗口）：
   base（原框+3px）1 改对 / 1 改坏；L（沿轴 −2.5h/+0.25h）1 改对 / **5 改坏**；
   S（方窗 3·max(w,h)）1 改对 / **7 改坏** —— 加宽会把邻行的名字吞进来（竖排引脚读成 `IN GND`、
   `1 IN 2 GND 3 OUT`）。**所以本通道只用紧框。**
4. **触发面**：`conf<0.85 或 w<=10` = 20 次调用（2 次/图），已覆盖全部 5 条真错里的 4 条；
   conf 单独当闸门不行（真错的 conf 有 0.994 / 0.927 的）。
5. **采信闸门**（`GUARD=1`，写代码前先用 `work/_pinapply.py` 离线裁定过）：
   答案必须**纯字母/括号/下划线/连字符、2~12 字**（引脚名），且
   ① 不等于当前预测名；② 不等于本元件位号；③ **不是位号形**（`U1`/`R12`：字母 ≤4 且**带数字**）；
   ④ **不与同元件别的脚已有的名字重复**。
   ④ 是必需的：case 4 的 `U2.3`（当前 `VIN` = GT，正确）有**两个**名字框被归给它，
   另一个 `[1398,232,1445,244]` 其实是 **pin 4 的 `VOUT`**，模型照实读成 `VOUT`
   —— 没有 ④ 就会把正确的 `VIN` 改成 `VOUT`（−0.08 分）。

★★ **端到端 A/B（官方 10 张，`work/_fast_out_pn{off,on}`，`-vlm-pin off`，2026-10）**：

    臂                          总分      元件 F1  引脚 F1   调用
    off（`FUSION_PINNAME=0`）   81.41     0.9963   0.7731     0
    on（`=1`，交付）            **81.75** 0.9963   **0.7864** 20     ← **+0.34 分**

逐案只有 case 7 变（**83.6435 → 86.9768**，引脚 F1 0.8000 → 0.9333，tp 6→7、fp 1→0），
其余 9 张**逐位相同**；20 次调用里只采纳 1 个槽位（`7/U1.4 CND→GND`）⇒ **零误伤**。
20 张广义回归（10GT + 200 池最差 10 张）：10GT **+0.34**，200 池上采信的 5 次全部**中性**
（同批内那三张分数逐位不变）；那条池子的 ±0.2~1.4 抖动与本通道无关（**没有采纳的三张**
也在抖：0092 单独重跑两次都是 0.71）⇒ 泛化安全。

**离线裁定**（写代码前先做，`work/_pinapply.py --rule g2 --conf 0.85 --maxw 10`）：
把 `work/_pinv10.jsonl` 的答案按上面规则写回预测再打分 = 81.14 → **81.47**；
conf<0.90 / 0.95 / 全问分别是 30/37/109 次调用，**分数都是 81.47**（多采纳的 2 条是中性改写：
改的仍是错名 ⇒ 尺子不动）⇒ 所以取最省的闸门。

旋钮：`FUSION_PINNAME`（**默认 1**）、`FUSION_PINNAME_CONF`（默认 0.85）、
`FUSION_PINNAME_MAXW`（框宽阈值，默认 10）、`FUSION_PINNAME_SCALE`（默认 4）、
`FUSION_PINNAME_PAD`（默认 3）、`FUSION_PINNAME_MAX`（槽位上限，默认 60）、
`FUSION_PINNAME_CALLS`（调用上限，默认 120）、`FUSION_PINNAME_GUARD`（默认 1）。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import cv2

#: 引脚名文字框周围余量（像素，图像坐标系）；与 `work/_pinv10.py` 一致
PAD = 3
#: prompt 与 `work/_pinv10.py` 逐字一致（改动会让 §5.86 的表失效）
PROMPT = ("这是电路原理图上芯片引脚旁边的**引脚名**文字（多为字母，例如 VDD、GND、EN、SDA、"
          "ADJ(GND)）。\n只输出你看到的字符串本身，不要解释、不要标点、不要规范化、不要补全。\n"
          "看不清就只输出 UNCERTAIN。")
#: 「像引脚名」的答案形状：纯字母/括号/下划线/连字符
_ANS = re.compile(r"^[A-Za-z_()/\-]{2,12}$")
#: 位号形（必须带数字，避免把 GND/EN 这种纯字母名误判成位号）
_DESIG = re.compile(r"^[A-Za-z]{1,4}\d{1,4}$")
_BAD = ("uncertain", "unclear", "unknown", "看不清", "不确定", "n/a", "none")


def _env_on(name: str, default: str) -> bool:
    v = (os.environ.get(name) or default).strip().lower()
    return v not in ("", "0", "false", "off", "no")


def _env_int(name: str, default: int) -> int:
    try:
        return int(float((os.environ.get(name) or "").strip() or default))
    except Exception:                                            # noqa: BLE001
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float((os.environ.get(name) or "").strip() or default)
    except Exception:                                            # noqa: BLE001
        return default


ON = _env_on("FUSION_PINNAME", "1")
CONF = _env_float("FUSION_PINNAME_CONF", 0.85)
MAXW = _env_float("FUSION_PINNAME_MAXW", 10.0)
SCALE = max(1, _env_int("FUSION_PINNAME_SCALE", 4))
MAX_SLOTS = _env_int("FUSION_PINNAME_MAX", 60)
MAX_CALLS = _env_int("FUSION_PINNAME_CALLS", 120)
GUARD = _env_on("FUSION_PINNAME_GUARD", "1")


def _first(txt: str) -> str:
    t = (txt or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    return t.splitlines()[0].strip() if t else ""


def looks_name(txt: str) -> bool:
    """答案是否形如一个引脚名（纯字母/括号），且不是弃权/解释。"""
    t = _first(txt)
    if not t or len(t) > 12:
        return False
    low = t.lower()
    if any(b in low for b in _BAD) or "=" in t or ":" in t or " " in t:
        return False
    return bool(_ANS.match(t))


def why(cur: str, conf, w: float) -> str:
    """触发判据：返回理由字符串，或 None（不问）。"""
    c = (cur or "").strip()
    try:
        cf = float(conf)
    except Exception:                                            # noqa: BLE001
        cf = 0.0
    if cf < CONF:
        return "lowconf"
    if w <= MAXW:
        return "narrow"
    return ""


def _box(val):
    b = (val or {}).get("box")
    if not b or len(b) < 4:
        return None
    try:
        return [float(v) for v in b[:4]]
    except Exception:                                            # noqa: BLE001
        return None


def _num_key(pins_sub, num: str):
    """在预测的 pins 子字典里按引脚号找键（键形如 `pin_7`；号本身可能被读错，故用文字比）。"""
    for pk in pins_sub or {}:
        try:
            from schematic_parser import metrics as _m              # noqa: PLC0415
            if _m.pin_key_to_number(pk) == str(num):
                return pk
        except Exception:                                        # noqa: BLE001
            pass
    return None


def slots(assigned, doc):
    """从 `_assigned` 账里挑出「有引脚名文字框 + 触发成立」的槽位（不调用模型）。"""
    out = []
    pins = (doc or {}).get("pins") or {}
    comps = (doc or {}).get("components") or {}
    for _cid, comp in ((assigned or {}).get("comps") or {}).items():
        if not isinstance(comp, dict):
            continue
        des = comp.get("designator") or {}
        key = des.get("canon") or des.get("text")
        if not key or key not in comps:
            continue
        # pin_f1 只统计 GT 里 box(IC) 元件的引脚；非 IC（发光管/二极管等）即使有名字也不碰
        try:
            from schematic_parser import metrics as _m              # noqa: PLC0415
            if _m.canon_type((comps.get(key) or {}).get("type")) not in _m.IC_TYPES:
                continue
        except Exception:                                        # noqa: BLE001
            pass
        psub = pins.get(key) or {}
        for _pk, pv in (comp.get("pins") or {}).items():
            nm = pv.get("name")
            if not isinstance(nm, dict):
                continue
            box = _box(nm)
            if not box:
                continue
            num = str((pv.get("number") or {}).get("text"))
            pk = _num_key(psub, num)
            if pk is None:
                continue                    # 预测里没有这根脚 ⇒ 改名拿不到分
            cur = (psub.get(pk) or {}).get("pinname")
            if cur in (None, ""):
                cur = nm.get("text")
            w = box[2] - box[0]
            r = why(str(cur or ""), nm.get("conf"), w)
            if not r:
                continue
            others = [((psub.get(o) or {}).get("pinname")) for o in psub if o != pk]
            out.append({"key": key, "pin_key": pk, "num": num, "box": box,
                        "cur": None if cur is None else str(cur), "conf": nm.get("conf"),
                        "assigned": nm.get("text"), "why": r, "others": others,
                        "desig": str(key)})
    return out


def _ask(client, img, prompt):
    try:
        txt, _dt = client.ask(img, prompt)
    except Exception:                                            # noqa: BLE001
        return ""
    return txt or ""


def apply(img, doc, assigned, client=None, opts=None):
    """就地改写 doc 里的引脚名。返回统计 dict（不抛异常）。"""
    opts = opts or {}
    st = {"slots": 0, "calls": 0, "adopted": 0, "unchanged": 0, "rejected": 0, "skipped": 0,
          "conf": CONF, "maxw": MAXW, "guard": int(bool(GUARD)), "trace": []}
    if client is None:
        try:
            from vlm import client as _c                          # noqa: PLC0415
            client = _c
        except Exception as exc:                                  # noqa: BLE001
            st["error"] = "no client: %s" % exc
            return st
    # ★ 与 pinzoom/valread 同因：不要用 `client.available()` 当闸门（本机端点无 KEY 时返回
    #   False，而它其实可用）。问不动就一个字都改不动，天然安全。
    if hasattr(client, "budget_exhausted") and client.budget_exhausted():
        st["error"] = "budget exhausted"
        return st
    max_slots = int(opts.get("max_slots", _env_int("FUSION_PINNAME_MAX", MAX_SLOTS)))
    max_calls = int(opts.get("max_calls", _env_int("FUSION_PINNAME_CALLS", MAX_CALLS)))
    scale = int(opts.get("scale", _env_int("FUSION_PINNAME_SCALE", SCALE)))
    pad = int(opts.get("pad", _env_int("FUSION_PINNAME_PAD", PAD)))
    guard = bool(opts.get("guard", GUARD))
    todo = slots(assigned, doc)[:max_slots]
    pins = doc.get("pins") or {}
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
        rec = {"key": t["key"], "num": t["num"], "cur": t["cur"], "ans": ans,
               "conf": t["conf"], "why": t["why"], "box": t["box"], "adopted": None}
        if not looks_name(ans):
            st["rejected"] += 1
            rec["note"] = "not_name"
        elif ans == t["cur"]:
            st["unchanged"] += 1
            rec["note"] = "same"
        elif guard and (ans.upper() == t["desig"].upper() or _DESIG.match(ans)):
            st["rejected"] += 1
            rec["note"] = "designator_like"
        elif guard and any(o and str(o).upper() == ans.upper() for o in (t["others"] or ())):
            # case 4 的 `U2.3`：两个名字框被归到同一根脚，另一个其实是 pin 4 的 VOUT
            st["rejected"] += 1
            rec["note"] = "dup_other_pin"
        else:
            pins[t["key"]][t["pin_key"]]["pinname"] = ans
            st["adopted"] += 1
            rec["adopted"] = True
        if rec["adopted"] is None:
            rec["adopted"] = False
        st["trace"].append(rec)
    st["slots"] = len(todo)
    return st


def _write_json(path, doc):
    """原子落盘：先写同目录 `.tmp` 再 `os.replace`（同卷、原子）。

    ★ 加固（与 `submission/run.py:_atomic_write_text` 同一条理由）：本通道**就地改写**已经
    落进 `final_results/` 的 `<case>_Pred.json`；判题机按墙钟硬杀是现实场景，被杀在
    `write_text` 一半就留下半截 JSON ⇒ 该 case 直接 0 分。先写 `.tmp` 再 `os.replace`，
    任何时刻被读到的一定是完整的旧文件或完整的新文件。
    """
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    os.replace(str(tmp), str(p))


def apply_file(png_path, assigned_json, src_json, dst_json, opts=None):
    """读图/文档/账本，改写引脚名，写回 dst_json。返回统计或 None（读不了图就 None）。"""
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
