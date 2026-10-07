# -*- coding: utf-8 -*-
"""把 LLM 的结论**定向回传**到 stage1/stage2 —— 跨阶段的那一段。

与邻居的分工（别混）：

- `fusion/llm_judge.py` / `fusion/audit.py`：只产**结论**（写进 claim 池）。
- `fusion/rescan.py`：在**文字**层面重读（把某个槽位的框再裁一次、再问一次）。
- **本模块**：当结论说明「上游的**几何**本来就错了」时，光重读文字没用，必须把
  参数带回 stage1 或 stage2 重跑。

### 为什么这件事非做不可

官方 `pin_f1`（`eda-pcb/schematic_parser/metrics.py:426-464`）的键是
`(元件, pin_key_to_number(pk))`，命中之后**还要 `pinname` 逐字相等**，不等就
`fp+=1` 且 `fn+=1`。而 GT 全库 9356 个 box 引脚的 `pinname` **没有一个是空的**。
所以「发了但名字读错」= 1 fp + 1 fn，「干脆不发」= 1 fn —— 名字是硬闸；
另一半是**号**（键）。

现状：几何已经能打通（pincol 让 0143 的 U1 从 GT 点命中 3/64 变成 63/64），
但提交文档里 U1 只有 13~45 个脚、且 `pin` F1 只有 0.25。

### 实测：VLM 读 pincol 的窗口是**全中**的，而且两个窗口各给一半

`work/tools/diag/_vlmnamewin.py --case 0143 --key c0003 --per 8 --cols 4 --dump`

| 窗 | `num` 字段 | `name` 字段 |
|---|---|---|
| `in_win`（框内） | 噪声 | **引脚名，逐字准** —— 54 个窗回 54/54：`VBAT / PC13/TRC_AF1 / PC14/OSC32_IN / PC15/OSC32_OUT / PH0/OSC_IN- / PH1/OSC_OUT / Nrst / PC0 …` |
| `out_win`（框外） | **引脚号，准** —— L 边 `1 2 3 4 5 6 7 8 10 11 … 33`，R 边 `64 63 62 61 60 56 49 48 … 34`（正是 GT 的 `64 VDD 63 VSS 62 PB9 61 PB8 60 BOOT0 … 34 PB13 33 PB12`） | 网络名（`3.3V / GND / NC / SCK`） |

对比 `eda-pcb/vlm/enrich.py:read_pins`（用它的 `adaptive_window(inner=160,
cross=58, outer=34)`）在同一个元件上**只回 13 个**、还大量重复 `PC0`。
⇒ **不是 VLM 不会读，是它的窗口没对准引脚。** 本模块直接用 pincol 的格点窗，
**号取自 `out_win` 的 `num`，名取自 `in_win` 的 `name`**。

### 关键顺序：必须改在 `build_doc` **之前**

`eda-pcb/stage2/assemble.py` 的 `BOX_PIN_NEED_NAME`（默认开）会把**没有名字**的
box 引脚整条丢掉。所以「先装配、再补名字」是补不回来的 —— 名字得先落进
`assigned[...]["pins"][k]["name"]`，`build_doc` 才会把这些脚放出来。
调用点在 `fusion/run.py` 里 `_assemble(...)` **之前**。
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

# ---- 开关（全部可环境变量覆盖）------------------------------------------------
ON = os.environ.get("FUSION_ITERATE", "1").strip().lower() not in (
    "", "0", "false", "off", "none")
NAME_WIN = os.environ.get("FUSION_ITERATE_WIN", "in_win")
NUM_WIN = os.environ.get("FUSION_ITERATE_NUM_WIN", "out_win")
PER = int(os.environ.get("FUSION_ITERATE_PER", "8"))
COLS = int(os.environ.get("FUSION_ITERATE_COLS", "4"))
SIDE = int(os.environ.get("FUSION_ITERATE_SIDE", "150"))
PAD = float(os.environ.get("FUSION_ITERATE_PAD", "3"))
MIN_PINS = int(os.environ.get("FUSION_ITERATE_MIN_PINS", "4"))
MAX_PINS = int(os.environ.get("FUSION_ITERATE_MAX_PINS", "120"))
# 只补「还没读到」的字段？默认**否** —— 实测 CRNN 读的名字（`P31-TRAH1`）与
# `assemble` 的位置号兜底都比 VLM 差得多。
KEEP_OURS = os.environ.get("FUSION_ITERATE_KEEP_OURS", "0").strip().lower() not in (
    "", "0", "false", "off", "none")
MIN_NAME_LEN = int(os.environ.get("FUSION_ITERATE_MIN_NAME_LEN", "2"))
# 号：`1` / `12` / `A1` / `AB12` 这种；纯字母（`NC`/`VSS`/`P`）一律拒收
NUM_RE = re.compile(r"^[A-Za-z]{0,2}\d{1,3}$")
SCORE = float(os.environ.get("FUSION_ITERATE_SCORE", "0.9"))


def _crop(bgr, win, pad: float = PAD):
    """按窗口裁一张，越界就夹回图内；太小的返回 None。"""
    if bgr is None or not win:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in win]
    except Exception:                                    # noqa: BLE001
        return None
    H, W = bgr.shape[:2]
    x0 = max(0.0, min(float(W), x0 - pad))
    y0 = max(0.0, min(float(H), y0 - pad))
    x1 = max(0.0, min(float(W), x1 + pad))
    y1 = max(0.0, min(float(H), y1 + pad))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return bgr[int(y0):int(y1), int(x0):int(x1)].copy()


def _good_name(s: str) -> bool:
    s = (s or "").strip()
    if len(s) < MIN_NAME_LEN:
        return False
    return any(ch.isalpha() for ch in s)


def _good_num(s: str) -> bool:
    s = (s or "").strip()
    if not NUM_RE.match(s):
        return False
    try:
        if len(s) > 4:
            return False
        d = int(re.sub(r"^[A-Za-z]+", "", s))
    except ValueError:
        return False
    return 1 <= d <= 2000


def _rec(assigned: dict, cid: str, k: int) -> Optional[dict]:
    comp = (assigned.get("comps") or {}).get(cid)
    if not isinstance(comp, dict):
        comp = assigned.get(cid)
    if not isinstance(comp, dict):
        return None
    pins = comp.setdefault("pins", {})
    r = pins.get(k)
    if r is None:
        r = pins.get(str(k))
    if r is None:
        r = {}
        pins[k] = r
        pins[str(k)] = r
    return r


def _write(assigned: dict, cid: str, k: int, field: str, val: str,
           score: float = SCORE) -> bool:
    """写 `assigned[cid]["pins"][k][field]`（`field` ∈ {`number`,`name`}）。"""
    r = _rec(assigned, cid, k)
    if r is None:
        return False
    old = ((r.get(field) or {}).get("text") or "").strip()
    if KEEP_OURS and old:
        return False
    r[field] = {"text": val, "score": score, "src": "iterate:%s" % field}
    return True


# ---------------------------------------------------------------------------
def _collect(stage1: dict, bgr, win_key: str, MAX: int = MAX_PINS):
    """按 `win_key` 把所有 pincol 格点裁成小图。

    **按元件分组**返回 `[(cid, [(k, 裁剪), ...]), ...]` —— 分组是必须的：
    0143 的 `c0001` 有 12 个窗、`c0003` 有 54 个，若跨元件按 `PER=8` 切批，
    一张拼图里会混进两个元件的脚，实测模型会把 `VBAT` 读成 `VDD`
    （单独只喂 `c0003` 的 54 个窗时是 54/54 全中）。
    """
    cols_doc = (stage1 or {}).get("pincol") or {}
    s1pins = (stage1 or {}).get("pins") or {}
    out: List[Tuple[str, List[Tuple[int, Any]]]] = []
    for cid, recs in cols_doc.items():
        if not recs:
            continue
        pts = s1pins.get(cid) or []
        # `merge_into` 把枚举点并进 `pins[cid]` 时是**保序**的：第 k 个枚举点就是
        # 第 k 个 stage1 引脚（多出来的老触点在列表尾部，不在枚举范围内）。
        n = min(len(recs), len(pts) if pts else len(recs), MAX)
        if n < MIN_PINS:
            continue
        items: List[Tuple[int, Any]] = []
        for k in range(n):
            img = _crop(bgr, recs[k].get(win_key))
            if img is not None:
                items.append((k, img))
        if items:
            out.append((cid, items))
    return out


def _ask(groups, client, log=None, tag=""):
    """分批问 VLM，返回 `[(cid, k, num, name), ...]`。"""
    from vlm import enrich as EN

    res: List[Tuple[str, int, str, str]] = []
    for cid, items in groups:
        crops = [im for _k, im in items]
        for s in range(0, len(crops), PER):
            sub = crops[s:s + PER]
            try:
                g = EN.tile(sub, cols=min(COLS, len(sub)), side=SIDE, pad=8,
                            label_h=26)
                txt, _dt = client.ask(g, EN.PINS_PROMPT, model=None)
            except Exception as exc:                      # noqa: BLE001
                if log:
                    log("  [iterate%s] %s 第 %d 批失败：%s"
                        % (tag, cid, s // PER + 1, exc))
                continue
            got = EN.parse_grid(txt or "")
            for i, val in got.items():
                j = s + i
                if not (0 <= j < len(items)):
                    continue
                if isinstance(val, (list, tuple)):
                    num, name = (list(val) + ["", ""])[:2]
                else:
                    num, name = "", ""
                res.append((cid, items[j][0], (num or "").strip(),
                            (name or "").strip()))
    return res


def _run(assigned: dict, stage1: dict, bgr, win_key: str, field: str,
         good, client, log, tag: str) -> Dict[str, Any]:
    import time as _t
    t0 = _t.time()
    rep: Dict[str, Any] = {"n_comp": 0, "n_win": 0, "n_ans": 0, "n_ok": 0,
                           "n_bad": 0, "sec": 0.0, "rows": []}
    if bgr is None:
        return rep
    groups = _collect(stage1, bgr, win_key)
    rep["n_comp"] = len(groups)
    rep["n_win"] = sum(len(it) for _c, it in groups)
    if not groups:
        rep["sec"] = round(_t.time() - t0, 2)
        return rep
    got = _ask(groups, client, log=log, tag=tag)
    rep["n_ans"] = len(got)
    for cid, k, num, name in got:
        val = (num if field == "number" else name).strip().strip("`'\"").rstrip(",;")
        if not good(val):
            rep["n_bad"] += 1
            continue
        if _write(assigned, cid, k, field, val):
            rep["n_ok"] += 1
            rep["rows"].append((cid, k, val))
    rep["sec"] = round(_t.time() - t0, 2)
    if log:
        log("  [iterate:%s] 元件 %d · 窗 %d · 回 %d · 写回 %d（坏 %d）· %.1fs"
            % (field, rep["n_comp"], rep["n_win"], rep["n_ans"], rep["n_ok"],
               rep["n_bad"], rep["sec"]))
    return rep


def pin_names(assigned: dict, stage1: dict, bgr, log=None, client=None,
              limit: int = 0) -> Dict[str, Any]:
    """按 pincol 的 `in_win` 问一次 VLM 的**引脚名**，写回 `assigned`。"""
    if not ON:
        return {"on": False}
    from vlm import client as _C
    rep = _run(assigned, stage1, bgr, NAME_WIN, "name", _good_name,
               client or _C, log, "name")
    rep["on"] = True
    return rep


def pin_numbers(assigned: dict, stage1: dict, bgr, log=None, client=None,
                limit: int = 0) -> Dict[str, Any]:
    """按 pincol 的 `out_win` 问一次 VLM 的**引脚号**，写回 `assigned`。

    这一步才是键（`pin_<号>`）的来路：`out_win` 的 `num` 字段在 0143 上给出
    L 边 `1..8,10..33`、R 边 `64,63,62,61,60,56,49,48,…` —— 正是 GT 的号。
    """
    if not ON:
        return {"on": False}
    from vlm import client as _C
    rep = _run(assigned, stage1, bgr, NUM_WIN, "number", _good_num,
               client or _C, log, "number")
    rep["on"] = True
    return rep


def _pos(rec: dict, side: str) -> float:
    """沿边位置：`L`/`R` 用 `y`，`T`/`B` 用 `x`。"""
    return float(rec.get("y", 0.0)) if side in ("L", "R") else float(rec.get("x", 0.0))


NUM_ONLY = re.compile(r"^\d{1,3}$")

#: 梯子最多只许离最近锚点 **GAP_MAX 格**。
#:
#: 全边外推（`GAP_MAX` 无穷大）只对**稠密升/降序**的符号成立（0143 的 STM32
#: LQFP：`1..32` / `64..33`）。实测 19 张定点档里的 GT 号大多是**稀疏 BGA 球号**：
#:
#:   * `0001 U1` GT `1,2,4,5,6,7,33,34,59,61..64,68,69,EP,391..400`
#:     —— 我们外推出稠密的 `1,2,3,4,6,7,8,10,13,16,…,38`，凭空造号；
#:   * `0011 U5` GT `2..9,21,31,39,40,41,1_1,1_2,317..327`、我们 `1,2,3,4,6,7,8`；
#:   * `0050 U1` GT `2..9,447,450..458,461..463,MCPF1412M06`、我们 `2..22`。
#:
#: 锚点（真读到的号）本身永远是对的，所以只把它两边的**小洞**补上、洞外一律留给
#: 下游的「位置兜底」，比造一串不存在的号更划算。`GAP_MAX >= 999` 即退回旧行为。
GAP_MAX = int(os.environ.get("FUSION_ITER_GAP", "2"))


def _smooth_numbers(assigned: dict, stage1: dict, min_anchor: int = 2,
                    min_support: float = 0.5) -> Dict[str, Any]:
    """用 **每条边** 上读到的号 + 该边的 `pitch`，把整条边的号**铺成等差梯子**。

    为什么不能逐点照抄、也不能最小二乘：

    - 逐点照抄：0143 的 R 边（GT 自上而下 `64,63,…,34`）只枚举出 22 个格点、中间
      有洞，读到的号是 `64,63,62,61,60,56,49,48,…` —— 洞那里会错位。
    - 最小二乘：一个离群的读法（L 边 index 8 读成 `10`，真值 `9`）会把**整条边**
      的截距拉偏 1，实测把 `1:VBAT` 拉成 `2:VDD`，tp 从 20 掉到 11。
    - **梯子**：`num_k = num_anchor + round((pos_k − pos_anchor)/pitch) · step`，
      `step ∈ {+1,−1}`。锚点就是**读到的号本身**，所以永不错位；离群读法只影响
      自己那一格，不影响别人。步长取「能让最多锚点自洽」的那个方向 —— 于是
      L 边（自上而下升序）与 R 边（自下而上升序）都自动正确。
    - 梯子**只补洞、不外推**：格点离最近锚点超过 `GAP_MAX` 格就留白（见 `GAP_MAX`
      的注释）—— 稀疏 BGA 球号上全边外推会凭空造号。
    """
    rep: Dict[str, Any] = {"sides": 0, "fit": 0, "written": 0, "skip": 0}
    cols_doc = (stage1 or {}).get("pincol") or {}
    s1pins = (stage1 or {}).get("pins") or {}
    for cid, recs in cols_doc.items():
        n = min(len(recs), len(s1pins.get(cid) or []) or len(recs))
        by_side: Dict[str, List[int]] = {}
        for k in range(n):
            by_side.setdefault(str(recs[k].get("side") or "?"), []).append(k)
        for side, ks in by_side.items():
            if len(ks) < 2:
                continue
            rep["sides"] += 1
            reads: List[Tuple[float, int]] = []
            for k in ks:
                r = _rec(assigned, cid, k) or {}
                v = str(((r.get("number") or {}).get("text")) or "").strip()
                if NUM_ONLY.match(v):
                    reads.append((_pos(recs[k], side), int(v)))
            if len(reads) < min_anchor:
                rep["skip"] += 1
                continue
            pitches = [float(recs[k].get("pitch") or 0.0) for k in ks]
            pitches = [p for p in pitches if p > 0.5]
            if pitches:
                pitches.sort()
                pitch = pitches[len(pitches) // 2]
            else:                                          # 退路：相邻位置差的中位
                ps = sorted(_pos(recs[k], side) for k in ks)
                ds = [b - a for a, b in zip(ps, ps[1:]) if b - a > 0.5]
                if not ds:
                    rep["skip"] += 1
                    continue
                ds.sort()
                pitch = ds[len(ds) // 2]
            best = None
            for step in (1, -1):
                for (ap, av) in reads:
                    hit = sum(1 for (p, v) in reads
                              if v == av + int(round((p - ap) / pitch)) * step)
                    if best is None or hit > best[0]:
                        best = (hit, ap, av, step)
            if best is None or best[0] < max(2, int(min_support * len(reads))):
                rep["skip"] += 1
                continue
            _hit, ap, av, step = best
            rep["fit"] += 1
            anchors = {int(round((p - ap) / pitch)) for (p, _v) in reads}
            for k in ks:
                d = int(round((_pos(recs[k], side) - ap) / pitch))
                if GAP_MAX < 999 and min(abs(d - i) for i in anchors) > GAP_MAX:
                    rep["gap_skip"] = rep.get("gap_skip", 0) + 1
                    continue
                v = av + d * step
                if not (1 <= v <= 2000):
                    continue
                if _write(assigned, cid, k, "number", str(v), score=0.75):
                    rep["written"] += 1
    return rep


def pin_fields(assigned: dict, stage1: dict, bgr, log=None,
               client=None) -> Dict[str, Any]:
    """两次都做：`out_win` 取号、`in_win` 取名，再按边把号外推补齐。"""
    if not ON:
        return {"on": False}
    from vlm import client as _C
    C = client or _C
    a = pin_numbers(assigned, stage1, bgr, log=log, client=C)
    sm = _smooth_numbers(assigned, stage1)
    b = pin_names(assigned, stage1, bgr, log=log, client=C)
    if log:
        log("  [iterate:number] 按边外推：%d 条边拟合成功 / %d 条跳过 · 补写 %d 个号"
            " · 洞外留白 %d（GAP_MAX=%d）"
            % (sm["fit"], sm["skip"], sm["written"], sm.get("gap_skip", 0), GAP_MAX))
    return {"on": True, "number": a, "name": b, "smooth": sm,
            "n_ok": int(a.get("n_ok", 0)) + int(b.get("n_ok", 0)),
            "sec": round(float(a.get("sec", 0.0)) + float(b.get("sec", 0.0)), 2)}


def redetect_hints(pred: dict, stage1: dict, tol: float = 6.0) -> List[dict]:
    """报告「哪些 `box` 元件的脚发得比枚举出来的少」—— 给外层重跑 stage1 用。

    只**报告**，不改任何东西：真正重跑 stage1 要动检测器的 `imgsz`/裁切，那是
    `submission/run.py` 的外层循环该决定的事（本模块不越权）。判据是**几何**：
    把 `pred['pins']` 的点翻回图像坐标，看每个 pincol 元件有多少个枚举点被发出去。
    """
    out: List[dict] = []
    H = ((stage1 or {}).get("image") or {}).get("h") or 0
    if not H:
        return out
    flat: List[Tuple[float, float]] = []
    for _key, pv in ((pred or {}).get("pins") or {}).items():
        for _pk, info in (pv or {}).items():
            pt = (info or {}).get("point") or {}
            try:
                flat.append((float(pt["x"]), float(H) - float(pt["y"])))
            except Exception:                             # noqa: BLE001
                pass
    for cid, recs in ((stage1 or {}).get("pincol") or {}).items():
        if not recs:
            continue
        n_hit = 0
        for r in recs:
            rx, ry = float(r.get("x", 0.0)), float(r.get("y", 0.0))
            if any((rx - x) ** 2 + (ry - y) ** 2 <= tol * tol for x, y in flat):
                n_hit += 1
        if len(recs) - n_hit >= max(2, int(0.25 * len(recs))):
            out.append({"cid": cid, "n_enum": len(recs), "n_sent": n_hit,
                        "why": "枚举出 %d 个格点，只发出去 %d 个" % (len(recs), n_hit)})
    return out


def _selftest() -> int:
    """不联网：只验纯逻辑（名字/号的判据、写回、裁窗夹回）。"""
    ok = 0
    bad = 0
    for s, want in (("VBAT", True), ("PC13/TRC_AF1", True), ("-", False),
                    ("1", False), ("", False), ("Nrst", True), ("VDD_", True)):
        got = _good_name(s)
        if got == want:
            ok += 1
        else:
            bad += 1
            print("  ✗ _good_name(%r) = %s，期望 %s" % (s, got, want))
    for s, want in (("1", True), ("12", True), ("A1", True), ("AB12", True),
                    ("NC", False), ("P", False), ("", False), ("VSS", False),
                    ("3.3V", False), ("9999", False)):
        got = _good_num(s)
        if got == want:
            ok += 1
        else:
            bad += 1
            print("  ✗ _good_num(%r) = %s，期望 %s" % (s, got, want))
    a: dict = {"comps": {"c0003": {}}}
    if not _write(a, "c0003", 7, "name", "PC0"):
        print("  ✗ _write 初次写回失败")
        bad += 1
    else:
        r = a["comps"]["c0003"]["pins"][7]
        if (r["name"]["text"] != "PC0"
                or a["comps"]["c0003"]["pins"]["7"] is not r):
            print("  ✗ _write 的两种键形态不一致")
            bad += 1
        else:
            ok += 1
    # number 与 name 必须能共存
    _write(a, "c0003", 7, "number", "8")
    if a["comps"]["c0003"]["pins"][7].get("name", {}).get("text") != "PC0":
        print("  ✗ _write 写 number 时把 name 冲掉了")
        bad += 1
    else:
        ok += 1
    try:
        import numpy as _np
        img = _np.zeros((20, 20, 3), dtype=_np.uint8)
        if _crop(img, (-5, -5, 5, 5), 0.0) is None or _crop(img, (1, 1, 2, 2), 0.0) is not None:
            print("  ✗ _crop 的越界夹回不对")
            bad += 1
        else:
            ok += 1
    except ImportError:
        pass
    # ---- 梯子：L 边升序 + 一个离群读法不许带偏整条边 ------------------------
    def _mk(side, ys, reads):
        recs = [{"side": side, "y": float(y), "x": 0.0, "pitch": 10.0} for y in ys]
        a = {"comps": {"c1": {"pins": {k: {"number": {"text": v}}
                                       for k, v in reads.items()}}}}
        return {"pincol": {"c1": recs}, "pins": {"c1": [[0.0, 0.0]] * len(ys)}}, a

    st, aa = _mk("L", [0, 10, 20, 30, 40, 50], {0: "1", 1: "2", 2: "3", 4: "6"})
    _smooth_numbers(aa, st)
    got = [int(aa["comps"]["c1"]["pins"][k]["number"]["text"]) for k in range(6)]
    if got != [1, 2, 3, 4, 5, 6]:
        print("  ✗ 梯子（L 升序 + 离群）得到 %s，期望 [1,2,3,4,5,6]" % got)
        bad += 1
    else:
        ok += 1
    st, ab = _mk("R", [0, 10, 20, 30, 40, 50], {0: "6", 1: "5", 3: "3"})
    _smooth_numbers(ab, st)
    got = [int(ab["comps"]["c1"]["pins"][k]["number"]["text"]) for k in range(6)]
    if got != [6, 5, 4, 3, 2, 1]:
        print("  ✗ 梯子（R 降序）得到 %s，期望 [6,5,4,3,2,1]" % got)
        bad += 1
    else:
        ok += 1
    # 一条边只有 1 个读法 -> 不拟合（宁可用装配层的位置号兜底）
    st, ac = _mk("L", [0, 10, 20], {1: "7"})
    if _smooth_numbers(ac, st)["fit"] != 0:
        print("  ✗ 锚点不足时不该拟合")
        bad += 1
    else:
        ok += 1
    print("iterate 自检：%d 过 / %d 不过" % (ok, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
