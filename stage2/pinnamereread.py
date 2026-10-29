"""`stage2/pinnamereread.py`：**装配之后**只重读脚名（`STAGE2_PINNAME_REREAD`，默认关；§5.467）。

## 为什么要单独做一个「事后」模块

§5.466 把「弱格子拿高分辨率权重补读」放进 `TextEngine.read()`，结果 case8 的读法一变，
`assign.py` 给裁片分的角色就跟着变（`slot1` 的号 `11`→`15`、`slot5` 的 `1`→`11`），
`pin_f1` 的键 `(元件, pin_key_to_number(pk))` 整片位移，名字明明更对了却掉 2.46 分。

这里的做法是**把「读数」和「绑定」彻底分开**：`assign()` 跑完、角色与号都定死之后，
才拿 `a["comps"][cid]["pins"][slot]["name"]["box"]` 这些**已经定好角色的裁片**去重读，
只覆盖 `name.text`，**绝不碰 `number`、绝不碰 `name.conf`、绝不增删槽**。

## 触发判据：字形段数 > 读出的字数（吞字），不是「名字短」

第一版按「空 / ≤3 字 / 纯数字」触发，把本来就对的短名也送进 64px 重读，于是
`BST`→`BSTI`（case7 −3.33）、`'5'`→`'A'`、`'c'`→`'8'`（case1 −0.68）——短名在这套图里合法。
真正的病因是 `_length_fix` 那条已经验证过的信号：**列投影的字形段数多于串长**（`TIM` 读成 `TM`、
`RLIM` 读成 `RLM`、`SHELL` 读成 `SHE`）。⇒ 只在「图上有 n 个字、我们只读出 < n 个字」时才重读，
且只接受**字数正好等于 n** 的结果。
"""
import os
from pathlib import Path

import numpy as np

from . import text as _text

REREAD = os.environ.get("STAGE2_PINNAME_REREAD", "1").strip().lower() in ("1", "true", "yes", "on")
# 高分辨率（64px、宽降采样 8）权重：`weights/textrec_h64.pt` 随仓库提交，
# 与 `textrec_best.pt`（32px）是**同一份合成+真实语料**重训出来的两份检查点。
_H64 = Path(__file__).resolve().parents[1] / "weights" / "textrec_h64.pt"
REREAD_CKPT = (os.environ.get("STAGE2_PINNAME_REREAD_CKPT", "").strip() or str(_H64))
REREAD_MINW = int(os.environ.get("STAGE2_PINNAME_REREAD_MINW", "2") or 2)
REREAD_MAXN = int(os.environ.get("STAGE2_PINNAME_REREAD_MAXN", "14") or 14)
REREAD_MINLEN = int(os.environ.get("STAGE2_PINNAME_REREAD_MINLEN", "2") or 2)
REREAD_MAXGAP = int(os.environ.get("STAGE2_PINNAME_REREAD_MAXGAP", "1") or 1)
REREAD_DBG = os.environ.get("STAGE2_PINNAME_REREAD_DBG", "0").strip().lower() in ("1", "true", "yes", "on")


def _nglyph(bgr: np.ndarray, box) -> int:
    """裁片里「有几个字形」（列投影的连通段数，与 `_length_fix` 同一统计量）。"""
    H, W = bgr.shape[:2]
    cb = _text.pad_box(tuple(float(v) for v in box[:4]))
    x0, y0, x1, y1 = (int(round(v)) for v in cb)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 3:
        return 0
    g = _text.to_gray(bgr)[y0:y1, x0:x1] if hasattr(_text, "to_gray") else None
    if g is None:
        g = np.asarray(bgr[y0:y1, x0:x1])
        if g.ndim == 3:
            g = g.mean(axis=2)
    if g.size == 0:
        return 0
    dark = g < max(1.0, float(g.mean()) * 0.8)
    try:
        runs = [t for t in _text._glyph_runs(dark) if (t[1] - t[0]) >= REREAD_MINW]   # noqa: SLF001
    except Exception:                                    # noqa: BLE001
        return 0
    return len(runs)


def fix(bgr: np.ndarray, doc: dict, a: dict, engine=None, ckpt: str = "") -> dict:
    """-> {"on":bool, "n":候选数, "chg":改写数, "log":[(cid,slot,旧,新,n)]}"""
    st = {"on": False, "n": 0, "chg": 0, "log": []}
    if not REREAD:
        return st
    st["on"] = True
    ck = ckpt or REREAD_CKPT
    if not ck or not os.path.exists(ck):
        st["error"] = "no ckpt: %r" % (ck,)
        return st
    try:
        eng2 = _text.TextEngine(ckpt=ck, device="auto")
    except Exception as e:                                  # noqa: BLE001
        st["error"] = "%s: %s" % (type(e).__name__, e)
        return st

    picks = []           # (cid, slot, box, cur_txt, n)
    for cid, rec in (a.get("comps") or {}).items():
        for slot, sl in (rec.get("pins") or {}).items():
            nm = (sl or {}).get("name") or {}
            box = nm.get("box")
            if not box or len(box) < 4:
                continue
            cur = (nm.get("text") or "").strip()
            n = _nglyph(bgr, box)
            if n < 2 or n > REREAD_MAXN or n <= len(cur):
                continue                     # 不是「吞字」形态 ⇒ 不碰
            if len(cur) < REREAD_MINLEN or (n - len(cur)) > REREAD_MAXGAP:
                continue                     # 单字名（`P`/`5`）与差两字以上一律不碰
            picks.append((cid, slot, tuple(float(v) for v in box[:4]), cur, n))
    st["n"] = len(picks)
    if not picks:
        return st

    boxes = [_text.tight_ink_box(bgr, p[2]) or p[2] for p in picks]
    out = eng2.read(bgr, boxes, roles=["pin_name"] * len(boxes))
    for p, r in zip(picks, out):
        cid, slot, _box, cur_t, n = p
        got = (r.get("reads") or {}).get("pin_name")
        if not got:
            continue
        t = (got[0] or "").strip()
        if not t or t == cur_t:
            continue
        if len(t) != n:
            continue                         # 只认「正好补满字形数」的结果
        (a["comps"][cid]["pins"][slot]["name"])["text"] = t
        st["chg"] += 1
        st["log"].append((cid, slot, cur_t, t, n))
    if REREAD_DBG:
        for row in st["log"]:
            print("[dbg pinname reread] %s %s %r->%r n=%d" % row)
    return st
