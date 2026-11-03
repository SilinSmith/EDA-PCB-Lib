# -*- coding: utf-8 -*-
"""box 件「型号名文字框」重读通道（`STAGE2_PARTNAME_OCR`，默认 `"0"` = 关）。

**病**：型号名（`partname` 文字框的 `text`）在管线里只被读一次。`fusion` 那边的两档
吸附（`modelset` 用引脚名集合反推、`namefix` 编辑距离 ≤2 的近失串）都只在「已经读到的
字符串」上纠错，救不了**整串读歪**的那类 —— `open_GT` 13 页实测：

    `P5306-435V` ← `IP5306-4.35V`（GT 原样）｜`TPE-C-SPIOT3` ← `TYPE-C 6P(073)`（库键）
    `SY813BADC` ← `SY8113BADC`｜`XC806P332MR` ← `XC6206P332MR`｜`TP404-MS` ← `TP4054-MS`

**药**：把同一个文字框**单独**重裁一次，走与引脚名通道**完全相同**的采样
（`stage2.isolated_ocr.prepare_isolated_sample` 的单样本独立识别 + `crop_gray(32, 384)`
共享灰度图，见 `stage2/pin_name_reading.py:150-153`），送给同一个 provider 再读一遍。
`assemble.build_doc` 在 `Name` 最终成型之后按档位决定要不要用这次读数覆盖它；被覆盖的
原值记在同一条目的 `_name_pp` 里（审计用，**只动 `Name` 这一个字段**）。

**档位**（`STAGE2_PARTNAME_OCR`）：

  0 —— 关（默认）。连 `cv2` 都不 import，产物与不挂这条通道时**逐字节相同**。
  1 —— 保守：只有当 `libpins.norm(PP 文本)` **精确等于**某个「形态良好」的库键时才采用，
       采用的是 **PP 读出的原字符串**（judge 比的是原串，库键可能带 `_` / 大小写差异）。
  2 —— 进取：`1` 之外再加「PP 文本与现名相同」或「`conf >= PP_CONF_HIGH` 且纯可打印
       ASCII 且 ≥3 字符」。

**两档共同的弃权**：PP 文本为空 / provider 不可用（`rapidocr` 缺包 ⇒ 整条通道返回空，
与不挂它时逐位相同）/ 读数等于该元件的**位号**（`case10 LED1 M-ED1→LED1` 那种把位号
读成型号名的）。

离线反事实（`work\\_partname_cf.py`，13 页 31 颗 box 件，只改 `Name` 再
`M.evaluate_case`）：基线 comp 轴 26.089 ｜「只认库键」26.188 ｜「+同名+高置信」26.287。
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

try:                                    # 包内导入（`from .assemble import build_doc`）
    from stage2 import libpins
except ImportError:                     # 直接跑脚本（`python eda-pcb\stage2\partname_ocr.py`）
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from stage2 import libpins

#: `=2` 档的高置信阈值。**来源 = 本次实测**：`work\_partname_cf.py` 的反事实 —— 13 页
#: 31 颗 box 件上，comp 轴「只认库键」26.188 → 再加这一档 26.287（+0.099）。
PP_CONF_HIGH = 0.90

#: 库键的「形态良好」判据（`=1` 档）：首字符必须是 ASCII 字母或数字，长度 ≥3，全串可打印
#: 且不含下面这些 OCR 噪声字符 —— `case6 U3` 的库键自己就带前导 `|`（`|78M12`），不能当证据。
_KEY_HEAD = re.compile(r"[A-Za-z0-9]")
_KEY_NOISE = frozenset("\"'`|")

#: 库键来源（与 `work\_partocr.py` / `work\_partname_cf.py` 的 `lib_keys()` 同源同法）。
_DATA_FILES = ("BoxICLibrary_200cases.json", "cpntLibrary.json")

_LIB_KEYS: Optional[dict] = None


def level() -> int:
    """`STAGE2_PARTNAME_OCR` 的档位；空值/非法值一律按 `0`（关）处理。"""
    raw = os.environ.get("STAGE2_PARTNAME_OCR", "0").strip().lower()
    if raw == "1":
        return 1
    if raw == "2":
        return 2
    return 0


def _well_formed(key) -> bool:
    """读数/库键能不能当证据用（首字符 ASCII 字母数字、长度 ≥3、可打印、无噪声字符）。

    `=2` 高置信档题面写的「纯可打印 ASCII 且长度 ≥3」被它包住（同 `_partname_cf.clean()`
    再加上首字符与噪声字符两条）。
    """
    s = str(key or "")
    if len(s) < 3 or not _KEY_HEAD.match(s[:1]):
        return False
    if any((not ch.isascii()) or (not ch.isprintable()) for ch in s):
        return False
    return not any(ch in _KEY_NOISE for ch in s)


def _lib_keys() -> dict:
    """`libpins.norm(库键) -> 原始库键`（首次调用建表，之后复用）。"""
    global _LIB_KEYS
    if _LIB_KEYS is not None:
        return _LIB_KEYS
    keys: dict = {}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in _DATA_FILES:
        try:
            with open(os.path.join(root, "data", name), encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):                  # 缺文件 / 坏 JSON 都只让本档变窄
            continue
        for key in raw:
            keys.setdefault(libpins.norm(key), key)
    _LIB_KEYS = keys
    return keys


class PartnameOCR:
    """一个 case 一条：懒加载原图与 provider，逐颗按档位裁决要不要覆盖 `Name`。"""

    __slots__ = ("_gray", "_keys", "_img_path", "_level", "_provider", "_tried", "stats")

    def __init__(self, stage1: dict, lvl: int) -> None:
        self._level = int(lvl)
        self._img_path = str(((stage1 or {}).get("image") or {}).get("path") or "")
        self._keys = _lib_keys() if self._level else {}
        self._provider = None
        self._gray = None
        self._tried = False
        self.stats = {"libkey": 0, "same": 0, "high": 0, "skip": 0}

    # -- 惰性就绪 ----------------------------------------------------------
    def _ready(self) -> bool:
        """懒建 provider + 灰度原图。任一步失败 ⇒ 本通道静默失效（不抛、不改任何东西）。"""
        if self._tried:
            return self._gray is not None and self._provider is not None
        self._tried = True
        try:
            from stage2.pin_name_ocr import get_pin_name_ocr
            self._provider = get_pin_name_ocr("auto")
        except Exception:                              # noqa: BLE001 —— 缺包就整条通道弃权
            self._provider = None
        if self._provider is not None and self._img_path:
            try:
                import cv2
                bgr = cv2.imread(self._img_path)
                if bgr is not None:
                    self._gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            except Exception:                          # noqa: BLE001
                self._gray = None
        return self._gray is not None and self._provider is not None

    # -- 读数 --------------------------------------------------------------
    def _read(self, info: dict):
        """裁 `partname.box` → 独立采样 → provider 读数；返回 `(text, conf)` 或 `None`。"""
        box = ((info or {}).get("partname") or {}).get("box")
        if not box or len(box) < 4:
            return None
        try:
            x0, y0, x1, y1 = [int(round(float(v))) for v in box[:4]]
        except (TypeError, ValueError):
            return None
        h, w = self._gray.shape[:2]
        # ±3 余量与「短边 <4 不读」是 `work\_partocr.py` 已经跑通的那两个数（与引脚名通道的
        # `padding=3` 同一个理由：CRNN 需要一点白边），它们是**采样口径**，不是可调参数。
        x0, y0 = max(0, x0 - 3), max(0, y0 - 3)
        x1, y1 = min(w, x1 + 3), min(h, y1 + 3)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        gray = self._gray[y0:y1, x0:x1].copy()
        try:
            import numpy as np
            from PIL import Image
            from stage2.isolated_ocr import prepare_isolated_sample
            from textrec.dataset import crop_gray
            canvas, bbox, _shared = prepare_isolated_sample({
                "gray": gray, "bbox": [0, 0, gray.shape[1], gray.shape[0]],
                "role": "pin_name", "rotation": 0, "padding": 3})
            normalized = np.asarray(crop_gray(Image.fromarray(canvas), bbox, 32, 384))
            result = self._provider.read_normalized(normalized)
        except Exception:                              # noqa: BLE001
            return None
        text = str(result.get("text") or "").strip()
        try:
            conf = float(result.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        return (text, conf)

    # -- 裁决 --------------------------------------------------------------
    def read(self, info: dict, cur_name=None, designator=None):
        """按档位裁决这颗元件要不要用 PP 读数覆盖 `Name`；返回 `{text, conf, by}` 或 `None`。"""
        if not self._level or not self._ready():
            self.stats["skip"] += 1
            return None
        got = self._read(info)
        if got is None:
            self.stats["skip"] += 1
            return None
        text, conf = got
        if not text:                                   # 空读数 ⇒ 弃权
            self.stats["skip"] += 1
            return None
        des = str(designator or "").strip()
        if des and text == des:                        # 把位号读成型号名 ⇒ 弃权
            self.stats["skip"] += 1
            return None
        # ★ 形态判据（`=1`/`=2` 都过）：读数本身得像型号名 —— 长度 ≥3、首字符是字母或数字、
        #   全是可打印 ASCII、不含引号/竖线这类噪声字符。题面给的「纯可打印 ASCII 且 ≥3」
        #   被它包住，所以高置信档不再重复判。
        #   也要按**原串**判：`norm()` 会把 `|` 这类噪声字符整个吃掉，于是 `case6` 的
        #   `|78M12`（旧名 `78M12`、GT 也是 `78M12`）能精确命中干净的库键 `78M12`，
        #   只按库键判形态就会把好名字改坏（实测 conf 0.9+ 时首字符 `|` 会漏进来）。
        if not _well_formed(text):
            self.stats["skip"] += 1
            return None
        key = self._keys.get(libpins.norm(text))
        if key is not None and _well_formed(key):
            self.stats["libkey"] += 1
            return {"text": text, "conf": conf, "by": "libkey"}
        if self._level >= 2:
            cur = str(cur_name or "").strip()
            if cur and text == cur:
                self.stats["same"] += 1
                return {"text": text, "conf": conf, "by": "same"}
            if conf >= PP_CONF_HIGH:
                self.stats["high"] += 1
                return {"text": text, "conf": conf, "by": "high"}
        self.stats["skip"] += 1
        return None

    # -- 统计 --------------------------------------------------------------
    def summary_line(self) -> str:
        """每个 case 一行的统计（`build_doc` 收尾时打印）。"""
        s = self.stats
        return ("[stage2] partname PP：采用 %d 个（库键 %d、同名 %d、高置信 %d；弃权 %d）"
                % (s["libkey"] + s["same"] + s["high"],
                   s["libkey"], s["same"], s["high"], s["skip"]))


# ★ 采用记录随 `assigned` 走一趟：`stage2/assemble.py` 采纳时调 `carry()` 把值写进
#   `partname.text` 并挂一条记录（`_assigned/<case>.json` 会把它落盘），fusion 侧在建池
#   之前用 `reassert()` 把被 `modelset`/`namefix` 重新吸附掉的文本压回去。
CARRIER = "_pp"


def carry(info: dict, hit: dict) -> bool:
    """把采用值写回 `partname.text`，并挂一条采用记录供 fusion 侧重新压回。"""
    pn = info.get("partname") if isinstance(info, dict) else None
    if not isinstance(pn, dict) or not hit:
        return False
    pn["text"] = hit["text"]
    pn[CARRIER] = {"now": hit["text"], "conf": float(hit["conf"]), "by": hit["by"]}
    return True


def reassert(assigned: dict) -> dict:
    """fusion 侧：吸附完、`CL.from_stage2` 建池**之前**，把已采纳的型号名重新压回去。

    为什么必须在这儿：交付的 `Name` 由 `comp.<cid>.partname` 这条 claim 决定 ——
    `fusion/claims.py` 在建池时从 `assigned` 抄一份候选值，`fusion/doc.py` 最后又拿
    claim 覆盖 `build_doc` 写好的 `Name`。所以 `Name` 在 `build_doc` 里改（哪怕改在
    「最终成型」那一处）都会被这两步打回原样：实测 case1 U1，stage2 采纳的
    `IP5306-4.35V`（= GT）被 `FUSION_MODEL_SET` 吸成 `IP5306`，交付 `Name` 跟着变回
    `IP5306`。`modelset`/`namefix` 只改 `partname.text`（旧值留 `_old_text`），不换字典，
    所以这条记录活得下来。
    """
    if level() <= 0:
        return {}
    comps = (assigned or {}).get("comps") or {}
    ids = []
    for cid, comp in comps.items():
        pn = comp.get("partname") if isinstance(comp, dict) else None
        rec = pn.get(CARRIER) if isinstance(pn, dict) else None
        if not isinstance(rec, dict):
            continue
        now = str(rec.get("now") or "")
        if not now or pn.get("text") == now:          # 没被动过 ⇒ 不用管
            continue
        pn["text"] = now
        ids.append(cid)
    if ids:
        print("[fusion] partname PP：压回 %d 个被重新吸附的型号名" % len(ids))
    return {"n": len(ids), "ids": ids}
