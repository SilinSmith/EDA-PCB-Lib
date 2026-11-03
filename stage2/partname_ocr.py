# -*- coding: utf-8 -*-
"""box 件「型号名文字框」重读通道（`STAGE2_PARTNAME_OCR`，默认 `"1"` = 档 1）。

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

  0 —— 关。连 `cv2` 都不 import，产物与不挂这条通道时**逐字节相同**。
     （2026-11-03 扫荡翻默认：13 页 84.4791→84.5203（+0.0412，只页1 +0.54）、
      池 29.0887→29.1158（+0.027）；单独一节记录见 `docs/RESULTS.md` §5.564。）
  1 —— 保守：只有当 `libpins.norm(PP 文本)` **精确等于**某个「形态良好」的库键时才采用，
       采用的是 **PP 读出的原字符串**（judge 比的是原串，库键可能带 `_` / 大小写差异）。
  2 —— 进取：`1` 之外再加「PP 文本与现名相同」或「`conf >= PP_CONF_HIGH` 且纯可打印
       ASCII 且 ≥3 字符」。

**两档共同的弃权**：PP 文本为空 / provider 不可用（`rapidocr` 缺包 ⇒ 整条通道返回空，
与不挂它时逐位相同）/ 读数等于该元件的**位号**（`case10 LED1 M-ED1→LED1` 那种把位号
读成型号名的）。

离线反事实（`work\\_partname_cf.py`，13 页 31 颗 box 件，只改 `Name` 再
`M.evaluate_case`）：基线 comp 轴 26.089 ｜「只认库键」26.188 ｜「+同名+高置信」26.287。

------------------------------------------------------------------------------
**同行拼接**（`STAGE2_PARTNAME_STITCH`，默认 `"0"` = 关）—— 上面那条通道只看**一个**
文字框，救不了「一整行型号名被 stage1 切成多个框」的那类。`open_GT` case11 `USB2`
（GT `AF 90\u00b0 WJDG`，box，5 脚）实测：页面上一整行型号名被切成

    [622,457,659,467] -> `AF-90`(0.66) | [661,457,665,461] -> `0`(0.64)（其实是被切下来的
    度数符号）| [670,457,710,467] -> `WMDG`(0.59)

于是 stage2 只拿到 `AF 90`（`key_for('AF 90')` -> `None`），fusion 的 `solve` 通道把它
吸附成 `SG90`（库键脚数 3），`INBOX_GUARD='visa'` 因为库键脚数对不上 5 脚件而拒绝贴名，
整颗件只兑现 1 根脚。

药：**不**把三个 crop 分别 OCR 再拼字符串（那会把各自的错字固化），而是取**同行邻居的
并集框**，在原图上用 `stage2.text.TextEngine` 重读**一次**。采用条件两条同时成立：
(1) 重读串能 **norm 级精确**命中库键（`key_for` 的模糊第三档不算）；(2) 该库键脚数与这颗
件的触点估计相差 <= `STITCH_TOL`（默认 1）。触点给**两个口径**（见 `contacts()`）：raw 触点
个数 与「不同网络数」—— 同一根脚在 stage1 里常有幻影双触点（case11 c0000：8 点 / 5 网），
任一估计落在 ±1 内即认为自洽。

**库键一致性护栏**（`STAGE2_PARTNAME_LIBGUARD`，默认 `"0"` = 关）：任何「重读/改写
partname」的通道在采用新串之前问一句 `guard(old, new, pair)` —— 新串的库键脚数与两个
触点估计**都**差 >=2，而原串能命中一个自洽的库键（或原串压根没有库键，见
`LIBGUARD_ORPHAN`）=> 拒绝改写。`SG90` 那次的 conf 0.803 并不低，靠置信度挡不住，只能
靠脚数一致性。
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
    raw = os.environ.get("STAGE2_PARTNAME_OCR", "1").strip().lower()
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


# ---------------------------------------------------------------------------
# 同行拼接 / 库键护栏（两条都**默认关**，关时逐字节等同不挂这两条通道）
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: str = "0") -> str:
    return (os.environ.get(name, default) or default).strip().lower()


#: `STAGE2_PARTNAME_STITCH`：默认 **`1` = 开**（§5.562：13 页 84.1735→84.3409）；
#: `0` = 关；`2` 与 `1` 行为相同（留 `2` 给以后细分）。
STITCH = _env_flag("STAGE2_PARTNAME_STITCH", "1")
#: `STAGE2_PARTNAME_REREAD`：单框也走同一套「重读 + 库键 + 脚数自洽」判据；默认 **`1` = 开**
#: （§5.563：13 页 84.3409→84.4791，池 29.0663→29.0887）。
REREAD = _env_flag("STAGE2_PARTNAME_REREAD", "1")
#: 同行判定：|cy 差| <= ROW * 行高。
STITCH_ROW = float(_env_flag("STAGE2_PARTNAME_STITCH_ROW", "0.6"))
#: 邻居水平间隙上限（* 行高）。
STITCH_GAP = float(_env_flag("STAGE2_PARTNAME_STITCH_GAP", "1.5"))
#: 一行最多并几个框（含自身）。
STITCH_MAXBOX = int(_env_flag("STAGE2_PARTNAME_STITCH_MAXBOX", "4"))
#: 并集宽度上限（* 行高）。题面给的 8 在 case11 上差 8px（并集 88 vs 8*10=80）——
#: stage1 的文字框是**按墨迹核心**取的，系统性比视觉行高小（`text.py:53-57`：中位
#: IoU 0.36），视觉行高约是框高的 2 倍 ⇒ 8*h 相当于「8 个字符位」。取 10。
STITCH_WMAX = float(_env_flag("STAGE2_PARTNAME_STITCH_WMAX", "10"))
#: 并集框上下各外扩多少像素再读（引擎自己还会 pad `PAD_MIN_Y=2`）。
STITCH_PADY = int(_env_flag("STAGE2_PARTNAME_STITCH_PADY", "2"))
#: 用哪个角色读并集框。`text.py:ROLES` 里**没有** `partname`，默认走 `value`。
STITCH_ROLE = (os.environ.get("STAGE2_PARTNAME_STITCH_ROLE", "value") or "value").strip()
#: 库键脚数与触点估计的容差（case11：库 5 脚、触点 8 点/5 网 -> 差 0）。
STITCH_TOL = int(_env_flag("STAGE2_PARTNAME_STITCH_TOL", "1"))
#: `STAGE2_PARTNAME_LIBGUARD`：库键一致性护栏。
LIBGUARD = _env_flag("STAGE2_PARTNAME_LIBGUARD")
#: 原串**压根没有库键**时，护栏要不要也拦（默认要：`AF 90` -> `SG90` 就是这种）。
LIBGUARD_ORPHAN = _env_flag("STAGE2_PARTNAME_LIBGUARD_ORPHAN", "1") in ("1", "2", "on", "true")


def stitch_level() -> int:
    """`STAGE2_PARTNAME_STITCH` 的档位；空值/非法值一律 0（关）。"""
    return 1 if STITCH in ("1", "2") else 0


def reread_level() -> int:
    """`STAGE2_PARTNAME_REREAD` 的档位（单框也重读）。"""
    return 1 if REREAD in ("1", "2") else 0


def libguard_on() -> bool:
    return LIBGUARD in ("1", "2", "on", "true")


_CTX = None


def _ctx():
    """`libpins` 的库上下文（第一次用时建，之后复用）。"""
    global _CTX
    if _CTX is None:
        _CTX = libpins.new_ctx()
    return _CTX


def exact_key(text):
    """**norm 级精确**命中库键就返回库键，否则 `None`（`key_for` 的模糊第三档不算）。"""
    t = str(text or "").strip()
    if not t:
        return None
    try:
        key = _ctx().key_for(t)
    except Exception:                                  # noqa: BLE001
        return None
    if key is None or libpins.norm(key) != libpins.norm(t):
        return None
    return key


def n_lib_pins(key) -> int:
    try:
        return len(_ctx().nums_all(key))
    except Exception:                                  # noqa: BLE001
        return 0


def contacts(stage1: dict, cid: str):
    """这颗件的触点估计 `(raw, net)`。

    `raw` = stage1 `pins[cid]` 的点数；`net` = 这些点所属的**不同网络数**。
    同一根脚常被 stage1 给两个幻影触点（case11 c0000：8 点 / 5 网，GT 5 脚），
    所以库键脚数只要落在**任一**估计的 ±TOL 内就算自洽 —— 单用 raw 会把 5 脚件
    的正确库键挡掉，单用 net 又会低估（多根脚共网时）。
    """
    pts = ((stage1 or {}).get("pins") or {}).get(cid) or []
    try:
        raw = len(pts)
    except TypeError:
        raw = 0
    idx = {}
    for net in (stage1 or {}).get("nets") or []:
        for item in net.get("pins") or []:
            try:
                pc, i = item[0], int(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if pc == cid:
                idx[i] = net.get("id")
    return raw, (len(set(idx.values())) if idx else raw)


def _fit(key, pair, tol=None) -> bool:
    """库键脚数与触点估计自洽？（`pair` 为空 ⇒ 没有触点信息，一律算自洽）"""
    if key is None:
        return False
    if not pair:
        return True
    tol = STITCH_TOL if tol is None else int(tol)
    n = n_lib_pins(key)
    raw, net = pair
    return min(abs(n - raw), abs(n - net)) <= tol


def guard(old_text, new_text, pair, tol=None) -> bool:
    """库键一致性护栏：返回 `True` = 允许把 `old_text` 改成 `new_text`。

    只在 `STAGE2_PARTNAME_LIBGUARD` 开时生效。拒绝条件（全部成立）：
      1. 新串 norm 级命中一个库键，且该库键脚数与触点两个估计**都**差 >= 2（自洽不了）；
      2. 原串能命中一个自洽的库键；或原串**一个库键都没有**（`LIBGUARD_ORPHAN`）。
    原串自己也自洽不了 ⇒ 不拿它当基准（否则会锁死一个本来就错的旧串）。
    """
    if not libguard_on():
        return True
    old = str(old_text or "").strip()
    new = str(new_text or "").strip()
    if not new or libpins.norm(new) == libpins.norm(old):
        return True
    k_new = exact_key(new)
    if k_new is None or _fit(k_new, pair, tol):
        return True
    k_old = exact_key(old)
    if k_old is not None and not _fit(k_old, pair, tol):
        return True
    if k_old is None and not LIBGUARD_ORPHAN:
        return True
    return False


def _key_clean(key) -> bool:
    """库键能不能直接当 `Name` 写出去（首字符字母数字、>=3、可打印、无噪声字符）。

    比 `_well_formed()` 宽一条：**不要求全 ASCII** —— 库键里的 `\u00b0` 这类符号正是
    GT 原样（case11 `AF 90\u00b0 WJDG`），写出去才与真值逐字符相同。
    """
    s = str(key or "")
    if len(s) < 3 or not _KEY_HEAD.match(s[:1]):
        return False
    if any((not ch.isprintable()) for ch in s):
        return False
    return not any(ch in _KEY_NOISE for ch in s)


class PartnameOCR:
    """一个 case 一条：懒加载原图与 provider，逐颗按档位裁决要不要覆盖 `Name`。"""

    __slots__ = ("_bgr", "_eng", "_gray", "_keys", "_img_path", "_level", "_provider",
                 "_texts", "_tried", "stats")

    def __init__(self, stage1: dict, lvl: int) -> None:
        self._level = int(lvl)
        self._img_path = str(((stage1 or {}).get("image") or {}).get("path") or "")
        self._keys = _lib_keys() if self._level else {}
        self._provider = None
        self._gray = None
        self._bgr = None
        self._eng = None
        self._texts = list((stage1 or {}).get("texts") or [])
        self._tried = False
        self.stats = {"libkey": 0, "same": 0, "high": 0, "skip": 0,
                      "stitch": 0, "stitch_skip": 0, "guard": 0}

    # -- 惰性就绪 ----------------------------------------------------------
    def _img_ready(self) -> bool:
        """懒加载原图（灰度 + BGR）。任一步失败 => 返回 False，不抛。"""
        if self._gray is None and self._bgr is None and self._img_path:
            try:
                import cv2
                bgr = cv2.imread(self._img_path)
                if bgr is not None:
                    self._bgr = bgr
                    self._gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            except Exception:                      # noqa: BLE001
                self._gray = None
        return self._gray is not None or self._bgr is not None

    def _provider_ready(self) -> bool:
        """懒建 PP provider。缺包/不可用 => False（本通道静默失效）。"""
        if self._provider is None and self._tried:
            return False
        if self._provider is not None:
            return True
        self._tried = True
        try:
            from stage2.pin_name_ocr import get_pin_name_ocr
            self._provider = get_pin_name_ocr("auto")
        except Exception:                          # noqa: BLE001
            self._provider = None
        return self._provider is not None

    def _ready(self) -> bool:
        """PP 通道就绪 = 原图 + provider 都在。"""
        return self._provider_ready() and self._img_ready()
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
                    self._bgr = bgr
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
    def read(self, info: dict, cur_name=None, designator=None, pair=None):
        """按档位裁决这颗元件要不要用 PP 读数覆盖 `Name`；返回 `{text, conf, by}` 或 `None`。

        `pair` = 这颗件的触点估计 `(raw, net)`（`contacts()`），只给
        `STAGE2_PARTNAME_LIBGUARD` 用；不给就等于没有触点信息（护栏不拦）。
        """
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
            if not guard(cur_name, text, pair):
                self.stats["guard"] += 1
                self.stats["skip"] += 1
                return None
            self.stats["libkey"] += 1
            return {"text": text, "conf": conf, "by": "libkey"}
        if self._level >= 2:
            cur = str(cur_name or "").strip()
            if cur and text == cur:
                self.stats["same"] += 1
                return {"text": text, "conf": conf, "by": "same"}
            if conf >= PP_CONF_HIGH:
                if not guard(cur_name, text, pair):
                    self.stats["guard"] += 1
                    self.stats["skip"] += 1
                    return None
                self.stats["high"] += 1
                return {"text": text, "conf": conf, "by": "high"}
        self.stats["skip"] += 1
        return None


    # -- 同行拼接（`STAGE2_PARTNAME_STITCH`）--------------------------------
    def _row(self, box):
        """`partname` 框 + 同行邻居框 -> `(选中的框列表, 并集框)`。"""
        try:
            x0, y0, x1, y1 = [float(v) for v in box[:4]]
        except (TypeError, ValueError):
            return [], None
        h = max(1.0, y1 - y0)
        cy = (y0 + y1) / 2.0
        cand = []
        for t in self._texts:
            bb = (t or {}).get("bbox_img")
            if not bb or len(bb) < 4:
                continue
            try:
                tb = [float(v) for v in bb[:4]]
            except (TypeError, ValueError):
                continue
            th = tb[3] - tb[1]
            tcy = (tb[1] + tb[3]) / 2.0
            if abs(tcy - cy) > STITCH_ROW * h:            # 不同行
                continue
            if th > 2.2 * h:                              # 明显是别的东西（大标题/框）
                continue
            if th < 0.35 * h and abs(tcy - cy) > 0.35 * h:  # 又小又偏（上下标以外不要）
                continue
            if max(0.0, max(x0, tb[0]) - min(x1, tb[2])) > STITCH_GAP * h:
                continue
            cand.append(tb)
        if not cand:
            return [], None
        cand.sort(key=lambda b: b[0])
        seed = min(range(len(cand)), key=lambda i: abs(cand[i][0] - x0))

        def _wide(lo, hi):
            return (max(cand[hi][2], x1) - min(cand[lo][0], x0)) > STITCH_WMAX * h

        lo = hi = seed
        while lo > 0 and (hi - lo + 2) <= STITCH_MAXBOX:
            if cand[lo][0] - cand[lo - 1][2] > STITCH_GAP * h or _wide(lo - 1, hi):
                break
            lo -= 1
        while hi + 1 < len(cand) and (hi - lo + 2) <= STITCH_MAXBOX:
            if cand[hi + 1][0] - cand[hi][2] > STITCH_GAP * h or _wide(lo, hi + 1):
                break
            hi += 1
        sel = cand[lo:hi + 1]
        uni = [min(b[0] for b in sel), min(b[1] for b in sel),
               max(b[2] for b in sel), max(b[3] for b in sel)]
        return sel, uni

    def _engine(self):
        """`stage2.text.TextEngine`（懒建，`device="cpu"`：这条通道只在离线单测里跑）。"""
        if self._eng is not None:
            return self._eng
        try:
            from stage2.text import TextEngine
            self._eng = TextEngine(device="cpu")
        except Exception:                              # noqa: BLE001
            self._eng = None
        return self._eng

    def _read_union(self, uni):
        """并集框在**原图**上重读**一次**（不是把几个框分别 OCR 再拼串）。"""
        if self._bgr is None:
            return None
        eng = self._engine()
        if eng is None:
            return None
        h, w = self._bgr.shape[:2]
        x0 = max(0, int(round(uni[0])))
        y0 = max(0, int(round(uni[1])) - STITCH_PADY)
        x1 = min(w, int(round(uni[2])))
        y1 = min(h, int(round(uni[3])) + STITCH_PADY)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        crop = self._bgr[y0:y1, x0:x1]
        try:
            got = eng.read(crop, [(0, 0, crop.shape[1], crop.shape[0])],
                           roles=(STITCH_ROLE,))
        except Exception:                              # noqa: BLE001
            return None
        rec = (got[0].get("reads") or {}).get(STITCH_ROLE) if got else None
        if not rec:
            return None
        return str(rec[0] or "").strip(), float(rec[1] or 0.0)

    def stitch(self, info: dict, cur_name=None, designator=None, pair=None):
        """同行多框 -> 并集重读 -> 命中库键且脚数自洽才返回 `{text, conf, by}`。

        `by="stitch"`。写出去的字符串优先用**库键原样**（`AF 90° WJDG` 与 GT 逐字符
        相同），库键形态不干净时退回重读串。
        """
        if stitch_level() <= 0 or not self._img_ready():
            return None
        box = ((info or {}).get("partname") or {}).get("box")
        if not box or len(box) < 4:
            self.stats["stitch_skip"] += 1
            return None
        sel, uni = self._row(box)
        if len(sel) < 2 and reread_level() <= 0:      # 只有一个框，且没开单框重读：不归这条通道管
            self.stats["stitch_skip"] += 1
            return None
        got = self._read_union(uni)
        if got is None or not got[0]:
            self.stats["stitch_skip"] += 1
            return None
        text, conf = got
        des = str(designator or "").strip()
        if des and text == des:
            self.stats["stitch_skip"] += 1
            return None
        old = str(cur_name or "").strip()
        if libpins.norm(text) == libpins.norm(old):    # 重读没带来新信息
            self.stats["stitch_skip"] += 1
            return None
        key = exact_key(text)
        if key is None or not _fit(key, pair):
            self.stats["stitch_skip"] += 1
            return None
        out = key if _key_clean(key) else text
        if not guard(old, out, pair):
            self.stats["guard"] += 1
            self.stats["stitch_skip"] += 1
            return None
        self.stats["stitch"] += 1
        return {"text": out, "conf": conf, "by": "stitch", "boxes": len(sel),
                "read": text}

    # -- 统计 --------------------------------------------------------------
    def summary_line(self) -> str:
        """每个 case 一行的统计（`build_doc` 收尾时打印）。"""
        s = self.stats
        line = ("[stage2] partname PP：采用 %d 个（库键 %d、同名 %d、高置信 %d；弃权 %d）"
                % (s["libkey"] + s["same"] + s["high"],
                   s["libkey"], s["same"], s["high"], s["skip"]))
        if stitch_level():
            line += ("｜拼接：采用 %d、弃权 %d｜护栏拦下 %d"
                     % (s["stitch"], s["stitch_skip"], s["guard"]))
        return line


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
    if hit.get("boxes"):
        pn[CARRIER]["boxes"] = int(hit["boxes"])
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
    # 拼接通道（`STAGE2_PARTNAME_STITCH`）也走同一条压回：只开拼接、不开 PP 时
    # 采用值同样会被 fusion 的吸附打回，必须一起放行。
    if level() <= 0 and stitch_level() <= 0:
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
