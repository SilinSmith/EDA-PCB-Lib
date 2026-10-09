# -*- coding: utf-8 -*-
"""官方元器件库（`cpntLibrary`）的**引脚补号 / 引脚改名**先验。

★ 默认**开**（`STAGE2_LIB_PINS=1`，2026-10-09 从「默认关」翻过来；依据见下）。关掉时本模块
  * 不读库文件、不建索引、不 import 额外依赖（只有 `envflag`，那是仓内既有模块）；
  * `complete_pins()` / `names_for()` 在入口第一行就返回，调用方拿到的东西与不挂接
    这条通道时**逐字节一致**。
⇒ 改默认的依据（220 张回归，`work\tools\diag\_libpins_cpu.py` + `_split220.py`，纯 CPU 32 s，
  输入 `work\_full220\runs\stage1` + `work\_f220_A\_assigned`）：官方 10GT 10 张
  **62.5743 → 65.1834（+2.6091，7 涨 0 跌**；pin tp 38→47 / fp 42→38 / fn 98→89）、另 10 张
  官方（`10_GTcase`）逐位不变、派生 200 张 **31.3691 → 31.4300（+0.0609，24 涨 25 跌，
  最差 0167 −2.924）**、官方 20 合计 **+1.3045**；交付链（真 `fusion.run`，官方 9 张）**+1.60**。
  门槛是「官方为正 且 派生不为负」。`STAGE2_LIB_PINS_FUZZY=1.0`（只用精确键）与默认 0.85
  在两档上几乎相同（派生 +0.0616 vs +0.0609），故保持 0.85。
⇒ 仍要记住的风险：库是 1.76 MB 的**外部数据**，2026-10-12 起随仓库提交在
  `eda-pcb/data/cpntLibrary.json`，路径解析走 `assets.cpnt_library()` 的候选链
  （环境变量 `STAGE2_CPNT_LIB` → **包内** → 本机 `work\data\` → `赛题资料\`），与 CWD
  无关；**旧默认指的就是本机那个 `work\data\`，容器里必然不存在 ⇒ 静默丢分**。而下面
  两条规则的
触发条件依赖「元件型号名被读对」。stage2 直出的 `Name` 是 OCR 原文（官方 9 张实测
17 个 box 里只有 1 个能命中库），真正被读对是**融合层**补强之后的事 —— 所以收益
主要在交付链（`build_doc` 也被 `fusion/run.py` 调用，见 `fusion/run.py:63/128`），
stage2 单跑几乎不触发。

两条规则（都**不看 GT**，都**只在唯一映射时动手**）
--------------------------------------------------
**规则 A（补号）** —— 某个 box/IC 的引脚桩**读到了引脚名、但没读到合法引脚号**
时，用 `库[型号].pins` 里 `name` 与它相等的记录；若该名字在库里**唯一映射到一个
号**，就用那个号（写成 `pin_<号>`），并让它通过「没读出号就丢」的闸门
（`BOX_PIN_DROP_UNREAD`）。库里同一 `number` 常对应多个 `name`（多符号实例聚合），
反之亦然，所以「唯一」的判据是：把该名字在库里出现过的**所有** `number` 收集起来
按 `norm()` 去重，集合大小 == 1。

**规则 B（改名）** —— 某个**号读到了**、但 `pinname` 为空或不在库里该型号的
名字集合里的引脚，用库的 `(型号, 号) → 名字`；**只有该号在库里唯一对应一个名字**
时才改写 `pinname`。`norm()` 后去重，集合大小 == 1 才动手。

**为什么必须「唯一」才动手**（而不是「挑一个像的」）：库是多实例聚合的，一个名字
可能对应 `1`/`2`/`A1`/`GND` 好几个号，一个号也可能对应 `VDD`/`VCC`/`VIN` 好几个名。
官方 `pin_f1` 的键是 `(元件, pin_key_to_number(pk))`，命中后还要求
`p.get('pinname') == g.get('pinname')` —— 猜错 = `fp += 1` **且** `fn += 1`，
比不动手（只 `fn += 1` 或干脆不参与）更差。多候选时保持现状是唯一稳的选择。

离线模拟量到的精度（`work\\tools\\diag\\_libsim.py`，官方 9 张；那是**直接改交付
JSON** 的模拟，不是本模块的实测）：规则 A 安全版补 6 个引脚 **6/6 号正确**
（66.3024 → 67.3085）、规则 B 改名 4 处 **4/4 正确**，A+B 合计 68.7659（+2.4635）。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:                                          # 作为包导入（正常路径）
    from envflag import env_on, env_str
except ImportError:                           # 直接跑脚本自检
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from envflag import env_on, env_str

# ── 开关 ──────────────────────────────────────────────────────────────────
# `STAGE2_LIB_PINS`（默认 "1"=开，2026-10-09 翻）：本通道总闸。见模块 docstring。
LIB_PINS = env_on("STAGE2_LIB_PINS", "1")

# 库文件路径。候选链见 `assets.cpnt_library()`：**包内 `eda-pcb/data/cpntLibrary.json`
# （1,755,587 B / 337 键，随仓库提交 ⇒ 进镜像）** → 本机 `WORKSPACE\work\data\` →
# `WORKSPACE\赛题资料\` → 数据集目录。相对路径按 `eda-pcb\stage2\libpins.py` 上溯两级
# 解析，所以**与当前工作目录无关**（stage2 会被 `cd eda-pcb` 之后以 `-m stage2.run`
# 调用，也会被 `submission\run.py` 从别处调用）。
_PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIB = _PKG_ROOT / "data" / "cpntLibrary.json"
LIB_PATH = env_str("STAGE2_CPNT_LIB", "") or ""

_NON_ALNUM = re.compile(r"[^0-9a-z]")


def norm(s) -> str:
    """归一化型号名 / 引脚名：小写 + 去掉所有非字母数字。

    与 `work\\tools\\diag\\_libcov.py:32` 的 `norm()` **逐字符相同**（模拟用的就是
    那一份），`1.25-2PWB` → `1252pwb`、`AMS1117-3.3` → `ams111733`。
    """
    return _NON_ALNUM.sub("", str(s or "").lower())


def lib_path() -> Path:
    """解析库文件路径：`STAGE2_CPNT_LIB` → 包内候选链（见 `assets.cpnt_library()`）。

    ★ 2026-10-12 修：**默认不再是 `WORKSPACE\\work\\data\\cpntLibrary.json`**。`work\\`
    不进仓库也不进镜像 ⇒ 容器里那条路径必然不存在，而本通道缺席时只打印一行就跳过，
    等于在判题机上白丢 +1.60（官方 9 张交付链实测）。现在与两个权重共用
    `assets.resolve()` 的规则（环境变量 → **包内相对候选** → 本机兜底），
    `eda-pcb/data/cpntLibrary.json` 随仓库提交，于是「拷整个 eda-pcb 进 /submission」
    这一条就够。
    """
    if LIB_PATH:
        return Path(LIB_PATH)
    try:
        from assets import cpnt_library            # 包根（eda-pcb）已在 sys.path 上
    except ImportError:                            # 直接按文件路径加载本模块时
        import sys as _s
        _s.path.insert(0, str(_PKG_ROOT))
        from assets import cpnt_library
    return Path(cpnt_library(must_exist=False))


# ── 惰性加载的库上下文 ────────────────────────────────────────────────────

class LibCtx:
    """库的**惰性**句柄：`STAGE2_LIB_PINS=0` 时永远不会真的读文件。

    为什么包成一个对象而不是模块级全局：`build_doc` 每次调用都可能被喂不同的
    `STAGE2_CPNT_LIB`（A/B 脚本会来回切），而库有 1.76 MB，每张图重读一次太浪费。
    调用方在**一次装配里**建一个 `LibCtx` 复用即可；它的 `load()` 是幂等的。
    """

    __slots__ = ("_raw", "_index", "_path", "_tried", "stats")

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else None
        self._raw: Optional[dict] = None
        self._index: Optional[dict] = None
        self._tried = False
        # 诊断计数（`STAGE2_LIB_PINS_DBG=1` 时由 `build_doc` 打印）。
        self.stats: Dict[str, int] = {"lib_miss": 0, "add": 0, "rename": 0}

    # -- 加载 ------------------------------------------------------------
    def load(self) -> bool:
        """读库 + 建两个唯一映射索引；文件不存在/读不动时记 `lib_miss` 并返回 False。"""
        if self._tried:
            return self._raw is not None
        self._tried = True
        p = self._path or lib_path()
        try:
            import json                                    # 只有真的要用才 import
            raw = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:                 # 缺文件 / 坏 JSON 都不该炸管线
            print("[libpins] 库不可用（%s）：%s —— 本通道自动跳过（引脚先验/改名全部失效，"
                  "官方 9 张约 −1.6 分）。用 STAGE2_CPNT_LIB 指定库文件；"
                  "包内应有 eda-pcb/data/cpntLibrary.json" % (p, e))
            return False
        self._raw = raw
        self._index = _build_index(raw)
        return True

    def key_for(self, comp_name: str) -> Optional[str]:
        """元件型号名 → 库键。

        三档与 `_libcov.Lib` 一致：**精确 → 归一化 → 模糊**。模糊档（默认阈值 0.85，
        可用 `STAGE2_LIB_PINS_FUZZY` 调）只用来**找到库键**，具体某条引脚仍走「唯一」
        判定 —— 实测官方 9 张里 `MT24/2` 靠模糊命中 `MT2492`。
        """
        if not self.load():
            self.stats["lib_miss"] += 1
            return None
        t = str(comp_name or "").strip()
        if not t:
            return None
        raw = self._raw or {}
        if t in raw:
            return t
        nt = norm(t)
        if not nt:
            return None
        cands = (self._index or {}).get("by_norm", {}).get(nt)
        if cands:
            return cands[0]
        cutoff = float(os.environ.get("STAGE2_LIB_PINS_FUZZY", "0.85"))
        if cutoff <= 0:
            return None
        import difflib
        hits = difflib.get_close_matches(nt, list((self._index or {}).get("by_norm", {})),
                                        n=1, cutoff=cutoff)
        if not hits:
            return None
        return (self._index or {})["by_norm"][hits[0]][0]

    # -- 两条规则 --------------------------------------------------------
    def num_for_name(self, key: str, pin_name: str) -> Optional[str]:
        """规则 A 的查表：`(库键, 引脚名) → 号`，**唯一映射**时才返回号。"""
        if not key or not pin_name:
            return None
        if not self.load():
            return None
        m = (self._index or {}).get("num_by_name", {}).get(key) or {}
        return m.get(norm(pin_name))

    def name_for_num(self, key: str, num) -> Optional[str]:
        """规则 B 的查表：`(库键, 引脚号) → 名字`，**唯一映射**时才返回名字。"""
        if not key or num is None or str(num) == "":
            return None
        if not self.load():
            return None
        m = (self._index or {}).get("name_by_num", {}).get(key) or {}
        return m.get(norm(num))

    def names_of(self, key: str) -> Dict[str, int]:
        """库键下**全部**引脚名的 `norm()` 集合（规则 B 判「这个名字库里有没有」）。"""
        if not key:
            return {}
        if not self.load():
            return {}
        return (self._index or {}).get("names", {}).get(key) or {}


def _build_index(raw: dict) -> dict:
    """把库扫一遍，建 4 张表（337 键 / ~1.2 万条引脚记录，实测 <0.3 s）。

    * `by_norm`      —— `norm(键) → [键…]`（归一化/模糊匹配用）
    * `num_by_name`  —— `键 → {norm(名) → 号}`，**只保留唯一映射**；
                       名字在库里对应 2 个及以上不同 `number` 的**整个丢掉**（规则 A）
    * `name_by_num`  —— `键 → {norm(号) → 名}`，同样只保留唯一映射（规则 B）；
                       改名前要落回**库里原始写法**（不是 `norm` 后的串）
    * `names`        —— `键 → {norm(名) → 出现次数}`（规则 B 判「当前名在不在库里」）
    """
    by_norm: Dict[str, List[str]] = {}
    num_by_name: Dict[str, Dict[str, str]] = {}
    name_by_num: Dict[str, Dict[str, str]] = {}
    names: Dict[str, Dict[str, int]] = {}
    for k in raw:
        by_norm.setdefault(norm(k), []).append(k)
        _n2n: Dict[str, set] = {}
        _n2n_raw: Dict[str, str] = {}
        _num2name: Dict[str, set] = {}
        _num2name_raw: Dict[str, str] = {}
        nm_cnt: Dict[str, int] = {}
        for p in (raw[k].get("pins") or []):
            nm = str(p.get("name") or "").strip()
            num = str(p.get("number") or "").strip()
            nn, nu = norm(nm), norm(num)
            if nn:
                nm_cnt[nn] = nm_cnt.get(nn, 0) + 1
            if nn and nu:
                _n2n.setdefault(nn, set()).add(nu)
                _n2n_raw.setdefault(nn, num)           # 首次出现的**原始号写法**
            if nn and nu:
                _num2name.setdefault(nu, set()).add(nn)
                _num2name_raw.setdefault(nu, nm)       # 首次出现的**原始名写法**
        num_by_name[k] = {n: _n2n_raw[n] for n, s in _n2n.items() if len(s) == 1}
        # 号→名：归一化后唯一才算数；返回值用**库里首次出现的原始写法**
        name_by_num[k] = {u: _num2name_raw[u]
                          for u, s in _num2name.items() if len(s) == 1}
        names[k] = nm_cnt
    return {"by_norm": by_norm, "num_by_name": num_by_name,
            "name_by_num": name_by_num, "names": names}


# ── 对外接口 ──────────────────────────────────────────────────────────────

def complete_pins(comp_name, stubs: Sequence[dict],
                  ctx: Optional[LibCtx] = None,
                  used=None) -> int:
    """**规则 A**：就地给「名读到、号没读出来」的桩补 `num`，返回补了几个。

    `stubs` 是 `assemble.build_doc` 里那份 `raw`（每项 `{"num","nm"[,"kept","fill"]}`），
    **就地改写**，不改长度、不动顺序（调用方按位置索引 `raw[k]` 取用）。

    「号没读出来」在本管线里是**两种**形态，都要抓：
      * `num == ""`（一个号都没读出来）；
      * `num` 是**位置补位号**（`fill=True`，OCR 读到的号没通过 1..999/不撞号检查、
        被 `tmp[i]=""` 之后按顺序填出来的）。位置号是**假的**，真正的证据是那个
        读到的名字 —— 模拟里 `_libsim.py` 补进去的 6 个引脚，在本管线里正是这种
        「有名字、号是位置号」的桩。`fill` 缺省时按 `kept is False` 认。

    只在下面四条**同时**成立时补：
      1. `STAGE2_LIB_PINS=1`（总闸）；
      2. 该桩读到了引脚名 `nm` 非空（「名读到」是规则 A 的前提，也是唯一能查表的键）；
      3. 元件型号名能命中库键（精确/归一化/模糊）；
      4. 该名字在库里**唯一映射**到一个号，且那个号没被这颗 IC 里**别处读出来的
         有效号**占用（`used`）。撞了就放弃，绝不改别的号 —— 补号的目的是把
         **假号**换成一个真号，不是把两个真脚挤到同一个 `pin_<号>` 键上。
    """
    if not LIB_PINS or not stubs:
        return 0
    ctx = ctx or LibCtx()
    n = 0
    key = ctx.key_for(comp_name)
    if not key:
        return 0
    used = used if used is not None else set()
    for r in stubs:
        # 「号没读出来」：空的，或者是位置补位号（`fill`）；**读出来且通过检查的号不动**
        if str(r.get("num") or "").strip() and not r.get("fill") and r.get("kept", True):
            continue
        nm = str(r.get("nm") or "").strip()
        if not nm:
            continue                       # 名也没读到 ⇒ 无从查表
        num = ctx.num_for_name(key, nm)
        if not num or num in used:
            continue
        r["num"] = num
        r["kept"] = True                   # 换成真号 ⇒ 不该再被当「没读出号」丢掉
        r["fill"] = False
        used.add(num)
        n += 1
    if n:
        ctx.stats["add"] = ctx.stats.get("add", 0) + n
    return n


def names_for(comp_name, num, ctx: Optional[LibCtx] = None) -> Optional[str]:
    """**规则 B**：`(元件型号, 引脚号) → 唯一引脚名`；不唯一/查不到时返回 `None`。

    返回的是**库里的原始写法**（如 `VDD`、`PA00`），不是 `norm()` 后的串 —— 交付
    产物要拿它直接当 `pinname` 用。
    """
    if not LIB_PINS:
        return None
    ctx = ctx or LibCtx()
    key = ctx.key_for(comp_name)
    if not key:
        return None
    return ctx.name_for_num(key, num)


def new_ctx(path=None) -> LibCtx:
    """建一个可复用的库句柄（总闸关时也安全：它一次文件都不会读）。"""
    return LibCtx(path)


def _selftest() -> int:
    """自检：用一个小合成库验两条规则的**唯一性**判据（不依赖真库）。"""
    import json
    import tempfile
    bad = []
    fake = {
        "AA1": {"pins": [
            {"name": "VDD", "number": "1"}, {"name": "VDD", "number": "1"},
            {"name": "PA00", "number": "2"},
            {"name": "X", "number": "3"}, {"name": "X", "number": "4"},   # 一名多号 → 不唯一
            {"name": "Z1", "number": "6"}, {"name": "Z2", "number": "6"}, # 一号多名 → 不唯一
        ]},
        "BB-2": {"pins": [{"name": "nRST", "number": "7"}]},
    }
    # 规则 A：VDD 唯一 → 1；X 对应两个号 → 不动手
    # 规则 B：号 1 唯一 → VDD；号 6 对应两个名 → 不动手；号 3 同一名字重复出现 → X
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "lib.json"
        f.write_text(json.dumps(fake, ensure_ascii=False), encoding="utf-8")
        globals()["LIB_PINS"] = True
        ctx = LibCtx(f)
        if ctx.num_for_name("AA1", "vdd") != "1":
            bad.append("规则A VDD→1")
        if ctx.num_for_name("AA1", "X") is not None:
            bad.append("规则A 一名多号时不该动手")
        if ctx.name_for_num("AA1", "1") != "VDD":
            bad.append("规则B 1→VDD")
        if ctx.name_for_num("AA1", "6") is not None:
            bad.append("规则B 号6 多名时不该动手")
        if ctx.name_for_num("AA1", "3") != "X":
            bad.append("规则B 号3 同名重复 → X")
        if ctx.key_for("bb2") != "BB-2":
            bad.append("归一化命中 BB-2")
        if ctx.name_for_num("AA1", 7) is not None:
            bad.append("不存在的号")
        stubs = [{"num": "", "nm": "VDD"}, {"num": "9", "nm": "nRST"},
                 {"num": "", "nm": "没这个名字"}]
        n = complete_pins("AA1", stubs, ctx, used={"9"})
        if n != 1 or stubs[0]["num"] != "1" or not stubs[0].get("kept"):
            bad.append("complete_pins 就地补号")
        if stubs[1]["num"] != "9" or stubs[2]["num"] != "":
            bad.append("complete_pins 不该动已读/查不到的桩")
        # `fill=True`（位置补位号）也要抓：位置号是假的，名字才是证据
        stubs2 = [{"num": "5", "nm": "PA00", "kept": False, "fill": True}]
        if complete_pins("AA1", stubs2, ctx, used=set()) != 1 or stubs2[0]["num"] != "2":
            bad.append("complete_pins 位置补位号也要换真号")
        stubs3 = [{"num": "5", "nm": "PA00", "kept": True, "fill": False}]
        if complete_pins("AA1", stubs3, ctx, used=set()) != 0:
            bad.append("complete_pins 不该动读出来的有效号")
        globals()["LIB_PINS"] = False
        if complete_pins("AA1", [{"num": "", "nm": "VDD"}], ctx) != 0:
            bad.append("总闸关时不该动手")
        if names_for("AA1", "1", ctx) is not None:
            bad.append("总闸关时不该改名")
        globals()["LIB_PINS"] = True
    print("libpins 自检：%s" % ("全部通过" if not bad else "失败 " + " / ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
