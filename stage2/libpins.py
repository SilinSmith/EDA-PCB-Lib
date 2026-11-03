# -*- coding: utf-8 -*-
"""官方元器件库（`BoxICLibrary_200cases` / `cpntLibrary`）的**引脚补号 / 引脚改名**先验。

★ 库 = **官方 box/IC 库 `BoxICLibrary_200cases.json`（360 键，默认）**，缺席时退到
  `cpntLibrary.json`（337 键）。依据（`work\\tools\\diag\\_libpins_cpu.py` 三臂 A/B，
  同一份 stage1 + `_assigned` 输入、只换库文件）：官方 9 张 68.7995（cpnt） → **69.0883**
  （boxic），派生 220 张 33.2120 → **33.4364**；两份库合并（BoxIC 优先 + cpnt 补缺键）
  在官方 9 张与 boxic **逐字节相同**、派生 220 张 33.4334（−0.003，无收益）⇒ 不合并。
  覆盖率（`tools/diag/_libcov.py`）：IC 名命中 官方 9 张 85.7% vs 71.4%、200 案例
  61.6% vs 30.9%；引脚名覆盖 99.2%/69.7% vs 99.0%/64.9%。官方 `pin_f1` 只考 box 元件。

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
⇒ 仍要记住的风险：库是 1.8 MB 的**外部数据**，2026-10-12/14 起随仓库提交在
  `eda-pcb/data/BoxICLibrary_200cases.json`（默认）与 `eda-pcb/data/cpntLibrary.json`
  （兜底），路径解析走 `assets.libraries()` 的候选链（环境变量 `STAGE2_BOXIC_LIB` /
  `STAGE2_CPNT_LIB` → **包内** → 本机 `work\data\` → `赛题资料\`），与 CWD
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
    from envflag import env_on, env_raw, env_str
except ImportError:                           # 直接跑脚本自检
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from envflag import env_on, env_raw, env_str

# ── 开关 ──────────────────────────────────────────────────────────────────
# `STAGE2_LIB_PINS`（默认 "1"=开，2026-10-09 翻）：本通道总闸。见模块 docstring。
LIB_PINS = env_on("STAGE2_LIB_PINS", "1")

# §5.150 (B) 的**细闸**：把本模块的两条规则分别关掉，用来给「库的整体贡献」做归属。
# 两条都**默认 1**（与历史逐位相同），只在 A/B 臂里显式设 0。
#   * `STAGE2_LIB_PINS_NUMFILL=0`  ⇒ 只关**规则 A**（`complete_pins`：按名字补号）
#   * `STAGE2_LIB_PINS_NAMEFILL=0` ⇒ 只关**规则 B**（按号补名）——**生效处是
#     `stage2/assemble.py:2467` 起的内联规则 B**（`_libpins.LIB_PINS_NAMEFILL` +
#     `_LIB_CTX.name_for_num`），**不是**本模块的 `names_for()`：后者
#     （`stage2/libpins.py:504`）全仓只有定义与自检两个引用、**没有管线调用点**
#     （§5.150 §4(B) 实测：早年把开关挂在它上面时是空开关）。
# 注意：`librefine` 走的是 `LibCtx.name_for_num` 方法，不走 `names_for`，
# 所以 `NAMEFILL=0` **只**影响 assemble 的规则 B —— 这正是做归属需要的正交性。
LIB_PINS_NUMFILL = env_on("STAGE2_LIB_PINS_NUMFILL", "1")
LIB_PINS_NAMEFILL = env_on("STAGE2_LIB_PINS_NAMEFILL", "1")

# 规则 A 的**型号名兜底**：型号名认不出来时，用读到的引脚名集合反查库键
# （`LibCtx.key_for_names`）。动机见该方法 docstring（主尺 case11 `USB2`：名读对、
# 型号读成 `SG90` ⇒ 规则 A 整颗不发）。
# 档位：`0`=关（默认，现行为逐位不变）；`1`=只在 `key_for()` 失败时兜底；
#       `2`=`key_for()` 命中的库键覆盖不到一半读到名字时，也允许换成按名字反查的键。
LIB_NAMEID = env_str("STAGE2_LIB_NAMEID", "0").strip()

# 库文件路径。**2026-10-14 起默认是官方 box/IC 库 `BoxICLibrary_200cases.json`**
# （用户 m42932「以新的这个为参考」）：官方 `pin_f1` 只考 box 元件，而这份库就是
# box/IC 那一档（360 键），同口径实测比 `cpntLibrary`（337 键）好 —— 官方 9 张
# 68.7995 → **69.0883**（+0.29）、派生 220 张 33.2120 → **33.4364**（+0.22）；
# 合并两份（BoxIC 优先 + cpnt 补缺键）**没有额外收益**（官方 9 张逐字节相同、
# 派生 220 张 −0.003）⇒ 不做合并，只按优先级取第一个存在的文件。
# 候选链（`assets.libraries()`）：`STAGE2_BOXIC_LIB` / `STAGE2_CPNT_LIB`（显式指定，
# **即使不存在也照用**，好让失败发声）→ **包内 `eda-pcb/data/BoxICLibrary_200cases.json`**
# → 包内 `eda-pcb/data/cpntLibrary.json`（BoxIC 缺席时的兜底）→ 本机 `work\data\` →
# `赛题资料\` → 数据集目录。相对路径按本文件上溯两级解析，**与当前工作目录无关**
# （stage2 会被 `cd eda-pcb` 之后以 `-m stage2.run` 调用，也会被 `submission\run.py`
# 从别处调用）。
_PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIB = _PKG_ROOT / "data" / "BoxICLibrary_200cases.json"   # 兼容旧引用
LIB_PATH = env_str("STAGE2_CPNT_LIB", "") or ""       # 兼容旧名：显式指定库文件
BOXIC_PATH = env_str("STAGE2_BOXIC_LIB", "") or ""    # 新名：显式指定 box/IC 库

_NON_ALNUM = re.compile(r"[^0-9a-z]")


def norm(s) -> str:
    """归一化型号名 / 引脚名：小写 + 去掉所有非字母数字。

    与 `work\\tools\\diag\\_libcov.py:32` 的 `norm()` **逐字符相同**（模拟用的就是
    那一份），`1.25-2PWB` → `1252pwb`、`AMS1117-3.3` → `ams111733`。
    """
    return _NON_ALNUM.sub("", str(s or "").lower())


def has_letter(s) -> bool:
    """字符串里是否至少有一个字母（用来识别「纯数字」的库名 / 读数）。"""
    return any(ch.isalpha() for ch in str(s or ""))


def n_letters(s) -> int:
    """字母个数（`known2` 档判「我方读数是个像样的名字」而不是单字乱码）。"""
    return sum(1 for ch in str(s or "") if ch.isalpha())


def has_digit(s) -> bool:
    """字符串里是否至少有一个数字（用来识别「编号式」的库名，如 `P67`/`L2`/`PA3`）。"""
    return any(ch.isdigit() for ch in str(s or ""))


#: `STAGE2_LIB_REFINE_DEGEN` 的合法档位（语义见 `stage2/librefine.py` 的模块 docstring）。
DEGEN_MODES = ("off", "letter", "shape", "known", "known2")

#: `STAGE2_LIB_PINS_NUMGATE` 的合法档位（语义见 `numgate_mode` 的 docstring）。
NUMGATE_MODES = ("0", "off", "conflict", "seen")


def degen_block(ctx, mode: str, cur, ref) -> bool:
    """库名回填守卫的**唯一判据**：`(我方读数 cur, 库名 ref)` 该不该挡。

    两条回填路径（`stage2/assemble.py` 的规则 B、`stage2/librefine.py`）都调这里，
    避免语义漂移（2026-10-12 §5.52 的教训：守卫挂错路径 = 等于没挂）。

      * `letter` —— 我方读数有字母、库名没有 ⇒ 挡（纯数字库名是库自己的误读）。
      * `known`  —— 同 `letter`，且我方读数必须是**库认识的名字**（乱码照旧让库纠正）。
      * `known2` —— `known` **加上**：我方读数是「≥2 个字母的库认识名字」，而**库名是
                    「编号式」（含数字）**时挡。`known` 挡不住的那一类是「库名也有字母、
                    但库里那颗元件的编号与图上根本对不上」——官方第 2 张 `STC11` 的
                    `pin_2` 就是 我方读对 `VCC`、库却给 `P67`（该键的 1..4 号在库里是
                    `P50/P67/…`，与图上的 `GND/VCC/SCL/SDA` 完全不是一回事）。
                    ★ 为什么必须**再加「库名含数字」这一道**：只看「我方名字库认识且 ≥2
                    字母」会把库的**正确**修复也挡掉 —— 派生 19 张实测 4 根由对变错
                    （`0011 U2.pin_9` `GN→GND_PAD`、`0029 U1.pin_2` `PGND→FAULTZ`、
                    `0190 U1.pin_2` `DGND→SFL`、`0190 U1.pin_27` `AD→AVDD`），
                    官方第 8 张 `U1.pin_7` `NC→VDDIO` 同病 ⇒ 只留「库给编号名」那一类
                    （派生那 4 根的库名都是功能名、不含数字；官方 `P67` 含数字）。
                    另外要求**我方名字也不含数字**：我方读数本身就是编号式时（`P67`/`L2`），
                    与库处在同一套编号体系里，让库去纠正。
    """
    if not mode or mode in ("off", "shape") or not cur or not ref:
        return False
    if mode == "letter":
        return has_letter(cur) and not has_letter(ref)
    if not (has_letter(cur) and ctx.is_known_name(cur)):
        return False                       # 乱码 / 纯数字读数 ⇒ 交给库纠正
    if not has_letter(ref):
        return True                        # `known` 的原判据：纯数字库名
    return (mode == "known2" and n_letters(cur) >= 2
            and has_digit(ref) and not has_digit(cur))


def degen_mode() -> str:
    """退化库名守卫的档位（`STAGE2_LIB_REFINE_DEGEN`，**默认 `known2`**）。

    ★ **唯一读取点**：库名回填有**两条**路径，都走这里读档，避免语义漂移 ——
      * `stage2/librefine.py`（融合层 ⑩ 的库名回填，`(型号,号) → 库名`）；
      * `stage2/assemble.py:1516` 的**规则 B**（`build_doc` 里的同名回填）。

    为什么规则 B 也要挂：2026-10-12 追到的回归正是发生在它这条路径上 ——
    融合层把型号名吸附成库键 `OUT`/`STC11` 之后，规则 B 用该键的**纯数字**引脚名
    `1`..`4` 把读对的 `VIN`/`TAB` 冲成 `3`/`4`；而 `librefine` 跑到时名字已经是
    数字了（`ref == cur`），一次都不会触发，所以只在 `librefine` 里挂守卫等于没挂。

    档位（`off`/`letter`/`shape`/`known`/`known2`）：
      * `off`    —— 老行为。
      * `letter` —— 库名不含字母、而我方读数含字母 ⇒ 该脚不改。
      * `shape`  —— 整颗元件：库给的名字全是纯数字、我方读数全不是 ⇒ 整颗不改。
      * `known`  —— `letter` + **我方读数必须是库认识的名字**（`LibCtx.is_known_name`）。
      * `known2` —— `known` + 「我方读数是 ≥2 个字母的库认识名字、**而库名是编号式（含数字）**」
                    ⇒ 拦。挡的是「库名有字母、但库里那颗元件的编号与图上对不上」那一类
                    （官方第 2 张 `STC11` 的 `pin_2`：我方读对 `VCC`，库给 `P67`）。
                    ★ 只按「≥2 字母」拦（不加「库名含数字」）实测**负收益**：派生 19 张
                    4 根由对变错（库的正确修复 `GND_PAD`/`FAULTZ`/`SFL`/`AVDD` 被挡）、
                    官方第 8 张 `VDDIO` 同病。判据的唯一实现在 `degen_block`，
                    两条回填路径都调它。

    ★ 2026-10-12 为什么默认选 `known`（两池实测，`docs/RESULTS.md` §5.52）：
      官方 10 张离线档 复刻尺 63.0897 → **63.2759**（protocol2 75.75 → **76.21**，
      tp 引脚 81 → 84）、派生 18 张 14.5290 → **14.5376**（protocol2 25.44 → **25.48**，
      tp 引脚 49 → 50），两池**逐脚 right→wrong 均为 0**。`letter` 官方同分但派生
      −0.0031（乱码 `VWPSTO2`/`IOW2T` 挡住库的合法纠正），`shape` 派生同分但官方
      少赚 0.0658（`STC11` 的库名表大多是字母，整颗判的「全纯数字」条件不成立）。
      ⇒ `known` 严格支配另外两档。精简 `-lean` 与完整档实测**不受影响**（逐位相同）。
    ★ 2026-10-13 默认再升到 `known2`（`docs/RESULTS.md` §5.53）：官方 10 张
      63.2759 → **63.3417**（protocol2 76.21 → **76.39**，tp 引脚 84 → 85，只动
      case 2：54.4862 → 55.1441），派生 18 张 14.5376 → **14.5376**（protocol2
      26.95 → **26.95**，逐位相同）、派生 **220 张**同向。宽判据（只看「≥2 字母」）
      实测负收益，见 `degen_block` 的 docstring。
    ★ 用 `env_raw` 而不是 `env_str`：`env_str` 把 `off`/`0`/`none` 归一成空串
      （它服务的是「开关是否打开」），而这里的模式值 `off` 本身是个**合法档位** ——
      用 `env_str` 会让 `STAGE2_LIB_REFINE_DEGEN=off` 静默变成默认档（2026-10-12 自检抓到）。
    """
    v = (env_raw("STAGE2_LIB_REFINE_DEGEN", "known2") or "known2").strip().lower()
    return v if v in DEGEN_MODES else "known2"


def numgate_mode() -> str:
    """库补**号**的一致性闸门档位（`STAGE2_LIB_PINS_NUMGATE`，**默认 `0`=关**）。

    为什么要有它：规则 A 会把「读到的名字」换算成库里的号。库里那份 `name ↔ number`
    是**真实器件**的编号，而标注里的编号不一定是它 —— §5.133 记的 `TPS2553DBVR`
    就是现成例子：库 `/FAULT`=4，而 case6 的 GT 写 3，于是把本来正确的
    `pin_3 FAULT` 改成 `pin_4 /FAULT`，**一处 −0.16 分**。

    档位（`env_raw`，因为 `0`/`off` 本身是合法档位）：
      * `0` / `off` —— 关闭（默认；与历史行为逐位相同）。
      * `conflict`  —— **图上与库冲突就不补**：若该元件图上**真读**到过这个号、
        且那个号读到的名字与本次要补的名字不是同一个（按 `norm()` 比），
        说明图上自己给出了另一套配对 ⇒ 信图，放弃这次补号。
        这条是**纯否定**判据：图没读到那个号时它不干涉（库仍是唯一证据）。
      * `seen`      —— 在 `conflict` 之上再加一条：库给的号必须出现在该元件图上
        **真读到的号**集合里，否则放弃。更保守，但也更可能丢掉本该补对的号
        （图上只读到少数几个号时尤其明显），所以默认不开，只作 A/B 臂。
    """
    v = (env_raw("STAGE2_LIB_PINS_NUMGATE", "0") or "0").strip().lower()
    return v if v in NUMGATE_MODES else "0"


def lib_path() -> Path:
    """解析库文件路径：显式环境变量 → 包内候选链（见 `assets.libraries()`）。

    ★ 2026-10-12 修：**默认不再是 `WORKSPACE\\work\\data\\cpntLibrary.json`**。`work\\`
    不进仓库也不进镜像 ⇒ 容器里那条路径必然不存在，而本通道缺席时只打印一行就跳过，
    等于在判题机上白丢 +1.60（官方 9 张交付链实测）。现在与两个权重共用
    `assets.resolve()` 的规则（环境变量 → **包内相对候选** → 本机兜底），
    `eda-pcb/data/` 下的两份库都随仓库提交，于是「拷整个 eda-pcb 进 /submission」
    这一条就够。

    ★ 2026-10-14：默认库换成 `BoxICLibrary_200cases.json`（用户 m42932）。候选链里
    **BoxIC 缺席时自动退到 `cpntLibrary.json`**，所以只带旧库的镜像也还是开着的。
    显式设了 `STAGE2_BOXIC_LIB` / `STAGE2_CPNT_LIB` 时**直接用那个路径**（不校验存在），
    这样 A/B 指错文件会以「库不可用」的形态暴露，而不是悄悄换成另一份库。
    """
    if BOXIC_PATH:
        return Path(BOXIC_PATH)
    if LIB_PATH:
        return Path(LIB_PATH)
    try:
        from assets import libraries                   # 包根（eda-pcb）已在 sys.path 上
    except ImportError:                                # 直接按文件路径加载本模块时
        import sys as _s
        _s.path.insert(0, str(_PKG_ROOT))
        from assets import libraries
    try:
        cands = [Path(p) for p in libraries()]         # 只返回**存在**的，BoxIC 在前
    except Exception:                                  # noqa: BLE001
        cands = []
    return cands[0] if cands else DEFAULT_LIB


# ── 惰性加载的库上下文 ────────────────────────────────────────────────────

def key_uniq(ctx, comp_name: str) -> Optional[str]:
    """`LibCtx.key_for_uniq` 的鸭子类型入口。

    自检/探针里的假 ctx 只实现 `key_for`（`pinlib._selftest` 的精确档桩就是这样），
    没有这个方法就退化成 `key_for` —— 保证「关档逐位等于原行为」在所有 ctx 上都成立。
    """
    fn = getattr(ctx, "key_for_uniq", None)
    if callable(fn):
        return fn(comp_name)
    return ctx.key_for(comp_name)


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
                  "官方 9 张约 −1.6 分）。用 STAGE2_BOXIC_LIB 指定 box/IC 库、"
                  "STAGE2_CPNT_LIB 指定旧库；包内应有 eda-pcb/data/BoxICLibrary_200cases.json"
                  "（兜底 eda-pcb/data/cpntLibrary.json）" % (p, e))
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

    def key_for_uniq(self, comp_name: str) -> Optional[str]:
        """`key_for` 的**唯一性复核版**（`STAGE2_PINLIB_KEY_UNIQ`，默认 **1 = 开**，§5.550）。

        与 `key_for` 的唯一区别：模糊档命中时要求「最优明显优于次优」，否则当作认不出。
        阈值直接复用 `stage2/pinlib.py` 的 `FUZZY`/`FUZZY_MARGIN`（同一对常量，来源是
        同名组模糊匹配），不新造数；关档时**逐位等于 `key_for`**。

        病（R-b 识别器回归，`open_GT case6 U1`）：图上印 `78M12`，读出 `789M12`，
        `difflib` 在 `78M12` / `79M12` 上**同分 0.909**（打平）且取了 `79M12`；两颗的
        引脚表正好相反（`78M12=IN1/GND2/OUT3` vs `79M12=GND1/IN2/OUT3`）⇒ 下游所有
        「按库改号/改名」的通道都会把本来读对的 `GND=2` 改坏。
        """
        _on = (os.environ.get("STAGE2_PINLIB_KEY_UNIQ", "1") or "0").strip().lower()
        if _on in ("", "0", "false", "off", "none"):
            return self.key_for(comp_name)
        k = self.key_for(comp_name)
        if not k or not self.load():
            return k
        t = str(comp_name or "").strip()
        if t == k or norm(t) == norm(k):
            return k                                     # 精确/归一化档：天然唯一
        from . import pinlib as _PB                          # 延迟 import：避免环形依赖
        import difflib
        nt = norm(t)
        best, second = 0.0, 0.0
        for kk in (self._raw or {}):
            r = difflib.SequenceMatcher(None, nt, norm(kk)).ratio()
            if r > best:
                best, second = r, best
            elif r > second:
                second = r
        if best >= _PB.FUZZY and (best - second) >= _PB.FUZZY_MARGIN:
            return k
        self.stats["key_ambig"] = self.stats.get("key_ambig", 0) + 1
        return None

    # -- 两条规则 --------------------------------------------------------
    def key_for_names(self, names: Sequence[str], min_hit: int = 3) -> Optional[str]:
        """型号名认不出来时，用**读到的引脚名集合**反查库键（`STAGE2_LIB_NAMEID`）。

        动机（主尺 case11）：`USB2` 四根脚的名读对了（`VCC/D-/D+/GND`），但型号名被
        读成 `SG90` ⇒ `key_for()` 命中不了库 ⇒ 规则 A 整颗不发，交付只剩 1 个脚。
        而库里 `AF 90° WJDG` 的 名→号 表正是 `VCC=1,D-=2,D+=3,GND=4,SHLD=5`。

        判据要严：至少 `min_hit` 个名字命中、且命中数 ≥ 读到名字数的一半，
        并且**最优唯一**（打平就放弃）—— 猜错库键 = 号整片错 = fp+fn 双罚。
        """
        if not self.load():
            return None
        # ★ 这里**不能用 `norm()`**：它把 `D-`/`D+` 都归成 `d`，于是 `VCC/D-/D+/GND`
        #   在库里会命中 `SN65HVD232D` 这种八竿子打不着的芯片（实测打平 ⇒ 永远放弃）。
        #   本函数要的是「名字就是那个名字」，所以只做「去空格、大写」。
        def _c(s) -> str:
            return "".join(str(s or "").split()).upper()

        want = sorted({_c(n) for n in names if _c(n)})
        if len(want) < min_hit:
            return None
        raw = self._raw or {}
        hits: List[tuple] = []
        sigs: Dict[str, tuple] = {}
        for k, v in raw.items():
            m: Dict[str, str] = {}
            dup: set = set()
            for p in (v.get("pins") or []):
                nm, num = _c((p or {}).get("name")), str((p or {}).get("number") or "").strip()
                if not nm or not num:
                    continue
                if nm in m and m[nm] != num:
                    dup.add(nm)
                m[nm] = num
            for d in dup:
                m.pop(d, None)
            h = sum(1 for n in want if n in m)
            if h >= min_hit and h * 2 >= len(want):
                hits.append((h, k))
                sigs[k] = tuple((n, m[n]) for n in want if n in m)
        if not hits:
            return None
        hits.sort(reverse=True)
        best = [k for h, k in hits if h == hits[0][0]]
        # 打平时**看结论是否一致**：USB 家族的 `USB_A`/`TYPE-B`/`U217-041N-4BV81`/`AF 90° WJDG`
        # 都把 `VCC/D-/D+/GND` 映射成 `1/2/3/4`，而 `CH340N`/`CH340C`（USB-串口芯片，
        # 名字碰巧一样）给的是别的号。取**多数签名**、且多数要占打平者的一半以上：
        # 这样「大家其实说的是同一件事」时仍能出手，真的各说各话时就放弃。
        if len(best) == 1:
            return best[0]
        cnt: Dict[tuple, int] = {}
        first: Dict[tuple, str] = {}
        for k in best:
            s = sigs[k]
            cnt[s] = cnt.get(s, 0) + 1
            first.setdefault(s, k)
        top = max(cnt.values())
        if top * 2 <= len(best):
            return None
        s = [sig for sig, c in cnt.items() if c == top][0]
        return first[s]

    def num_for_name_raw(self, key: str, pin_name: str) -> Optional[str]:
        """`key_for_names` 配套的查表：**只去空格大写**，保留 `D-`/`D+` 的区别。

        `num_for_name()` 走 `norm()`，会把 `D-`/`D+` 归成同一个键 `d`，
        对连接器这类「名字只差一个符号」的脚是致命的（实测 `VCC/D-/D+/GND`
        在 `AF 90° WJDG` 里 4 个名字只归一成 3 个）。
        """
        if not key or not pin_name:
            return None
        if not self.load():
            return None
        m: Dict[str, str] = {}
        dup: set = set()
        for p in ((self._raw or {}).get(key) or {}).get("pins") or []:
            nm = "".join(str((p or {}).get("name") or "").split()).upper()
            num = str((p or {}).get("number") or "").strip()
            if not nm or not num:
                continue
            if nm in m and m[nm] != num:
                dup.add(nm)
            m[nm] = num
        for d in dup:
            m.pop(d, None)
        return m.get("".join(str(pin_name).split()).upper())

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

    def key_by_nums(self, nums: Sequence) -> Optional[str]:
        """按**已读脚号集合**反查库键：只有唯一一个库键把这些号全包含时才返回。

        动机：`key_for(partname)` 只认名字，partname 被 OCR 读坏（case11 `TYPE-C 6P(073)`
        读成 `TYPE-C-SPIO73`、`AF 90° WJDG` 读成 `AF 90`）就整颗查不到库，H1 的
        「号→名」确认随之失效。而脚号本身常常已经读出来了（`A5/A12/B5/B9/B12/7`），
        用它们反查是**不依赖名字**的第二条路。要求 `len(nums) >= 3` 由调用方把关。
        """
        if not nums or not self.load():
            return None
        want = set()
        for n in nums:
            t = str(n or "").strip()
            if t:
                want.add(t.upper())
        if not want:
            return None
        hit = None
        cands = []
        for key, body in (self._raw or {}).items():
            have = set()
            for p in (body or {}).get("pins") or []:
                t = str((p or {}).get("number") or "").strip().upper()
                if t:
                    have.add(t)
            if want <= have:
                cands.append((len(have), key))
        if not cands:
            return None
        cands.sort()
        # 最小超集若并列（同一个连接器的两条别名，如 `TYPE-C 6P(073)` / `tpyec母头`），
        # 只要它们在这些号上的**号→名映射完全一致**就等价，取第一个；否则不给答案。
        if len(want) < 4:
            return None                     # 号太少时最小超集必然乱中（实测 2 个号就能撞上）
        best = cands[0][0]
        tied = [k for n, k in cands if n == best]
        if len(tied) == 1:
            return tied[0]
        # 最小超集并列（同一连接器的别名条目，如 `TYPE-C 6P(073)` / `tpyec母头`）：
        # 用「号→名唯一解」逐号表决，`None`（该键里这个号多名歧义）当通配；有冲突就放弃。
        agg = {}
        for key in tied:
            for num in want:
                v = self.name_for_num(key, num)
                if v is None:
                    continue
                cur = agg.get(num)
                if cur is None:
                    agg[num] = v
                elif norm(cur) != norm(v):
                    return None
        if not agg:
            return None
        score = {}
        for key in tied:
            score[key] = sum(1 for num in want if self.name_for_num(key, num) is not None)
        return max(tied, key=lambda k: (score[k], -tied.index(k)))

    def nums_all(self, key: str) -> List[str]:
        """该库键下**全部引脚号**（保序、去重）。"""
        if not key or not self.load():
            return []
        out = []
        for p in ((self._raw or {}).get(key) or {}).get("pins") or []:
            num = str((p or {}).get("number") or "").strip()
            if num and num not in out:
                out.append(num)
        return out

    def names_of(self, key: str) -> Dict[str, int]:
        """库键下**全部**引脚名的 `norm()` 集合（规则 B 判「这个名字库里有没有」）。"""
        if not key:
            return {}
        if not self.load():
            return {}
        return (self._index or {}).get("names", {}).get(key) or {}

    def is_known_name(self, name) -> bool:
        """`norm(name)` 是不是**库里任何键**用过的引脚名（全局词表 2401 条）。

        用途：`STAGE2_LIB_REFINE_DEGEN=known` 档的判据 —— 只有当我方读数**是库认识的
        名字**（官方 5 U3 的 `VIN`/`TAB`、2 U3 的 `SDA`）时才拦「库的纯数字名覆盖它」；
        `VWPSTO2`/`IOW2T` 这种库完全不认识的乱码照旧让库来纠正 —— 派生 19 张里正是
        这一类**必须**被覆盖（`0011 U2.pin_3` GT=`3`、`0151 U2.pin_2` GT=`2`），
        所以 `letter` 档（不看词表）在派生集上是 −0.0031、`known` 档不掉。
        """
        if not self.load():
            return False
        return norm(name) in ((self._index or {}).get("all_names") or ())


def _build_index(raw: dict) -> dict:
    """把库扫一遍，建 4 张表（337 键 / ~1.2 万条引脚记录，实测 <0.3 s）。

    * `by_norm`      —— `norm(键) → [键…]`（归一化/模糊匹配用）
    * `num_by_name`  —— `键 → {norm(名) → 号}`，**只保留唯一映射**；
                       名字在库里对应 2 个及以上不同 `number` 的**整个丢掉**（规则 A）
    * `name_by_num`  —— `键 → {norm(号) → 名}`，同样只保留唯一映射（规则 B）；
                       改名前要落回**库里原始写法**（不是 `norm` 后的串）
    * `names`        —— `键 → {norm(名) → 出现次数}`（规则 B 判「当前名在不在库里」）
    * `all_names`    —— **全局**引脚名词表（`norm(名)` 的集合，2401 条）。
                       给 `STAGE2_LIB_REFINE_DEGEN=known` 档用：判「我方这个读数是不是
                       库认识的名字」，见 `LibCtx.is_known_name`。
    """
    by_norm: Dict[str, List[str]] = {}
    num_by_name: Dict[str, Dict[str, str]] = {}
    name_by_num: Dict[str, Dict[str, str]] = {}
    names: Dict[str, Dict[str, int]] = {}
    all_names: set = set()
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
                all_names.add(nn)
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
            "name_by_num": name_by_num, "names": names, "all_names": all_names}


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
    if not LIB_PINS or not LIB_PINS_NUMFILL or not stubs:
        return 0
    ctx = ctx or LibCtx()
    n = 0
    key = ctx.key_for(comp_name)
    _by_names = False
    if LIB_NAMEID not in ("", "0", "off", "false", "no"):
        # 型号名认不出来（`=1`）或**认得出来但明显不对**（`=2`：该库键覆盖不到一半的读到
        # 名字，而按名字反查的键能覆盖 ≥3 个）时，用引脚名集合反查库键。
        # `USB2` 就是 `=2` 的情形：`key_for('SG90')` 命中了库里的伺服 `SG90`（3 脚），
        # 而我们的 4 个名字一个都不在里面 ⇒ 规则 A 一个号都补不出来。
        _names = [str(r.get("nm") or "") for r in stubs]
        _hit = 0
        if key:
            _hit = sum(1 for _n in _names if _n and ctx.num_for_name(key, _n))
        _want = len([_n for _n in _names if _n])
        if not key or (LIB_NAMEID >= "2" and _want and _hit * 2 < _want):
            _k2 = ctx.key_for_names(_names)
            if _k2 and _k2 != key:
                _h2 = sum(1 for _n in _names if _n and ctx.num_for_name_raw(_k2, _n))
                if _h2 > _hit:
                    key, _by_names = _k2, True
                    ctx.stats["nameid"] = ctx.stats.get("nameid", 0) + 1
    if not key:
        return 0
    used = used if used is not None else set()
    gate = numgate_mode()
    # ★ 注意：档位 `"0"` 是非空**字符串**（真值）⇒ 不能写 `if gate:`，否则「关」档也会跑冲突规则。
    # 这个坑是本节自检抓出来的（官方 10 张上冲突规则从不触发，所以 A/B 看不出来）。
    gated = gate in ("conflict", "seen")
    # 图上自己给出的 (号 → 名) 配对，只取**真读**（非位置补号、且通过检查）：
    seen: set = set()
    read_pair: Dict[str, str] = {}
    if gated:
        for r in stubs:
            s = str(r.get("num") or "").strip()
            if not s or r.get("fill") or not r.get("kept", True):
                continue
            seen.add(norm(s))
            nmr = str(r.get("nm") or "").strip()
            if nmr:
                read_pair.setdefault(norm(s), norm(nmr))
    for r in stubs:
        # 「号没读出来」：空的，或者是位置补位号（`fill`）；**读出来且通过检查的号不动**
        if str(r.get("num") or "").strip() and not r.get("fill") and r.get("kept", True):
            continue
        nm = str(r.get("nm") or "").strip()
        if not nm:
            continue                       # 名也没读到 ⇒ 无从查表
        num = (ctx.num_for_name_raw(key, nm) if _by_names
               else ctx.num_for_name(key, nm))
        if not num or num in used:
            continue
        if gated:
            nu = norm(num)
            if nu in read_pair and read_pair[nu] != norm(nm):
                # 图上给这个号读到了**别的名字** ⇒ 图上与库冲突，信图（§5.133 的 −0.16 就是这一类）
                ctx.stats["numgate_conflict"] = ctx.stats.get("numgate_conflict", 0) + 1
                continue
            if gate == "seen" and seen and nu not in seen:
                ctx.stats["numgate_notseen"] = ctx.stats.get("numgate_notseen", 0) + 1
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
    if not LIB_PINS or not LIB_PINS_NAMEFILL:
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
        # `degen_block` 的探针用（`known2` 要判「我方读数是库认识的名字」，
        # 以及「库给的是编号式名字」那一类）。
        "CC1": {"pins": [{"name": "VCC", "number": "1"},
                         {"name": "P67", "number": "2"},
                         {"name": "VIN", "number": "3"},
                         {"name": "GN", "number": "4"},
                         {"name": "DGND", "number": "5"},
                         {"name": "SFL", "number": "6"},
                         {"name": "VDDIO", "number": "7"}]},
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
        # ★ `known` 档用到的全局引脚名词表 + 档位读取
        for nm, want in (("vdd", True), ("pa00", True), ("nrst", True),
                         ("vw psto2", False), ("iow2t", False), ("", False)):
            if ctx.is_known_name(nm) is not want:
                bad.append("is_known_name(%r) 应为 %s" % (nm, want))
        for mode, want in (("off", "off"), ("letter", "letter"),
                           ("shape", "shape"), ("known", "known"),
                           ("known2", "known2"), ("KNOWN2", "known2"),
                           ("KNOWN", "known"), ("乱写", "known2"), ("", "known2")):
            os.environ["STAGE2_LIB_REFINE_DEGEN"] = mode
            if degen_mode() != want:
                bad.append("degen_mode(%r) 应为 %r，得到 %r"
                           % (mode, want, degen_mode()))
        os.environ.pop("STAGE2_LIB_REFINE_DEGEN", None)
        if degen_mode() != "known2":
            bad.append("degen_mode 默认应为 known2")
        if has_letter("4") or has_letter("") or not has_letter("VIN"):
            bad.append("has_letter 判据")
        if n_letters("VCC") != 3 or n_letters("4") != 0 or n_letters("") != 0:
            bad.append("n_letters 判据")
        if not has_digit("P67") or has_digit("VIN") or has_digit(""):
            bad.append("has_digit 判据")
        # `degen_block`：两档 ×（乱码/单字母/多字母已知/纯数字）×（库名有无字母）
        for mode, cur, ref, want in (
                ("off", "VCC", "P67", False),
                ("letter", "VCC", "P67", False),          # 库名有字母 ⇒ letter 不管
                ("known", "VCC", "P67", False),           # 库名有字母 ⇒ known 也不管
                ("known2", "VCC", "P67", True),           # ★ 本档新增的那一类（库名编号式）
                ("known2", "VCC", "VCC_PAD", False),      # ★ 库给功能名 ⇒ 不挡（派生 4 根的坑）
                ("known2", "GN", "GND_PAD", False),
                ("known2", "DGND", "SFL", False),
                ("known2", "VDDIO", "NC", False),
                ("known2", "P67", "L2", False),           # 我方名字本身是编号式 ⇒ 让库修
                ("known2", "VCC", "4", True),
                ("known2", "VIN", "3", True),
                ("known2", "P", "PP", False),             # 单字母乱码 ⇒ 让库修
                ("known2", "VWPSTO2", "3", False),        # 库不认识的乱码 ⇒ 让库修
                ("known2", "IOW2T", "2", False),
                ("known2", "3", "VWPSTO2", False),        # 我方纯数字 ⇒ 让库修
                ("known", "VWPSTO2", "3", False),
                ("known", "VIN", "3", True),
                ("letter", "VIN", "3", True)):
            if degen_block(ctx, mode, cur, ref) is not want:
                bad.append("degen_block(%r,%r,%r) 应为 %s" % (mode, cur, ref, want))
        # ★ 库补**号**闸门 `numgate_mode`：档位读取（默认关、大小写/空白容错、乱写回落 0）
        for mode, want in ((None, "0"), ("", "0"), ("off", "off"), ("0", "0"),
                           ("conflict", "conflict"), ("SEEN", "seen"),
                           (" seen ", "seen"), ("乱写", "0")):
            if mode is None:
                os.environ.pop("STAGE2_LIB_PINS_NUMGATE", None)
            else:
                os.environ["STAGE2_LIB_PINS_NUMGATE"] = mode
            if numgate_mode() != want:
                bad.append("numgate_mode(%r) 应为 %r，得到 %r" % (mode, want, numgate_mode()))
        # ★ 闸门行为（官方 10 张上它一次都不触发 ⇒ 只能靠这里保证新 GT 上真能用）：
        #   图上给同一个号读到了**别的名字** ⇒ conflict/seen 都不补，默认档照补（旧行为）。
        def _gate(mode, stubs):
            if mode is None:
                os.environ.pop("STAGE2_LIB_PINS_NUMGATE", None)
            else:
                os.environ["STAGE2_LIB_PINS_NUMGATE"] = mode
            c2 = LibCtx(f)
            return complete_pins("AA1", stubs, c2, used=set()), stubs, c2

        n, s, _c = _gate(None, [{"num": "", "nm": "VDD"},
                                {"num": "1", "nm": "PA00", "kept": True}])
        if n != 1 or s[0]["num"] != "1":
            bad.append("numgate 默认关：与图冲突时也要照补（旧行为）")
        for mode in ("conflict", "seen"):
            n, s, c2 = _gate(mode, [{"num": "", "nm": "VDD"},
                                    {"num": "1", "nm": "PA00", "kept": True}])
            if n != 0 or s[0]["num"] != "" or c2.stats.get("numgate_conflict") != 1:
                bad.append("numgate=%s：图上该号读到别的名字时不该补" % mode)
        # 图上只给了一个「没名字的号」：conflict 管不着，seen 必须挡住（库号在图上没见过）
        n, s, c2 = _gate("seen", [{"num": "", "nm": "VDD"},
                                  {"num": "5", "nm": "", "kept": True}])
        if n != 0 or c2.stats.get("numgate_notseen") != 1:
            bad.append("numgate=seen：库号在图上没见过时不该补")
        n, s, _c = _gate("conflict", [{"num": "", "nm": "VDD"},
                                      {"num": "5", "nm": "", "kept": True}])
        if n != 1 or s[0]["num"] != "1":
            bad.append("numgate=conflict：只差 seen 的那一类不该被挡")
        n, s, _c = _gate("seen", [{"num": "", "nm": "VDD"},
                                  {"num": "1", "nm": "", "kept": True}])
        if n != 1 or s[0]["num"] != "1":
            bad.append("numgate=seen：库号确实在图上读到过就该放行")
        # 「真读」的判据：位置补位号（fill）与没通过检查（kept=False）的号都不算证据。
        # 注意这两个桩**自身**也是「号没读出来」⇒ 两桩都会被补（这正是位置号规则的本意），
        # 关键判据是 **不记 numgate_conflict**。
        for extra in ({"num": "1", "nm": "PA00", "kept": False, "fill": True},
                      {"num": "1", "nm": "PA00", "kept": False}):
            n, s, c2 = _gate("conflict", [{"num": "", "nm": "VDD"}, dict(extra)])
            if n != 2 or s[0]["num"] != "1" or "numgate_conflict" in c2.stats:
                bad.append("numgate：fill/kept=False 的号不算真读，不该被当成冲突（应两桩都补）")
        os.environ.pop("STAGE2_LIB_PINS_NUMGATE", None)
        if numgate_mode() != "0":
            bad.append("numgate_mode 收尾应回到默认 0")
        # ★ 两个细闸确实挂在对应的对外接口上
        globals()["LIB_PINS_NUMFILL"] = False
        if complete_pins("AA1", [{"num": "", "nm": "VDD"}], LibCtx(f), used=set()) != 0:
            bad.append("LIB_PINS_NUMFILL=0 时规则A不该动手")
        globals()["LIB_PINS_NUMFILL"] = True
        globals()["LIB_PINS_NAMEFILL"] = False
        if names_for("AA1", "1", LibCtx(f)) is not None:
            bad.append("LIB_PINS_NAMEFILL=0 时规则B不该动手")
        globals()["LIB_PINS_NAMEFILL"] = True
    print("libpins 自检：%s" % ("全部通过" if not bad else "失败 " + " / ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
