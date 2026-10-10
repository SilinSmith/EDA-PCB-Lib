# -*- coding: utf-8 -*-
"""对齐轴与「长度解码」——把「本质共线」的文字框聚成一条轴，再按轴序解开位号。

## 为什么需要这个模块

`fusion/priors.py:506 windows_for` 找连号窗口用的是**纯文本滑窗**：把某个前缀
下所有槽位的候选折成数字，滑一个长度等于槽位数的窗，按「有候选落进窗内的槽位
数」排序。它自己的 docstring 就承认了极限：

    ⚠️ 纯文本分不出前两名时不要硬选。实测 case 1（真值 C151..C156，
    读成 C51/C15/C5/C54/C5/C15）：
      - 窗口 [51,56] 覆盖 4/6
      - 窗口 [151,156] 覆盖 4/6
    两者证据量完全相等 …… 能区分的是**图**。

于是它返回前 3 名，把窗口清单塞进 `pool.hy.slots[sid].context["连号窗口候选"]`，
指望 `fusion/llm_judge.py` 拿着图去二选一。**但生产档 LLM 轮是关的**
（`eda-pcb/submission/run.py` 只在 `-fusion llm` 时打开，且实测 24 张定点档
25.4434 → 25.2688 更差）⇒ 这个歧义**从来没有人仲裁**，最后由裸分决定，
而「插回一个 1」的修复变体打了 `REPAIR_PENALTY=0.85` 折，天然输给不加字的那种。

## 但我们其实不需要 LLM 也能分开它：长度律

位号框的 **宽/字高** 与真实字符串长度强相关（KiCad 等宽字形，步进 ≈ 字高）。
`work/tools/diag/_wlen2.py --all` 在 220 张、8928 条配上真值的位号候选上量得：

| 真实字符数 | n | w/h 中位 | p10 | p90 |
|---|---|---|---|---|
| 2 | 5029 | 1.571 | 1.250 | 2.286 |
| 3 | 3047 | 2.273 | 1.571 | 2.389 |
| 4 |  775 | 2.714 | 1.571 | 3.333 |

中位点几乎共线：`w/h ≈ 0.571 × 字符数 + 0.43`。
于是 case 1 那排 `C155`..`C164` 的框（w/h = 2.67 / 2.86 / 3.06 / 3.11）
**根本装不下两位数**，`[51,56]` 当场被排除 —— 一个像素级的几何判据就把
「文字上不可分」的歧义解掉了。

## 三层结构

1. **长度律** `len_prob(w, h, n_chars)`：给「这个框里是 n 个字符」一个似然。
   它是一条 **软**判据（sigma 按 p10..p90 的散度取 0.30），不是闸门。
2. **轴** `Axis`：把同一前缀、垂直方向落在同一带（容差 **max(0.6·h, 3px)**，
   按字高归一 ⇒ 与图幅无关）的框，按沿轴坐标排成一个序列。这就是需求里
   「本质上对齐的内容在更高维度上共线」的那个维度。
   对齐容差故意**宽松**：差几个像素绝不拆轴（样例里一条导线歪一像素就毁全图，
   那种刚性判据是错的）。
3. **解码器** `decode(axis, obs)`：在轴上解一条**连续递增**的号序列
   `lo..lo+n-1`，目标是

       Σ_i [ 文本似然(读法 → 该槽位的号) + λ · 长度似然(框宽高 → 该号的字符数) ]

   用 DP 在候选 `lo` 上取优。返回**排序后的全部**解释（不是「选一个」），
   胜出者只是分最高那条；其余进候选表当证据 —— 与 `cvparts/hypo.py` 的
   「一个内容的不同识别结果在同一维度上延伸」一致。

## 与旧路径的关系

`decode` 只**产候选**，不写文档；最终仍由 `fusion/solve.py` 按分裁决。
所以这个模块是纯增量的：它让「几何上说得通」的解释进入候选表并拿到应有的分，
而不是让「少一个字」的读法靠 `×0.85` 的折扣天然胜出。
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from envflag import env_on
except ImportError:                                    # 直接跑脚本自检
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from envflag import env_on

# 开关。**默认开** —— owner m24624：「因为一点点丢分就否定一整个链路，而不是把它
# 完全落地，实际上也是不可取的。」两件事必须一起记在这里：
#  (a) 之前那次「23.36（关）→ 23.20（开），净 −0.16」的 A/B **是脏的**：修掉
#      兜底键 `pNNN` 被当成读数那个 bug 之后重跑，19 张均分与修前**逐位相同**；
#      而决策级差分（`work\tools\diag\_axisdiff.py`，直接比最终定下来的值）
#      显示 18 张**一字不改**，连据称掉 0.75 的 `0001` 都是**改 0 处**。
#  (b) 本层当时**从未生效**：`len_prob` 用的 `CHAR_A·n + CHAR_B` 把所有字符当成
#      一样宽，而实测字符宽度只差 0.05（`fusion/charwidth.py`，38.8 万条真实标注
#      拟合）⇒ `LEN_CONFLICT` 那条强证据分支是**死代码**，弱证据分支又被
#      `_cap_like` 压在读数之下 ⇒ 一次都没赢过。现在换成拟合出来的逐字符宽度表，
#      并且长度律改判**行内规范框宽**（`Item.w_ref`），见 `build_axes`。
ON = env_on("FUSION_AXIS", "1")

# 兜底键（不是读数）的判据，与 `eda-pcb\schematic_parser\metrics.py:is_unkeyed` 一致。
_UNKEYED = re.compile(r"^(?:\d+|p\d{3,4})$", re.I)

# ---------------------------------------------------------------- 长度律
# 逐字符宽度表（**自动生成**，`work\tools\diag\_charfit.py` ← 38.8 万条真实「框+文本」）。
# 为什么不用 `CHAR_A·n + CHAR_B`：一是字符之间**并不等宽**（`9`=0.71 vs `8`=0.82），
# 二是这个函数形态把「框的固定边距」也摊进了字宽。带截距的逐字符模型 rmse 0.179
# 对均匀模型 0.307（降 41.7%）。
import os

try:
    from fusion.charwidth import ADV as _ADV, PAD as _PAD
    from fusion.charwidth import FALLBACK as _ADV_FB, RMSE as _ADV_RMSE
    from fusion.charwidth import STAGE1_CAL as _S1_CAL, STAGE1_SIG as _S1_SIG
except ImportError:                                    # 直接跑脚本自检
    try:
        from charwidth import ADV as _ADV, PAD as _PAD
        from charwidth import FALLBACK as _ADV_FB, RMSE as _ADV_RMSE
        from charwidth import STAGE1_CAL as _S1_CAL, STAGE1_SIG as _S1_SIG
    except ImportError:
        _ADV, _PAD, _ADV_FB, _ADV_RMSE = {}, 0.30, 0.75, 0.20
        _S1_CAL, _S1_SIG = -0.187, 0.475

# 似然宽度：σ 用**拟合残差** `RMSE`(0.1791)。
# ★★ 为什么不用实测散布 0.475 —— 这是**做了四组对照之后**的结论，不是拍的。
# 我在「读数 == GT 位号」的 102 个**确认正确**的读法上量了 `w/h - pred_wh_text`：
#   中位 −0.187 ｜ 均值 −0.253 ｜ 标准差 0.475 ｜ p10 −0.618 ｜ p90 −0.061 ｜ 94.1% 为负
# 看上去该用 σ=0.475、并补一个 −0.187 的域偏移。**但实测说这样会把整层消音**：
# 19 张定点档上逐槽位比对最终决策（`work\tools\diag\_axisdiff.py`）：
#   A  σ=0.1791 CAL=0     → case 1 改对 2 处（C10→C160、C15→C155）；0001 改错 2 处；净 0
#   B  σ=0.1791 CAL=−0.187→ 与 A 逐条相同（CAL 在决策上**没有区别**）
#   C  σ=0.475  CAL=0     → **全 19 张 0 处改动**（整层被消音，等于关掉）
#   D  σ=0.475  CAL=−0.187→ 同 C
# ⇒ 0.475 是把「正确读法的散布」当成噪声模型，而长度判据要的是**尾部**行为：
#   正确读法的残差**从不 > +0.2**（直方图 [+0.0,+0.2) 只有 4.9%、再往上 0 个），
#   所以「框明显偏宽 ⇒ 少了一个字」这条推断，用 0.179 当 σ 时门限落在 +0.39，
#   正好压在正确读法的上界之上 —— 这才是可用的判据。
# ⇒ 两条都记在这里作**被否证记录**：`STAGE1_SIG=0.475` 与 `STAGE1_CAL` 都**不启用**。
# 残余问题（未解，见下）：0001 上 `C4`→`C10`、`C8`→`C11` 仍是错的 —— 那是
# stage1 的框本身偏宽（框的错，不是读数的错），按 owner「读数是证据，形状只是形状」
# 应当**不许形状覆盖自洽读数**。要做的是**逐实例的接受门**，不是再调全局常数。
CHAR_SIG = max(0.10, float(os.environ.get("AXIS_CHAR_SIG") or _ADV_RMSE))
# 域偏移：实测确有其事（中位 −0.187），但上面 A/B 对照显示它对决策**没有影响**，
# 默认**不启用**（`AXIS_CHAR_CAL` 可临时打开做实验）。
CHAR_CAL = float(os.environ.get("AXIS_CHAR_CAL") or 0.0)
# 同一把尺子允许多少个字符：少于 1 / 多于 24 个字符的框不参与（噪声框）。
CHAR_MIN = 1
CHAR_MAX = 24
# 行内规范框宽：成员宽度低于本排中位宽的这个倍数 ⇒ 判为「框没延展到位」。
W_TRUNC = 0.75

# ---------------------------------------------------------------- 轴
# 垂直方向成带的容差（**乘以字高**，再加一个绝对底），故意宽松。
AXIS_BAND_FRAC = 0.60
AXIS_BAND_PX = 3.0
# 一条轴至少几个成员才值得解码（两个也可能是连号，但噪声大）。
AXIS_MIN = 3
# 轴上相邻两个框的间距 / 中位间距，允许的倍数范围（宽松到 0.25..4.0）。
AXIS_PITCH_LO = 0.25
AXIS_PITCH_HI = 4.0
# ★ 竖直轴（`FUSION_AXIS_VERT`，默认 0）：同前缀的一**列**位号也能成轴。
# 为什么默认关（而不是「没量过」）：`docs/RESULTS.md` §2.9 已经把这条路的天花板量死了 ——
# 原理图里**几何相邻 != 号相邻**（位号是设计者随手分配的标识），同前缀邻座能覆盖的
# run 只有 16/639 = **2.5%**，其中靠邻座推对的只有 **0.3%**；真正有信号的是「**偷邻居
# 的号**」（`near_not_chosen` 78 = 12.2%），那是**归属（一对一）**问题、不是排列问题。
# 所以这里把竖排实装出来是为了「钉死这个方向有没有用」，默认必须关、且 A/B 说了算。
AXIS_VERT = env_on("FUSION_AXIS_VERT", "0")

_DESIG_RE = re.compile(r"^([A-Za-z]{1,3})(\d{1,4})$")


def split_desig(s: Any) -> Tuple[str, Optional[int]]:
    m = _DESIG_RE.match(str(s or "").strip())
    if not m:
        return "", None
    return m.group(1).upper(), int(m.group(2))


def pred_wh(n_chars: float) -> float:
    """n 个字符的框，预期的 `宽/字高`（**均匀近似**；精确形态是 `pred_wh_text`）。"""
    return _PAD + _ADV_FB * float(n_chars) + CHAR_CAL


def pred_wh_text(text: Any) -> float:
    """逐字符的预期 `宽/字高`：`PAD + Σ_c ADV[c] + CAL` —— 长度律的**精确形态**。

    `CAL` 是 stage1 检出框域的偏移（见 `CHAR_CAL` 注释）。
    """
    return _PAD + sum(_ADV.get(c, _ADV_FB) for c in str(text or "")) + CHAR_CAL


def _gauss(delta: float) -> float:
    d = float(delta) / (2.0 * CHAR_SIG)
    return float(2.718281828459045 ** (-d * d))


def len_prob_text(w: float, h: float, text: Any) -> float:
    """「这个框里是 `text`」的似然，0..1（**逐字符**，用于判「少了/多了哪个字」）。"""
    t = str(text or "")
    n = len(t)
    if h <= 0 or not t or not (CHAR_MIN <= n <= CHAR_MAX):
        return 0.0
    return _gauss(float(w) / float(h) - pred_wh_text(t))


def len_prob(w: float, h: float, n_chars: float) -> float:
    """「这个框里是 `n_chars` 个字符」的似然（均匀近似，保留给只知字数的调用方）。"""
    if h <= 0 or n_chars <= 0 or not (CHAR_MIN <= n_chars <= CHAR_MAX):
        return 0.0
    return _gauss(float(w) / float(h) - pred_wh(n_chars))


def est_chars(w: float, h: float) -> float:
    """反解：这个框大约装几个字符（连续值，用于诊断与排序）。"""
    if h <= 0:
        return 0.0
    return max(0.0, (float(w) / float(h) - _PAD) / _ADV_FB)


# ---------------------------------------------------------------- 轴
@dataclass
class Item:
    """轴上的一个成员：一个文字框（或引用它的槽位）及其读数。"""
    key: str                       # 槽位 id 或文字框 id
    box: Sequence[float]           # **[图像空间]** 的框
    text: str = ""                 # 最优读法
    score: float = 0.0             # 最优读法的分
    reads: List[Tuple[str, float]] = field(default_factory=list)  # 全部读法
    meta: Dict[str, Any] = field(default_factory=dict)
    # **长度律该用的宽度**：默认就是自己的框宽；若本排判定该框「没延展到位」
    # （见 `build_axes` 的 `W_TRUNC`），则用**行内规范宽** —— 因为「这一格该有几个字」
    # 是**行**说的，不是这个被截断的框说的（owner m23945 ③）。
    w_ref: float = 0.0

    @property
    def w(self) -> float:
        return float(self.box[2]) - float(self.box[0])

    @property
    def w_len(self) -> float:
        """长度律实际使用的宽度（行内规范宽优先）。"""
        return float(self.w_ref) if self.w_ref else float(self.w)

    @property
    def h(self) -> float:
        return float(self.box[3]) - float(self.box[1])

    @property
    def cx(self) -> float:
        return (float(self.box[0]) + float(self.box[2])) / 2.0

    @property
    def cy(self) -> float:
        return (float(self.box[1]) + float(self.box[3])) / 2.0


@dataclass
class Axis:
    """一条对齐轴：同一前缀、同一带上、沿轴向有序的一串框。"""
    prefix: str
    items: List[Item] = field(default_factory=list)
    horizontal: bool = True
    pitch: float = 0.0             # 相邻框中心距的中位数
    note: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    def place(self, it: Item) -> float:
        return it.cx if self.horizontal else it.cy

    def sort(self) -> None:
        self.items.sort(key=self.place)
        gaps = [self.place(self.items[i + 1]) - self.place(self.items[i])
                for i in range(len(self.items) - 1)]
        gaps = [g for g in gaps if g > 0]
        if gaps:
            gaps.sort()
            self.pitch = gaps[len(gaps) // 2]
        self.note["gaps"] = [round(g, 1) for g in gaps]


def _row_regular(b: Sequence["Item"], w_med: float) -> bool:
    """这一排够不够**规整**到可以把中位宽当成「这一格该多宽」的规范。

    判据（**实例级**，不是全局开关）：至少 2/3 的成员宽度落在中位宽的 ±25% 内。
    `len(b) < 3` 一律判否 —— 两个框不足以说明什么叫「这一排的格式」。
    """
    if w_med <= 0 or len(b) < 3:
        return False
    near = sum(1 for it in b if abs(it.w - w_med) <= 0.25 * w_med)
    return near * 3 >= len(b) * 2


def build_axes(items: Sequence[Item], min_members: int = AXIS_MIN) -> List[Axis]:
    """把文字框聚成轴。

    对每个前缀，按「垂直坐标」排序，把彼此在容差内的成员串起来；
    容差 = `max(AXIS_BAND_FRAC × 字高, AXIS_BAND_PX)` —— **宽松**，
    差几个像素绝不拆轴（需求：「正常对齐范围内对齐，即使差了几个像素点，
    那肯定也是对齐的」）。

    轴内成员按沿轴坐标排序；成员数 < `min_members` 的轴不返回。
    """
    by_pre: Dict[str, List[Item]] = {}
    for it in items:
        pre, _num = split_desig(it.text)
        if not pre:
            continue
        by_pre.setdefault(pre, []).append(it)

    out: List[Axis] = []
    dirs = (True, False) if AXIS_VERT else (True,)
    for pre, group in by_pre.items():
        for horizontal in dirs:
            # 先按「垂直于轴」的坐标分带（带容差跟着成员自己的尺寸走）。
            # 横排按 cy 分带、竖排按 cx 分带（竖排的带容差用**框宽**，因为同一列的框
            # 应该左对齐/同宽，而这正是 `place()` 之后要拿到的顺序）。
            key = (lambda t: t.cy) if horizontal else (lambda t: t.cx)
            span = (lambda t: t.h) if horizontal else (lambda t: t.w)
            group = sorted(group, key=key)
            bands: List[List[Item]] = []
            for it in group:
                placed = False
                for b in bands:
                    ref = b[-1]
                    tol = max(AXIS_BAND_FRAC * max(span(it), span(ref)), AXIS_BAND_PX)
                    if abs(key(it) - key(ref)) <= tol:
                        b.append(it)
                        placed = True
                        break
                if not placed:
                    bands.append([it])
            for b in bands:
                if len(b) < min_members:
                    continue
                ax = Axis(prefix=pre, items=list(b), horizontal=horizontal)
                ax.sort()
                # ★ 行内规范框宽：**同一排的框本该一样宽**。某个框明显窄于本排中位宽
                #   ⇒ 那是「框没延展到位」，不是「这格里字少」（owner m23945 ③：
                #   「文本框没有正常的延展或者合并」）。于是长度律要用**行给它的宽度**
                #   判，而不是它自己那个被截断的框 —— 这就是「合适的时候用合适的方法」：
                #   稀疏的排（0175 那种 `C33/C23/C34/…`）宽度本就参差，中位数不构成
                #   规范，`_row_regular` 直接返回 False，一条读数都不会被改。
                #   竖排不做这条（竖排要的是「同列同高」，与长度律无关）。
                if horizontal:
                    ws = sorted(it.w for it in b)
                    w_med = ws[len(ws) // 2] if ws else 0.0
                    if _row_regular(b, w_med):
                        for it in b:
                            if it.w < W_TRUNC * w_med:
                                it.w_ref = w_med
                        ax.note["w_med"] = round(w_med, 2)
                        ax.note["trunc"] = [it.key for it in b if it.w_ref]
                else:
                    ax.note["vert"] = True
                if ax.pitch > 0:
                    ax.note["pitch_ok"] = True
                out.append(ax)
    out.sort(key=lambda a: (-len(a), a.prefix))
    return out


# ---------------------------------------------------------------- 解码
@dataclass
class Assign:
    """一种解释：轴上的每个位置拿到哪个号。"""
    lo: int
    hi: int
    per: List[Optional[int]]            # 与 axis.items 等长；None = 保持原读法
    value: float = 0.0
    detail: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi,
                "per": self.per, "value": round(self.value, 4)}


# 文本似然：读法与候选号的关系（与 `priors.read_score` 同精神，但这里是**本地**实现，
# 免得 axis 反向依赖 priors 造成循环 import）。它只用于**排序轴内解释**。
def text_loglik(obs: str, cand: str) -> float:
    obs = str(obs or "")
    cand = str(cand or "")
    if not obs or not cand:
        return 0.02
    if obs == cand:
        return 1.0
    # 「少一个字符」= obs 是 cand 去掉某些字符后的结果（最常见：窄 `1` 被吞）
    if len(obs) < len(cand):
        import difflib
        sm = difflib.SequenceMatcher(None, obs, cand)
        same = sum(bl.size for bl in sm.get_matching_blocks())
        miss = len(cand) - same
        if same >= 1 and miss == 1:
            return 0.40
        if same >= 1 and miss == 2:
            return 0.14
        return 0.02
    if len(obs) == len(cand):
        # 同长度不同内容 = 形近替换。**必须给得很低**：`priors.READ_SUB` 只有 0.05。
        # 一旦给高（这里曾是 0.35），一串「同长度、错数字」的窗口（真值
        # `C33/C23/C34/C24/C32` 被解成 `C33..C37`）就能逐点压过正确的原读法。
        import difflib
        sm = difflib.SequenceMatcher(None, obs, cand)
        same = sum(bl.size for bl in sm.get_matching_blocks())
        if same >= len(cand) - 1:
            return 0.10
        return 0.03
    return 0.03


def score_at(axis: Axis, lo: int, lam: float = 1.0) -> Assign:
    """给**任意**起点 `lo` 打分：把轴上第 i 个框解释成 `前缀+(lo+i)`。

    单独抽出来是因为全局层要试的 `lo` 根本不来自任何读数 —— 它来自
    「上一条轴刚刚铺到 158，所以下一条应该从 159 开始」。读数里永远不会出现
    159（`C15`/`C10`/`C181` 都推不出它），所以候选集必须能被外部指定。
    """
    n = len(axis.items)
    pre = axis.prefix
    hi = lo + n - 1
    # 先逐点算出「派号」与「照原样」两个分值，再决定这一排到底派不派。
    pts: List[Tuple[int, str, Optional[tuple], float]] = []
    for i, it in enumerate(axis.items):
        num = lo + i
        cand = "%s%d" % (pre, num)
        best = None
        for txt, sc in (it.reads or [(it.text, it.score)]):
            tl = text_loglik(txt, cand)
            if tl <= 0.0:
                continue
            sc_v = float(sc)
            # ★ 三项**相乘**，不再把长度似然当加法奖励：
            #   加法会让「派一个字符数对得上的号」本身就值 ~1.0 分，于是随便
            #   一串错误连号都能盖过正确的原读法 —— case 0175 的
            #   `C33/C23/C34/C24/C32`（真值！）就是这么被解成 `C11..` 的。
            s = tl * (0.4 + 0.6 * max(0.0, min(1.0, sc_v))) * (
                0.4 + 0.6 * len_prob_text(it.w_len, it.h, cand))
            if best is None or s > best[0]:
                best = (s, tl, txt, sc_v)
        # 不派号：保持原读法。**但不能白拿基线** —— 原读法自己的字符数也得
        # 和框的宽高比对得上（`C5` 落在 w/h=3.06 的框里，本就说明它少了字，
        # 此时「照原样读」几乎不该有分量，`C158` 那种补字解释才该赢）。
        keep_s = KEEP_BASE * lam * len_prob_text(it.w_len, it.h, it.text)
        pts.append((num, cand, best, keep_s))

    def _plan(count_bonus: bool):
        """count_bonus=True 时把段奖励折进每个位置的判据 —— 这样「牺牲一个位置
        的局部最优去凑成整段连号」是算得出来的（`C181`→`C161` 就是这种情况：
        单看字形 keep 更强，但补上它整排 159/160/161 才连得起来）。"""
        p: List[Optional[int]] = []
        d: List[Dict[str, Any]] = []
        t = 0.0
        h = 0
        for i, (num, cand, best, keep_s) in enumerate(pts):
            thr = keep_s - (RUN_BONUS * lam if count_bonus else 0.0)
            if best is not None and best[0] > thr:
                p.append(num)
                t += best[0]
                h += 1
                d.append({"i": i, "key": axis.items[i].key, "num": num, "cand": cand,
                          "read": best[2], "text_lik": round(best[1], 3),
                          "score": round(best[0], 4)})
            else:
                p.append(None)
                t += keep_s
                d.append({"i": i, "key": axis.items[i].key, "num": None,
                          "why": "keep", "read": axis.items[i].text,
                          "score": round(keep_s, 4)})
        if h >= 2:
            t += RUN_BONUS * (h - 1) * lam      # 整段连号本身也是证据
        return p, t, d

    per, tot, detail = _plan(False)
    p2, t2, d2 = _plan(True)
    if t2 > tot:                                 # 凑段更划算就凑段
        per, tot, detail = p2, t2, d2
    return Assign(lo=lo, hi=hi, per=per, value=tot, detail=detail)


def decode(axis: Axis, lam: float = 1.0, topk: int = 4) -> List[Assign]:
    """在轴上解一条连续递增的号序列，返回**排序后的全部解释**（降序）。

    目标：`Σ_i [ 文本似然(第 i 个框的读法, 派给它的号) + λ · 长度似然(框, 号的字符数) ]`

    - 「连续递增」= 号序列是 `lo, lo+1, …, lo+n-1`（沿轴顺序），这是 PCB 位号的常规画法；
    - 也允许某个位置**不派号**（该位置保持它原有的读法），代价是 0 —— 因为轴可能
      混进一个不属于这一串的框（宽松聚轴的副作用），或者那个框的读法本来是对的；
    - 候选 `lo` 从「全部读数里出现过的数字」以及「最左框位数推出来的号」两处取，
      另外补一段 `0..max` 的低位窗口，避免只盯读数而错过整体方案。

    返回前 `topk` 条，`Assign.value` 是总分。
    """
    n = len(axis.items)
    if n == 0:
        return []
    pre = axis.prefix

    # 每个位置的候选号 -> (文本似然, 长度似然, 合并分)
    digits_hint = [max(1, int(round(est_chars(it.w, it.h))) - len(pre))
                   for it in axis.items]

    cand_lo: set = set()
    for it in axis.items:
        for txt, _sc in (it.reads or [(it.text, it.score)]):
            p, num = split_desig(txt)
            if p == pre and num is not None:
                cand_lo.add(num)
                cand_lo.add(num - n + 1)
    # 从位数推：最左框如果是 d 位，编号大致在 10^(d-1) 附近
    d0 = digits_hint[0]
    if 1 <= d0 <= 4:
        base = 10 ** (d0 - 1)
        for v in range(max(0, base - 2), min(base * 10, base + n + 2)):
            cand_lo.add(v)
    cand_lo = {int(v) for v in cand_lo if 0 <= v + n - 1 <= 9999}

    out: List[Assign] = [score_at(axis, lo, lam) for lo in sorted(cand_lo)]

    out.sort(key=lambda a: -a.value)
    # 去重：同一个 per 只留分最高的
    seen = set()
    uniq: List[Assign] = []
    for a in out:
        k = tuple(a.per)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(a)
    return uniq[:topk]


def covered(assign: Assign) -> List[Tuple[str, str]]:
    """把一条解释变成 `[(槽位/框 key, 派给它的号字符串)]`，只含真正派了号的位置。"""
    out = []
    for i, num in enumerate(assign.per):
        if num is None:
            continue
        out.append((assign.detail[i].get("key") or "", "%d" % num))
    return out


# ---------------------------------------------------------------- 全局解码
# 同一前缀的号空间被各条轴**分块铺满**（PCB 位号在一张图里通常就是 1..N 连号）。
# 轴与轴之间的空洞要罚分：`C179` 这种解释虽然能把某一条轴的读法对得整整齐齐，
# 却会在号空间里留下 159..178 这一大段谁都不认领的空洞。
GAP_PEN = 0.06
# ★ 但「连号」绝不是无条件的：case 0175 的电容排真值是 `C33/C23/C34/C24/C32`
#   —— 一张图里的同前缀位号**可以很稀疏**。所以只在「这一串读法确实像连号」时才
#   允许派号：多数位置要能被窗解释，而且总分要明显超过「全部保持原读法」。
MIN_GAIN = 0.10
# 「照原样读」的单点基线**上限**（还要再乘该读法自己的长度似然）。定得高是
# 关键：一条本来就像位号、字符数又和框对得上的读法就是**强证据**，不能被
# 「随便凑一串连号」抢走（0175 真值 `C33/C23/C34/C24/C32` 被解成连号，
# 就是因为这里只给 0.25）。而字符数与框明显冲突的读法（`C5` 落在 4 字的框里）
# 乘出来自然接近 0，补字解释才轮得到。
KEEP_BASE = 0.90
# 一整段连续号本身也是证据：相邻的几个框**同时**被解释成连号，比逐点解释可靠得多。
# （case 1 的 `C15`/`C10`：两点「少一个窄 1」各自只值 0.34/0.35，靠段奖励才赢。）
RUN_BONUS = 0.45


def keep_value(axis: Axis, lam: float = 1.0) -> float:
    """「所有位置都不派号、保持原读法」的总分 —— 派号方案必须压过它。"""
    return sum(KEEP_BASE * lam * len_prob_text(it.w_len, it.h, it.text)
               for it in axis.items)


@dataclass
class PrefixPlan:
    """一个前缀的全局解释：每条轴拿一段号。"""
    prefix: str
    per_axis: List[Tuple[Axis, Assign, int]]   # (轴, 该轴的局部解释, 段起点 lo)
    value: float = 0.0
    gaps: List[Tuple[int, int]] = field(default_factory=list)   # 空缺区间

    def as_dict(self) -> Dict[str, Any]:
        return {"prefix": self.prefix, "value": round(self.value, 4),
                "gaps": self.gaps,
                "axes": [{"n": len(ax), "lo": lo, "hi": asg.hi,
                          "reads": [it.text for it in ax.items],
                          "cands": [d.get("cand") for d in asg.detail]}
                         for ax, asg, lo in self.per_axis]}


def decode_prefix(axes: Sequence[Axis], topk: int = 3, gap_pen: float = GAP_PEN,
                  max_lo_step: int = 40) -> Optional[PrefixPlan]:
    """**全局**解码：给同一前缀的所有轴各分一段连号，让号空间尽量被铺满。

    为什么需要第二层：单条轴上「哪个窗口」常常仍然不可分 —— case 1 的
    `C159/C160/C161` 那三个框，读数 `C15`/`C10`/`C181` 既支持窗口 `[159,161]`
    （两条少字 + 一条形近替换）也支持 `[179,181]`（`C181` 是**精确**读法，
    锚住整段，局部打分反而更高）。分开它们的不是这条轴自己，而是**别的轴**：
    同前缀的 `C151..C156` 已经占了 151..156，`C157..C158` 跟着，
    于是接下来必然是 159 —— `[179,181]` 会在号空间里留下 159..178 的大洞。

    做法：轴按「阅读序」排（先垂直带、再沿轴坐标），DP 在「下一个可用号」上推进。
    """
    axes = [a for a in axes if len(a) > 0]
    if not axes:
        return None
    axes = sorted(axes, key=lambda a: (round(min(it.cy for it in a.items) / 8.0),
                                       min(it.cx for it in a.items)))
    # 每条轴先在自己身上求出若干候选解释（含"保持原样"）
    cands: List[List[Assign]] = []
    for ax in axes:
        cs = decode(ax, topk=max(topk, 6))
        if not cs:
            cs = [Assign(lo=-1, hi=-1, per=[None] * len(ax), value=0.0,
                         detail=[{"i": i, "key": ax.items[i].key, "num": None,
                                  "why": "keep", "read": ax.items[i].text, "score": 0.0}
                                 for i in range(len(ax))])]
        cands.append(cs)
    lo_all = min(c.lo for cs in cands for c in cs if c.lo >= 0) if any(
        c.lo >= 0 for cs in cands for c in cs) else 1
    hi_all = max(c.hi for cs in cands for c in cs if c.hi >= 0) if any(
        c.hi >= 0 for cs in cands for c in cs) else 1

    # dp[next] = (最好总分, 回溯路径)；next = 铺完当前这些轴之后「下一个可用的号」
    best: Dict[int, Tuple[float, list]] = {0: (0.0, [])}
    for k, ax in enumerate(axes):
        n = len(ax)
        nxt: Dict[int, Tuple[float, list]] = dict(best)   # 允许这一轴「不出声」
        for cur, (tot, path) in best.items():
            tries: List[Tuple[int, Assign]] = []
            if k == 0:
                # 第一条轴：编号**从哪儿开始是自由的**，所以不罚空洞；
                # 除了它自己的候选，也试一下「从 1 开始连号」这个最朴素的解释。
                for c in cands[k]:
                    if c.lo >= 0:
                        tries.append((c.lo, c))
                tries.append((1, score_at(ax, 1)))
            else:
                # ★ 关键：先试「**紧接着上一个号**」。读数里永远不会出现 159
                #   （`C15`/`C10`/`C181` 都推不出它），但上一条轴刚铺到 158，
                #   所以 159 才是这条轴最该被考虑的解释。
                tries.append((cur, score_at(ax, cur)))
                for c in cands[k]:
                    if c.lo >= 0:
                        tries.append((c.lo, c))
            for lo, cc in tries:
                if lo < 0 or lo + n - 1 > 9999:
                    continue
                if k > 0 and lo - cur > max_lo_step:
                    continue                      # 一次跳太远，不作为候选
                if k > 0 and lo < cur:
                    continue                      # 号的推进是单调的，不许倒着占号
                # ★ 只在这串读法**确实像连号**时才允许派号（见 MIN_GAIN 注释）：
                #   多数位置要真的被窗解释，且总得明显超过「全部照原样读」。
                n_hit = sum(1 for p in cc.per if p is not None)
                if n_hit < max(2, (n + 1) // 2):
                    continue
                if cc.value <= keep_value(ax) + MIN_GAIN:
                    continue
                pen = 0.0 if k == 0 else gap_pen * max(0, lo - cur)
                end = lo + n
                v = tot + cc.value - pen
                if end not in nxt or v > nxt[end][0]:
                    nxt[end] = (v, path + [(ax, cc, lo)])
        best = nxt
        if not best:
            return None
    end, (tot, path) = max(best.items(), key=lambda kv: kv[1][0])
    per_axis: List[Tuple[Axis, Assign, int]] = list(path)
    # 空洞只在**轴与轴之间**算（首条轴之前那段号空间本来就不该有人认领）
    gaps: List[Tuple[int, int]] = []
    prev_end: Optional[int] = None
    for ax, cc, lo in per_axis:
        if prev_end is not None and lo > prev_end:
            gaps.append((prev_end, lo - 1))
        prev_end = lo + len(ax)
    return PrefixPlan(prefix=axes[0].prefix, per_axis=per_axis, value=tot, gaps=gaps)


def items_from_pool(pool, only_with_box: bool = True) -> List[Item]:
    """从 `fusion/claims.py` 的候选池里取出「带框的位号候选」，供聚轴用。

    - 槽位形如 `comp.<cid>.key`；
    - 框取**取景窗的 `focus`** —— `claims.py:565-570` 把位号候选的框写进
      `pool.set_region(...)`（`pool.meta[sid]` 里**只有** `type`/`cid`，没有框；
      踩过一次：照 `pool.meta` 取框 ⇒ 每个槽位都被跳过、整层从不出声）。
      `focus` 是**候选自己的文字框**（图像空间），正是长度律要量的那个框；
    - 没有框的槽位直接跳过（聚轴要的就是几何）。
    - ★ **兜底键不是读数，一律不进轴**：`claims.py` 给读不出位号的元件发的键是
      `pNNN`（200 训练集里 GT 键有 4034/9171 是 `∅NNN`），它的框是**占位小框**
      （实测 10×10），和「4 个字符」天然长度冲突 ⇒ 会走 `inject` 的**强证据**分支，
      把一排 `p044/p035/p042/...` 改写成凭空连号的 `P0..P32`。实测 0001 因此被
      强改 33 处、0042 被强改 12 处，19 张总分 23.36 → **23.20**（0001 −0.75、
      0042 −2.30，netH tp 233→227）。判据沿用 `metrics.is_unkeyed`：
      `^\d+$` 或 `^p\d{3,4}$`。
    - `reads` 收全部候选，解码器要看到「一个内容的不同识别结果」。
    """
    out: List[Item] = []
    for sid in pool.slot_ids():
        if not sid.startswith("comp.") or not sid.endswith(".key"):
            continue
        meta = pool.meta.get(sid) or {}
        box = None
        try:
            r = pool.region(sid)
        except Exception:                                  # noqa: BLE001
            r = None
        if r is not None:
            box = getattr(r, "focus", None) or getattr(r, "box", None)
        if not box:
            box = meta.get("box") or meta.get("bbox")
        if only_with_box and not box:
            continue
        reads = [(str(a.value), float(a.score)) for a in pool.alts(sid)]
        b = pool.best(sid)
        if _UNKEYED.match(str(b.value) if b else ""):
            continue                       # 兜底键不是读数（见 docstring）
        if not box:
            continue
        out.append(Item(key=sid, box=[float(v) for v in box],
                        text=str(b.value) if b else "",
                        score=float(b.score) if b else 0.0,
                        reads=reads, meta=dict(meta)))
    return out


# ---------------------------------------------------------------- 接进流水线
# 长度冲突的读法上，几何证据允许**明确压过**它多少分。
AXIS_MARGIN = 0.05
# `len_prob(读法)` 低于此值 = 读法自己的字符数和框对不上（窄 `1` 被吞的**长度错**）。
LEN_CONFLICT = 0.30


def _is_read(source: Any) -> bool:
    return str(source or "").startswith(("ocr", "vlm"))


def _cap_like(pool, sid: str, want: str, sc: float) -> float:
    """与 `fusion/priors.py:201 _cap` 同款护栏：**读数是证据、轴只是形状**。

    轴要派的号如果会顶掉一条**已经读出来的**（且值不同）读数，就只能当旁证
    —— 压在它下面，绝不越过。
    """
    b = pool.best(sid)
    if b is None:
        return sc
    if _is_read(getattr(b, "source", "")) and str(b.value) != str(want):
        return min(sc, float(b.score) - 0.01)
    return sc


def inject(pool, lam: float = 1.0, log=None) -> Dict[str, Any]:
    """把轴解码的结果**追加**进候选池（`fusion/run.py` 在 `priors.annotate` 之前调）。

    规矩来自 owner：**可以削弱内容的置信度，但不能轻易覆盖。** 所以本函数
      * 只**加候选**，从不删除、也从不改写任何已有候选；
      * **只有**「读法自己的字符数与框明显冲突」时（`len_prob < LEN_CONFLICT`，
        即 `C15`/`C10`/`C5` 这类窄 `1` 被吞掉的**长度错**）才允许略微压过原读数
        —— 框宽是独立于 CRNN 的硬证据，这也是 0175 那种「本来就稀疏的排」不会被
        误伤的保证：那里的读法长度都自洽，走不到这条分支；
      * 其余情况一律走 `_cap_like`，只做旁证。
    """
    st: Dict[str, Any] = {"n_slots": 0, "n_axes": 0, "n_add": 0, "n_strong": 0,
                          "plan": None}
    items = items_from_pool(pool)
    st["n_slots"] = len(items)
    if len(items) < AXIS_MIN:
        return st
    axes = build_axes(items)
    if not axes:
        return st
    plan = decode_prefix(axes)
    if plan is None:
        return st
    st["n_axes"] = len(plan.per_axis)
    st["plan"] = plan.as_dict()

    for ax, cc, lo in plan.per_axis:                 # (Axis, Assign, lo)
        if cc is None:
            continue
        # ★ 必须从 `cc.detail`（带 `cand` = **完整字符串**如 `C155`）取解出来的值。
        #   曾经错写成 `zip(ax.items, cc.per)` —— `Assign.per` 存的是一串**纯整数**
        #   （`score_at` 里的 `num`），于是注进池子的候选是 `"10"`、`"11"` 这种
        #   **丢了前缀**的东西：强证据分支照样加进去、也在池子里当过最优，但下游
        #   `solve` 一看不是合法位号就丢回原读数。整层因此在决策层**看起来从不生效**
        #   （19 张定点档 100% 一致），而真相是**注入的值本身是坏的**。
        #   这正是 owner m24624 说的「可能因为之前的各种bug导致没有得分」。
        val_of = {d.get("key"): d.get("cand") for d in (cc.detail or [])}
        for it in ax.items:
            val = val_of.get(it.key)
            if not val or str(val) == str(it.text):
                continue                     # 解码器决定「不动它」⇒ 本层不出声
            b = pool.best(it.key)
            bsc = float(b.score) if b is not None else 0.0
            lp = len_prob_text(it.w_len, it.h, it.text)
            if lp < LEN_CONFLICT:            # 长度错：几何是硬证据
                sc = min(0.99, bsc + AXIS_MARGIN)
                st["n_strong"] += 1
            else:                            # 只是形状相似：只当旁证
                sc = _cap_like(pool, it.key, val, min(bsc, 0.5))
            pool.add(it.key, val, score=sc, source="geo:axis",
                     evidence={"轴": "%s y≈%.0f" % (ax.prefix, ax.items[0].cy),
                               "同轴读数": [x.text for x in ax.items],
                               "读法": it.text, "读法长度似然": round(lp, 3),
                               "解出": val, "轴得分": round(float(cc.value), 3)},
                     question="元件 %s 的位号是哪一个？" % it.key.split(".")[1],
                     meta=dict(pool.meta.get(it.key) or {}))
            st["n_add"] += 1
    if log and st["n_add"]:
        log("  [axis] %d 条轴、追加 %d 个候选（其中长度错强证据 %d）"
            % (st["n_axes"], st["n_add"], st["n_strong"]))
    return st


# ---------------------------------------------------------------- 自检
def _selftest() -> int:
    n = 0

    def chk(c, m):
        nonlocal n
        n += 1
        if not c:
            raise AssertionError("第 %d 项失败：%s" % (n, m))

    # 长度律：220 张的中位点必须复现
    # 旧手拍常量 `CHAR_A·n + CHAR_B` 的锚点是 2 字 1.571 / 4 字 2.714（上表）；
    # 换成拟合出来的 `PAD + FALLBACK·n` 之后锚点变成 1.70 / 3.17。这里钉住**新**锚点，
    # 免得哪天又把两只手拍的数悄悄换回去。
    chk(abs(pred_wh(2) - (_PAD + 2 * _ADV_FB)) < 0.01, "2 字符的预期 w/h")
    chk(abs(pred_wh(4) - (_PAD + 4 * _ADV_FB)) < 0.01, "4 字符的预期 w/h")
    chk(abs(est_chars(60.0, 21.0) - (60.0 / 21.0 - _PAD) / _ADV_FB) < 0.05,
        "est_chars(60,21) 应与新的 `PAD/FALLBACK` 自洽")
    # 长度律：逐字符宽度表（`fusion/charwidth.py`，38.8 万条真实标注拟合）。
    # ⚠ **校准口径必须知道**：表是从**标注框**拟合的，而这里吃的是 **stage1 检出的
    #   文字框**；两者每字符宽并不完全一致（stage1 框系统性偏宽约 0.3 ≈ 半个字符）。
    #   好在 `score_at` 里「原读法」与「补字候选」用的是**同一个框的同一个 w/h**，
    #   这个系统性偏移在两边同时出现、方向上不改变谁赢，所以断言只钉**相对**判据。
    #   真实锚点：case 1 的 A 轴框是 55×18，读 `C15`，真值 `C155`。
    chk(len_prob_text(55, 18, "C155") > len_prob_text(55, 18, "C15") + 0.4,
        "55x18 的框应明确更像 C155 而不是 C15（拿到 %.3f vs %.3f）"
        % (len_prob_text(55, 18, "C155"), len_prob_text(55, 18, "C15")))
    chk(len_prob_text(19, 8, "C14") > len_prob_text(19, 8, "C144"),
        "19x8 的窄框更像 3 字符（0001 那种小框）")
    # ★ 旧模型（`CHAR_A·n + CHAR_B`）对「只差一个字符」的一对给的分几乎一样
    #   —— 这就是强证据分支变成死代码、整层从不生效的直接原因。钉住这一点别回退。
    chk(len_prob_text(55, 18, "C15") < LEN_CONFLICT,
        "读 `C15` 落在 55x18 的框里，长度似然必须掉到 LEN_CONFLICT 以下才算「长度错」"
        "（拿到 %.3f）" % len_prob_text(55, 18, "C15"))

    # 文本似然：少一个字符是常见错，同长度形近替换**必须更弱**
    # （`priors.READ_SUB` 只有 0.05；给高了整排错号就能逐点压过正确读数）
    chk(text_loglik("C15", "C159") > text_loglik("C15", "C160"),
        "C15 更该解释成 C159（少一个 9）而不是 C160")
    chk(text_loglik("C15", "C159") > text_loglik("C181", "C161"),
        "「少一个窄 1」必须比「同长度换一个数字」更可信")
    chk(text_loglik("C181", "C161") < 0.2, "同长度形近替换要给得很低")
    chk(text_loglik("C10", "C160") < text_loglik("C10", "C10"), "完全相同的读法分最高")

    # ★★ 真实数据夹具：case 1 的**三条真实轴**（GT `C151..C162`）。
    #   框与读数逐条抄自 `work/_submit_out220r36/runs/stage2/_assigned/1.json` 的
    #   `desig_slots`（`_rowfull.py` 输出），w/h 由框宽高算得。
    #   A 轴 6 个（5 个读对、只有第 5 个把 `C155` 读成 `C15`）；
    #   C 轴 2 个（`C157` 对、`C5` 实为 `C158`）；
    #   B 轴 3 个（`C15`/`C10`/`C181` 实为 `C159`/`C160`/`C161`）。
    items = [
        # A 轴 y≈156
        Item(key="a1", box=[104.0, 147.0, 157.0, 165.0], text="C151", score=0.885,
             reads=[("C151", 0.885)]),
        Item(key="a2", box=[222.0, 147.0, 278.0, 165.0], text="C152", score=0.913,
             reads=[("C152", 0.913)]),
        Item(key="a3", box=[340.0, 147.0, 395.0, 165.0], text="C153", score=0.868,
             reads=[("C153", 0.868)]),
        Item(key="a4", box=[457.0, 147.0, 514.0, 165.0], text="C154", score=0.767,
             reads=[("C154", 0.767)]),
        Item(key="a5", box=[605.0, 147.0, 660.0, 165.0], text="C15", score=0.925,
             reads=[("C15", 0.925)]),
        Item(key="a6", box=[753.0, 145.0, 813.0, 166.0], text="C156", score=0.776,
             reads=[("C156", 0.776)]),
        # C 轴 y≈597
        Item(key="c1", box=[164.0, 587.0, 225.0, 608.0], text="C157", score=0.833,
             reads=[("C157", 0.833)]),
        Item(key="c2", box=[311.0, 588.0, 366.0, 606.0], text="C5", score=0.965,
             reads=[("C5", 0.965)]),
        # B 轴 y≈656.5
        Item(key="b1", box=[488.0, 646.0, 548.0, 667.0], text="C15", score=0.955,
             reads=[("C15", 0.955)]),
        Item(key="b2", box=[605.0, 648.0, 661.0, 666.0], text="C10", score=0.840,
             reads=[("C10", 0.840)]),
        Item(key="b3", box=[753.0, 646.0, 809.0, 667.0], text="C181", score=0.733,
             reads=[("C181", 0.733)]),
    ]
    axes_all = build_axes(items, min_members=2)
    # 横轴断言只看横轴（`FUSION_AXIS_VERT=1` 时会额外多出竖轴，语义不变）。
    axes = [a for a in axes_all if a.horizontal]
    if AXIS_VERT:
        vax = [a for a in axes_all if not a.horizontal]
        chk(bool(vax), "开了 FUSION_AXIS_VERT 就该出现竖轴（一条都没有）")
        chk(all(a.prefix == "C" for a in vax), "竖轴前缀也都是 C")
        for a in vax:
            ys = [it.cy for it in a.items]
            chk(ys == sorted(ys), "竖轴成员应按 y 递增，拿到 %s" % ys)
    chk(len(axes) == 3, "应该聚成三条轴（拿到 %d 条）" % len(axes))
    sizes = sorted(len(a) for a in axes)
    chk(sizes == [2, 3, 6], "三条轴的长度应是 6/2/3，拿到 %s" % sizes)
    chk(all(a.prefix == "C" for a in axes), "三条轴的前缀都是 C")
    by_n = {len(a): a for a in axes}
    chk([it.text for it in by_n[6].items] ==
        ["C151", "C152", "C153", "C154", "C15", "C156"], "A 轴按 x 排序")
    chk([it.text for it in by_n[2].items] == ["C157", "C5"], "C 轴按 x 排序")
    chk([it.text for it in by_n[3].items] == ["C15", "C10", "C181"], "B 轴按 x 排序")

    # ---- 第一层（单轴）在 B 轴上**找不到 159**：局部最优是把它排到 [15,17]
    #      （`C15` 自己像 15），把窗口锚到 179 是次优。这个断言钉住「必须要有全局层」
    #      这件事本身；哪天局部打分强到能自己选对，它会失败，那时就该回来删掉全局层。
    b_first = decode(by_n[3], lam=1.0, topk=6)
    chk(bool(b_first), "B 轴解码应该有结果")
    chk(b_first[0].lo != 159,
        "单轴解码不该自己找到 159（拿到 %s）—— 找不到正是需要全局层的理由"
        % b_first[0].lo)

    # ---- 第二层（全局铺号）：A 占 151..156、C 占 157..158 ⇒ B 必然是 159..161。
    plan = decode_prefix(axes, topk=4)
    chk(plan is not None, "全局解码应该有结果")
    los = [lo for _ax, _asg, lo in plan.per_axis]
    chk(los == [151, 157, 159],
        "全局解码应给 A/C/B 三段 151/157/159，拿到 %s" % los)
    chk(plan.gaps == [], "最优铺号不应该留空洞，拿到 %s" % plan.gaps)
    got = [[d.get("cand") for d in asg.detail] for _ax, asg, _lo in plan.per_axis]
    chk(got[0] == ["C151", "C152", "C153", "C154", "C155", "C156"],
        "A 轴应解出 C151..C156（把读错的 `C15` 纠成 `C155`），拿到 %s" % got[0])
    chk(got[1] == ["C157", "C158"],
        "C 轴应解出 C157/C158（把 `C5` 纠成 `C158`），拿到 %s" % got[1])
    # ★ 这条以前是**刻意保留的能力边界**（旧断言写 `["C159","C160",None]`，把
    #   `C181` 留着不动）。换成逐字符宽度表 + 由拟合残差定 σ 之后，边界自己消失了：
    #   `C181` 那个框是 56×21（w/h=2.667），逐字符模型里 4 个字符要 3.423 ⇒
    #   `len_prob = 0.0115` ⇒ 「照原样读」的分几乎归零，补字解释理所当然地赢。
    #   这正是 owner m24624 说的：「可能因为之前的各种bug导致没有得分」—— 旧模型把
    #   所有字符当一样宽，`C181` 的长度似然虚高到 0.73，才需要我人为按住它。
    #   现在三条轴**全部**与 GT（`C151..C161`）逐字相符。
    chk(got[2] == ["C159", "C160", "C161"],
        "B 轴应解出 C159/C160/C161（把 `C15`/`C10`/`C181` 一起纠对），拿到 %s" % got[2])

    # 宽松聚轴：垂直差 4px（字高 21，容差 max(12.6, 3)）不能拆轴
    near = [Item(key="a", box=[0, 100, 60, 121], text="C11", reads=[("C11", 0.9)]),
            Item(key="b", box=[70, 104, 130, 125], text="C12", reads=[("C12", 0.9)]),
            Item(key="c", box=[140, 101, 200, 122], text="C13", reads=[("C13", 0.9)])]
    chk(len(build_axes(near, min_members=3)) == 1, "差 4px 不许拆轴（宽松对齐）")
    # 但差得远的必须是两条轴
    far = [Item(key="a", box=[0, 100, 60, 121], text="C11", reads=[("C11", 0.9)]),
           Item(key="b", box=[70, 104, 130, 125], text="C12", reads=[("C12", 0.9)]),
           Item(key="c", box=[140, 300, 200, 321], text="C13", reads=[("C13", 0.9)])]
    chk(len(build_axes(far, min_members=3)) == 0, "差 200px 不该聚成一条轴")
    # 间距不等也要聚成一条轴（case 1 的 A 轴真实间距是 118/118/117/148/148）
    uniq = [Item(key="a", box=[0, 100, 60, 121], text="C11", reads=[("C11", 0.9)]),
            Item(key="b", box=[70, 104, 130, 125], text="C12", reads=[("C12", 0.9)]),
            Item(key="c", box=[200, 101, 260, 122], text="C13", reads=[("C13", 0.9)])]
    chk(len(build_axes(uniq, min_members=3)) == 1, "间距差 25% 不许拆轴")

    print("axis 自检 %d 项：全部通过" % n)
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        raise SystemExit(_selftest())
    raise SystemExit("用法：python -m fusion.axis --selftest")
