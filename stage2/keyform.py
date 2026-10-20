# -*- coding: utf-8 -*-
"""键形成层（`STAGE2_KEY_CHANNEL`）：位号（键）只由「专职通道 + 稳定几何」形成。

为什么要有这一层（证据链：`docs/RESULTS.md` §5.289 / §5.291 / §5.306.3 / §5.307–§5.310）：

* 键是**主键**。键错一个字符，这颗元件的引脚、网络、值全部跟着记错项。§5.291 的
  oracle 分解：没键 +2.96、键错 +4.01、全对 +7.01（50 评测案，`--no-vlm`）。
* 识别侧已封盘：九臂（换权重、禁借读、推理裁片训练）在 50 评测案上全负，最好的
  `nb0`（禁借读）仍 −0.11（§5.310）。亏损不在「读得出吗」，而在**键怎么被形成**。
* 今天的做法把三件事混在一个标量里（`assign()` 的 `_score`）：谁的框（距离项）、
  读得对不对（fit × conf）、允不允许借读（`_BORROW_PREF`）。于是出现
  `U6 → J2`（借读了别人的脚名）这种「读得清楚、归属错了」的键；它比「读不清、
  退回 `pNNN`」更伤——具名键拿不到官方坐标兜底（`metrics.py:347` 只对无名元件
  开放），键一错连引脚一起判死。
* 反过来，「只信自己名下的框」也不够：`owned` 是**最近元件 + 半径**划的，大 IC 的
  位号常印在体外 40px 开外，于是被划成「自由文字」（`assign.py:1350` 那段注释就是
  这个坑），一味收紧会把本来对的键丢掉（本次 v1 实测：键命中 689 → 628）。

所以本层不继承 `owned/steal` 的任何划分，**自己**从几何重建「这块印字是谁的位号」：

1. **专职通道**：只有「位号形态 + 类型前缀互证」（`canon_designator_meaning` 的
   `strict` 路径）的读法能进键通道。值、脚名、网络名、型号的读法一概不进——
   那是别的角色的活（禁借读的口径，§5.306.3 的 `nb0` 已经证明读数本身不是瓶颈）。
2. **稳定几何**：印字与元件体的关联只看页面局部关系——位号印在体**上方**（或紧贴
   体侧），与体在另一个轴上重叠，且框到框距离很小。这三项写进 `geom()`，是 0–1 的
   连续分，与读置信度相乘。不用全图统计量，所以不随图密度漂移。
3. **一对一**：全局贪心（分数降序 → cid → 框身份），一个元件最多一个键、一块框最多
   归一颗元件、一个位号串最多用一次。撞框/撞号的那一票**直接作废**（元件退回
   `pNNN`），绝不找替代——替代就是「借」，正是本层要禁的东西。

`pick()` 是纯函数（不读文件、不写文件、不碰全局环境以外的状态），输入输出都用
`assign()` 已经在用的字典形状，便于逐位对拍与 A/B。
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

LAST: dict = {}

# 几何参数（px；图坐标是 1000 级别，见 stage1 的 bbox_img 口径）
_ABOVE_LO = -4.0        # 文字底边比体顶边高多少算「贴着上方」（负 = 略压进体里）
_ABOVE_HI = 70.0        # 再高就当它不像这颗的位号（可能是上一行的字）
_ABOVE_SLACK = 40.0     # 超出 _ABOVE_HI 后每超这么多 px，几何分衰减一档
_KEY_R = 45.0           # 框到框的硬上限（超出这个距离，任何几何分都归零）


def _env_f(name: str, dft: float) -> float:
    try:
        v = os.environ.get(name)
        return float(v) if v not in (None, "") else dft
    except Exception:
        return dft


def _ov(a, b, ax: int) -> float:
    """一维重叠占**窄者**长度的比例（0–1）。`ax=0` 横向、`ax=1` 纵向。"""
    lo = max(float(a[ax]), float(b[ax]))
    hi = min(float(a[ax + 2]), float(b[ax + 2]))
    if hi <= lo:
        return 0.0
    w = min(abs(float(a[ax + 2]) - float(a[ax])), abs(float(b[ax + 2]) - float(b[ax])))
    return max(0.0, min(1.0, (hi - lo) / (w or 1.0)))


def _r2r(a, b) -> float:
    """框到框的切比雪夫间距（重叠时为 0）。"""
    dx = max(0.0, max(float(a[0]), float(b[0])) - min(float(a[2]), float(b[2])))
    dy = max(0.0, max(float(a[1]), float(b[1])) - min(float(a[3]), float(b[3])))
    return max(dx, dy)


def geom(text_box, body) -> float:
    """这块印字「像不像这颗元件体的位号」，0–1。

    位号在官方数据上的印刷位置很固定（`assign.py` 头部引用的 `_text_geom.py`：
    位号 p50 距离 6px、几乎都在体外）：体上方、横向与体重叠、框到框很近。
    """
    if not text_box or not body or len(text_box) != 4 or len(body) != 4:
        return 0.0
    r = _r2r(text_box, body)
    if r > _KEY_R:
        return 0.0
    above = float(body[1]) - float(text_box[3])      # >0 表示文字在体上方
    if above < _ABOVE_LO:
        base = 0.55                                   # 压进体里：仍可能，但降权
    elif above <= _ABOVE_HI:
        base = 1.0 - 0.25 * (above / _ABOVE_HI)       # 越贴着体越好
    else:
        base = 0.75 / (1.0 + (above - _ABOVE_HI) / _ABOVE_SLACK)
    cross = max(_ov(text_box, body, 0), _ov(text_box, body, 1))
    near = 1.0 - min(1.0, r / _KEY_R)                 # 距离项（线性）
    return max(0.0, min(1.0, base * (0.30 + 0.40 * cross + 0.30 * near)))


def _bid(rec: dict):
    """文字框的物理身份：`box_i`（跨元件唯一的框序号）优先，缺了退回四舍五入的框。"""
    try:
        bi = int(rec.get("_i", -1))
    except Exception:
        bi = -1
    if bi >= 0:
        return ("box", bi)
    b = rec.get("bbox") or rec.get("box")
    if b and len(b) == 4:
        return ("rect", tuple(round(float(v), 1) for v in b))
    return None


def _char_width(texts, canon_fn, comps, gmin, min_conf) -> float:
    """本页位号「每个字符占多少 px」的中位数（用来识破**掉字的读数**）。

    识别器偶尔把 `C13` 读成 `C1`、`C157` 读成 `C19`，置信度还很高；但框的宽度没变，
    所以「读数宽度 / 字符数」会明显偏大。取全页窄框读数的中位字符宽做基准。
    """
    if not comps:
        return 0.0
    ctype = comps[0].get("type") or "r"
    vals = []
    for rec in texts or []:
        b = rec.get("bbox") or rec.get("box")
        if not b or len(b) != 4:
            continue
        w = abs(float(b[2]) - float(b[0]))
        h = max(1.0, abs(float(b[3]) - float(b[1])))
        if w / h > 6.0 or w < 6.0:
            continue
        for v in (rec.get("reads") or {}).values():
            if not v or not v[0] or float(v[1] or 0.0) < min_conf:
                continue
            can = canon_fn(str(v[0]), ctype)[0]
            if can and 2 <= len(can) <= 5:
                vals.append(w / len(can))
    if len(vals) < 20:
        return 0.0
    vals.sort()
    return vals[len(vals) // 2]


def pick(texts: Sequence[dict], comps: Sequence[dict], canon_fn,
         gmin: float = None, min_conf: float = None,
         rescue: bool = None) -> List[Tuple[str, dict]]:
    """键形成。返回 `[(cid, info), ...]`（按 cid 升序）。

    * `texts`：`stage2.text.TextEngine.read` 的记录（要有 `bbox` 与 `reads`）。
    * `comps`：`[{"id","type","bbox"}]`，调用方负责剔掉 `NO_DESIGNATOR` 类型。
    * `canon_fn`：`canon_designator_meaning(text, ctype, rescue=...)`
      → `(canon, fid, type_fix, how)`；只有 `how == "strict"` 的读法进键通道
      （`rescue=True` 时额外放行类型翻修路径，默认关）。
    * info 字段与旧路径的候选同构：`text/canon/conf/d/fit/box/box_i/type_fix`。
    """
    global LAST
    if gmin is None:
        gmin = _env_f("STAGE2_KEY_GMIN", 0.30)
    if min_conf is None:
        min_conf = _env_f("STAGE2_KEY_MINCONF", 0.55)
    if rescue is None:
        rescue = os.environ.get("STAGE2_KEY_RESCUE", "0").strip() not in ("", "0")

    wcheck = os.environ.get("STAGE2_KEY_WCHECK", "1").strip() not in ("", "0")
    anyrole = os.environ.get("STAGE2_KEY_ANYROLE", "0").strip() not in ("", "0")
    n_pair = n_wide = n_shape = n_rescue = n_wpen = 0
    cw = _char_width(texts or [], canon_fn, comps or [], gmin, min_conf)
    pairs = []
    for rec in texts or []:
        b = rec.get("bbox") or rec.get("box")
        if not b or len(b) != 4:
            continue
        w = abs(float(b[2]) - float(b[0]))
        h = max(1.0, abs(float(b[3]) - float(b[1])))
        if w / h > 6.0:                       # 长条框（网络名/说明文字）不进键通道
            n_wide += 1
            continue
        reads = rec.get("reads") or {}
        cxs = ((float(b[0]) + float(b[2])) * 0.5, (float(b[1]) + float(b[3])) * 0.5)
        for c in comps:
            g = geom(b, c["bbox"])
            if g < gmin:
                continue
            best = None
            # ★ 专职通道：**只认 `designator` 这一种读法**。同一块框在别的角色下会被
            #   裁成别的样子（实测 0001 的 `C13` 框：designator 读 `C13` conf 0.998，
            #   而另一角色读 `C1` conf 0.999）——按分数挑就会挑到掉字的那一版，正是
            #   老路径 `U6 → J2` 之类错的来源。`STAGE2_KEY_ANYROLE=1` 可回退成全角色。
            _roles = (list(reads.items()) if anyrole
                      else [(r, v) for r, v in reads.items() if r == "designator"])
            for role, v in _roles:
                if not v or not v[0]:
                    continue
                txt, conf = str(v[0]), float(v[1] or 0.0)
                if conf < min_conf:
                    continue
                can, fid, tfix, how = canon_fn(txt, c["type"])
                if not can:
                    continue
                if how and how != "strict":
                    if not rescue:
                        n_rescue += 1
                        continue
                s = float(fid) * conf * g
                if wcheck and cw > 0:
                    n = len(can)
                    ratio = (cw * n) / max(1.0, w)
                    if ratio < 0.60:
                        s *= max(0.05, ratio / 0.60)
                        n_wpen += 1
                if best is None or s > best[0]:
                    best = (s, txt, conf, can, fid, tfix, how)
            if best is None:
                continue
            n_shape += 1
            s, txt, conf, can, fid, tfix, how = best
            pairs.append((s, c["id"], rec, {
                "text": txt, "canon": can, "conf": round(conf, 4),
                "d": round(float(_dist(cxs, c["bbox"])), 1),
                "fit": round(float(fid), 4),
                "box": [round(float(x), 1) for x in b],
                "box_i": int(rec.get("_i", -1)),
                "type_fix": tfix, "key_geom": round(g, 4),
                "key_src": "keyform", "how": how or "strict",
            }))
            n_pair += 1

    pairs.sort(key=lambda p: (-p[0], p[1], str(_bid(p[2]))))
    used_cid, used_box, used_canon = set(), set(), set()
    out: List[Tuple[str, dict]] = []
    n_box = n_canon = 0
    for s, cid, rec, info in pairs:
        if cid in used_cid:
            continue
        bi = _bid(rec)
        if bi is not None and bi in used_box:
            n_box += 1
            continue
        if info["canon"] in used_canon:
            n_canon += 1
            continue
        used_cid.add(cid)
        if bi is not None:
            used_box.add(bi)
        used_canon.add(info["canon"])
        out.append((cid, info))
    out.sort(key=lambda x: x[0])
    LAST = {"n_comp": len(comps or []), "n_text": len(texts or []),
            "n_pair": n_pair, "n_shape": n_shape, "n_key": len(out),
            "n_wide": n_wide, "n_skip_rescue": n_rescue,
            "n_drop_box": n_box, "n_drop_canon": n_canon,
            "n_wpen": n_wpen, "char_w": round(cw, 1),
            "gmin": gmin, "min_conf": min_conf, "rescue": bool(rescue)}
    return out


def _dist(center, body) -> float:
    """文字中心到元件框的距离（`assign._score` 里的那个 `d` 的口径）。"""
    cx, cy = center
    dx = max(float(body[0]) - cx, 0.0, cx - float(body[2]))
    dy = max(float(body[1]) - cy, 0.0, cy - float(body[3]))
    return (dx * dx + dy * dy) ** 0.5
