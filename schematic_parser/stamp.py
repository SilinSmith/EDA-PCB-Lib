"""图章贴框：把检测框换成官方标注格式的「规范图章」。

为什么要它
---------
官方 GT 里，符号类元件的 bbox 不是「符号外框」，而是**每类一个固定尺寸的图章**
（`r` 20x8、`v` 10x10、`c` 20x16、`gnd` 20x19、`pin`/`testpoint` 20x20、
`net_input` 30x10 …；横竖两种朝向各一个值）。只有 `box`/`switch`/`other`/
`buzzer`/`block`/`jumper` 是逐实例测量的。

而官方评分对 bbox 的要求是（problem_statement.txt:175-184, :358）：

    bbox 不用于 component 匹配，只在 key matched component 上计算坐标误差，
    默认阈值 20px —— 即 max(|pred[i] - gt[i]| for i in 0..3) <= 20

20px 容差对一个 20x8 的图章来说很宽。所以**只要类别对、中心对**，把预测框换成
以预测中心为中心的规范图章，就一定落在容差内。检测器的任务因此退化成
「认类 + 定中心」，而不是「精确回归框的宽高」。

★★ 2026-10-07：图章从**正方形**改成**长方形（长, 短）+ 轴向**
------------------------------------------------------------
`work\\tools\\diag\\_stampshape.py` 在全部官方 case 上量出来的事实：

* **每个图章类的 (长边, 短边) 是 100% 唯一的**（例如 `r` 1626 个实例全是 20x8，
  `c` 1668 个全是 20x16，`l` 115 个全是 34x6）—— 所以「长方形模板」是精确的，
  不是近似。旧表取 `side = max(w, h)` 把这条信息压掉了。
* 用「完美居中的真值框」比较两种贴法的**角点切比雪夫误差**（按实例数加权）：
      **正方形 2.74px  /  长方形 0.00px**
  单项最大：`l` 14.0px、`net_input`/`net_output`/`net_bidirection` 10.0px、
  `r` 6.0px、`fuse` 6.0px、`bjt`/`mosfet` 4.5px。
* 危害不只是「误差大小」：`r`/`l`/`net_*` 被压成正方形之后，**符号的长短边
  （= 引脚轴向）在提交文档里就没了**，凡是按形状判定的下游与诊断都会被骗
  （我自己就曾被「`r` 的 IoU 恒为 0」骗过一轮）。
* 20px 容差确实很宽，所以这是**余量**问题而不是硬失败：`l` 方形 14px + 中心偏
  6px 就出界了；长方形把这块余量还了回来。
* 轴向比例（长边竖直 / 全部），官方语料实测，用作「框接近正方时」的兜底先验：
  `v`/`pin`/`testpoint`/`amp_*`/`crystal_4pin` 100%、`c` 78.9%、`d` 61.7%、
  `r` 54.6%、`led` 53.2%、`bjt` 13.7%、`net_output` 10.5%、`fuse` 5.0%、
  `gnd` 20.3%、`l` 23.5%、`net_input` 16.2%。
  正常情况下**不用这个先验** —— 轴向由检出的原始框（`box_raw`）的宽高比决定。

派生方式
--------
表由 `derive(json_root)` 从官方 JSON 直接统计：把尺寸按 2px 归并后取值 1~2 种的类
视为图章，取 `(max(w,h), min(w,h))`。`box`/`switch`/`other` 等不在表里的类，
`snap_box` 原样返回。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_JSON_ROOT = Path(__file__).resolve().parent.parent / "yolo_part" / "Dataset(waitforpreprocess)"

# 官方 220 个 case 派生：type -> (长边, 短边)。括号内是官方语料里的实例数。
# ★ 每个类的 (长, 短) 在这份语料里都是**唯一**的（`_stampshape.py` 实测 100%）。
STAMP_WH: dict[str, tuple[int, int]] = {
    "v": (10, 10),              # 1152
    "crystal_2pin": (15, 10),   # 44
    "crystal_3pin": (15, 10),   # 6
    "crystal": (16, 15),        # 1
    "c": (20, 16),              # 1668
    "r": (20, 8),               # 1626
    "gnd": (20, 19),            # 2229
    "d": (20, 14),              # 201
    "led": (20, 14),            # 186
    "pin": (20, 20),            # 175
    "testpoint": (20, 20),      # 43
    "fuse": (22, 10),           # 20
    "bjt": (29, 20),            # 51
    "bjt_npn": (29, 20),        # 2
    "bjt_pnp": (29, 20),        # 2
    "mosfet": (29, 20),         # 66
    "mosfet_npn": (29, 20),     # 6
    "mosfet_pnp": (29, 20),     # 1
    "net_input": (30, 10),      # 339
    "net_output": (30, 10),     # 76
    "net_bidirection": (30, 10),  # 13
    "l": (34, 6),               # 115
    "amp_3pin": (30, 30),       # 10
    "amp_5pin": (40, 40),       # 11
    "crystal_4pin": (40, 40),   # 8
    # 下面是实例数很少的类，保留但别当准的（且都不在 `SNAP_TYPES` 里，实际不生效）
    "motor": (79, 79),
    "opt": (81, 81),
    "seg": (903, 903),          # 1 个实例，几乎肯定是错的
}

# 兼容旧调用点：`STAMP_SIDE` = 旧的正方形边长 = 长边（`max(w,h)`）。
# ⚠ 新代码请用 `STAMP_WH` / `stamp_wh()`；只有「造一个正方形候选框」这种场合
#   才该用 `STAMP_SIDE`（例如 `stage1/detect.py` 里 CV 候选的自建槽位）。
STAMP_SIDE: dict[str, int] = {t: wh[0] for t, wh in STAMP_WH.items()}

# 「长边竖直」的比例（官方语料实测），只在检出的原始框接近正方时兜底
AXIS_VERT_PRIOR: dict[str, float] = {
    "v": 1.0, "pin": 1.0, "testpoint": 1.0, "amp_3pin": 1.0, "amp_5pin": 1.0,
    "crystal_4pin": 1.0, "c": 0.789, "d": 0.617, "r": 0.546, "led": 0.532,
    "crystal_2pin": 0.432, "mosfet": 0.394, "mosfet_npn": 0.333, "l": 0.235,
    "gnd": 0.203, "net_input": 0.162, "bjt": 0.137, "net_output": 0.105,
    "net_bidirection": 0.077, "fuse": 0.05, "bjt_npn": 0.0, "crystal": 0.0,
    "crystal_3pin": 0.833, "mosfet_pnp": 1.0, "bjt_pnp": 0.5,
}

# 只有这些类做贴框；其余（box/switch/other/buzzer/block/jumper/amp/esd/dc/... 等
# 逐实例测量的类）保持预测框原样
SNAP_TYPES: frozenset[str] = frozenset({
    "v", "c", "r", "gnd", "d", "led", "pin", "testpoint", "fuse",
    "crystal", "crystal_2pin", "crystal_3pin", "crystal_4pin",
    "bjt", "bjt_npn", "bjt_pnp", "mosfet", "mosfet_npn", "mosfet_pnp",
    "net_input", "net_output", "net_bidirection",
    "amp_3pin", "amp_5pin", "l",
})


def stamp_wh(t: str):
    """该类的规范图章 `(长边, 短边)`；不是图章类返回 None。"""
    return STAMP_WH.get(t) if t in SNAP_TYPES else None


def is_snap_type(t: str) -> bool:
    return t in SNAP_TYPES and t in STAMP_WH


def _axis_vertical(box, t: str) -> bool:
    """由**检出的原始框**（贴章之前的宽高比）定「长边是竖直的吗」。

    接近正方（|w-h| <= 1）时用官方语料的先验兜底 —— 这种情况只可能出现在
    正方形模板上（那时轴向无所谓）或检测器完全没给出形状时。
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = abs(x2 - x1), abs(y2 - y1)
    if abs(w - h) <= 1.0:
        return AXIS_VERT_PRIOR.get(t, 1.0) >= 0.5
    return h > w


def snap_box(box, t: str, axis_vertical=None):
    """把 `box` 换成以它中心为中心的规范图章**长方形**；非图章类原样返回。

    box: (x1, y1, x2, y2) 任意序列，左上角原点图像像素
    axis_vertical: 长边是否竖直。None ⇒ 从 `box` 自己的宽高比推断
                   （所以**必须传贴章之前的原始框**，见 `stage1/detect.py`）。
    """
    wh = stamp_wh(t)
    if wh is None:
        return tuple(float(v) for v in box)
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    lo, sh = float(wh[0]), float(wh[1])
    if lo == sh:
        w = h = lo
    else:
        if axis_vertical is None:
            axis_vertical = _axis_vertical((x1, y1, x2, y2), t)
        w, h = (sh, lo) if axis_vertical else (lo, sh)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def snap_detections(dets, names=None):
    """就地风格地返回新列表。dets: 可迭代 (cls_id_or_name, box[, score])。

    names 给定时 cls 按 id 解释，否则按字符串解释。
    """
    out = []
    for d in dets:
        cls, box = d[0], d[1]
        rest = list(d[2:])
        t = names[int(cls)] if (names is not None and isinstance(cls, (int, float))) else str(cls)
        out.append((cls, snap_box(box, t), *rest))
    return out


def derive(json_root=DEFAULT_JSON_ROOT, tol: int = 2, max_forms: int = 2,
           min_instances: int = 10):
    """从官方 JSON 重新统计图章表。返回 {type: (长边, 短边, n_instances)}。"""
    sizes: dict[str, Counter] = defaultdict(Counter)
    for jp in Path(json_root).rglob("*.json"):
        try:
            doc = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        comps = doc.get("components")
        if not isinstance(comps, dict):
            continue
        for _, c in comps.items():
            t, b = c.get("type"), c.get("bbox")
            if not t or not isinstance(b, list) or len(b) != 4:
                continue
            sizes[t][(round(abs(b[2] - b[0])), round(abs(b[3] - b[1])))] += 1
    table = {}
    for t, cnt in sizes.items():
        coarse = Counter()
        for (w, h), k in cnt.items():
            coarse[(round(w / tol) * tol, round(h / tol) * tol)] += k
        n = sum(cnt.values())
        if len(coarse) <= max_forms and n >= min_instances:
            # 出现次数最多的那个形态 = 模板
            (mw, mh), _k = cnt.most_common(1)[0]
            table[t] = (max(mw, mh), min(mw, mh), n)
    return table


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    table = derive()
    print(f"{'type':<18}{'(长,短)':>12}{'表里':>12}{'n':>7}   in SNAP_TYPES")
    for t, (lo, sh, n) in sorted(table.items(), key=lambda kv: -kv[1][2]):
        cur = STAMP_WH.get(t)
        mark = "" if cur == (lo, sh) else ("  ← 表是 %s" % (cur,))
        print(f"{t:<18}{str((lo, sh)):>12}{str(cur):>12}{n:>7}   "
              f"{t in SNAP_TYPES}{mark}")
    print("\n表里没有但 STAMP_WH 有的（未被自动派生接受）:",
          sorted(set(STAMP_WH) - set(table)))
