# -*- coding: utf-8 -*-
"""GT 体检（**只读**）：逐案扫 GT json，按「内部一致性 / schema 规则」报违规。

设计纪律（用户 m00074 指令）：
  * **只读**——本文件绝不写任何 GT 文件；修复在 `_gtfix.py` 里单独做、单独提交。
  * 每条规则只依赖 **GT 自身的内部证据**（schema、`type_list.md`、图片尺寸、
    同一 case 内其它标注），**不参考我方产物**。
  * 10GT 只体检、不修改（公开评分参照）。

规则清单（rule id 是稳定的，报告与修复脚本共用）：

  file/parse
    file_json_unreadable      JSON 解析失败
    file_no_main_json         目录里没有任何含 `components` 的 json
    file_json_ignored         目录里除主 json 外还有 json（会被 `discover` 的
                              `sorted(jsons)[0]` 或我们的取主逻辑忽略——`10GT` 的
                              `*.json.bak-bbox` 不算）
  structure
    struct_missing_section    components/pins/nets 缺失
    struct_section_not_dict   三个顶层键不是 dict
  components
    comp_key_duplicate        key 与另一个 key 冲突（大小写/空白归一后相同）
    comp_key_empty            key 为空串
    comp_name_missing         `Name` 缺失或空
    comp_type_missing         `type` 缺失或空
    comp_type_unknown         `type` 不在 `type_list.md` 的规范类型集里
                              （大小写不敏感；`bjt_npn`/`crystal_4pin` 这类由
                              `canon_type` 折叠的写法单独记为 comp_type_noncanonical）
    comp_type_noncanonical    能折到规范类型但不是规范拼写（官方 canon_type 会折叠）
    comp_bbox_missing         `bbox` 缺失或长度≠4
    comp_bbox_nonfinite       bbox 含 NaN/inf 或非数
    comp_bbox_degenerate      bbox 面积 ≤ 0 或宽/高为 0
    comp_bbox_inverted        x1>x2 或 y1>y2
    comp_bbox_out_of_image    bbox 任一角超图片范围（容差 tol_px）
    comp_bbox_overlap         两个元件 bbox 的 IoU > iou_thr（几何重合）
  pins
    pin_group_missing         components 里的元件在 pins 里没有同名组
    pin_group_orphan          pins 里有组但 components 里没有该元件
    pin_group_not_dict        pins[comp] 不是 dict
    pin_point_missing         缺 `point`
    pin_point_not_dict        `point` 不是 dict
    pin_point_nonfinite       point.x/y 非数/NaN/inf
    pin_point_out_of_image    point 超图片范围（容差 tol_px）
    pin_name_missing          `pinname` 缺失
    pin_name_empty            `pinname` 为空串
    pin_number_empty          脚号（key 去掉 `pin_` 前缀）为空
    pin_number_odd            脚号含空白/控制字符（如 `pin_2K6 3 29`）
    pin_number_duplicate      同一元件内两个 pin key 归一后脚号相同
    pin_key_collision         同一元件内两个 pin key 归一后**完全相同**
    pin_far_from_comp         引脚点距元件 bbox 的距离 > near_px（>0 表示在框外）
    pin_pin_coincident        同一 case 内不同元件的引脚点完全重合
  nets
    net_not_dict              nets[x] 不是 dict
    net_hypergraph_missing    缺 `hyperGraph`
    net_hypergraph_not_str    `hyperGraph` 不是字符串
    net_hypergraph_malformed  `hyperGraph` 非空但不是 `(...)` 形态，或 token 无 `.`
    net_hypergraph_dupref     `hyperGraph` 里同一引用写了两次
    net_hypergraph_empty      `hyperGraph` 为空 / 无有效引用
    net_ref_unknown_comp      引用了 components 里不存在的元件
    net_ref_unknown_pin       引用了该元件 pins 里不存在的脚号
    net_edges_missing         缺 `edges`
    net_edges_not_dict        `edges` 不是 dict
    net_edge_not_pair         edge 不是长度=2 的点列
    net_edge_nonfinite        edge 端点含 NaN/inf
    net_edge_out_of_image     edge 端点超图片范围（容差 tol_px）
    net_edge_zero_length      两端点完全重合（退化边）
    net_edge_duplicate        同一 net 内两条边端点集合相同（几何重合）
    net_edges_ge_2_missing    `edges` 项数 < 2（只有 0 或 1 条边）
    net_no_edges_but_refs     hyperGraph 有引用但 `edges` 为空
    net_isolated_gt_pin       hyperGraph 只有 1 个引用（单端网）
    net_dup_membership        同一对 (元件,脚) 出现在多个 net 里（跨 net 重复）
    net_dup_key_across_nets   两个 net 的引用集合完全相同（重复网）

用法：
    python -X utf8 tools\\diag\\_gtlint.py --out work\\_gen220\\gt_lint.json
    python -X utf8 tools\\diag\\_gtlint.py --split 10GT --verbose
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "yolo_part" / "Dataset(waitforpreprocess)"
SPLITS = ("10GT", "10_GTcase", "200_train_cases")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# 规范类型集来自 `10GT/type_list.md` 的第一列，加上 `data.IC_TYPES` / `data.VALUE_TYPES`
# 里的官方别名。
EXTRA_TYPES = {"box", "ic", "boxic", "r", "resistor", "c", "capacitor"}

# 判分器的官方 type 折叠（`metrics.canon_type`）。**必须 import 实现**而不是抄一份：
# 抄写会随实现漂移，直接导致把「判分器能折叠的写法」误报成 error
# （`amp_3pin/amp_5pin` 21 条 / 11 case 已复现）。失败时退化为恒等函数。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:                                                                   # noqa: SIM105
    from schematic_parser.metrics import canon_type                 # type: ignore
except Exception:                                                      # noqa: BLE001
    def canon_type(t):
        return t


def load_type_list(path: Path):
    """`| \\`gnd\\` | 接地 | ... |` -> {'gnd', ...}"""
    if not path.exists():
        return set(EXTRA_TYPES)
    text = path.read_text(encoding="utf-8")
    types = {m.group(1).strip() for m in re.finditer(r"^\|\s*`([^`]+)`\s*\|", text, re.M)}
    return types | EXTRA_TYPES


# --------------------------------------------------------------------- utils

def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def finite(v) -> bool:
    return is_num(v) and math.isfinite(float(v))


def point_of(p):
    """-> (x, y) or None; `bad` 标记非数/NaN。"""
    if not isinstance(p, dict):
        return None, "not_dict"
    x, y = p.get("x"), p.get("y")
    if not (finite(x) and finite(y)):
        return None, "nonfinite"
    return (float(x), float(y)), ""


def pin_number_of(key: str) -> str:
    """与 `data.pin_key_to_number` 一致：去掉 `pin_` 前缀。"""
    return key[4:] if key.startswith("pin_") else key


def resolve_ref(tok, comps, pins):
    """把 hyperGraph 里的一条引用 `<comp>.<num>` 解到 (comp, num)。

    难点：**元件名自身可以含 `.`**（多单元器件 `U8.1`/`U10.2`，见 200_train_cases
    的 0108/0134/0138/0169/0197），而脚号**也可以含 `.`**（`U1.4.7 μF`、`U5.VCC3.3V`、
    `U7.0618.1`，见 0016/0035）。因此单点切分一定会误判，必须**双向消歧**：

      1. 先试 `第一个 `.` 之前` 为元件名；若该元件存在**且**余下部分正好是它的
         一个真实脚号 -> 合法（`ok`）。
      2. 否则把余下部分逐段并入元件名（贪心向右扩展），每扩展一段就检查
         「元件存在 且 剩余尾部 == 真实脚号」-> 合法（`greedy`）。
      3. 都不成立时，按第一个 `.` 切分给出真实违规（`unknown_comp` / `unknown_pin`）。

    返回 (status, comp, num)，status ∈ {ok, greedy, unknown_comp, unknown_pin, malformed}。
    这条规则是修 linter 自身误报用的（首版把 41 条合法的多单元引用全报成了
    `net_ref_unknown_comp` / `net_ref_unknown_pin`）。
    """
    if "." not in tok:
        return "malformed", tok, ""
    first, _, rest = tok.partition(".")

    def nums_of(name):
        grp = pins.get(name)
        return {pin_number_of(k) for k in grp} if isinstance(grp, dict) else set()

    if first in comps and rest in nums_of(first):
        return "ok", first, rest
    # 贪心：把 rest 的头部逐段并进元件名
    comp, num = first, rest
    while "." in num:
        head, _, tail = num.partition(".")
        cand = comp + "." + head
        if cand in comps:
            if tail in nums_of(cand):
                return "greedy", cand, tail
            comp, num = cand, tail
            continue
        break
    if first in comps:
        return "unknown_pin", first, rest
    return "unknown_comp", first, rest


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = ua + ub - inter
    return inter / den if den > 0 else 0.0


def point_bbox_dist(p, bb) -> float:
    """点到 bbox 的距离（在框内为 0）。"""
    x, y = p
    x1, y1, x2, y2 = bb
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def norm_key(k: str) -> str:
    return re.sub(r"\s+", "", str(k)).lower()


# --------------------------------------------------------------------- linter

class Lint:
    def __init__(self, types, tol_px=1.0, iou_thr=0.80, near_px=200.0,
                 coincident_px=0.0, image_inset=0.0):
        self.types = types
        self.tol_px = tol_px
        self.iou_thr = iou_thr
        self.near_px = near_px
        self.coincident_px = coincident_px
        self.image_inset = image_inset
        self.rows = []                    # 逐条违规
        self.case_counts = collections.Counter()   # case key -> 违规条数
        self.type_counts = collections.Counter()   # rule -> 违规条数
        self.case_rules = collections.defaultdict(set)
        # 全局可观测表（供第 3 步归因用）
        self.case_stats = {}

    # -- 记录 ---------------------------------------------------------------
    def bad(self, split, case, rule, evidence, node="", severity="error"):
        self.rows.append({"split": split, "case": case, "rule": rule,
                          "node": str(node), "evidence": str(evidence)[:300],
                          "severity": severity})
        self.type_counts[rule] += 1
        self.case_counts["%s/%s" % (split, case)] += 1
        self.case_rules["%s/%s" % (split, case)].add(rule)

    # -- 主入口 -------------------------------------------------------------
    def lint_case(self, split, case_name, case_dir, main_json, n_extra_json, verbose):
        key = "%s/%s" % (split, case_name)
        pngs = sorted(p for p in case_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in IMAGE_EXT)
        image_size = self.image_size(pngs[0] if pngs else None)
        if main_json is None:
            self.bad(split, case_name, "file_no_main_json",
                     "json 文件：%s" % ", ".join(sorted(p.name for p in case_dir.glob("*.json"))))
            return
        if n_extra_json:
            self.bad(split, case_name, "file_json_ignored",
                     "%d 个额外 json 被忽略：%s" % (
                         n_extra_json,
                         ", ".join(sorted(p.name for p in case_dir.glob("*.json")
                                          if p != main_json)[:4])), severity="warn")
        try:
            doc = json.loads(main_json.read_text(encoding="utf-8"))
        except Exception as exc:                                       # noqa: BLE001
            self.bad(split, case_name, "file_json_unreadable", repr(exc))
            return
        self.lint_doc(split, case_name, doc, main_json.name, image_size, verbose)

    def image_size(self, png):
        if png is None:
            return None
        try:
            import struct
            with png.open("rb") as fh:
                head = fh.read(33)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return (float(w), float(h))
        except Exception:                                              # noqa: BLE001
            pass
        try:
            from PIL import Image
            with Image.open(png) as im:
                return (float(im.size[0]), float(im.size[1]))
        except Exception:                                              # noqa: BLE001
            return None

    def out_of_image(self, x, y, image_size):
        if image_size is None:
            return None
        w, h = image_size
        if x < -self.tol_px or y < -self.tol_px:
            return "negative"
        if x > w + self.tol_px or y > h + self.tol_px:
            return "beyond w=%.0f h=%.0f" % (w, h)
        return None

    # -- 一份文档 -----------------------------------------------------------
    def lint_doc(self, split, case, doc, json_name, image_size, verbose):
        if not isinstance(doc, dict):
            self.bad(split, case, "struct_section_not_dict", "顶层不是 dict")
            return
        for sec in ("components", "pins", "nets"):
            if sec not in doc:
                self.bad(split, case, "struct_missing_section", "缺顶层键 %s" % sec)
            elif not isinstance(doc[sec], dict):
                self.bad(split, case, "struct_section_not_dict",
                         "%s 不是 dict（%s）" % (sec, type(doc[sec]).__name__))
        comps = doc.get("components") if isinstance(doc.get("components"), dict) else {}
        pins = doc.get("pins") if isinstance(doc.get("pins"), dict) else {}
        nets = doc.get("nets") if isinstance(doc.get("nets"), dict) else {}

        stat = {"json": json_name, "image": image_size,
                "n_comp": len(comps), "n_pin_group": len(pins),
                "n_pin": sum(len(g) for g in pins.values() if isinstance(g, dict)),
                "n_net": len(nets),
                "n_edge": sum(len(n.get("edges") or {}) for n in nets.values()
                              if isinstance(n, dict)),
                "rules": {}, "n_bad": 0}

        self.lint_components(split, case, comps, pins, image_size)
        self.lint_pins(split, case, comps, pins, image_size)
        self.lint_nets(split, case, comps, pins, nets, image_size)
        self.case_stats["%s/%s" % (split, case)] = stat

    # -- components ---------------------------------------------------------
    def lint_components(self, split, case, comps, pins, image_size):
        keys = list(comps)
        seen = {}
        for k in keys:
            if k == "":
                self.bad(split, case, "comp_key_empty", "components 里有空 key")
            nk = norm_key(k)
            if nk in seen:
                self.bad(split, case, "comp_key_duplicate", "%r 与 %r 归一后相同" % (k, seen[nk]), k)
            seen.setdefault(nk, k)

        boxes = []
        for k, c in comps.items():
            if not isinstance(c, dict):
                self.bad(split, case, "comp_bbox_missing", "元件项不是 dict: %r" % (type(c).__name__), k)
                continue
            name = c.get("Name")
            if not isinstance(name, str) or not name.strip():
                self.bad(split, case, "comp_name_missing", "Name=%r" % (name,), k)
            ctype = c.get("type")
            if not isinstance(ctype, str) or not ctype.strip():
                self.bad(split, case, "comp_type_missing", "type=%r" % (ctype,), k)
            else:
                t = ctype.strip()
                if t not in self.types:
                    # 判分器自己有一层官方别名折叠（`metrics.TYPE_ALIASES`，
                    # 例：`amp_5pin/amp_3pin/amp_7pin -> amp`、`crystal_3pin -> crystal`、
                    # `ic/boxic -> box`、`resistor -> r`）。能被它折叠的写法**不影响判分**，
                    # 所以只能算 soft（原本错报成 error；`amp_5pin` 21 条 / 11 case 已复现）。
                    canon = canon_type(t)
                    if canon in self.types:
                        self.bad(split, case, "comp_type_noncanonical",
                                 "type=%r（官方 canon_type -> %r，判分不受影响）"
                                 % (ctype, canon), k, severity="soft")
                    else:
                        self.bad(split, case, "comp_type_unknown",
                                 "type=%r 不在 type_list.md/%d 类里，且官方别名表也折不动"
                                 % (ctype, len(self.types)), k, severity="error")
            bb = c.get("bbox")
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                self.bad(split, case, "comp_bbox_missing", "bbox=%r" % (bb,), k)
                continue
            if not all(finite(v) for v in bb):
                self.bad(split, case, "comp_bbox_nonfinite", "bbox=%r" % (bb,), k)
                continue
            x1, y1, x2, y2 = (float(v) for v in bb)
            if x2 < x1 or y2 < y1:
                self.bad(split, case, "comp_bbox_inverted", "bbox=%r" % (bb,), k)
            if (x2 - x1) * (y2 - y1) <= 0:
                self.bad(split, case, "comp_bbox_degenerate",
                         "bbox=%r w=%.1f h=%.1f" % (bb, x2 - x1, y2 - y1), k)
            for cx, cy, tag in ((x1, y1, "x1y1"), (x2, y2, "x2y2")):
                why = self.out_of_image(cx, cy, image_size)
                if why:
                    self.bad(split, case, "comp_bbox_out_of_image",
                             "bbox=%r %s=%s（%s）" % (bb, tag, why, "超界"), k)
            boxes.append((k, (x1, y1, x2, y2), ctype))
        # 几何重合：并查集式扫对（O(n^2)，200 张图元件数 ≤ 数百，可接受）
        n = len(boxes)
        for i in range(n):
            for j in range(i + 1, n):
                iou = bbox_iou(boxes[i][1], boxes[j][1])
                if iou > self.iou_thr:
                    self.bad(split, case, "comp_bbox_overlap",
                             "%s(%s) vs %s(%s) IoU=%.2f" % (boxes[i][0], boxes[i][2],
                                                             boxes[j][0], boxes[j][2], iou),
                             "%s|%s" % (boxes[i][0], boxes[j][0]))
        # pins 组与 components 键的对账
        for k in comps:
            if k not in pins:
                self.bad(split, case, "pin_group_missing", "components 有 %r，pins 没有" % k, k)
        for k in pins:
            if k not in comps:
                self.bad(split, case, "pin_group_orphan", "pins 有 %r，components 没有" % k, k)

    # -- pins ---------------------------------------------------------------
    def lint_pins(self, split, case, comps, pins, image_size):
        seen_pts = {}
        for comp, group in pins.items():
            if not isinstance(group, dict):
                self.bad(split, case, "pin_group_not_dict",
                         "pins[%r] 不是 dict（%s）" % (comp, type(group).__name__), comp)
                continue
            comp_bb = None
            c = comps.get(comp)
            bb = (c or {}).get("bbox") if isinstance(c, dict) else None
            if isinstance(bb, (list, tuple)) and len(bb) == 4 and all(finite(v) for v in bb):
                comp_bb = tuple(float(v) for v in bb)
            num_seen = {}
            for pk, info in group.items():
                num = pin_number_of(pk)
                if not str(num).strip():
                    self.bad(split, case, "pin_number_empty", "key=%r" % pk, "%s.%s" % (comp, pk))
                if re.search(r"\s", str(num)) or any(ord(ch) < 32 for ch in str(num)):
                    self.bad(split, case, "pin_number_odd", "key=%r -> 脚号 %r" % (pk, num),
                             "%s.%s" % (comp, pk), severity="warn")
                nn = norm_key(num)
                if pk in num_seen:
                    self.bad(split, case, "pin_key_collision",
                             "key %r 与 %r 完全相同" % (pk, num_seen[pk]), "%s.%s" % (comp, pk))
                elif nn in {norm_key(x) for x in num_seen.values()}:
                    self.bad(split, case, "pin_number_duplicate",
                             "%r 与 %r 归一后脚号都是 %r" % (pk, [k for k in num_seen
                                                                  if norm_key(num_seen[k]) == nn][0], num),
                             "%s.%s" % (comp, pk))
                num_seen.setdefault(pk, num)
                if not isinstance(info, dict):
                    self.bad(split, case, "pin_point_missing",
                             "pins[%r][%r] 不是 dict" % (comp, pk), "%s.%s" % (comp, pk))
                    continue
                if "pinname" not in info:
                    self.bad(split, case, "pin_name_missing", "无 pinname", "%s.%s" % (comp, pk),
                             severity="warn")
                else:
                    pn = info.get("pinname")
                    if not isinstance(pn, str):
                        self.bad(split, case, "pin_name_missing",
                                 "pinname 非字符串：%r" % (pn,), "%s.%s" % (comp, pk), severity="warn")
                    elif not pn.strip():
                        self.bad(split, case, "pin_name_empty", "pinname 为空串", "%s.%s" % (comp, pk),
                                 severity="warn")
                if "point" not in info:
                    self.bad(split, case, "pin_point_missing", "无 point", "%s.%s" % (comp, pk))
                    continue
                pt, why = point_of(info.get("point"))
                if pt is None:
                    rule = "pin_point_not_dict" if why == "not_dict" else "pin_point_nonfinite"
                    self.bad(split, case, rule, "point=%r" % (info.get("point"),), "%s.%s" % (comp, pk))
                    continue
                ooi = self.out_of_image(pt[0], pt[1], image_size)
                if ooi:
                    self.bad(split, case, "pin_point_out_of_image",
                             "point=(%.0f,%.0f) %s" % (pt[0], pt[1], ooi), "%s.%s" % (comp, pk))
                if comp_bb is not None:
                    d = point_bbox_dist(pt, comp_bb)
                    if d > self.near_px:
                        self.bad(split, case, "pin_far_from_comp",
                                 "距 bbox >%.0fpx：d=%.1f bbox=%s point=(%.0f,%.0f)"
                                 % (self.near_px, d, comp_bb, pt[0], pt[1]),
                                 "%s.%s" % (comp, pk), severity="warn")
                key2 = (round(pt[0], 1), round(pt[1], 1))
                if key2 in seen_pts and seen_pts[key2] != ("%s.%s" % (comp, pk)):
                    self.bad(split, case, "pin_pin_coincident",
                             "%s 与 %s 坐标完全重合 (%.1f,%.1f)"
                             % (seen_pts[key2], "%s.%s" % (comp, pk), pt[0], pt[1]),
                             "%s.%s" % (comp, pk))
                seen_pts.setdefault(key2, "%s.%s" % (comp, pk))

    # -- nets ---------------------------------------------------------------
    def lint_nets(self, split, case, comps, pins, nets, image_size):
        membership = collections.defaultdict(list)     # (comp,num) -> [net,...]
        net_refsets = {}
        for name, net in nets.items():
            if not isinstance(net, dict):
                self.bad(split, case, "net_not_dict",
                         "nets[%r] 不是 dict（%s）" % (name, type(net).__name__), name)
                continue
            hg = net.get("hyperGraph")
            refs = []
            if hg is None:
                self.bad(split, case, "net_hypergraph_missing", "缺 hyperGraph", name)
            elif not isinstance(hg, str):
                self.bad(split, case, "net_hypergraph_not_str", "hyperGraph=%r" % (hg,), name)
            else:
                s = hg.strip()
                if s and not (s.startswith("(") and s.endswith(")")):
                    self.bad(split, case, "net_hypergraph_malformed",
                             "非 `(...)` 形态：%r" % hg[:80], name)
                body = s[1:-1] if s.startswith("(") and s.endswith(")") else s
                toks = [t.strip() for t in body.split(",") if t.strip()]
                for tok in toks:
                    if "." not in tok:
                        self.bad(split, case, "net_hypergraph_malformed",
                                 "token 无 `.`：%r（hyperGraph=%r）" % (tok, hg[:80]), name)
                        continue
                    # 元件名本身可以含 `.`（多单元 `U8.1`），脚号也可以含 `.`
                    # （`U1.4.7 μF`）。必须双向消歧：先按第一个 `.` 试，
                    # 不行就把元件名贪心向右扩展。见 resolve_ref 的 docstring。
                    st, comp, num = resolve_ref(tok, comps, pins)
                    refs.append((comp, ".", num))
                    if st == "greedy":
                        self.bad(split, case, "net_ref_multiunit_greedy",
                                 "%s 解为 元件=%s 脚号=%s（多单元点号切分歧义，非违规）"
                                 % (tok, comp, num), name, severity="soft")
                if len(refs) != len(set(refs)):
                    self.bad(split, case, "net_hypergraph_dupref",
                             "同一引用重复：%r" % hg[:120], name)
                if not refs:
                    self.bad(split, case, "net_hypergraph_empty",
                             "hyperGraph 为空/无有效引用：%r" % (hg,), name)
            for comp, _, num in refs:
                membership[(comp, num)].append(name)
                if comp not in comps:
                    self.bad(split, case, "net_ref_unknown_comp",
                             "%s 不在 components（net=%r）" % (comp, name), name)
                else:
                    grp = pins.get(comp)
                    if isinstance(grp, dict):
                        nums = {pin_number_of(pk) for pk in grp}
                        if num not in nums:
                            self.bad(split, case, "net_ref_unknown_pin",
                                     "%s.%s 不在 pins[%r]（共 %d 脚）" % (comp, num, comp, len(nums)),
                                     name)

            edges = net.get("edges")
            if edges is None:
                self.bad(split, case, "net_edges_missing", "缺 edges", name)
                n_edges = 0
            elif not isinstance(edges, dict):
                self.bad(split, case, "net_edges_not_dict",
                         "edges 不是 dict（%s）" % type(edges).__name__, name)
                n_edges = 0
            else:
                n_edges = len(edges)
                seen_edges = {}
                for ek, e in edges.items():
                    if not isinstance(e, (list, tuple)) or len(e) != 2:
                        self.bad(split, case, "net_edge_not_pair",
                                 "edge 长度≠2：%r" % (e,), "%s.%s" % (name, ek))
                        continue
                    pts = []
                    ok = True
                    for p in e:
                        pt, why = point_of(p)
                        if pt is None:
                            self.bad(split, case, "net_edge_nonfinite",
                                     "端点 %r（%s）" % (p, why), "%s.%s" % (name, ek))
                            ok = False
                            break
                        pts.append(pt)
                    if not ok:
                        continue
                    for pt in pts:
                        ooi = self.out_of_image(pt[0], pt[1], image_size)
                        if ooi:
                            self.bad(split, case, "net_edge_out_of_image",
                                     "端点 (%.0f,%.0f) %s" % (pt[0], pt[1], ooi),
                                     "%s.%s" % (name, ek))
                    if pts[0] == pts[1]:
                        self.bad(split, case, "net_edge_zero_length",
                                 "两端点重合 (%.1f,%.1f)" % pts[0], "%s.%s" % (name, ek),
                                 severity="warn")
                    # 同 net 内两条边**几何重合**（方向可反）——判定键是
                    # 排序后的两个端点（1px 量化），而不是有序列表：
                    # `(607,784)-(609,784)` 与 `(609,784)-(607,784)` 是同一条线段。
                    ends = sorted(((round(pts[0][0], 1), round(pts[0][1], 1)),
                                   (round(pts[1][0], 1), round(pts[1][1], 1))))
                    sig = tuple(ends)
                    if sig in seen_edges:
                        self.bad(split, case, "net_edge_duplicate",
                                 "%s 与 %s 几何重合（端点集合相同，方向无关）"
                                 % (seen_edges[sig], "%s.%s" % (name, ek)),
                                 "%s.%s" % (name, ek), severity="warn")
                    seen_edges.setdefault(sig, "%s.%s" % (name, ek))
                if n_edges < 2:
                    # 只有 1 条边的 net 在这批 GT 里**是常态**（`(U1.7)` 单端网 + 一根线），
                    # 不能当违规。只统计，供归因。
                    self.bad(split, case, "net_edges_lt_2",
                             "edges 只有 %d 条边（refs=%d）" % (n_edges, len(refs)),
                             name, severity="warn")
                if n_edges == 0 and refs:
                    self.bad(split, case, "net_no_edges_but_refs",
                             "hyperGraph 有 %d 个引用但 edges 为空" % len(refs), name)
            if len(refs) == 1:
                self.bad(split, case, "net_isolated_gt_pin",
                         "单端网：%r" % (refs[0],), name, severity="warn")
            net_refsets[name] = frozenset(refs)

        for pair, where in membership.items():
            if len(where) > 1:
                self.bad(split, case, "net_dup_membership",
                         "%s.%s 同时属于 %s" % (pair[0], pair[1], "/".join(where)),
                         pair[0])
        # 引用集合完全相同的两个 net
        by_sig = collections.defaultdict(list)
        for name, sig in net_refsets.items():
            if sig:
                by_sig[sig].append(name)
        for sig, names in by_sig.items():
            if len(names) > 1:
                self.bad(split, case, "net_dup_key_across_nets",
                         "引用集合相同的网：%s" % "/".join(sorted(names)),
                         names[0])


# --------------------------------------------------------------------- driver

def pick_main_json(case_dir: Path):
    """返回 (主 json path, 额外 json 数)。主 json = 含 `components` 的那个。"""
    jsons = sorted(p for p in case_dir.iterdir()
                   if p.is_file() and p.suffix.lower() == ".json")
    main = None
    for p in jsons:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                              # noqa: BLE001
            continue
        if isinstance(d, dict) and "components" in d:
            main = p
            break
    if main is None and jsons:
        main = jsons[0]
    return main, max(0, len(jsons) - 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--split", action="append", default=None)
    ap.add_argument("--out", default="", help="违规全量 JSON 落盘路径")
    ap.add_argument("--tol-px", type=float, default=1.0, help="越界容差")
    ap.add_argument("--iou-thr", type=float, default=0.80, help="元件 bbox 重合阈值")
    ap.add_argument("--near-px", type=float, default=200.0, help="引脚离 bbox 多远算异常")
    ap.add_argument("--top", type=int, default=0, help="只打印违规最多的前 N 个 case")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.data_root)
    splits = tuple(a.split) if a.split else SPLITS
    types = load_type_list(root / "10GT" / "type_list.md")
    lint = Lint(types, tol_px=a.tol_px, iou_thr=a.iou_thr, near_px=a.near_px)

    n_cases = 0
    for split in splits:
        sd = root / split
        if not sd.is_dir():
            print("!! split 不存在：%s" % sd)
            continue
        for case_dir in sorted(p for p in sd.iterdir() if p.is_dir()):
            main_json, n_extra = pick_main_json(case_dir)
            lint.lint_case(split, case_dir.name, case_dir, main_json, n_extra, a.verbose)
            n_cases += 1

    print("=" * 78)
    print("GT 体检：%d 个 case，规范类型 %d 个（含别名），容差 %.1fpx，IoU>%.2f"
          % (n_cases, len(types), a.tol_px, a.iou_thr))
    print("=" * 78)
    print("--- 违规类型直方图（rule -> 条数 / 涉及 case 数）")
    rule_cases = collections.Counter()
    for r in lint.rows:
        rule_cases[r["rule"]] += 1
    by_rule = collections.defaultdict(set)
    for r in lint.rows:
        by_rule[r["rule"]].add("%s/%s" % (r["split"], r["case"]))
    for rule, cnt in sorted(lint.type_counts.items(), key=lambda kv: -kv[1]):
        print("  %-28s %6d 条  / %3d case" % (rule, cnt, len(by_rule[rule])))
    if not lint.type_counts:
        print("  （无违规）")

    print("--- 按 split（严重度分离：hard=必须修，soft=计数量级但单独列）")
    case_split = {k: k.split("/", 1)[0] for k in lint.case_stats}
    for split in splits:
        keys = [k for k in case_split if case_split[k] == split]
        if not keys:
            continue
        # soft = 一切非 error 的严重度（`warn` / `soft`）。只比 `"warn"` 会漏掉
        # `comp_type_noncanonical` 这类被判分器折叠的 soft 记录。
        cases_hard = {r["case"] for r in lint.rows
                      if r["split"] == split and r["severity"] == "error"}
        cases_soft = {r["case"] for r in lint.rows
                      if r["split"] == split and r["severity"] != "error"}
        n_hard = sum(1 for r in lint.rows if r["split"] == split and r["severity"] == "error")
        n_soft = sum(1 for r in lint.rows if r["split"] == split and r["severity"] != "error")
        print("  %-18s case=%3d  hard: %4d 条 / %3d case   soft: %5d 条 / %3d case  干净 case=%3d"
              % (split, len(keys), n_hard, len(cases_hard), n_soft, len(cases_soft),
                 len(keys) - len(cases_hard | cases_soft)))

    print("--- 违规最多的 case（前 15）")
    for k, cnt in lint.case_counts.most_common(15):
        print("  %-40s %5d 条  %s" % (k, cnt, ",".join(sorted(lint.case_rules[k])[:5])))

    if a.verbose:
        print("--- 违规明细（前 400 条）")
        for r in lint.rows[:400]:
            print("  [%s] %s/%s %s=%s  %s" % (r["severity"], r["split"], r["case"],
                                              r["rule"], r["node"], r["evidence"]))

    if a.out:
        outp = Path(a.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_cases": n_cases, "splits": list(splits),
            "params": {"tol_px": a.tol_px, "iou_thr": a.iou_thr, "near_px": a.near_px},
            "rule_counts": dict(lint.type_counts),
            "rule_cases": {k: sorted(v) for k, v in by_rule.items()},
            "case_counts": dict(lint.case_counts),
            "case_rules": {k: sorted(v) for k, v in lint.case_rules.items()},
            "case_stats": lint.case_stats,
            "violations": lint.rows,
        }
        outp.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8", newline="\n")
        print("违规明细 -> %s（%d 条）" % (outp, len(lint.rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
