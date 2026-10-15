# -*- coding: utf-8 -*-
"""队友 13 个「引脚读取器」的**接口契约检查器**（2026-10-16 统一后新增）。

为什么需要它：这 13 个模块是 workflow 上可重组的部件（节点 ⑦c–⑦g），只有接口一致，
`stage2/run.py` 的 `PIN_READERS` 注册表和 workflow 的连线才不会随模块改动悄悄失效。
这个脚本把契约**写成可执行的断言**，改动后跑一次就知道有没有破约。

契约（与各模块 docstring 顶部的【接口】块、`PIN_READERS` 一致）：
    · 读数器  read_xxx(doc, reads, engine, bgr, *, min_conf=0.0, **opts) -> List[dict]
    · 裁片器  xxx_boxes(doc, image, reads=None, **opts)                 -> (List[dict], image)
    · 返回    纯追加；关掉开关时下游逐位不变
    · 开关    只在 `stage2/run.py` 读（模块源码里不得出现 os.environ/getenv）

用法：
    cd Y:\\workspace\\EDA\\eda-pcb
    python -X utf8 tools\\dev\\pinread_contract.py          # 打印表格
    python -X utf8 tools\\dev\\pinread_contract.py -v       # 附每个入口的实测签名
不合规 exit 1。
"""
from __future__ import annotations

import importlib
import inspect
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

# (模块, 入口, 种类, 冒烟调用追加 kwargs)
#   种类 reader  : (doc, reads, engine, bgr, *, min_conf, **opts)
#   种类 boxes   : (doc, image, reads=None, **opts) -> (list, image)
#   种类 collect : 阶段一/证据类收集器，顺序固定但不吃 engine
#   种类 helper  : 纯库函数，只登记签名
TARGETS = [
    ("stage1.pincolor", "extract_color_pins", "collect", {}),
    ("stage1.pin_evidence", "build_pin_evidence", "collect", {}),
    ("stage2.color_crops", "color_number_boxes", "boxes", {}),
    ("stage2.color_crops", "read_color_numbers", "reader", {}),
    ("stage2.pin_text_crops", "pin_text_boxes", "boxes", {}),
    ("stage2.pin_text_crops", "read_pin_texts", "reader", {}),
    ("stage2.pin_lane_crops", "pin_lane_boxes", "boxes", {}),
    ("stage2.pin_lane_reading", "read_pin_lanes", "reader", {}),
    ("stage2.pin_number_crops", "number_repair_boxes", "boxes", {}),
    ("stage2.pin_number_reading", "read_pin_numbers", "reader", {}),
    ("stage2.pin_name_reading", "read_pin_names", "reader", {"mode": "off"}),
    ("stage2.pin_name_ocr", "get_pin_name_ocr", "helper", {"mode": "off"}),
    ("stage2.mono_pin_crops", "mono_pin_boxes", "boxes", {}),
    ("stage2.mono_pin_reading", "read_mono_pin_texts", "reader", {"mode": "off"}),
    ("stage2.isolated_ocr", "prepare_isolated_sample", "helper", {}),
    ("stage2.isolated_ocr", "read_isolated_many", "helper", {}),
]

READER_HEAD = ("doc", "reads", "engine", "bgr")
BOXES_HEAD = ("doc", "image")


def kind_ok(kind: str, sig: inspect.Signature) -> str:
    """返回 "" 表示合规，否则返回失败原因。"""
    params = list(sig.parameters.values())
    pos = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    kw = [p for p in params if p.kind == p.KEYWORD_ONLY]
    names = [p.name for p in pos]
    if kind == "reader":
        if names[:4] != list(READER_HEAD):
            return "位置参数应为 %s，实际 %s" % (list(READER_HEAD), names)
        if len(pos) > 4:
            return "读数器不得再有位置参数（多出 %s）" % names[4:]
        for p in kw:
            if p.default is p.empty:
                return "关键字参数 %s 没有默认值" % p.name
        if "min_conf" not in [p.name for p in kw]:
            return "缺少 min_conf 关键字参数"
    elif kind == "boxes":
        if names[:2] != list(BOXES_HEAD):
            return "位置参数应以 %s 开头，实际 %s" % (list(BOXES_HEAD), names)
        rest = pos[2:]
        if rest and not (len(rest) == 1 and rest[0].name == "reads"
                         and rest[0].default is None):
            return "第三个位置参数只能是 reads=None，实际 %s" % [(p.name, p.default) for p in rest]
        for p in kw:
            if p.default is p.empty:
                return "关键字参数 %s 没有默认值" % p.name
    elif kind == "collect":
        if len(pos) < 2:
            return "收集器至少要有 2 个位置参数，实际 %s" % names
        for p in kw:
            if p.default is p.empty:
                return "关键字参数 %s 没有默认值" % p.name
    return ""


class _StubEngine:
    """冒烟用的假识别器：`recognizer()` 返回 None ⇒ 读数器应立即返回 []。"""

    @staticmethod
    def recognizer():
        return None

    @staticmethod
    def read(_bgr, boxes, *_a, **_k):
        return [{"bbox": b, "reads": {}} for b in boxes]


def smoke(kind: str, fn, extra: dict):
    doc = {}
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    if kind == "reader":
        out = fn(doc, [], _StubEngine(), img, min_conf=0.0, **extra)
        if not isinstance(out, list):
            return "返回值应为 list，实际 %s" % type(out).__name__
    elif kind == "boxes":
        out = fn(doc, img, reads=[])
        if not (isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[0], list)):
            return "返回值应为 (list, image)，实际 %s" % (type(out).__name__,)
    elif kind == "collect":
        try:
            fn([], {}, np.zeros((8, 8), dtype=np.uint8), **extra)
        except TypeError:
            fn([], {}, **extra)
    return ""


def registry_pairs():
    """从 run.py 的 PIN_READERS 里抽出 (模块, 函数) 对，用于防止清单漂移。

    用 AST 而不是正则：注册表项里混着开关变量、角色元组和 lambda，
    正则会把 `("pin_name", "pin_number")` 这样的角色元组也当成一对。
    """
    import ast
    path = os.path.join(ROOT, "stage2", "run.py")
    src = io.open(path, encoding="utf-8").read()
    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "PIN_READERS" for t in node.targets):
            for elt in node.value.elts:
                mod, fn = elt.elts[2], elt.elts[3]
                if isinstance(mod, ast.Constant) and isinstance(fn, ast.Constant):
                    out.add((str(mod.value), str(fn.value)))
    return out


def main(argv) -> int:
    verbose = "-v" in argv
    rows, bad = [], 0
    mods = sorted({m for m, _f, _k, _e in TARGETS})
    sources = {}
    for mod in mods:
        rel = mod.replace(".", os.sep) + ".py"
        try:
            sources[mod] = io.open(os.path.join(ROOT, rel), encoding="utf-8").read()
        except OSError:
            sources[mod] = ""
    for mod, fname, kind, extra in TARGETS:
        note = ""
        try:
            m = importlib.import_module(mod)
            fn = getattr(m, fname)
            sig = inspect.signature(fn)
            note = kind_ok(kind, sig)
            if not note:
                note = smoke(kind, fn, extra)
        except Exception as e:                       # noqa: BLE001
            note = "%s: %s" % (type(e).__name__, e)
        env_hits = len(re.findall(r"os\.environ|getenv", sources.get(mod, "")))
        if env_hits:
            note = (note + "；" if note else "") + "模块源码里读 env %d 处（应只在 run.py）" % env_hits
        ok = not note
        bad += 0 if ok else 1
        rows.append((ok, mod, fname, kind, note, sig if verbose else None))
        if verbose:
            print("   %-28s %s" % (mod, sig))

    print("模块入口契约（读数器 / 裁片器 / 收集器 / helper）")
    print("-" * 100)
    for ok, mod, fname, kind, note, _s in rows:
        print("%s %-26s %-24s %-7s %s"
              % ("PASS" if ok else "FAIL", mod, fname, kind, note))
    print("-" * 100)
    print("共 %d 个入口：通过 %d，失败 %d" % (len(rows), len(rows) - bad, bad))

    pairs = registry_pairs()
    if not pairs:
        print("FAIL 解析 run.py 的 PIN_READERS 失败")
        return 1
    pairs = {("stage2." + m, f) for m, f in pairs}
    listed = {(m, f) for m, f, _k, _e in TARGETS}
    helpers = {("stage1.pincolor", "extract_color_pins"),
               ("stage1.pin_evidence", "build_pin_evidence"),
               ("stage2.pin_lane_crops", "pin_lane_boxes"),
               ("stage2.pin_name_ocr", "get_pin_name_ocr"),
               ("stage2.mono_pin_crops", "mono_pin_boxes"),
               ("stage2.isolated_ocr", "prepare_isolated_sample"),
               ("stage2.isolated_ocr", "read_isolated_many")}
    miss = sorted(pairs - listed)
    extra_reg = sorted(listed - pairs - helpers)
    if miss:
        print("FAIL 注册表里这些项没被契约覆盖：%s" % (miss,))
        bad += len(miss)
    if extra_reg:
        print("提示 契约表里有注册表未挂载的部件（允许，如裁片器）：%s" % (extra_reg,))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
