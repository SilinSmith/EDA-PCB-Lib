# -*- coding: utf-8 -*-
"""§5.114 取证：队友分支（`work/_wt_pr3` @ c3d3684）相对我方主线的**模块依赖差**。

回答用户的「本地的现有版本不一定提供了准确的 IO 支持」这一问：把队友侧
**新增/改动的 .py** 逐个解析 import，只看**本地模块**（其名字能在队友树上
找到文件），再看这个模块在我方 `eda-pcb` 里**有没有**、以及
`from X import Y` 的符号在不在 —— 缺的/形状不同的就是「依赖没被提供」的清单。

用法（从 Y:\\workspace\\EDA 根跑）：
    python -X utf8 work\\_depgap.py [--symbols] [--all]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from pathlib import Path

THEIRS = Path(r"Y:\workspace\EDA\work\_wt_pr3")
OURS = Path(r"Y:\workspace\EDA\eda-pcb")
SKIP = {"tests", "tools", "docs", "work", ".git", "yolo_part", "weights", "assets"}

ap = argparse.ArgumentParser()
ap.add_argument("--symbols", action="store_true", help="还查 from X import Y 的符号是否在")
ap.add_argument("--all", action="store_true", help="不过滤「只列新增/改动」，列全部")
a = ap.parse_args()


def modmap(tree: Path):
    """模块名 -> 文件路径（含包）。最外层名字优先。"""
    out = {}
    for p in tree.rglob("*.py"):
        if any(part in SKIP for part in p.relative_to(tree).parts[:-1]):
            continue
        out.setdefault(p.stem if p.stem != "__init__" else p.parent.name, p)
    return out


their, our = modmap(THEIRS), modmap(OURS)
print("队友侧本地模块 %d 个，我方 %d 个" % (len(their), len(our)))


def digest(p: Path):
    return hashlib.sha1(p.read_bytes()).hexdigest()[:12] if p.exists() else "-"


def symbols(p: Path):
    try:
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return out


changed, missing_import, new_files = [], {}, []
for name, p in sorted(their.items()):
    if name not in our:
        new_files.append((name, p))
    else:
        if digest(p) != digest(our[name]):
            changed.append((name, p))
print("我方没有的模块 %d 个；双方都有但内容不同 %d 个" % (len(new_files), len(changed)))
print("--- 我方没有的模块（队友新增，需要一起搬） ---")
for name, p in new_files:
    lines = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
    print("    +%-28s %5d 行  %s" % (name, lines, p.relative_to(THEIRS)))

want = [(n, p) for n, p in (new_files + changed)] if a.all else (new_files + changed)
rows = []
for name, p in sorted(want):
    try:
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as e:
        rows.append((name, "?", "SYNTAX: %s" % e))
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                rows.append((name, al.name.split(".")[0], None))
        elif isinstance(node, ast.ImportFrom):
            if node.level:      # 相对 import：留给包内自洽
                continue
            if node.module:
                rows.append((name, node.module.split(".")[0],
                             [al.name for al in node.names]))

seen, out = set(), []
for src, mod, syms in rows:
    if mod not in their:          # 不是本地模块（stdlib / 三方）
        continue
    key = (src, mod)
    if key in seen:
        continue
    seen.add(key)
    if mod not in our:
        out.append((src, mod, "**缺整个模块**", syms))
        continue
    if a.symbols and syms:
        have = symbols(our[mod])
        gone = [s for s in syms if s != "*" and s not in have]
        if gone:
            out.append((src, mod, "缺符号 " + ",".join(gone), syms))
print("--- 本地依赖差（队友的新增/改动文件 → 依赖的本地模块 → 我方状态） ---")
if not out:
    print("    （无：所有本地依赖在我方都存在）")
for src, mod, why, syms in sorted(out):
    print("    %-34s → %-24s %s" % (src, mod, why))
