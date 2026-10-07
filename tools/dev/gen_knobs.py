# -*- coding: utf-8 -*-
"""扫描源码里所有环境开关，生成 `docs/KNOBS.md`。

本项目的调参全部走环境变量（`STAGE1_* / STAGE2_* / STAGE3_* / FUSION_* / VLM_* /
TEXTREC_* / CV_*`，共 200+ 个），散落在各模块顶部。这个脚本把它们连**默认值**和
**定义位置**一起抽出来，避免「有个开关但没人知道它叫什么 / 默认是什么」。

用法：
    python tools/dev/gen_knobs.py            # 打印到 stdout
    python tools/dev/gen_knobs.py --write    # 写入 docs/KNOBS.md
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKGS = ["stage1", "stage2", "stage3", "fusion", "vlm", "textrec",
        "cvparts", "pipeline", "submission", "schematic_parser", "synth"]

# `os.environ.get("X", "y")` / `env_on("X", "1")` / `envflag.falsey("X")`
PAT = re.compile(
    r"""(?:os\.environ\.get|env_on|env_int|env_float|envflag\.\w+)\s*\(\s*"""
    r"""["']([A-Z][A-Z0-9_]{3,})["']\s*(?:,\s*([^)\n]{0,60}))?""")


def scan() -> "collections.OrderedDict[str, list]":
    out = collections.OrderedDict()
    for pkg in PKGS:
        rows = {}
        for p in sorted((ROOT / pkg).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            src = p.read_text(encoding="utf-8", errors="ignore")
            for m in PAT.finditer(src):
                line = src[:m.start()].count("\n") + 1
                rows.setdefault(m.group(1), []).append(
                    (str(p.relative_to(ROOT)).replace("\\", "/"), line,
                     (m.group(2) or "").strip()))
        if rows:
            out[pkg] = sorted(rows.items())
    return out


def render() -> str:
    data = scan()
    total = sum(len(v) for v in data.values())
    L = ["# 环境开关总表",
         "",
         "> 本文件由 `python tools/dev/gen_knobs.py --write` **自动生成**，不要手改。",
         "",
         "全部调参都走环境变量。假值词表（`0` / `off` / `false` / `none` / `no` 一律视为关）",
         "统一实现在 `envflag.py`；PowerShell 里 `$env:X=''` 是**删除**变量而不是关功能，",
         "要关某个开关请写具体假值，例如 `$env:STAGE2_PINNUM_TIGHT='0'`。",
         "",
         "共 **%d** 个开关。" % total,
         "",
         "## 目录",
         ""]
    for pkg, rows in data.items():
        L.append("- [%s](#%s)（%d）" % (pkg, pkg, len(rows)))
    L.append("")
    for pkg, rows in data.items():
        L += ["<a id=\"%s\"></a>" % pkg, "", "## %s" % pkg, "",
              "| 开关 | 默认值 | 定义位置 |", "|---|---|---|"]
        for name, locs in rows:
            f, ln, dflt = locs[0]
            d = ("`%s`" % dflt) if dflt else "—"
            if len(locs) > 1:
                d += " *(%d 处)*" % len(locs)
            L.append("| `%s` | %s | `%s:%d` |" % (name, d, f, ln))
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="写入 docs/KNOBS.md")
    a = ap.parse_args()
    txt = render()
    if a.write:
        out = ROOT / "docs" / "KNOBS.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt, encoding="utf-8")
        print("wrote", out)
    else:
        print(txt)


if __name__ == "__main__":
    main()
