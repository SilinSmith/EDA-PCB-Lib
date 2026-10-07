# -*- coding: utf-8 -*-
"""生成 `docs/TOOLS.md` —— `tools/` 下每个脚本的索引（含首行 docstring）。

`tools/` 是本项目「怎么量」的全部家当：打分尺子、错例归因、A/B 工具。
脚本很多（300+），单靠 `ls` 找不到东西，所以把**文件名 + 行数 + 一句话说明**
抽成一张表。新增脚本后重跑本生成器即可。

用法：
    python tools/dev/gen_toolindex.py            # 打印
    python tools/dev/gen_toolindex.py --write    # 写入 docs/TOOLS.md
"""
from __future__ import annotations

import argparse
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"

SUB_DESC = {
    "eval": "**打分尺子**：端到端 A/B、批量评测、结果归档与对比。",
    "diag": "**诊断与归因**：把某一项失分拆到可行动的最小单元，再逐一证伪对策。",
    "data": "数据集构建与体检（合成数据、标签审计、官方口径评测）。",
    "dev": "开发辅助（本目录的生成器）。",
    "debug": "单样例对账：把官方评测器的打分过程逐条摊开。",
    "flow": "`docs/workflow.html` 的构建与校验。",
    "gnd": "电源/地（`∅NNN`）符号专项。",
    "viz": "可视化：把错例裁片、对照图渲染成人眼可核对的图。",
    "recover": "中断恢复工具。",
}


def first_doc(path: pathlib.Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return ""
    doc = ast.get_docstring(tree) or ""
    for line in doc.split("\n"):
        if line.strip():
            return line.strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    L = ["# 工具索引",
         "",
         "> 由 `python tools/dev/gen_toolindex.py --write` **自动生成**，不要手改。",
         "",
         "`tools/` 是本项目「怎么量」的全部家当。每个回答一个具体问题的脚本都在这里，",
         "配套的用法说明见 `docs/DIAGNOSTICS.md`。",
         ""]

    n = 0
    for sub in sorted(p for p in TOOLS.iterdir() if p.is_dir()):
        files = sorted(sub.glob("*.py"))
        if not files:
            continue
        n += len(files)
        L += ["<a id=\"%s\"></a>" % sub.name, "",
              "## `tools/%s/`（%d 个脚本）" % (sub.name, len(files)), "",
              SUB_DESC.get(sub.name, ""), "",
              "| 脚本 | 行 | 说明 |", "|---|---:|---|"]
        for f in files:
            src = f.read_text(encoding="utf-8", errors="ignore")
            L.append("| `%s` | %d | %s |" % (f.stem, src.count("\n") + 1,
                                             first_doc(f).replace("|", "\\|")))
        L.append("")

    top = sorted(p for p in TOOLS.glob("*.py"))
    if top:
        L += ["## `tools/` 根目录", "", "| 脚本 | 行 | 说明 |", "|---|---:|---|"]
        for f in top:
            src = f.read_text(encoding="utf-8", errors="ignore")
            L.append("| `%s` | %d | %s |" % (f.stem, src.count("\n") + 1,
                                             first_doc(f).replace("|", "\\|")))
        L.append("")

    L.insert(6, "共 **%d** 个脚本。\n" % n)
    txt = "\n".join(L)
    if a.write:
        out = ROOT / "docs" / "TOOLS.md"
        out.write_text(txt, encoding="utf-8")
        print("wrote", out, "(%d scripts)" % n)
    else:
        print(txt)


if __name__ == "__main__":
    main()
