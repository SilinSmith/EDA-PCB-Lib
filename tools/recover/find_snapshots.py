# -*- coding: utf-8 -*-
"""从 DSH 会话日志里回收被误改的源文件。

背景：2026-10-06 07:33 我用一段 PowerShell 做「假值词表统一」时写错了循环，
把 `$p[0]`/`$p[1]` 当成了搜索/替换串，实际取到的是字符串的首两个字符，
于是对三个文件做了**全局字符替换**：

    stage1\\pipeline.py :  'B' -> 'I'
    stage1\\wires.py    :  'i' -> 'f'
    stage1\\detect.py   :  'i' -> 'f'

会话日志（`%USERPROFILE%\\.dsh\\sessions\\--Y-workspace-EDA--\\*\\session.v4.jsonl`）
里保留了**每一次 write 的完整内容**与**每一次 read 的完整快照**，所以可以精确回收。

用法：
    python work\\tools\\recover\\find_snapshots.py            # 列出所有快照
    python work\\tools\\recover\\find_snapshots.py --dump     # 导出最佳快照到 work\\_recover\\
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

LOG_ROOT = Path(os.path.expanduser("~")) / ".dsh" / "sessions" / "--Y-workspace-EDA--"
OUT = Path(r"Y:\workspace\EDA\work\_recover")

TARGETS = {
    "wires.py": r"stage1\wires.py",
    "pipeline.py": r"stage1\pipeline.py",
    "detect.py": r"stage1\detect.py",
}

# 会话日志里 read 结果的形状（原始 JSONL 文本，换行被转义成两字符 `\n`）：
#   <path>...</path>|<type>file</type>|<content>\n1: xxx\n2: yyy
READ_RE = re.compile(r"<path>([^<]*\\([^\\<>]+\.py))</path><type>file</type><content>\\n(.*?)</content>",
                     re.S)
LINE_RE = re.compile(r"^(\d+): ?(.*)$")
_NL = "\\n"          # 字面量反斜杠 + n（JSON 转义后的换行）


def parse_read(body: str):
    """把 read 结果的 `\\nN: text` 还原成 {行号: 文本}。"""
    lines = {}
    body = body.replace("|1: ", _NL + "1: ")     # 保险：极少数情况下分隔符是竖线
    for raw in body.split(_NL):
        m = LINE_RE.match(raw)
        if not m:
            continue
        lines[int(m.group(1))] = m.group(2)
    return lines


def iter_log_lines():
    for f in sorted(glob.glob(str(LOG_ROOT / "**" / "*.jsonl"), recursive=True)):
        if os.path.getsize(f) == 0:
            continue
        with open(f, encoding="utf-8", errors="replace") as fh:
            for ln, line in enumerate(fh):
                yield f, ln, line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--prefix", default="", help="只处理文件名含该串的日志")
    a = ap.parse_args()

    best: dict = {}          # fname -> (time, seq, nlines, maxline, lines)
    for f, ln, line in iter_log_lines():
        if a.prefix and a.prefix not in os.path.basename(f):
            continue
        for m in READ_RE.finditer(line):
            path, body = m.group(1), m.group(2)
            base = os.path.basename(path.replace("\\\\", "\\"))
            if base not in TARGETS:
                continue
            lines = parse_read(body)
            if not lines:
                continue
            try:
                obj = json.loads(line)
                t = obj.get("time", 0)
                seq = obj.get("seq", 0)
            except Exception:
                t, seq = 0, 0
            prev = best.get(base)
            key = (len(lines), max(lines))
            if prev is None or key > (len(prev[3]), max(prev[3])):
                best[base] = (t, seq, os.path.basename(f), lines)
                print("[best] %-12s t=%d seq=%d %s  行数=%d 最大行号=%d"
                      % (base, t, seq, os.path.basename(f)[:20], len(lines), max(lines)))

    if a.dump and best:
        OUT.mkdir(parents=True, exist_ok=True)
        for base, (t, seq, src, lines) in best.items():
            p = OUT / ("recovered_" + base)
            mx = max(lines)
            txt = "\n".join(lines.get(i, "") for i in range(1, mx + 1))
            p.write_text(txt, encoding="utf-8")
            print("wrote %s  (%d 行, %d 字符, 缺口 %d 行)"
                  % (p, mx, len(txt), mx - len(lines)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
