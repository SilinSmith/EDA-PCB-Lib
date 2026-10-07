# -*- coding: utf-8 -*-
"""从 DSH 会话日志里回收被误改的源文件。

背景：2026-10-06 07:33 我用一段 PowerShell 做「假值词表统一」时写错了循环 ——
`$p[0]`/`$p[1]` 取到的是字符串的首两个字符（不是搜索串与替换串），于是对三个文件
做了**全局字符替换**：

    stage1\\pipeline.py :  'B' -> 'I'
    stage1\\wires.py    :  'i' -> 'f'
    stage1\\detect.py   :  'i' -> 'f'

会话日志（`%USERPROFILE%\\.dsh\\sessions\\--Y-workspace-EDA--\\*\\session.v4.jsonl`）
里按 seq 顺序保留了每一次 `write` 的**完整内容**、每一次 `edit` 的
`old_string`/`new_string`、以及每一次 `read` 的完整快照，因此可以精确重放到
损坏前一刻。

用法：
    python work\\tools\\recover\\replay.py            # 只报告
    python work\\tools\\recover\\replay.py --dump     # 导出到 work\\_recover\\
    python work\\tools\\recover\\replay.py --restore  # 直接写回源文件（先备份 .corrupt）
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

TARGETS = ("wires.py", "pipeline.py", "detect.py")

READ_MARK = "\\n<type>file</type>\\n<content>\\n"
LINE_RE = re.compile(r"^(\d+): ?(.*)$")
_NL = "\\n"


def parse_read_body(body: str):
    lines = {}
    for raw in body.split(_NL):
        m = LINE_RE.match(raw)
        if m:
            lines[int(m.group(1))] = m.group(2)
    return lines


def read_snapshots(line: str):
    """从一条日志行里抓出所有 read 结果：(basename, {行号: 文本})。"""
    out = []
    i = 0
    while True:
        p = line.find("</path>", i)
        if p < 0:
            break
        q = line.rfind("<path>", max(0, p - 500), p)
        path = line[q + 6:p] if q >= 0 else ""
        c = line.find(READ_MARK, p)
        if c < 0:
            i = p + 7
            continue
        c2 = line.find("</content>", c)
        if c2 < 0:
            i = p + 7
            continue
        lines = parse_read_body(line[c + len(READ_MARK):c2])
        if lines:
            out.append((os.path.basename(path.replace("\\\\", "\\")), lines))
        i = c2 + 10
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--root", default=str(OUT), help="存放已解压 *.jsonl 的目录")
    ap.add_argument("--only", default="", help="只处理文件名含该串的日志")
    a = ap.parse_args()

    events = []          # (seq, kind, payload)
    files = sorted(glob.glob(str(Path(a.root) / "**" / "*.jsonl"), recursive=True))
    if a.only:
        files = [f for f in files if a.only in os.path.basename(f)]
    for f in files:
        if os.path.getsize(f) == 0:
            continue
        with open(f, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                if ".py" not in raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                t = obj.get("type")
                seq = obj.get("seq", 0)
                data = obj.get("data") or {}
                if t == "tool/call":
                    nm = data.get("name")
                    if nm not in ("write", "edit"):
                        continue
                    try:
                        args = json.loads(data.get("arguments") or "{}")
                    except Exception:
                        continue
                    fp = str(args.get("file_path") or "")
                    base = os.path.basename(fp)
                    if base not in TARGETS:
                        continue
                    if nm == "write":
                        events.append((seq, "write", (base, args.get("content") or "", os.path.basename(f))))
                    else:
                        events.append((seq, "edit", (base, args.get("old_string") or "",
                                                     args.get("new_string") or "", os.path.basename(f))))
                else:
                    for base, lines in read_snapshots(raw):
                        if base in TARGETS:
                            events.append((seq, "read", (base, lines)))
    events.sort(key=lambda e: e[0])
    print("事件数 %d" % len(events))

    state = {}
    for seq, kind, payload in events:
        if kind == "write":
            base, content, src = payload
            state[base] = content
            print("seq=%d write %-12s %d 字符  (%s)" % (seq, base, len(content), src[:18]))
        elif kind == "edit":
            base, old, new, src = payload
            cur = state.get(base)
            if cur is None:
                print("seq=%d edit  %-12s [无基线，跳过]" % (seq, base))
                continue
            n = cur.count(old)
            if n == 1:
                state[base] = cur.replace(old, new)
                print("seq=%d edit  %-12s ok" % (seq, base))
            else:
                print("seq=%d edit  %-12s 命中 %d 次 %s" % (seq, base, n, "（跳过）" if n == 0 else "（全部替换）"))
                if n:
                    state[base] = cur.replace(old, new)
        else:
            base, lines = payload
            got = state.get(base)
            mx = max(lines)
            if got is None or mx > got.count("\n"):
                state[base] = "\n".join(lines[i] for i in range(1, mx + 1))
                print("seq=%d read  %-12s 行数=%d（用作基线）" % (seq, base, len(lines)))

    OUT.mkdir(parents=True, exist_ok=True)
    for base, txt in state.items():
        print("--- %-12s %d 字符 / %d 行" % (base, len(txt), txt.count("\n") + 1))
        if a.dump or a.restore:
            p = OUT / ("recovered_" + base)
            p.write_text(txt, encoding="utf-8")
            print("    wrote %s" % p)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
