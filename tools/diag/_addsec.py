# -*- coding: utf-8 -*-
"""把一节 markdown 追加进 `eda-pcb/docs/RESULTS.md`（保持 CRLF、无 BOM），带重复自检。

用法：`python -X utf8 work/_addsec.py --file work/_sec5120.md`
自检要点：RESULTS.md 的行尾是 CRLF；用 `splitlines()` 精确整行匹配会踩空，
必须用 `startswith`。追加前先确认这节的标题不在文件里。
"""
from __future__ import print_function

import argparse
import io
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
TARGET = r"Y:\workspace\EDA\eda-pcb\docs\RESULTS.md"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    with io.open(a.file, "r", encoding="utf-8") as fh:
        body = fh.read().replace("\r\n", "\n")
    body = body.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
    first = body.lstrip("\n").split("\n", 1)[0].strip()
    head = first[:24]
    raw = open(a.target, "rb").read()
    has_bom = raw[:3] == b"\xef\xbb\xbf"
    lone_lf = raw.replace(b"\r\n", b"").count(b"\n")
    text = raw.decode("utf-8-sig")
    dup = any(ln.startswith(head) for ln in text.splitlines())
    if dup and not a.force:
        print("！已经存在以 %r 开头的行，未追加（--force 可强行追加）" % head)
        return 1
    if not text.endswith("\n"):
        text += "\n"
    if not text.endswith("\n\n"):
        text += "\n"
    text += body.rstrip("\n") + "\n"
    with io.open(a.target, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    after = open(a.target, "rb").read()
    print("追加完成：%s" % a.file)
    print("  目标：%s" % a.target)
    print("  标题行：%s" % first)
    print("  大小：%d → %d 字节" % (len(raw), len(after)))
    print("  BOM：%s（追加前 %s）｜单个 LF：%d → %d"
          % (after[:3] == b"\xef\xbb\xbf", has_bom, lone_lf,
             after.replace(b"\r\n", b"").count(b"\n")))
    lines = after.decode("utf-8-sig").split("\n")
    print("  文件行数：%d" % len(lines))
    for i, ln in enumerate(lines, 1):
        if ln.startswith(head):
            print("  标题所在行：%d" % i)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
