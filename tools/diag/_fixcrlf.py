# -*- coding: utf-8 -*-
"""把 `eda-pcb/docs/RESULTS.md` 的孤立 LF 统一成 CRLF（保持无 BOM）。"""
from __future__ import print_function

import io
import sys

sys.stdout.reconfigure(encoding="utf-8")
T = r"Y:\workspace\EDA\eda-pcb\docs\RESULTS.md"

raw = open(T, "rb").read()
lone = raw.replace(b"\r\n", b"").count(b"\n")
text = raw.decode("utf-8-sig")
text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
with io.open(T, "w", encoding="utf-8", newline="") as fh:
    fh.write(text)
after = open(T, "rb").read()
print("孤立 LF：%d → %d ｜BOM：%s ｜字节：%d → %d"
      % (lone, after.replace(b"\r\n", b"").count(b"\n"),
         after[:3] == b"\xef\xbb\xbf", len(raw), len(after)))
