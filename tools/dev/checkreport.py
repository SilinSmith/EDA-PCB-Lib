# -*- coding: utf-8 -*-
"""校验生成的 Word 报告结构（段落/章节/表格/装饰残留）。"""
import re
import sys

import docx

p = sys.argv[1] if len(sys.argv) > 1 else r"eda-pcb\docs\技术报告-赛题六.docx"
d = docx.Document(p)
ps = [x.text for x in d.paragraphs]
print("段落", len(ps), "| 表格", len(d.tables), "| 正文字符", sum(len(x) for x in ps))
tick = chr(96)
bad = [x for x in ps if tick in x or "**" in x]
print("markdown 残留段落:", len(bad))
for x in bad[:5]:
    print("   !", x[:90])
heads = [x for x in ps if re.match(r"^[0-9]+(\.[0-9]+)*\s|^附录", x)]
print("章节标题:", len(heads))
for x in heads:
    print("   ", x)
print("表格行数:", [len(t.rows) for t in d.tables])
print("空段落(分页/间距):", sum(1 for x in ps if not x.strip()))
