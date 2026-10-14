# -*- coding: utf-8 -*-
"""把 .docx 按「段落 / 表格」顺序转成纯文本（只用标准库）。

用法：python -X utf8 work\\_docxdump.py <in.docx> <out.txt>
"""
import sys
import xml.etree.ElementTree as ET
import zipfile

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def para_text(p):
    out = []
    for node in p.iter():
        if node.tag == W + "t":
            out.append(node.text or "")
        elif node.tag == W + "tab":
            out.append("\t")
        elif node.tag == W + "br":
            out.append("\n")
    return "".join(out)


def walk(container, lines):
    for child in container:
        if child.tag == W + "p":
            lines.append(para_text(child))
        elif child.tag == W + "tbl":
            for row in child.findall(W + "tr"):
                cells = []
                for cell in row.findall(W + "tc"):
                    sub = []
                    walk(cell, sub)
                    cells.append(" ".join(s.strip() for s in sub if s.strip()))
                lines.append(" | ".join(cells))
            lines.append("")


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with zipfile.ZipFile(src) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    body = root.find(W + "body")
    lines = []
    walk(body, lines)
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("lines=%d chars=%d -> %s" % (len(text.split("\n")), len(text), dst))


if __name__ == "__main__":
    main()
