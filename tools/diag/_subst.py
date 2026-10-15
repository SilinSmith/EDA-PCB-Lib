# -*- coding: utf-8 -*-
"""按「字面子串」替换文件内容（避开 PowerShell 引号地狱）。

用法：
    python work/tools/diag/_subst.py --file <路径> --old-file <存旧串的文件> --new-file <存新串的文件>
或直接 import 用 `subst(path, pairs)`。
"""
import argparse
import io
import sys
from pathlib import Path


def subst(path, pairs, write=True):
    """pairs = [(old, new), ...]；返回 (命中数, 总对数, 文本)。"""
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    hits = 0
    for old, new in pairs:
        n = s.count(old)
        if n:
            s = s.replace(old, new)
            hits += 1
        else:
            print("MISS: %r" % (old[:60],))
    if write and hits:
        p.write_text(s, encoding="utf-8", newline="")
    return hits, len(pairs), s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--old-file", required=True)
    ap.add_argument("--new-file", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    old = io.open(a.old_file, encoding="utf-8").read()
    new = io.open(a.new_file, encoding="utf-8").read()
    h, t, _ = subst(a.file, [(old, new)], write=not a.dry)
    print("命中 %d/%d" % (h, t))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
