# -*- coding: utf-8 -*-
"""开关普查：把 eda-pcb 里所有「环境开关」连默认值一起列出来，判定默认开/关。

目的：回答「链路里哪些节点根本没发挥作用」。默认关的开关 = 静默失效的节点/子节点。
用法:  python work/tools/diag/_flags.py [关键字过滤]
"""
import re
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA\eda-pcb")
SKIP = {"eda_yolo", "yolo_part", ".git", "__pycache__", "synth"}

# env_on("NAME", "1") / env_off("NAME","0") / env_str("NAME","cap") / env_raw(...)
PAT_FN = re.compile(r'\benv_(on|off|str|raw)\(\s*"([A-Z0-9_]+)"\s*(?:,\s*"([^"]*)")?')
# os.environ.get("NAME", "default")  /  os.environ.get("NAME","default")
PAT_OS = re.compile(r'os\.environ\.get\(\s*"([A-Z0-9_]+)"\s*(?:,\s*(?:"([^"]*)"|([A-Za-z_][\w.]*)))?')

FALSE = frozenset(("", "0", "off", "false", "none", "no", "disable", "disabled"))


def norm(v):
    return ("" if v is None else str(v)).strip().lower()


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else None
    rows = {}
    for p in sorted(ROOT.rglob("*.py")):
        if any(s in p.parts for s in SKIP):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = p.relative_to(ROOT).as_posix()
        for i, line in enumerate(txt.splitlines(), 1):
            for m in PAT_FN.finditer(line):
                kind, name, dflt = m.group(1), m.group(2), m.group(3)
                if kind == "off":
                    on = norm(dflt) == "0" if dflt is not None else True
                elif kind == "on":
                    on = norm(dflt if dflt is not None else "1") not in FALSE
                else:
                    on = None       # 字符串模式：只要非空即开
                rows.setdefault(name, []).append((rel, i, kind, dflt, on))
            for m in PAT_OS.finditer(line):
                name, dflt, ident = m.group(1), m.group(2), m.group(3)
                if ident is not None:
                    on = None
                else:
                    on = norm(dflt if dflt is not None else "") not in FALSE
                rows.setdefault(name, []).append((rel, i, "os.get", dflt, on))

    off, on_, unknown = [], [], []
    for name, hits in sorted(rows.items()):
        if filt and filt.upper() not in name:
            continue
        verdicts = [h[4] for h in hits]
        if all(v is False for v in verdicts if v is not None) and any(v is not None for v in verdicts):
            off.append((name, hits))
        elif all(v is True for v in verdicts if v is not None) and any(v is not None for v in verdicts):
            on_.append((name, hits))
        else:
            unknown.append((name, hits))

    def dump(title, items):
        print("\n===== %s（%d）=====" % (title, len(items)))
        for name, hits in items:
            locs = ", ".join("%s:%d" % (h[0], h[1]) for h in hits[:4])
            d = hits[0][3]
            print("  %-34s 默认=%-8s %s" % (name, repr(d), locs))

    dump("⚠ 默认【关】—— 静默失效", off)
    dump("默认【开】", on_)
    dump("多档/未定（字符串模式或变量默认）", unknown)
    print("\n合计开关名 %d 个（默认关 %d）" % (len(off) + len(on_) + len(unknown), len(off)))


if __name__ == "__main__":
    main()
