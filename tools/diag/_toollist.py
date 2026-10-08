import ast, pathlib, sys

ROOT = pathlib.Path(r"Y:\workspace\EDA\eda-pcb\tools")
for sub in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    files = sorted(sub.glob("*.py"))
    if not files:
        continue
    print("##", sub.name, len(files))
    for f in files:
        try:
            d = (ast.get_docstring(ast.parse(f.read_text(encoding="utf-8", errors="ignore"))) or "")
        except Exception:
            d = ""
        first = ""
        for ln in d.split("\n"):
            if ln.strip():
                first = ln.strip()
                break
        n = len(f.read_text(encoding="utf-8", errors="ignore").split("\n"))
        print("   %-30s %4d  %s" % (f.stem, n, first[:95]))
