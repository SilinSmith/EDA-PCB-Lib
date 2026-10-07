import pathlib, re
t = pathlib.Path("work/viz/index.html").read_text(encoding="utf-8")
print("len", len(t))
for m in re.finditer(r"<h3>(.*?)</h3>", t): print("  §", m.group(1))
rows = re.findall(r"<tr class='it'>(.*?)</tr>", t, re.S)
print("rows:", len(rows))
for r in rows[:4]:
    cells = re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)
    print("   ", " | ".join(re.sub(r"<[^>]+>", "", c) for c in cells))
