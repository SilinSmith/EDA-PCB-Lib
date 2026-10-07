import pathlib, re
t = pathlib.Path("work/viz/case_0001.html").read_text(encoding="utf-8")
for m in re.finditer(r"__[A-Z]+__", t):
    print("hit:", m.group(0), "| ctx:", t[max(0,m.start()-70):m.end()+40].replace("\n","\\n"))
print("total", len(re.findall(r"__[A-Z]+__", t)))
