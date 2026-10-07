import pathlib, re, glob
bad=0
for p in glob.glob("work/viz/*.html"):
    t=pathlib.Path(p).read_text(encoding="utf-8")
    m=re.findall(r"__[A-Z]+__", t)
    if m: print(p, set(m)); bad+=1
print("files with leftovers:", bad)
