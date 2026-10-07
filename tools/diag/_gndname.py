import sys, json, re
from collections import Counter
from pathlib import Path
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.stdout.reconfigure(encoding="utf-8")
from schematic_parser import data as S

cls = Path(r"Y:\workspace\EDA\PCB训练集\classes.txt").read_text(encoding="utf-8").split()
print("classes containing gnd/ground/pwr/power:", [c for c in cls if re.search("gnd|grd|ground|pwr|power|vss|earth", c, re.I)])
print("total classes:", len(cls))

hits = Counter(); vals = Counter(); namepat = Counter()
for c in S.discover(S.DEFAULT_DATA_ROOT):
    try: doc = S.load_json(c.json_path)
    except Exception: continue
    for k, v in (doc.get("components") or {}).items():
        if str(v.get("type","")).strip() != "gnd": continue
        nm = str(v.get("Name") or ""); va = str(v.get("value") or "")
        hits[type(v.get("Name")).__name__] += 1
        for s in (nm, va):
            m = re.search(r"(A?D?P?GND|VSS|V_?SS|EARTH|GNDA|GNDD)", s, re.I)
            if m: vals[m.group(1).upper()] += 1
        if re.fullmatch(r"\u2205\d+", nm): namepat["empty-set-prefixed"] += 1
        elif re.fullmatch(r"[A-Za-z]*\d+", nm): namepat["designator-like"] += 1
        else: namepat["other"] += 1
print("Name python type count:", dict(hits))
print("gnd-text tokens found in Name/value:", dict(vals))
print("Name shape:", dict(namepat))
