import re
from pathlib import Path
s = Path(r"Y:\workspace\EDA\work\flow\workflow.html").read_text(encoding="utf-8")
m = re.search(r"<script>(.*?)</script>", s, re.S)
js = m.group(1)
print("js chars:", len(js))
i = js.find("showChanges")
print(js[max(0,i-200): i+2500])
