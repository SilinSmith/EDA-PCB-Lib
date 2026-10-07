import re
from pathlib import Path
s = Path(r"Y:\workspace\EDA\work\flow\workflow.html").read_text(encoding="utf-8")
m = re.search(r'<script type="application/json" id="graphdata">(.*?)</script>', s, re.S)
out = s[:m.start()] + "<<<GRAPHDATA>>>" + s[m.end():]
# strip <style> and <script> bodies for readability
out = re.sub(r"<style>.*?</style>", "<style>…</style>", out, flags=re.S)
out = re.sub(r"<script>.*?</script>", "<script>…</script>", out, flags=re.S)
print("non-graphdata chars:", len(out))
print(out)
