import json, re, pathlib

P = pathlib.Path(r'Y:\workspace\EDA\work\flow\workflow.html')
s = P.read_text(encoding='utf-8')
m = re.search(r'<script type="application/json" id="graphdata">(.*?)</script>', s, re.S)
raw = m.group(1)
d = json.loads(raw)

out = pathlib.Path(r'Y:\workspace\EDA\work\flow\_graphdata.json')
out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
print('graphdata ->', out, out.stat().st_size)

# JS part after the JSON
head = s[:m.start()]
tail = s[m.end():]
pathlib.Path(r'Y:\workspace\EDA\work\flow\_graphhead.html').write_text(head, encoding='utf-8')
pathlib.Path(r'Y:\workspace\EDA\work\flow\_graphtail.js').write_text(tail, encoding='utf-8')
print('head lines', head.count('\n'), 'tail lines', tail.count('\n'))
