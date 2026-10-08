import re, pathlib, collections

pat = re.compile(r"""(?:os\.environ\.get|env_on|envflag\.\w+)\s*\(\s*["']([A-Z][A-Z0-9_]{3,})["']\s*(?:,\s*([^)]{0,40}))?""")
d = collections.defaultdict(dict)
pkgs = ['stage1', 'stage2', 'stage3', 'fusion', 'vlm', 'textrec',
        'cvparts', 'pipeline', 'submission', 'schematic_parser', 'synth']
for pkg in pkgs:
    for p in sorted(pathlib.Path(pkg).rglob('*.py')):
        s = p.read_text(encoding='utf-8', errors='ignore')
        for m in pat.finditer(s):
            d[pkg].setdefault(m.group(1), (m.group(2) or '').strip()[:34])
for pkg in pkgs:
    if not d[pkg]:
        continue
    print('##', pkg, len(d[pkg]))
    for k, v in sorted(d[pkg].items()):
        print('   %-34s %s' % (k, v))
allk = set(k for v in d.values() for k in v)
print()
print('TOTAL', len(allk))
