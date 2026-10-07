# -*- coding: utf-8 -*-
"""Check image properties across the dataset: size, mode, colour palette, uniqueness."""
import os, glob, collections, sys
from PIL import Image

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

DS = r'Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)'

def probe(p):
    im = Image.open(p)
    return im

groups = collections.defaultdict(list)
for sub in ['10GT', '10_GTcase', '200_train_cases']:
    for p in glob.glob(os.path.join(DS, sub, '*', '*.png')):
        groups[sub].append(p)

for sub, ps in groups.items():
    print('=' * 70)
    print(sub, len(ps), 'images')
    sizes = collections.Counter()
    modes = collections.Counter()
    for p in ps:
        im = probe(p)
        sizes[im.size] += 1
        modes[im.mode] += 1
    print('  modes:', dict(modes))
    print('  distinct sizes:', len(sizes))
    for s, n in sizes.most_common(8):
        print('     %sx%s  x%d' % (s[0], s[1], n))

# colour analysis of one representative image from each source
print('=' * 70)
samples = [
    os.path.join(DS, '10GT', '1', '1.png'),
    os.path.join(DS, '200_train_cases', '0001', '0001-KiCad.png'),
    os.path.join(DS, '200_train_cases', '0002', '0002-Datasheet.png'),
]
for p in samples:
    im = Image.open(p).convert('RGB')
    cols = collections.Counter(im.getdata())
    print('---', os.path.basename(p), im.size, 'distinct colours:', len(cols))
    for c, n in cols.most_common(12):
        print('     rgb%-18s %8d  (%.2f%%)' % (str(c), n, 100.0 * n / (im.size[0] * im.size[1])))
