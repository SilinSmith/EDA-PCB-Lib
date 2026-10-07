# -*- coding: utf-8 -*-
"""把某个 case 的几个图层拍平到一张图上看效果（给开发期自检用）。

用法：python work\\tools\\viz\\_peek.py <case> [layer ...]
默认 layer = gt ours texts regions pins net
"""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
VIZ = ROOT / "work" / "viz"


def main():
    cid = sys.argv[1]
    layers = sys.argv[2:] or ["gt", "ours", "texts", "regions", "pins", "net"]
    d = VIZ / "assets" / cid
    base = Image.open(d / "base.png").convert("RGBA")
    for L in layers:
        p = d / (L + ".png")
        if not p.exists():
            print("  跳过缺文件 %s" % p.name)
            continue
        base = Image.alpha_composite(base, Image.open(p).convert("RGBA"))
    out = VIZ / ("_peek_%s.png" % cid)
    base.convert("RGB").save(out)
    print("→ %s  %s" % (out, base.size))


if __name__ == "__main__":
    main()
