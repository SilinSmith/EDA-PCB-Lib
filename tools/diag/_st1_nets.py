# -*- coding: utf-8 -*-
"""stage1 的 nets / wires 结构摸底（第二阶段组装要用）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ST1 = Path(r"Y:\workspace\EDA\work\stage1_all14")


def main():
    for name in ("0059", "0050"):
        p = ST1 / ("%s.stage1.json" % name)
        if not p.is_file():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        print("=" * 60)
        print(name, "image", d.get("image"))
        w = d.get("wires") or {}
        print("wires keys:", list(w.keys()))
        for k in ("segments", "n_segments", "n_polylines", "scale"):
            if k in w:
                v = w[k]
                print("  wires[%s] = %s" % (k, len(v) if isinstance(v, list) else v))
        nets = d.get("nets") or []
        print("nets n=%d" % len(nets))
        for i in (0, 1, 2):
            if i < len(nets):
                s = json.dumps(nets[i], ensure_ascii=False)
                print("  net[%d]: %s" % (i, s[:300]))
        sizes = sorted((len(n.get("pins") or []) for n in nets), reverse=True)
        print("  每 net 引脚数 top8:", sizes[:8],
              " 单引脚 net 占比 %.1f%%"
              % (100.0 * sum(1 for s in sizes if s == 1) / max(1, len(sizes))))
        pins = d.get("pins") or {}
        comps = d.get("components") or []
        empty = [c["id"] for c in comps if not (pins.get(c["id"]) or [])]
        print("  components=%d  pins groups=%d  无引脚元件=%d"
              % (len(comps), len(pins), len(empty)))
        if comps:
            print("  comp0:", json.dumps(comps[0], ensure_ascii=False)[:260])


if __name__ == "__main__":
    main()
