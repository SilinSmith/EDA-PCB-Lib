# -*- coding: utf-8 -*-
"""stage1 产物 schema 摸底 + 官方 GT 的 Name/位号约定。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ST1 = Path(r"Y:\workspace\EDA\work\stage1_all14")
GT_ROOT = Path(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)")


def main():
    p = ST1 / "0050.stage1.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    print("top keys:", list(d.keys()))
    t = d.get("texts")
    print("texts: %s n=%s" % (type(t).__name__, len(t) if t else 0))
    if t:
        ex = t[:3] if isinstance(t, list) else list(t.items())[:3]
        print("  sample:", json.dumps(ex, ensure_ascii=False)[:300])
    for k in ("pins", "nets"):
        v = d.get(k)
        print("%s: %s n=%s" % (k, type(v).__name__, len(v) if v else 0))
        if isinstance(v, dict) and v:
            kk = list(v)[0]
            print("  sample:", json.dumps({kk: v[kk]}, ensure_ascii=False)[:400])
    wd = d.get("wire_detail") or {}
    print("wire_detail keys:", list(wd.keys()))
    seg = (wd.get("segments") or [None])[0]
    print("  seg0:", seg)
    print("  groups:", type(wd.get("groups")).__name__,
          (len(wd["groups"]) if wd.get("groups") else 0))

    # ---- GT 的 Name 与 key 的关系 ----
    same = Counter()
    tot = Counter()
    fake = Counter()
    val_ex = {}
    for split in ("10GT", "10_GTcase", "200_train_cases"):
        for jf in (GT_ROOT / split).glob("*/*.json"):
            try:
                g = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            for key, c in (g.get("components") or {}).items():
                ty = c.get("type")
                tot[ty] += 1
                if key.startswith("\u2205"):
                    fake[ty] += 1
                if str(c.get("Name")) == key:
                    same[ty] += 1
                if ty not in val_ex:
                    val_ex[ty] = {"key": key, "Name": c.get("Name"),
                                  "value": c.get("value"), "bbox": c.get("bbox")}
    print("\n=== GT Name == key 的比例（按 type） ===")
    print("%-16s %6s %6s %8s %s" % ("type", "n", "假号∅", "Name==key", "样例(key/Name/value)"))
    for ty, n in tot.most_common(18):
        e = val_ex.get(ty, {})
        s = "%s / %s / %s" % (e.get("key"), e.get("Name"), e.get("value"))
        print("%-16s %6d %6d %7.1f%%  %s"
              % (ty, n, fake[ty], 100.0 * same[ty] / max(1, n), s[:70]))
    print("总计 %d，假号 %d，Name==key %d"
          % (sum(tot.values()), sum(fake.values()), sum(same.values())))


if __name__ == "__main__":
    main()
