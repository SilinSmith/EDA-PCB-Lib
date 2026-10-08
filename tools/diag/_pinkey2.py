# -*- coding: utf-8 -*-
"""stage2 文档里 `pins` 是按 cid 还是按位号存的？以及 `assigned['comps'][cid]['pins']` 的规模。

    python work\\tools\\diag\\_pinkey2.py [stage2doc.json ...]
"""
import json, sys, os, glob
sys.stdout.reconfigure(encoding="utf-8")

paths = sys.argv[1:]
if not paths:
    paths = sorted(glob.glob(r"Y:\workspace\EDA\work\_submit_out220r36\runs\stage2\*.json"))
    paths = [p for p in paths if "_assigned" not in p][:3]
for p in paths:
    print("=" * 72)
    print(p)
    d = json.load(open(p, encoding="utf-8"))
    print("顶层键：%s" % sorted(d.keys()))
    pins = d.get("pins") or {}
    comps = d.get("components") or []
    cids = [c.get("id") if isinstance(c, dict) else c for c in comps]
    print("components %d 个，id 前 8：%s" % (len(comps), cids[:8]))
    print("pins 类型 %s，%d 个键，前 10：%s"
          % (type(pins).__name__, len(pins), list(pins)[:10]))
    if pins:
        k0 = list(pins)[0]
        print("  pins[%r] = 前 200 字：%s"
              % (k0, json.dumps(pins[k0], ensure_ascii=False)[:200]))
        inter = set(pins) & set(cids)
        print("  ★ pins 的键里有 %d 个能对上 component 的 id（共 %d 个 component）"
              % (len(inter), len(comps)))
    else:
        print("  ★ pins 是空的")
    # 每颗元件的引脚数
    if comps and isinstance(comps[0], dict):
        c0 = comps[0]
        print("  第一个元件字段：%s" % sorted(c0.keys()))
        for f in ("pins", "pin_count", "pins_img", "pins_json"):
            if f in c0:
                v = c0[f]
                print("    %s -> %s %s" % (f, type(v).__name__,
                      (len(v) if hasattr(v, "__len__") else v)))
    # GT 的 pins 结构
    print()


print("#" * 72)
print("对照：官方 GT 的 pins 结构")
gt = sorted(glob.glob(r"Y:\workspace\EDA\eda-pcb\yolo_part\Dataset(waitforpreprocess)\10GT\*\*.json"))
gt = [g for g in gt if "_Pred" not in g]
for g in gt[:1]:
    print(g)
    G = json.load(open(g, encoding="utf-8"))
    print("GT 顶层键：%s" % sorted(G.keys()))
    gp = G.get("pins")
    if isinstance(gp, dict):
        print("GT pins 键前 10：%s" % list(gp)[:10])
    elif isinstance(gp, list):
        print("GT pins 是 list，%d 条，第一条：%s"
              % (len(gp), json.dumps(gp[0], ensure_ascii=False)[:200]))
