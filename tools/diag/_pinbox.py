# -*- coding: utf-8 -*-
"""核对 boxscan 的第 ③ 条：引脚取景框到底是不是 None。

`_wfall.md` 的 boxscan 节点列了「顺带要修」的第 3 条：
`claims.py` 里 `pinpts.get(cid)` 用的是 `cid`，而 `doc["pins"]` 的键是位号 ⇒ 取景框恒 None。
本脚本直接对 `runs/stage1/<case>.stage1.json` 验证 `pins` 的键到底是 cid 还是位号，
再顺带统计 `runs/stage2/_assigned/<case>.json` 里引脚槽位的个数。
"""
import json
import os
import sys

ROOT = r"Y:\workspace\EDA"
OUT = os.path.join(ROOT, "work", "_fast_out_reg9", "runs")


def main():
    cases = sys.argv[1:] or ["1", "4", "7", "10"]
    for case in cases:
        p = os.path.join(OUT, "stage1", "%s.stage1.json" % case)
        if not os.path.exists(p):
            print("%-5s stage1 产物不在 %s" % (case, p))
            continue
        d = json.load(open(p, encoding="utf-8"))
        pins = d.get("pins") or {}
        comps = d.get("components") or []
        cids = [c["id"] for c in comps]
        ks = list(pins)
        print("== case %s：%d 个元件 / pins 有 %d 个键" % (case, len(cids), len(ks)))
        print("   components[i].id 例子：", cids[:4])
        print("   pins 的键例子      ：", ks[:4])
        hit = [k for k in cids if k in pins]
        miss = [k for k in cids if k not in pins]
        print("   cid 能在 pins 里找到：%d/%d；找不到：%s"
              % (len(hit), len(cids), miss[:6]))
        # 有键的里面，点是不是真的（非 None）
        good = 0
        tot = 0
        for k in hit:
            for pt in pins[k]:
                tot += 1
                if pt and pt[0] is not None and pt[1] is not None:
                    good += 1
        print("   这些键下的点：%d/%d 是真的坐标" % (good, tot))
        # 位号当键能不能找到（取证：stage2 文档才是按位号）
        a = os.path.join(OUT, "stage2", "_assigned", "%s.json" % case)
        if os.path.exists(a):
            ad = json.load(open(a, encoding="utf-8"))

            def _dv(x):
                # assigned 里的 designator 可能是 {"text": ...} 也可能是裸字符串
                if isinstance(x, dict):
                    return x.get("text")
                return x

            desig = {_dv(c.get("designator")) for c in (ad.get("comps") or {}).values()}
            desig.discard(None)
            print("   assigned 里的位号例子：", sorted(desig)[:4],
                  "→ 位号当键命中 pins：%d" % len([x for x in desig if x in pins]))
        print()


if __name__ == "__main__":
    main()
