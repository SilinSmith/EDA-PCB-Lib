"""逐槽位对照表：**GT / 我们（stage2 装配）/ VLM 引脚通道** 三方的 (号, 名)。

用法：
    python work\\tools\\diag\\_pincase.py 0105 [U1,U2]       # 派生集
    python work\\tools\\diag\\_pincase.py 1 --official        # 官方 10GT

为什么要有这个：`metrics.pin_f1()` 按 **(元件, 键里的数字)** 配对、再比 `pinname`
（`schematic_parser/metrics.py:442-456`）。一个槽位有「号」和「名」两半，VLM 引脚
通道两半都能改；要判断某张图**掉分是哪一半造成的**，只看总分不够，得看逐槽位。

三方数据来源：
  * GT     —— `10GT\\<case>\\<case>.json` 或 `200_train_cases\\<case>\\*_target_named.json`
  * 我们   —— `work\\_f220_A\\<case>.json`（= 引脚通道关着的交付预测，含号与名）
  * VLM    —— `work\\_vlm_pinfix220\\<case>.json` 的 `pins`（号只活在键名 `pin_<号>`）
             与该文档里 `comps[<cid>]` 的 `pins`（enrich 重写过的那份）
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

ROOT = r"Y:\workspace\EDA"
DER = os.path.join(ROOT, "eda-pcb", "yolo_part",
                   "Dataset(waitforpreprocess)", "200_train_cases")
OFF = os.path.join(ROOT, "eda-pcb", "yolo_part",
                   "Dataset(waitforpreprocess)", "10GT")
A = os.path.join(ROOT, "work", "_f220_A")
VLM = os.path.join(ROOT, "work", "_vlm_pinfix220")


def keynum(pk):
    m = re.search(r"(\d+)", str(pk))
    return int(m.group(1)) if m else None


def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return None


def gt_of(case, official):
    if official:
        return load(os.path.join(OFF, case, case + ".json"))
    g = glob.glob(os.path.join(DER, case, "*_target_named.json"))
    return load(g[0]) if g else None


def main(argv):
    case = argv[0]
    official = "--official" in argv
    want = None
    for a in argv[1:]:
        if not a.startswith("--"):
            want = {x.strip() for x in a.split(",") if x.strip()}
    gt = gt_of(case, official)
    if gt is None:
        print("!! 找不到 GT", case)
        return 1
    ours = load(os.path.join(A, case + ".json")) or {}
    vlm = load(os.path.join(VLM, case + ".json")) or {}
    print("case %s | GT %s" % (case, "官方" if official else "派生"))

    def nz(d, cid):
        """把 `{cid: {pin_k: info}}` 或 `{cid: {k: info}}` 归一成 {num: info}。"""
        g = (d.get("pins") or {}).get(cid) or {}
        out = {}
        for k, v in g.items():
            n = keynum(k)
            if n is not None:
                out[n] = v or {}
        return out

    gtset = gt.get("pins") or {}
    for cid in sorted(gtset):
        if want and cid not in want:
            continue
        g = nz(gt, cid)
        o = nz(ours, cid)
        v = nz(vlm, cid)
        if not g and not o and not v:
            continue
        gn = [str((x or {}).get("pinname")) for x in g.values()]
        print("\n== %s  GT %d 个键 %s | 名 %s"
              % (cid, len(g), sorted(g)[:16], gn[:8]))
        # 我们的：`pins` 是点表，号/名在 `components[cid]` 里？—— A 目录是预测文档，
        # 与 GT 同构，所以直接读
        for n in sorted(set(g) | set(o) | set(v)):
            row = "  键 %-5s" % n
            row += " GT:%-10s" % (repr(g[n].get("pinname"))[1:-1] if n in g else "—")
            row += " 我们:%-10s" % (repr(o[n].get("pinname"))[1:-1]
                                    if n in o else "—")
            # VLM 的键就是它的号；名在 pinname
            row += " VLM:%-10s" % (repr(v[n].get("pinname"))[1:-1]
                                   if n in v else "—")
            mark = ""
            if n in g and n not in o:
                mark = "  ← GT 有、我们缺"
            print(row + mark)
        # 点对齐：我们的槽 k -> VLM 的槽，看「同一个物理点」上三方的名
        po = (ours.get("pins") or {}).get(cid) or {}
        pv = (vlm.get("pins") or {}).get(cid) or {}
        if po and pv:
            h = float(((ours.get("image") or {}).get("h")) or 0.0)
            print("  -- 按点对齐（我们槽 -> VLM 槽，距离<3px）--")
            for ko, io in (po.items() if isinstance(po, dict) else []):
                try:
                    qx = float((io or {}).get("point", {}).get("x"))
                    qy = float((io or {}).get("point", {}).get("y"))
                except Exception:                              # noqa: BLE001
                    continue
                num = keynum(ko)
                best, bd = None, 9e9
                for kv, iv in (pv.items() if isinstance(pv, dict) else []):
                    try:
                        dx = float((iv or {}).get("point", {}).get("x"))
                        dy = float((iv or {}).get("point", {}).get("y"))
                    except Exception:                          # noqa: BLE001
                        continue
                    d = abs(dx - qx) + abs(dy - (h - qy) if False else dy - qy)
                    if d < bd:
                        best, bd = kv, d
                if best is None or bd > 3.0:
                    continue
                gname = (g.get(num) or {}).get("pinname")
                print("     我们键 %-5s 名=%-8s | VLM 键 %-6s 名=%-8s | GT(键%s) 名=%s"
                      % (num, (o.get(num) or {}).get("pinname"),
                         keynum(best), (v.get(keynum(best)) or {}).get("pinname"),
                         num, gname))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
