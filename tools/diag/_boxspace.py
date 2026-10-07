# -*- coding: utf-8 -*-
"""确认 `_assigned` 里各种 box 的坐标空间（图像空间 vs JSON 空间）。

判据：拿同一条目的 box 与它所属元件的 `bbox_img` / `bbox_json` 比「哪个包含它」。
用法：python work\\tools\\diag\\_boxspace.py [case ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "work" / "_submit_out220r36" / "runs"


def inside(inner, outer):
    if not inner or not outer:
        return None
    return (outer[0] - 2 <= inner[0] and outer[1] - 2 <= inner[1]
            and inner[2] <= outer[2] + 2 and inner[3] <= outer[3] + 2)


def main():
    for cid in (sys.argv[1:] or ["0001"]):
        s1 = json.loads((RUN / "stage1" / ("%s.stage1.json" % cid)).read_text(encoding="utf-8"))
        asg = json.loads((RUN / "stage2" / "_assigned" / ("%s.json" % cid)).read_text(encoding="utf-8"))
        H = float(s1["image"]["h"])
        texts = s1.get("texts") or []
        print("\n==== %s  H=%s  cmp=%d  texts=%d" % (cid, H, len(s1["components"]), len(texts)))

        # ---- 决定性判据：desig 的 box 与该文字框的 bbox_img 比 ----
        n_eq = n_flip = n_other = 0
        shown = 0
        for cidk, sl in (asg.get("desig_slots") or {}).items():
            for e in (sl or []):
                bi = e.get("box_i")
                b = e.get("box")
                if bi is None or not b or bi >= len(texts):
                    continue
                tb = texts[bi].get("bbox_img")
                flat = [tb[0], H - tb[3], tb[2], H - tb[1]]
                if max(abs(b[i] - tb[i]) for i in range(4)) <= 1.0:
                    n_eq += 1
                elif max(abs(b[i] - flat[i]) for i in range(4)) <= 1.0:
                    n_flip += 1
                else:
                    n_other += 1
                if shown < 6:
                    shown += 1
                    print("   desig   %s box=%-30s texts[%d].bbox_img=%-30s 翻转=%-30s => %s"
                          % (cidk, [round(x, 1) for x in b], bi,
                             [round(x, 1) for x in tb], [round(x, 1) for x in flat],
                             "等于原框(图像空间)" if max(abs(b[i] - tb[i]) for i in range(4)) <= 1.0
                             else ("等于翻转(JSON空间)" if max(abs(b[i] - flat[i]) for i in range(4)) <= 1.0
                                   else "都不是")))
        print("   desig_slots[].box  vs  stage1.texts[box_i].bbox_img：等于原框 %d｜等于翻转 %d｜都不是 %d"
              % (n_eq, n_flip, n_other))

        # ---- 引脚框：看它与该元件 bbox_img / bbox_json 的包含关系 ----
        s1c = {c["id"]: c for c in s1["components"]}
        stat = {"pin_number": [0, 0, 0], "pin_name": [0, 0, 0]}
        shown2 = 0
        for cidk, c in (asg.get("comps") or {}).items():
            c1 = s1c.get(cidk)
            if c1 is None:
                continue
            bi, bj = c1.get("bbox_img"), c1.get("bbox_json")
            for k, p in (c.get("pins") or {}).items():
                for kind in ("number", "name"):
                    e = (p or {}).get(kind)
                    if not (e and e.get("box")):
                        continue
                    b = e["box"]
                    a, b2 = inside(b, bi), inside(b, bj)
                    if a is None:
                        continue
                    stat["pin_" + kind][0 if a and not b2 else (1 if b2 and not a else 2)] += 1
                    if shown2 < 8 and kind == "name":
                        shown2 += 1
                        print("   pinname %s k=%-3s box=%-30s img内含=%s json内含=%s  elem img=%s json=%s"
                              % (cidk, k, [round(x, 1) for x in b], a, b2,
                                 [round(x, 1) for x in bi], [round(x, 1) for x in bj]))
        for kind, (only_i, only_j, both) in stat.items():
            print("   %-11s 只被 bbox_img 包含 %4d｜只被 bbox_json 包含 %4d｜两个都包含 %4d"
                  % (kind, only_i, only_j, both))


if __name__ == "__main__":
    main()
