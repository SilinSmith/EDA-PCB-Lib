# -*- coding: utf-8 -*-
"""官方尺下的「缺什么」归因：把 component/pin 的 fp/fn 拆成可修/不可修两类。

回答两个问题（DROP_UNKEYED=1 之后）：

  component  missing 112 =
      GT 的 ∅NNN 内部编号（键不可推导 ⇒ 结构上拿不到）
    + 我们**探测到了但位号读错/没读出来**（键不匹配 ⇒ 有机会靠读字/绑定救回）
    + 我们**根本没探测到**（检测漏框）

  pin        missing 143 =
      属于 GT ∅ 元件的脚（同上，结构上拿不到）
    + 属于我们没输出的元件的脚
    + 键对上了但尖端 >5px（tip_position）
    + 键对上了但脚名不符（pin_name）

用法（仓库根）：
    python -X utf8 work\\tools\\dev\\_misswhy.py --run dropunk [--cases "1,2,3"] [--show 8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GT_DIR = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
TOL = 5.0
UNKEYED = "\u2205"          # ∅


def load(case: str, run: str):
    gt = json.loads((GT_DIR / case / (case + ".json")).read_text(encoding="utf-8"))
    pr = json.loads((ROOT / "work" / ("_fast_out_" + run) / "final_results"
                     / (case + "_Pred.json")).read_text(encoding="utf-8"))
    return gt, pr


def maxerr(a, b) -> float:
    return max(abs(float(a[i]) - float(b[i])) for i in range(4))


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def cd(a, b) -> float:
    ca, cb = center(a), center(b)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def pins_of(doc) -> dict:
    out = {}
    ps = doc.get("pins") or {}
    if not isinstance(ps, dict):
        return out
    for ck, d in ps.items():
        if not isinstance(d, dict):
            continue
        for pid, rec in d.items():
            if isinstance(rec, dict) and isinstance(rec.get("point"), dict):
                p = rec["point"]
                out[(ck, pid)] = (str(rec.get("pinname", "")),
                                  (float(p.get("x", 0.0)), float(p.get("y", 0.0))))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dropunk")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args(argv)
    cases = [c.strip() for c in a.cases.split(",") if c.strip()]

    C = {}          # 分类 -> 计数
    P = {}
    examples = {}

    def bump(d, k, n=1):
        d[k] = d.get(k, 0) + n

    for case in cases:
        gt, pr = load(case, a.run)
        gc, pc = gt.get("components") or {}, pr.get("components") or {}
        gp, pp = pins_of(gt), pins_of(pr)

        # ---------------- component ----------------
        tp_keys, mism = [], []
        for k, g in gc.items():
            p = pc.get(k)
            if p is None:
                continue
            if (str(p.get("type", "")) == str(g.get("type", ""))
                    and str(p.get("Name", "")) == str(g.get("Name", ""))
                    and str(p.get("value", "")) == str(g.get("value", ""))
                    and maxerr(p.get("bbox", [0, 0, 0, 0]), g.get("bbox", [0, 0, 0, 0])) <= TOL):
                tp_keys.append(k)
            else:
                mism.append(k)
        miss = [k for k in gc if k not in pc]
        extra = [k for k in pc if k not in gc]
        for k in mism:
            bump(C, "attr_mismatch")
        for k in miss:
            if k.startswith(UNKEYED):
                bump(C, "gt_unkeyable")
                continue
            g = gc[k]
            best, bi = None, 0.0
            for e in extra:
                v = iou(g.get("bbox", [0, 0, 0, 0]), pc[e].get("bbox", [0, 0, 0, 0]))
                if v > bi:
                    best, bi = e, v
            if best is not None and bi >= 0.3:
                bump(C, "detected_but_miskeyed")
                examples.setdefault("miskeyed", []).append(
                    (case, k, str(g.get("Name", "")), best, str(pc[best].get("Name", "")),
                     round(bi, 2)))
            elif best is not None and cd(g.get("bbox", [0, 0, 0, 0]),
                                         pc[best].get("bbox", [0, 0, 0, 0])) <= 25.0:
                bump(C, "detected_nearby_badbox")
                examples.setdefault("nearby", []).append(
                    (case, k, str(g.get("Name", "")), best, round(bi, 2),
                     round(cd(g.get("bbox", [0, 0, 0, 0]),
                              pc[best].get("bbox", [0, 0, 0, 0])), 1)))
            else:
                bump(C, "not_detected")
                examples.setdefault("notdet", []).append(
                    (case, k, str(g.get("type", "")), str(g.get("Name", ""))))
        for k in extra:
            bump(C, "extra")

        # ---------------- pin ----------------
        for key, (gname, gpt) in gp.items():
            ck, pid = key
            if ck not in pc:                      # 元件键没配上 ⇒ 结构上拿不到
                if ck.startswith(UNKEYED):
                    bump(P, "gt_unkeyable")
                else:
                    bump(P, "owner_missing")
                    examples.setdefault("owner", []).append(
                        (case, ck, pid, str(gname)))
                continue
            if (ck, pid) not in pp:
                bump(P, "pin_not_emitted")
                examples.setdefault("notemit", []).append(
                    (case, ck, pid, str(gname), (round(gpt[0], 1), round(gpt[1], 1))))
                continue
            pname, ppt = pp[(ck, pid)]
            d = ((gpt[0] - ppt[0]) ** 2 + (gpt[1] - ppt[1]) ** 2) ** 0.5
            if d > TOL:
                bump(P, "tip_position")
                examples.setdefault("tip", []).append(
                    (case, ck, pid, round(d, 2), str(gname), str(pname)))
            elif pname != gname:
                bump(P, "pin_name")
                examples.setdefault("pname", []).append(
                    (case, ck, pid, str(gname), str(pname)))
            else:
                bump(P, "tp")
        for key in pp:
            ck, pid = key
            if ck not in gc:
                bump(P, "extra")
                continue
            g = gp.get(key)
            if g is None:
                bump(P, "extra")
                continue
            pname, ppt = pp[key]
            d = ((g[1][0] - ppt[0]) ** 2 + (g[1][1] - ppt[1]) ** 2) ** 0.5
            if not (d <= TOL and pname == g[0]):
                bump(P, "penalized_extra")

    print("== component 归因（%s，官方 10 张）==" % a.run)
    tot = sum(C.values())
    for k in sorted(C, key=lambda x: -C[x]):
        print("   %-26s %4d   %5.1f%%" % (k, C[k], 100.0 * C[k] / max(tot, 1)))
    print("   合计 %d" % tot)
    print()
    print("== pin 归因 ==")
    totp = sum(P.values())
    for k in sorted(P, key=lambda x: -P[x]):
        print("   %-26s %4d   %5.1f%%" % (k, P[k], 100.0 * P[k] / max(totp, 1)))
    print("   合计 %d" % totp)
    print()
    for name, title in (("miskeyed", "探测到了但位号读错"),
                        ("nearby", "位置近但框不对"),
                        ("notdet", "根本没有检测框"),
                        ("owner", "元件键没配上（owner_missing）"),
                        ("notemit", "GT 有、我方完全没发这个脚"),
                        ("tip", "尖端位置 >5px"),
                        ("pname", "脚名不符")):
        rows = examples.get(name) or []
        if not rows:
            continue
        print("-- %s（%d 条，前 %d）--" % (title, len(rows), a.show))
        for r in rows[:a.show]:
            print("   ", r)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
