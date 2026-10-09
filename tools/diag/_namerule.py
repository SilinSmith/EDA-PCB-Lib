# -*- coding: utf-8 -*-
"""引脚名救援：在 _namerescue.py 的明细上离线比较「发不发、发哪个」的规则。

只用明细（--dump），不碰模型；每条规则给出 (出/对/错) 与在官方 9 张上的
pinF1 / 总分增量估计。判定口径：发出且名字与 GT 相同 => tp+1、fn-1；
发出但名字错 => fp+1（fn 不变）—— 错发的代价是 +1 fp，正发是 +2 单位。
"""
import collections
import json
import re
import sys

NORM = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())  # noqa: E731
CLEAN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9+._\-/]{0,11}$")
TP, FP, FN = 37, 64, 51


def f1(tp, fp, fn):
    return 2.0 * tp / max(1e-9, 2.0 * tp + fp + fn)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"Y:\workspace\EDA\work\_namerescue_dump.json"
    recs = json.loads(open(path, encoding="utf-8").read())
    base = f1(TP, FP, FN)
    print("明细 %d 条；基线 pinF1 %.4f（tp %d fp %d fn %d）"
          % (len(recs), base, TP, FP, FN))
    report = []

    def rep(name, emits):
        right = sum(1 for r, t in emits if NORM(t) == NORM(r["gt"]))
        wrong = len(emits) - right
        f = f1(TP + right, FP + wrong, FN - right)
        report.append((0.25 * (f - base), name, len(emits), right, wrong,
                       f - base))
        print("  %-38s 出 %2d 对 %2d 错 %2d  pinF1 %.4f (%+.4f) 总分 %+.3f"
              % (name, len(emits), right, wrong, f, f - base, 0.25 * (f - base)))

    def txt(r, arm):
        v = (r["cands"] or {}).get(arm)
        return v[0] if v else ""

    def cf(r, arm):
        v = (r["cands"] or {}).get(arm)
        return float(v[1]) if v else 0.0

    def g(r):
        return (r["gt"] or "").strip()

    print("\n-- 明细 --")
    for r in recs:
        lb = (r.get("lib_best") or {}).get("grp4")
        print("  %-4s %-4s %-4s gt=%-7s ink=%-4d part=%-14s lib=%-4s "
              "inlib=%-5s g4=%-8s(%.2f) g6=%-8s s2=%-8s libB=%s"
              % (r["case"], r["comp"], r["pin"], g(r), r["ink"],
                 (r.get("part") or "")[:14], r.get("lib_n"),
                 str(r.get("gt_in_lib")), txt(r, "grp4"), cf(r, "grp4"),
                 txt(r, "grp6"), txt(r, "s_nl2"), lb))

    print("\n-- 规则 A：单臂 + 置信度/字符集门槛 --")
    for arm in ("grp4", "grp6", "s_nl2", "s_nl4", "det", "strip"):
        for thr in (0.0, 0.5, 0.6, 0.7, 0.8):
            rep("%s conf>=%.1f" % (arm, thr),
                [(r, txt(r, arm)) for r in recs
                 if txt(r, arm) and cf(r, arm) >= thr])
        rep("%s + clean" % arm,
            [(r, txt(r, arm)) for r in recs
             if CLEAN.match(txt(r, arm) or "")])
        rep("%s + clean + conf>=0.6" % arm,
            [(r, txt(r, arm)) for r in recs
             if CLEAN.match(txt(r, arm) or "") and cf(r, arm) >= 0.6])

    print("\n-- 规则 B：多臂共识 --")
    for a1, a2 in (("grp4", "grp6"), ("grp4", "s_nl2"), ("grp4", "strip"),
                   ("s_nl2", "strip")):
        rep("%s==%s" % (a1, a2),
            [(r, txt(r, a1)) for r in recs
             if NORM(txt(r, a1)) and NORM(txt(r, a1)) == NORM(txt(r, a2))])
    rep("grp4==grp6 or ==s_nl2",
        [(r, txt(r, "grp4")) for r in recs
         if NORM(txt(r, "grp4")) and
         (NORM(txt(r, "grp4")) == NORM(txt(r, "grp6")) or
          NORM(txt(r, "grp4")) == NORM(txt(r, "s_nl2")))])

    print("\n-- 规则 C：官方库约束 --")
    for rho in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        rep("grp4 库模糊>=%.1f" % rho,
            [(r, (r["lib_best"]["grp4"][0] if r.get("lib_best") and
                  "grp4" in r["lib_best"] else ""))
             for r in recs
             if r.get("lib_best") and "grp4" in r["lib_best"] and
             r["lib_best"]["grp4"][1] >= rho])
    rep("grp4 库里精确命中",
        [(r, txt(r, "grp4")) for r in recs
         if r.get("gt_in_lib") and NORM(txt(r, "grp4")) and
         r.get("libkey")])

    print("\n-- 规则 D：组合 --")
    rep("grp4 (clean&conf>=0.6) 或 grp4==grp6",
        [(r, txt(r, "grp4")) for r in recs
         if (CLEAN.match(txt(r, "grp4") or "") and cf(r, "grp4") >= 0.6) or
         (NORM(txt(r, "grp4")) and NORM(txt(r, "grp4")) == NORM(txt(r, "grp6")))])
    rep("grp4 clean&conf>=0.5 或 (lib>=0.8&clean)",
        [(r, txt(r, "grp4")) for r in recs
         if (CLEAN.match(txt(r, "grp4") or "") and cf(r, "grp4") >= 0.5) or
         (r.get("lib_best") and "grp4" in r["lib_best"] and
          r["lib_best"]["grp4"][1] >= 0.8 and CLEAN.match(txt(r, "grp4") or ""))])
    report.sort(reverse=True)
    print("\n-- 排名（按总分增量）--")
    for d, name, ne, ri, wr, df in report[:12]:
        print("  %+.3f  %-38s 出 %2d 对 %2d 错 %2d" % (d, name, ne, ri, wr))
    cnt = collections.Counter(NORM(g(r)) for r in recs)
    print("\nGT 名分布：%s" % dict(cnt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
