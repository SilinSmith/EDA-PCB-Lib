"""逐张对比两个 stage2_eval --json 产物（用法：python work\\_cmp9.py <A.json> <B.json>）。"""
import json
import sys

A = json.load(open(sys.argv[1], encoding="utf-8"))
B = json.load(open(sys.argv[2], encoding="utf-8"))


def per(d):
    for key in ("cases", "per_case", "details", "items"):
        v = d.get(key)
        if isinstance(v, list) and v:
            out = {}
            for c in v:
                n = c.get("case_id") or c.get("case") or c.get("name") or c.get("id")
                out[str(n)] = c
            return out
    return {}


pa, pb = per(A), per(B)
if not pa:
    print("顶层键:", list(A.keys()))
    print(json.dumps(A, ensure_ascii=False)[:800])
    sys.exit(0)

print("%-26s %8s %8s %8s   %s" % ("case", "A", "B", "d", "pinA->pinB"))
sa_tot = sb_tot = 0.0
worse = 0
for k in sorted(pa, key=lambda s: (len(s), s)):
    a, b = pa[k], pb.get(k, {})
    sa = float(a.get("score") or 0.0)
    sb = float(b.get("score") or 0.0)
    sa_tot += sa
    sb_tot += sb
    if sb < sa - 1e-9:
        worse += 1
    pa_ = float(a.get("pin_f1") or a.get("pin") or 0.0)
    pb_ = float(b.get("pin_f1") or b.get("pin") or 0.0)
    print("%-26s %8.4f %8.4f %+8.3f   %.3f->%.3f" % (k, sa, sb, sb - sa, pa_, pb_))
n = max(len(pa), 1)
print("%-26s %8.4f %8.4f %+8.3f   变差 %d/%d 张"
      % ("均分", sa_tot / n, sb_tot / n, (sb_tot - sa_tot) / n, worse, n))

# ★ 按 split 汇总（2026-10 补）：全量 220 = `10GT`(裸数字名) + `10_GTcase`(`00xx-xxx`) +
#   `200_train_cases`(四位数字名)。只报均分会被最大那一档吃掉方向 —— 必须先拆开看。
#   （踩过：引脚通道全量 220 均分 −0.072，拆开却是 10GT +2.59 / GTcase +0.76 / 200 张 −0.25。）
import collections                                                    # noqa: E402
import re                                                             # noqa: E402


def split_of(k):
    if re.fullmatch(r"\d{4}", k):
        return "200_train_cases"
    if re.fullmatch(r"\d{4}-.*", k):
        return "10_GTcase"
    return "10GT"


agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0, 0])
for k in pa:
    if k not in pb:
        continue
    sa = float(pa[k].get("score") or 0.0)
    sb = float(pb[k].get("score") or 0.0)
    g = agg[split_of(k)]
    g[0] += 1
    g[1] += sa
    g[2] += sb
    g[3] += 1 if sb > sa + 1e-9 else 0
    g[4] += 1 if sb < sa - 1e-9 else 0
print()
print("%-18s %5s %9s %9s %9s  %s" % ("split", "n", "A", "B", "d均分", "变好/变差"))
for s in sorted(agg):
    cnt, x, y, bet, wor = agg[s]
    print("%-18s %5d %9.4f %9.4f %+9.4f  %d/%d"
          % (s, cnt, x / cnt, y / cnt, (y - x) / cnt, bet, wor))
