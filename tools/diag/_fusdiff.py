"""逐张对比两个 `stage2_eval.py --json` 结果，找出融合层掉分/加分的 case。

用法（workdir=Y:\workspace\EDA）：
    python work\tools\diag\_fusdiff.py <base.json> <new.json> [--top 25] [--dump]

`--dump` 时按掉分幅度打印前 N 张的 counts/diagnostic 细目。
只依赖标准库。
"""
import json
import sys
from pathlib import Path


def load(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    return {c["case_id"]: c for c in d["cases"]}


def row(c):
    return (float(c.get("score") or 0.0),
            float(c.get("component_f1") or 0.0),
            float(c.get("pin_f1") or 0.0),
            float(c.get("net_hypergraph_f1") or 0.0),
            float(c.get("net_line_f1") or 0.0))


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    top = 20
    if "--top" in argv:
        top = int(argv[argv.index("--top") + 1])
    dump = "--dump" in argv
    a, b = load(argv[1]), load(argv[2])
    keys = sorted(set(a) & set(b))
    rows = []
    for k in keys:
        ra, rb = row(a[k]), row(b[k])
        rows.append((rb[0] - ra[0], k, ra, rb))
    rows.sort()
    na = sum(r[2][0] for r in rows) / max(1, len(rows))
    nb = sum(r[3][0] for r in rows) / max(1, len(rows))
    print("共同 case %d ｜ 基线 %.4f ｜ 新件 %.4f ｜ 差 %+.4f" % (len(rows), na, nb, nb - na))
    print("分项均分（基线 -> 新件）：comp %.4f->%.4f  pin %.4f->%.4f  netH %.4f->%.4f  line %.4f->%.4f" % (
        sum(r[2][1] for r in rows) / len(rows), sum(r[3][1] for r in rows) / len(rows),
        sum(r[2][2] for r in rows) / len(rows), sum(r[3][2] for r in rows) / len(rows),
        sum(r[2][3] for r in rows) / len(rows), sum(r[3][3] for r in rows) / len(rows),
        sum(r[2][4] for r in rows) / len(rows), sum(r[3][4] for r in rows) / len(rows)))
    print()
    print("=== 掉分最狠的 %d 张 ===" % top)
    for d, k, ra, rb in rows[:top]:
        print("  %-26s %+8.3f (%.2f->%.2f)  comp %.3f->%.3f pin %.3f->%.3f" %
              (k, d, ra[0], rb[0], ra[1], rb[1], ra[2], rb[2]))
        if dump:
            ca = a[k]["counts"]["component"]
            cb = b[k]["counts"]["component"]
            print("        comp tp %s->%s fp %s->%s fn %s->%s ｜ pin tp %s->%s fp %s->%s" % (
                ca["tp"], cb["tp"], ca["fp"], cb["fp"], ca["fn"], cb["fn"],
                a[k]["counts"]["pin"]["tp"], b[k]["counts"]["pin"]["tp"],
                a[k]["counts"]["pin"]["fp"], b[k]["counts"]["pin"]["fp"]))
    print()
    print("=== 加分最多的 %d 张 ===" % top)
    for d, k, ra, rb in rows[-top:][::-1]:
        print("  %-26s %+8.3f (%.2f->%.2f)" % (k, d, ra[0], rb[0]))
    neg = [r for r in rows if r[0] < -1e-9]
    pos = [r for r in rows if r[0] > 1e-9]
    print()
    print("掉分 %d 张（合计 %+.3f）｜加分 %d 张（合计 %+.3f）｜持平 %d 张" %
          (len(neg), sum(r[0] for r in neg), len(pos), sum(r[0] for r in pos),
           len(rows) - len(neg) - len(pos)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
