# -*- coding: utf-8 -*-
"""把两个 stage2_eval --json（220 张）按 suite 拆分汇总。

用法：
    python -X utf8 work\tools\diag\_split220.py <A.json> <B.json>

suite：10GT（名字 1..10）、10_GTcase（带来源后缀的名字）、train200（0001..0200）。
官方 20 = 10GT + 10_GTcase。
"""
import json
import re
import sys

W = {'component_f1': 0.30, 'pin_f1': 0.25, 'net_hypergraph_f1': 0.35, 'net_line_f1': 0.10}


def load(p):
    d = json.load(open(p, encoding='utf-8'))
    return {str(c['case_id']): c for c in d['cases']}


def suite(n):
    if re.match(r'^\d{1,2}$', n):
        return '10GT'
    if re.match(r'^\d{4}$', n):
        return 'train200'
    return '10_GTcase'


def w(c):
    return 100 * sum(W[k] * float(c.get(k) or 0) for k in W)


def main():
    A, B = load(sys.argv[1]), load(sys.argv[2])
    groups = {}
    for n in A:
        groups.setdefault(suite(n), []).append(n)
    for g in sorted(groups):
        ns = sorted(groups[g])
        sa = sum(w(A[n]) for n in ns) / len(ns)
        sb = sum(w(B[n]) for n in ns) / len(ns)
        f = lambda c, k: sum(float(c[n][k]) for n in ns) / len(ns)
        d = sorted(((n, w(B[n]) - w(A[n])) for n in ns), key=lambda t: t[1])
        up = sum(1 for _, x in d if x > 1e-9)
        dn = sum(1 for _, x in d if x < -1e-9)
        print('%-10s n=%3d  A %.4f -> B %.4f  d %+.4f | pin %.4f->%.4f | netH %.4f->%.4f | up %d down %d'
              % (g, len(ns), sa, sb, sb - sa, f(A, 'pin_f1'), f(B, 'pin_f1'),
                 f(A, 'net_hypergraph_f1'), f(B, 'net_hypergraph_f1'), up, dn))
        print('            worst: %s' % ', '.join('%s %+.3f' % t for t in d[:5]))
        for met in ('pin', 'net_hypergraph'):
            ca = [A[n]['counts'][met] for n in ns if A[n].get('counts', {}).get(met)]
            cb = [B[n]['counts'][met] for n in ns if B[n].get('counts', {}).get(met)]
            s = lambda cs, k: sum(int(c.get(k) or 0) for c in cs)
            print('            %-14s A tp %d fp %d fn %d  |  B tp %d fp %d fn %d'
                  % (met, s(ca, 'tp'), s(ca, 'fp'), s(ca, 'fn'), s(cb, 'tp'), s(cb, 'fp'), s(cb, 'fn')))
    off = [n for n in A if suite(n) != 'train200']
    if off:
        print()
        print('official%-4d A %.4f  B %.4f  d %+.4f'
              % (len(off), sum(w(A[n]) for n in off) / len(off),
                 sum(w(B[n]) for n in off) / len(off),
                 sum(w(B[n]) - w(A[n]) for n in off) / len(off)))
    print('all%-9d A %.4f  B %.4f  d %+.4f'
          % (len(A), sum(w(A[n]) for n in A) / len(A), sum(w(B[n]) for n in A) / len(A),
             sum(w(B[n]) - w(A[n]) for n in A) / len(A)))


if __name__ == '__main__':
    main()
