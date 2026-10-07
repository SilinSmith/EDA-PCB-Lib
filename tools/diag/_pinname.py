# -*- coding: utf-8 -*-
"""Why does `pin.name_mismatch` fire?  (`p.get('pinname') != g.get('pinname')`)

For every GT IC component that the official matcher pairs with one of ours, walk
the pins whose *number* we got right and bucket the `pinname` comparison:

    both_empty      : GT has no pinname, we have none      -> should be a tp
    gt_empty        : GT empty, we invented a name         -> pure self-inflicted
    we_empty        : GT has a name, we produced none      -> reachable by reading
    equal           : both non-empty and identical
    differ          : both non-empty, different            -> CRNN/VLM misread or
                                                              we attached the
                                                              neighbour's name

Usage::

    python work\\tools\\diag\\_pinname.py --pred work\\_submit_out220r21\\runs\\stage2
    python work\\tools\\diag\\_pinname.py --pred ... --top 20 --show 0004
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'eda-pcb'))

from schematic_parser import metrics as M                        # noqa: E402
from schematic_parser.data import IC_TYPES, discover, load_json  # noqa: E402


def _num(pk):
    try:
        return M.pin_key_to_number(pk)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--split', default='200_train_cases')
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--show', default='')
    a = ap.parse_args()

    cases = {c.case_id: c for c in discover() if c.split == a.split}
    cfg = M.EvalConfig()
    cnt = Counter()
    samples = {'gt_empty': [], 'we_empty': [], 'differ': []}

    for cid in sorted(cases):
        p = os.path.join(a.pred, '%s.json' % cid)
        if not os.path.exists(p):
            continue
        gt = load_json(cases[cid].json_path)
        pr = load_json(p)
        _f1, km = M.assign_components(gt, pr, cfg)
        # reverse: gt key -> our key
        back = {g: k for k, g in km.items()}
        gcomp = gt.get('components') or {}
        gpins = gt.get('pins') or {}
        ppins = pr.get('pins') or {}
        for gk, gv in gcomp.items():
            if gv.get('type') not in IC_TYPES:
                continue
            pk = back.get(gk)
            if pk is None:
                continue
            gpn = gpins.get(gk) or {}
            ppn = ppins.get(pk) or {}
            gnum = {}
            for k, v in gpn.items():
                n = _num(k)
                if n is not None:
                    gnum[n] = v
            pnum = {}
            for k, v in ppn.items():
                n = _num(k)
                if n is not None:
                    pnum[n] = v
            for n, g in gnum.items():
                if n not in pnum:
                    cnt['number_absent'] += 1
                    continue
                pv = pnum[n]
                gn = (g or {}).get('pinname') or ''
                pn = (pv or {}).get('pinname') or ''
                gn = str(gn).strip(); pn = str(pn).strip()
                if not gn and not pn:
                    cnt['both_empty'] += 1
                elif not gn:
                    cnt['gt_empty'] += 1
                    if len(samples['gt_empty']) < 4000:
                        samples['gt_empty'].append((cid, gk, n, repr(pn)))
                elif not pn:
                    cnt['we_empty'] += 1
                    if len(samples['we_empty']) < 4000:
                        samples['we_empty'].append((cid, gk, n, repr(gn)))
                elif gn == pn:
                    cnt['equal'] += 1
                else:
                    cnt['differ'] += 1
                    if len(samples['differ']) < 4000:
                        samples['differ'].append((cid, gk, n, gn, pn))

    tot = sum(cnt.values())
    print('pred =', a.pred)
    print('GT IC pins on matched components = %d' % tot)
    for k in ('both_empty', 'equal', 'gt_empty', 'we_empty', 'differ',
              'number_absent'):
        print('  %-14s %6d  %6.1f%%'
              % (k, cnt[k], 100.0 * cnt[k] / max(1, tot)))
    for k in ('gt_empty', 'we_empty', 'differ'):
        print()
        print('--- %s samples ---' % k)
        for row in samples[k][:a.top]:
            print('   ', row)

    if a.show:
        cid = a.show
        gt = load_json(cases[cid].json_path)
        pr = load_json(os.path.join(a.pred, '%s.json' % cid))
        _f1, km = M.assign_components(gt, pr, cfg)
        back = {g: k for k, g in km.items()}
        for gk, gv in (gt.get('components') or {}).items():
            if gv.get('type') not in IC_TYPES or gk not in back:
                continue
            pk = back[gk]
            gpn = (gt.get('pins') or {}).get(gk) or {}
            ppn = (pr.get('pins') or {}).get(pk) or {}
            print()
            print('=== %s : GT %s <- our %s  (%d GT pins / %d our pins) ==='
                  % (cid, gk, pk, len(gpn), len(ppn)))
            for k, v in sorted(gpn.items(), key=lambda kv: str(kv[0])):
                n = _num(k)
                got = None
                for k2, v2 in ppn.items():
                    if _num(k2) == n:
                        got = v2
                        break
                print('   pin %-6s GT name=%-16s our=%-16s %s'
                      % (n, repr((v or {}).get('pinname')),
                         repr((got or {}).get('pinname')),
                         '' if got is not None else '<number absent>'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
