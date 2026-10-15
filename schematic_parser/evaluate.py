# -*- coding: utf-8 -*-
"""Score a prediction directory against the ground truth.

Prediction layout (the layout the competition asks for)::

    <pred_root>/case_001/case_001.json
or a flat directory of ``<case_id>.json`` files.  Cases are matched to the
ground truth by their numeric id, so ``case_001`` finds GT case ``0001``.

Usage::

    python -m schematic_parser.evaluate --pred submission --split 10_GTcase
    python -m schematic_parser.evaluate --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

from .data import DEFAULT_DATA_ROOT, SPLITS, Case, discover, load_json
from .metrics import EvalConfig, CaseScore, aggregate, evaluate_case


def _norm_id(text: str) -> str:
    """``'case_001'`` / ``'001'`` / ``'1'`` -> ``'1'`` (leading zeros dropped)."""
    m = re.search(r'(\d+)', text)
    return str(int(m.group(1))) if m else text


def load_predictions(pred_root: str) -> Dict[str, dict]:
    """Collect every prediction JSON under *pred_root*, keyed by normalised id."""
    out: Dict[str, dict] = {}
    if not os.path.isdir(pred_root):
        raise SystemExit('prediction root not found: %s' % pred_root)
    for name in sorted(os.listdir(pred_root)):
        path = os.path.join(pred_root, name)
        if os.path.isdir(path):
            inner = [f for f in sorted(os.listdir(path))
                     if f.lower().endswith('.json')]
            if not inner:
                continue
            out[_norm_id(name)] = load_json(os.path.join(path, inner[0]))
        elif name.lower().endswith('.json'):
            out[_norm_id(name)] = load_json(path)
    return out


def run(pred_root: str, data_root: str = DEFAULT_DATA_ROOT,
        splits: Sequence[str] = SPLITS, cfg: Optional[EvalConfig] = None,
        verbose: bool = True) -> dict:
    cases = [c for c in discover(data_root, tuple(splits)) if c.json_path]
    preds = load_predictions(pred_root) if pred_root else {}

    scores: List[CaseScore] = []
    missing: List[str] = []
    for case in cases:
        gt = load_json(case.json_path)
        pred = preds.get(_norm_id(case.case_id))
        if pred is None:
            missing.append(case.key)
            pred = {}
        scores.append(evaluate_case(gt, pred, case.key, cfg))

    summary = aggregate(scores)
    summary['missing_predictions'] = len(missing)
    summary['missing_examples'] = missing[:10]

    if verbose:
        print('%-34s %8s %8s %8s %8s %8s' %
              ('case', 'score', 'compF1', 'pinF1', 'netHF1', 'lineF1'))
        print('-' * 78)
        for s in scores:
            print('%-34s %8.2f %8.4f %8.4f %8.4f %8.4f' %
                  (s.case_id, s.score, s.component.f1, s.pin.f1,
                   s.net_hypergraph.f1, s.net_line.f1))
        print('-' * 78)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return {'summary': summary, 'scores': scores}


def self_test(data_root: str = DEFAULT_DATA_ROOT, cfg: Optional[EvalConfig] = None):
    """Sanity checks that do not need a trained model."""
    cfg = cfg or EvalConfig()
    cases = [c for c in discover(data_root) if c.json_path]
    print('dataset: %d cases with ground truth' % len(cases))
    if not cases:
        raise SystemExit('no ground truth found under %s' % data_root)

    # 1. GT against itself must be a perfect score.
    bad = []
    for c in cases:
        gt = load_json(c.json_path)
        s = evaluate_case(gt, gt, c.key, cfg)
        if abs(s.score - 100.0) > 1e-6:
            bad.append((c.key, s))
    print('GT-vs-GT perfect on %d/%d cases' % (len(cases) - len(bad), len(cases)))
    for key, s in bad[:5]:
        print('   not perfect: %s -> %.4f  comp=%s pin=%s net=%s line=%s' %
              (key, s.score, s.component.as_dict(), s.pin.as_dict(),
               s.net_hypergraph.as_dict(), s.net_line.as_dict()))

    # 2. Empty prediction must be zero.
    worst = 0.0
    for c in cases:
        gt = load_json(c.json_path)
        s = evaluate_case(gt, {}, c.key, cfg)
        worst = max(worst, s.score)
    print('empty prediction max score = %.4f (expect 0.0)' % worst)

    # 3. Copy-GT-with-a-renamed-key must lose exactly that component.
    c = cases[0]
    gt = load_json(c.json_path)
    import copy
    broken = copy.deepcopy(gt)
    if broken['components']:
        k = next(iter(broken['components']))
        broken['components']['ZZZ_' + k] = broken['components'].pop(k)
        s = evaluate_case(gt, broken, c.key, cfg)
        print('renamed one component key: compF1 %.4f -> %.4f (expect a dip)' %
              (1.0, s.component.f1))

    # 4. bbox / pin tolerance sensitivity
    for mode in ('corners', 'center', 'iou'):
        c2 = EvalConfig(bbox_mode=mode)
        vals = []
        for cc in cases:
            g = load_json(cc.json_path)
            vals.append(evaluate_case(g, g, cc.key, c2).component.f1)
        print('bbox_mode=%-8s GT-vs-GT mean ComponentF1 = %.6f' %
              (mode, sum(vals) / len(vals)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', default=None, help='prediction root directory')
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--split', action='append', default=None,
                    help='restrict to a split (repeatable); default all')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--bbox-mode', default='corners',
                    choices=('corners', 'center', 'iou'))
    ap.add_argument('--net-mode', default='micro', choices=('micro', 'exact'))
    ap.add_argument('--out', default=None, help='write per-case JSON here')
    ap.add_argument('--profile', choices=('latest', 'protocol2', 'protocol'), default='latest')
    args = ap.parse_args(argv)

    cfg = EvalConfig.for_profile(args.profile)
    cfg.bbox_mode, cfg.net_mode = args.bbox_mode, args.net_mode
    if args.self_test:
        self_test(args.data_root, cfg)
        return 0
    if not args.pred:
        ap.error('--pred is required unless --self-test is given')
    res = run(args.pred, args.data_root, args.split or SPLITS, cfg)
    if args.out:
        from .data import save_json
        save_json({
            'config': vars(cfg),
            'summary': res['summary'],
            'cases': [s.as_dict() for s in res['scores']],
        }, args.out)
        print('wrote %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
