"""Re-score frozen 10GT predictions with old/new rules and list real errors.

No inference, label rewriting, network merging or output repair is performed.
Reports contain input hashes, explicit rule profiles and per-case ledgers.
All scores are local reconstructions, not official competition results.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schematic_parser import metrics
from tools.debug.ledger import build_ledger
from tools.eval.pin_network_ab import CASE_IDS, _unique_file


def _read(path):
    raw = path.read_bytes()
    return json.loads(raw.decode('utf-8-sig')), hashlib.sha256(raw).hexdigest()


def audit(gt_root, prediction_dir, output):
    if not metrics._HAVE_SCIPY:
        raise ValueError('SciPy Hungarian matching is required')
    gt_root, prediction_dir, output = (Path(p).resolve() for p in (gt_root, prediction_dir, output))
    if output.exists():
        raise ValueError('Output directory already exists; use a fresh directory: %s' % output)
    configs = {name: metrics.EvalConfig.for_profile(name) for name in ('protocol2', 'latest')}
    docs, sources = {}, []
    for cid in CASE_IDS:
        gp = _unique_file((gt_root / cid / (cid + '.json'), gt_root / (cid + '.json')), 'GT ' + cid)
        pp = _unique_file((prediction_dir / (cid + '_Pred.json'), prediction_dir / (cid + '.json')),
                          'prediction ' + cid)
        gt, gh = _read(gp)
        pred, ph = _read(pp)
        docs[cid] = gt, pred
        sources.append({'case_id': cid, 'gt': str(gp), 'gt_sha256': gh,
                        'prediction': str(pp), 'prediction_sha256': ph})
    scores, ledgers = {name: [] for name in configs}, {}
    for cid, (gt, pred) in docs.items():
        for name, cfg in configs.items():
            scores[name].append(metrics.evaluate_case(gt, pred, cid, cfg))
        check = metrics.evaluate_case(gt, gt, cid, configs['latest'])
        if abs(check.score - 100) > 1e-8:
            raise ValueError('GT self-check failed: ' + cid)
        led = build_ledger(gt, pred, cid, configs['latest'])
        if not all(row['ok'] for row in led['selfcheck']):
            raise ValueError('Diagnostic/scorer mismatch: ' + cid)
        ledgers[cid] = led
    report = {
        'schema': 'eda-pcb-scoring-rules-audit-v1',
        'evidence_boundary': 'Same frozen predictions; local reconstructed scoring only. Score changes measure rule changes, not recognition improvement. Power geometry follows configured bbox matching; official detailed scorer unavailable.',
        'input_policy': 'Read-only GT and predictions; no inference or changes to annotations, coordinates or networks.',
        'sources': sources,
        'profiles': {name: {'config': asdict(configs[name]),
                            'aggregate': metrics.aggregate(rows),
                            'cases': [s.as_dict() for s in rows]}
                     for name, rows in scores.items()},
        'latest_gt_self_checks': len(docs),
        'latest_ledger_self_checks': {cid: led['selfcheck'] for cid, led in ledgers.items()},
        'errors': {},
    }
    for cid, led in ledgers.items():
        report['errors'][cid] = {
            section: [row for row in led[section] if row['verdict'] != 'tp']
            for section in ('components', 'pins', 'nets', 'lines')}
    # Verify unchanged inputs before publishing any report.
    for row in sources:
        for kind in ('gt', 'prediction'):
            if hashlib.sha256(Path(row[kind]).read_bytes()).hexdigest() != row[kind + '_sha256']:
                raise ValueError('Input changed during scoring: ' + row[kind])
    output.mkdir(parents=True, exist_ok=False)
    for cid, ledger in ledgers.items():
        (output / (cid + '-ledger.json')).write_text(
            json.dumps(ledger, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'comparison.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-root', required=True, type=Path)
    parser.add_argument('--prediction-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Fresh report directory')
    args = parser.parse_args(argv)
    try:
        report = audit(args.gt_root, args.prediction_dir, args.output)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    for name, profile in report['profiles'].items():
        print(name + ': ' + json.dumps(profile['aggregate'], ensure_ascii=False))
    print('GT and diagnostic self-checks: all 10 passed. Inputs unchanged.')
    print('Report: ' + str(args.output / 'comparison.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
