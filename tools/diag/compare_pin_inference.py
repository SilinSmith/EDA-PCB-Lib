"""Inference-only worker: GT is never opened by this program."""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--stage1', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--initial', type=Path, required=True)
p.add_argument('--capture', action='store_true')
p.add_argument('--mono', choices=('on', 'off', 'auto'), default='on')
p.add_argument('--cases', default='1,2,3,4,5,6,7,8,9,10')
args = p.parse_args()
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', EDA_GPU_CAP='0',
                  STAGE2_PINNUM_CONSENSUS='1', STAGE2_BOX_PIN_DUPLICATE_BEST='1',
                  STAGE2_PIN_TRUST_BIND='1', STAGE2_PIN_NAME_EDGE_MIDPOINT='0',
                  STAGE2_PIN_NAME_OCR='on', STAGE2_MONO_PIN_OCR=args.mono)
sys.path.insert(0, str(args.source.resolve()))
from stage2 import run
from stage2.text import TextEngine

args.out.mkdir(parents=True, exist_ok=True)
args.initial.mkdir(parents=True, exist_ok=True)
engine = TextEngine(device='cpu', allow_missing=False)
real_read = engine.read
current = None
calls = 0
input_checks = []
def controlled_read(image, boxes, roles=None):
    global calls
    calls += 1
    if calls != 1:
        raise AssertionError('Unexpected additional initial OCR call')
    if args.capture:
        records = real_read(image, boxes) if roles is None else real_read(image, boxes, roles)
        (args.initial / f'{current}.json').write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
    else:
        records = json.loads((args.initial / f'{current}.json').read_text(encoding='utf-8'))
        expected = [[float(v) for v in b] for b in boxes]
        frozen = [[float(v) for v in r['bbox']] for r in records]
        if expected != frozen:
            raise AssertionError(f'{current}: initial OCR boxes changed; capture fresh reads')
        input_checks.append(dict(case=current, boxes=len(expected), exact_box_order_verified=True))
        if roles is not None:
            wanted = set(roles)
            for r in records:
                r['reads'] = {k:v for k,v in r['reads'].items() if k in wanted}
    return copy.deepcopy(records)
engine.read = controlled_read
stats = []
for case in args.cases.split(','):
    current, calls = case.strip(), 0
    result = run.process(current, args.stage1, args.out, engine, texts='both', recut='off')
    assert calls == 1
    stats.append(result)
    print(f'{current}: {result["t_total"]:.2f}s', flush=True)
origins = {n: str(Path(m.__file__).resolve()) for n,m in sys.modules.items()
           if n.startswith('stage2') and getattr(m, '__file__', None)}
assert all(Path(v).is_relative_to(args.source.resolve()) for v in origins.values())
(args.out / 'inference-verification.json').write_text(json.dumps(dict(
    source=str(args.source.resolve()), mono=args.mono, stats=stats, module_origins=origins,
    initial_capture=args.capture, gt_used_in_inference=False), ensure_ascii=False, indent=2), encoding='utf-8')
(args.out / 'initial-input-verification.json').write_text(json.dumps(
    input_checks, ensure_ascii=False, indent=2), encoding='utf-8')
