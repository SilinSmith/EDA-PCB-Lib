# -*- coding: utf-8 -*-
"""交付级等价 A/B（**纯 CPU**，不碰 GPU / LM Studio）；`--lib` 换库文件（换默认库就是用它量的）。

为什么它等价于交付链：`fusion/run.py:128` 就是 `base = build_doc(s1_doc, assigned, case)`，
本脚本用**同一份输入**（`work\_fast_out_reg9\runs\stage1\<case>.stage1.json` +
`work\_fnew\_assigned\<case>.json`）直接调同一个函数，只切 `STAGE2_LIB_PINS`。
写文件用 `stage2/run.py:249` 的同一写法（`json.dumps(pred, ensure_ascii=False)`），
所以两份产物可以逐字节比。

用法：
    python -X utf8 work\tools\diag\_libpins_cpu.py --out work\_libpins_cpu
    python -X utf8 work\tools\eval\stage2_eval.py work\_libpins_cpu\off --quiet
    python -X utf8 work\tools\eval\stage2_eval.py work\_libpins_cpu\on  --quiet
"""
import argparse
import hashlib
import importlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]        # Y:\workspace\EDA
EPCB = ROOT / 'eda-pcb'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default=str(ROOT / 'work' / '_fast_out_reg9' / 'runs' / 'stage1'))
    ap.add_argument('--assigned', default=str(ROOT / 'work' / '_fnew' / '_assigned'))
    ap.add_argument('--out', default=str(ROOT / 'work' / '_libpins_cpu'))
    ap.add_argument('--cases', default='1,2,3,4,5,7,8,9,10')
    ap.add_argument('--lib', default='')
    a = ap.parse_args()
    cases = [c.strip() for c in a.cases.split(',') if c.strip()]
    out = pathlib.Path(a.out)

    sys.path.insert(0, str(EPCB))
    if a.lib:
        os.environ['STAGE2_CPNT_LIB'] = a.lib

    report = {'cases': [], 'total': {'add': 0, 'rename': 0, 'net_hg': 0}}
    for mode in ('off', 'on'):
        os.environ['STAGE2_LIB_PINS'] = '1' if mode == 'on' else '0'
        os.environ['STAGE2_LIB_PINS_DBG'] = '0'
        # 每次都重新 import，确保模块级 LIB_PINS 常量跟着环境变量走
        for m in ('stage2.assemble', 'stage2.libpins'):
            sys.modules.pop(m, None)
        importlib.invalidate_caches()
        assemble = importlib.import_module('stage2.assemble')
        d = out / mode
        d.mkdir(parents=True, exist_ok=True)
        docs = {}
        for c in cases:
            s1 = json.loads(pathlib.Path(a.stage1, '%s.stage1.json' % c).read_text(encoding='utf-8'))
            asg = json.loads(pathlib.Path(a.assigned, '%s.json' % c).read_text(encoding='utf-8'))
            doc = assemble.build_doc(s1, asg, c)
            docs[c] = doc
            (d / ('%s.json' % c)).write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
        if mode == 'off':
            off_docs = docs
        else:
            on_docs = docs

    for c in cases:
        A, B = off_docs[c], on_docs[c]
        same = (json.dumps(A, sort_keys=True) == json.dumps(B, sort_keys=True))
        pa, pb = A.get('pins') or {}, B.get('pins') or {}
        added, renamed = [], []
        for ck in sorted(set(pb) | set(pa)):
            ga, gb = pa.get(ck) or {}, pb.get(ck) or {}
            for pk in sorted(set(gb) | set(ga)):
                if pk not in ga:
                    added.append('%s.%s=%s' % (ck, pk, (gb[pk] or {}).get('pinname')))
                elif pk not in gb:
                    pass
                elif (ga[pk] or {}).get('pinname') != (gb[pk] or {}).get('pinname'):
                    renamed.append('%s.%s:%r->%r' % (ck, pk, (ga[pk] or {}).get('pinname'),
                                                     (gb[pk] or {}).get('pinname')))
        na, nb = A.get('nets') or {}, B.get('nets') or {}
        n_net = sum(1 for k in set(na) | set(nb)
                    if (na.get(k) or {}).get('hyperGraph') != (nb.get(k) or {}).get('hyperGraph'))
        report['cases'].append({'case': c, 'identical': same, 'n_add': len(added),
                                'n_rename': len(renamed), 'n_net_hg': n_net,
                                'added': added, 'renamed': renamed})
        report['total']['add'] += len(added)
        report['total']['rename'] += len(renamed)
        report['total']['net_hg'] += n_net
    (out / '_diff.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    for r in report['cases']:
        print('case %-3s 一致=%-5s 补号 %d 改名 %d 受影响网络 %d'
              % (r['case'], r['identical'], r['n_add'], r['n_rename'], r['n_net_hg']))
        for x in r['added']:
            print('    +', x)
        for x in r['renamed']:
            print('    ~', x)
    t = report['total']
    print('合计 补号 %d / 改名 %d / 受影响网络 %d' % (t['add'], t['rename'], t['net_hg']))
    # 逐字节：off 目录的 md5（用于与其他 run 对照）
    for mode in ('off', 'on'):
        hs = [hashlib.md5((out / mode / ('%s.json' % c)).read_bytes()).hexdigest() for c in cases]
        print('%s md5 合并 %s' % (mode, hashlib.md5(''.join(hs).encode()).hexdigest()[:16]))


if __name__ == '__main__':
    main()
