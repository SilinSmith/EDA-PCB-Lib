# -*- coding: utf-8 -*-
"""电源节点「怎么放」的稳健性实验：把电源符号写进净图有多少种放法，各种放法在
**新旧两套评分口径**下分别值多少分。

背景（两条已实测的事实）：
  * `stage2/assemble.py:197 STAGE2_NET_DROP_UNKEYED` 默认 `1` = **把电源/地这类
    `∅NNN` 未键元件从 `hyperGraph` 里删掉**。当时的依据是旧（字面）口径下它们
    「永远配不上 ⇒ 纯 FP」（220 张 `=0` 21.1165 → `=1` 22.9068）。
  * 但最新 QA 把非 box 元件的引脚号折成通配（`net_pin_collapse_nonbox=True`）之后，
    `(p012,*)` 经坐标匹配被改写成 `(∅200,*)` **就能配上了**：官方 9 张上
    `drop=1` 62.5201 → `drop=0` **64.8641（+2.344）**。
  * 风险：万一官方评分仍是字面口径，同一个改动会 **−4.44**（57.0646 vs 61.5084）。

所以本工具量的是「**两套口径都能接受**的折中放法」：
  `keep`  原样（= `drop=0`）
  `nonet` 丢掉「整条 net 全是电源符号」的孤立网（那些只可能是 FP）
  `uniq`  在 `nonet` 基础上，再保证**每个电源符号只出现在一条 net 里**
          （GT 的每个电源符号恰好只属于一条 net；我们一个符号散在多条碎网里）
  `uniq1` `uniq` 之外再把「该 net 里没有 IC 引脚」的电源成员也丢掉（只把电源挂到
          真正有 IC 的网上）

用法：
    python tools\\diag\\_pwrfilter.py --pred ..\\work\\_fdrop0 --cases 1,2,3,4,5,7,8,9,10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from schematic_parser import metrics as M                            # noqa: E402
from schematic_parser.data import (DEFAULT_DATA_ROOT, SPLITS,        # noqa: E402
                                   build_hypergraph, discover, load_json,
                                   parse_hypergraph)

PWR_TYPES = frozenset({'gnd', 'v', 'vcc', 'power', 'gnd_ref', 'earth'})
IC_TYPES = frozenset(M.IC_TYPES)


def _is_pwr(comps, key):
    return M.canon_type((comps.get(key) or {}).get('type')) in PWR_TYPES


def _is_ic(comps, key):
    return M.canon_type((comps.get(key) or {}).get('type')) in IC_TYPES


def variant(doc: dict, how: str) -> dict:
    """按 `how` 重排 `nets[*].hyperGraph`；只动 net 段，`components`/`pins` 不动。"""
    comps = doc.get('components') or {}
    if how == 'keep':
        return doc
    # 1) 解析每条 net 的成员
    raw = {}
    for name, net in (doc.get('nets') or {}).items():
        hg = (net or {}).get('hyperGraph') if isinstance(net, dict) else None
        raw[name] = list(dict.fromkeys(parse_hypergraph(hg))) if hg else []
    # 2) 每个电源符号出现在哪些 net 里
    where = {}
    for name, mem in raw.items():
        for k, _p in mem:
            if _is_pwr(comps, k):
                where.setdefault(k, []).append(name)
    keep_net = {}
    for k, names in where.items():
        # 选一条「最有信息」的 net：非电源成员最多 → 网最大 → 名字最小（字典序）
        best = max(sorted(names), key=lambda n: (sum(1 for kk, _ in raw[n]
                                                   if not _is_pwr(comps, kk)),
                                                 len(raw[n])))
        keep_net[k] = best
    out_nets = {}
    for name, mem in raw.items():
        members = []
        for k, p in mem:
            if _is_pwr(comps, k):
                if how in ('uniq', 'uniq1'):
                    if keep_net.get(k) != name:
                        continue
                if how == 'uniq1' and not any(_is_ic(comps, kk) for kk, _ in mem):
                    continue
            members.append((k, p))
        if how in ('nonet', 'uniq', 'uniq1') and members and \
                all(_is_pwr(comps, k) for k, _ in members):
            continue                      # 电源孤网 ⇒ 丢
        out_nets[name] = members
    nets = {}
    for i, (name, mem) in enumerate(sorted(out_nets.items()), 1):
        src = (doc.get('nets') or {}).get(name) or {}
        nn = dict(src)
        nn['hyperGraph'] = build_hypergraph(mem)
        nets['net_%d' % i] = nn
    out = dict(doc)
    out['nets'] = nets
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="电源节点放法 × 两套口径")
    ap.add_argument("--pred", required=True, help="含电源节点的产物目录（<case>.json）")
    ap.add_argument("--ref", default=None, help="对照臂（默认无 = 不出现电源节点的臂）")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--cases", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    idx = {}
    for c in discover(a.data_root, SPLITS):
        if c.json_path:
            idx.setdefault(c.case_id, c)
    want = [s.strip() for s in a.cases.split(",")] if a.cases else None
    files = [p for p in sorted(Path(a.pred).glob('*.json'),
                              key=lambda p: (len(p.stem), p.stem))
             if "." not in p.stem]
    if want:
        files = [p for p in files if p.stem in want]
    cfg_old = M.EvalConfig(net_pin_collapse_nonbox=False)
    cfg_new = M.EvalConfig()
    hows = ('keep', 'nonet', 'uniq', 'uniq1')
    acc = {h: {'old': [], 'new': [], 'nh_old': [], 'nh_new': [], 'nets': []}
           for h in hows}
    ref = {'old': [], 'new': []}
    for p in files:
        case = idx.get(p.stem)
        if case is None:
            print("跳过 %s（GT 无此 case）" % p.name)
            continue
        gt = load_json(case.json_path)
        pred = json.loads(p.read_text(encoding='utf-8'))
        if a.ref:
            rp = Path(a.ref) / p.name
            if rp.exists():
                rd = json.loads(rp.read_text(encoding='utf-8'))
                ref['old'].append(M.evaluate_case(gt, rd, p.stem, cfg_old).score)
                ref['new'].append(M.evaluate_case(gt, rd, p.stem, cfg_new).score)
        for h in hows:
            d = variant(pred, h)
            so = M.evaluate_case(gt, d, p.stem, cfg_old)
            sn = M.evaluate_case(gt, d, p.stem, cfg_new)
            acc[h]['old'].append(so.score)
            acc[h]['new'].append(sn.score)
            acc[h]['nh_old'].append(so.net_hypergraph.f1)
            acc[h]['nh_new'].append(sn.net_hypergraph.f1)
            acc[h]['nets'].append(len(d.get('nets') or {}))

    if not acc['keep']['old']:
        print("没配上任何 case")
        return 1
    n = len(acc['keep']['old'])
    mean = lambda v: sum(v) / len(v)                                  # noqa: E731
    print("产物 %s（%d 张）" % (a.pred, n))
    print("%-8s %9s %9s %9s %9s %9s %7s" %
          ("放法", "旧口径", "新口径", "netH旧", "netH新", "net 数", "Δ新"))
    print("-" * 66)
    if a.ref and ref['old']:
        print("%-8s %9.4f %9.4f" % ("[对照]", mean(ref['old']), mean(ref['new'])))
    for h in hows:
        print("%-8s %9.4f %9.4f %9.4f %9.4f %9.1f %+7.4f" %
              (h, mean(acc[h]['old']), mean(acc[h]['new']),
               mean(acc[h]['nh_old']), mean(acc[h]['nh_new']),
               mean(acc[h]['nets']), mean(acc[h]['new']) - mean(acc['keep']['new'])))
    if a.ref and ref['old']:
        base_new = mean(ref['new'])
        base_old = mean(ref['old'])
        print("-" * 66)
        for h in hows:
            print("  %-7s 相对对照：旧 %+7.4f   新 %+7.4f" %
                  (h, mean(acc[h]['old']) - base_old,
                   mean(acc[h]['new']) - base_new))
    print("\n逐张（新口径）：")
    print("  %-6s %8s %8s %8s %8s | %6s %6s" %
          ("case", "keep", "nonet", "uniq", "uniq1", "net数", "net数uniq"))
    for i, p in enumerate(files):
        if i >= len(acc['keep']['new']):
            break
        print("  %-6s %8.2f %8.2f %8.2f %8.2f | %6d %6d" %
              (p.stem, acc['keep']['new'][i], acc['nonet']['new'][i],
               acc['uniq']['new'][i], acc['uniq1']['new'][i],
               acc['keep']['nets'][i], acc['uniq']['nets'][i]))
    if a.json:
        Path(a.json).write_text(json.dumps(
            {h: {k: v for k, v in acc[h].items()} for h in hows},
            ensure_ascii=False, indent=1), encoding='utf-8')
        print("\n写到 %s" % a.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
