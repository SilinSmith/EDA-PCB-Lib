# -*- coding: utf-8 -*-
"""量「净 hyperGraph」的两种口径，看差多少分。

背景（**最新 QA 原文**，2026-10）：

    Q: 对非 box 元件，pinnumber 不参与评分，是否不要求 HyperGraph 中的
       component.pin_number 和 GT 完全一致了？
    A: 非 box 元件 pinnumber 不计分，所以 rcl 等在 hygraph 中相当于**不区分引脚**；
       gnd/vcc 同理；vcc 和 gnd 默认属于同一个 net；**输出规范参考 10gt**。

`metrics.net_pin_set()` 现在把 `hyperGraph` 里的 `组件.引脚号` **按字面**当节点，
对 box（IC）是对的（官方 Q2 只认 box 的引脚号），但对 r/c/l/gnd/v 就太严了：
GT 里电源引脚号是**随机生产的**（`∅113.111`），我们无论输出什么号都配不上。
本工具用「把非 box 元件的引脚号折成一个通配 `*`」的预处理，量这个口径差多少。

用法：
    python tools\\diag\\_netconv.py --pred ..\\work\\_f9_B_pin --cases 1,2,3,4,5,7,8,9,10
    python tools\\diag\\_netconv.py --pred ..\\work\\_fast_out_reg9\\runs\\stage2 --json ..\\work\\_netconv.json
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

WILD = '*'
PWR_TYPES = frozenset({'gnd', 'v', 'vcc', 'power', 'gnd_ref', 'earth'})


def collapse_nonbox(doc: dict, pwr: str = 'off') -> dict:
    """把非 box（IC 以外）元件的引脚号折成通配 `*`。

    只动 `nets[*].hyperGraph`；`pins` 段保持不变（pin_f1 本来就只统计 box）。

    `pwr` 处理 QA 里「vcc 和 gnd 默认属于同一个 net」这句的另一种读法：
      - `off`：电源符号和其他非 box 一样只折引脚号（默认）
      - `type`：再按类型合并（所有 gnd 一个节点、所有 v 一个节点）
      - `all`：再合并成一个节点（gnd 和 v 同节点，字面读法）
    """
    comps = doc.get('components') or {}
    nets = {}
    for name, net in (doc.get('nets') or {}).items():
        if not isinstance(net, dict):
            nets[name] = net
            continue
        nn = dict(net)
        hg = net.get('hyperGraph')
        if isinstance(hg, str) and hg:
            refs, seen = [], set()
            for c, p in parse_hypergraph(hg):
                t = M.canon_type((comps.get(c) or {}).get('type'))
                if t in M.IC_TYPES:
                    key, pin = c, p
                else:
                    key, pin = c, WILD
                    if pwr != 'off' and t in PWR_TYPES:
                        key = 'PWR_all' if pwr == 'all' else 'PWR_' + str(t)
                        pin = WILD
                r = (key, pin)
                if r not in seen:
                    seen.add(r)
                    refs.append(r)
            nn['hyperGraph'] = build_hypergraph(refs)
        nets[name] = nn
    out = dict(doc)
    out['nets'] = nets
    return out


def net_sizes(doc: dict):
    return [len(set(parse_hypergraph((n or {}).get('hyperGraph', ''))))
            for n in (doc.get('nets') or {}).values()]


def main(argv=None):
    ap = argparse.ArgumentParser(description="净 hyperGraph 口径对比")
    ap.add_argument("--pred", required=True, help="预测目录（<case>.json）")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--cases", default=None, help="逗号分隔；默认目录里全部")
    ap.add_argument("--net-mode", default="micro", choices=("micro", "exact"))
    ap.add_argument("--pwr", default="type", choices=("off", "type", "all"),
                    help="电源符号「属于同一个 net」的读法，见 collapse_nonbox")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    idx = {}
    for c in discover(a.data_root, SPLITS):
        if c.json_path:
            idx.setdefault(c.case_id, c)
    want = [s.strip() for s in a.cases.split(",")] if a.cases else None
    preds = sorted(Path(a.pred).glob('*.json'),
                   key=lambda p: (len(p.stem), p.stem))
    if want:
        preds = [p for p in preds if p.stem in want]
    cfg = M.EvalConfig(net_mode=a.net_mode, net_pin_collapse_nonbox=False)
    cfg_col = M.EvalConfig(net_mode=a.net_mode)
    rows = []
    for p in preds:
        case = idx.get(p.stem)
        if case is None:
            print("跳过（GT 里没有这个 case）：%s" % p.name)
            continue
        gt = load_json(case.json_path)
        pred = json.loads(p.read_text(encoding='utf-8'))
        s_old = M.evaluate_case(gt, pred, p.stem, cfg)
        s_met = M.evaluate_case(gt, pred, p.stem, cfg_col)
        s_new = M.evaluate_case(collapse_nonbox(gt), collapse_nonbox(pred),
                                p.stem, cfg)
        s_pwr = s_new
        if a.pwr != 'off':
            s_pwr = M.evaluate_case(collapse_nonbox(gt, a.pwr),
                                    collapse_nonbox(pred, a.pwr), p.stem, cfg)
        rows.append(dict(
            case=p.stem, score_old=s_old.score, score_met=s_met.score,
            score_new=s_new.score, score_pwr=s_pwr.score,
            netH_old=s_old.net_hypergraph.f1, netH_met=s_met.net_hypergraph.f1,
            netH_new=s_new.net_hypergraph.f1, netH_pwr=s_pwr.net_hypergraph.f1,
            netL_old=s_old.net_line.f1, netL_new=s_new.net_line.f1,
            netL_pwr=s_pwr.net_line.f1,
            compF1=s_old.component.f1, pinF1=s_old.pin.f1,
            n_net_gt=len(gt.get('nets') or {}),
            n_net_pred=len(pred.get('nets') or {}),
            gt_sizes=sorted(net_sizes(gt), reverse=True)[:6],
            pred_sizes=sorted(net_sizes(pred), reverse=True)[:6],
        ))

    if not rows:
        print("没配上任何 case")
        return 1

    print("预测目录 %s（%d 张，net_mode=%s，pwr=%s）" %
          (a.pred, len(rows), a.net_mode, a.pwr))
    print("%-6s %8s %8s %8s %8s %8s %8s %8s" %
          ("case", "字面", "折叠(实测)", "折叠(自实现)", "折叠+电源",
           "netH字面", "netH折叠", "netH电"))
    print("-" * 78)
    for r in rows:
        print("%-6s %8.2f %8.2f %8.2f %8.2f %8.4f %8.4f %8.4f" %
              (r['case'], r['score_old'], r['score_met'], r['score_new'],
               r['score_pwr'], r['netH_old'], r['netH_new'], r['netH_pwr']))
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n          # noqa: E731
    print("-" * 78)
    print("均值：字面 %.4f | 折叠实测 %.4f | 折叠自实现 %.4f | 折叠+电源合并 %.4f" %
          (mean('score_old'), mean('score_met'), mean('score_new'),
           mean('score_pwr')))
    print("      Δ实测 %+.4f  Δ自实现 %+.4f  Δ电源 %+.4f" %
          (mean('score_met') - mean('score_old'),
           mean('score_new') - mean('score_old'),
           mean('score_pwr') - mean('score_old')))
    print("      netH 字面 %.4f 折叠 %.4f 电 %.4f | netL 字面 %.4f 折叠 %.4f 电 %.4f" %
          (mean('netH_old'), mean('netH_new'), mean('netH_pwr'),
           mean('netL_old'), mean('netL_new'), mean('netL_pwr')))
    print("\n每张的 net 数 / 最大 net 尺寸（截前 6）：")
    for r in rows:
        print("  %-6s GT %2d %s\n         预 %2d %s" %
              (r['case'], r['n_net_gt'], r['gt_sizes'],
               r['n_net_pred'], r['pred_sizes']))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                encoding='utf-8')
        print("\n写到 %s" % a.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
