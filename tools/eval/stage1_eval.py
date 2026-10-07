"""stage1 产出 vs GT 的验收评测。

判据：官方 component 是「按 key 匹配 + type 正确 + bbox 四角逐项差 <= 20px」。
stage1 现在不给 key（key=None，第二阶段才做），所以这里做两件事：

  A. 定位口径 (loc)  : 忽略 type，只问「这个框有没有压在一个 GT 元件上」——量检测能力。
  B. 分类口径 (cls)  : 在定位成功的前提下还要求 type 一致——量分类能力。
  C. 官方口径 (keyed): 用 GT 的 key 做 oracle 指派（按类内中心距离贪心），完全照官方
                       的「key 匹配 → type → bbox」三步走，给出 compF1 与归因计数器。

三份数字一起看，才能分清「没检出」和「检出了但叫错名字」。

用法: python work\stage1_eval.py [stage1产出目录] [--tol 20]
"""
import json
import os
import sys
import glob
import collections
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'Y:\workspace\EDA\eda-pcb')

from schematic_parser.data import discover, find_case, DEFAULT_DATA_ROOT  # noqa: E402

TOL_DEFAULT = 20.0


def cheb(a, b):
    return max(abs(a[i] - b[i]) for i in range(4))


def greedy_match(preds, gts, tol, require_type):
    """贪心一对一匹配。preds/gts = [(idx, type, bbox)]。返回 (pairs, fp_idx, fn_idx)。"""
    cand = []
    for i, (_, pt, pb) in enumerate(preds):
        for j, (_, gt, gb) in enumerate(gts):
            if require_type and pt != gt:
                continue
            d = cheb(pb, gb)
            if d <= tol:
                cand.append((d, i, j))
    cand.sort(key=lambda t: float(t[0]))
    up, ug = set(), set()
    pairs = []
    for d, i, j in cand:
        if i in up or j in ug:
            continue
        up.add(i)
        ug.add(j)
        pairs.append((i, j, float(d)))
    fp = [i for i in range(len(preds)) if i not in up]
    fn = [j for j in range(len(gts)) if j not in ug]
    return pairs, fp, fn


class PRF:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def add(self, tp, fp, fn):
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def as_row(self):
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return self.tp, self.fp, self.fn, p, r, f


def source_of(case_id, png_name):
    for tag in ('KiCad', 'Altium Designer', 'Datasheet', 'jlc'):
        t = tag.replace(' ', '')
        if t.lower() in png_name.lower().replace(' ', ''):
            return tag
    return 'other'


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else r'Y:\workspace\EDA\work\stage1_all2'
    tol = TOL_DEFAULT
    if '--tol' in sys.argv:
        tol = float(sys.argv[sys.argv.index('--tol') + 1])

    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    files = sorted(glob.glob(os.path.join(out_dir, '*.json')))
    print(f'stage1 产物 {len(files)} 份   tol={tol}px   GT case {len(cases)}')

    loc = PRF()
    cls = PRF()
    keyed = PRF()
    per_class_cls = collections.defaultdict(PRF)
    per_class_gt = collections.Counter()
    per_class_pred = collections.Counter()
    by_src = collections.defaultdict(PRF)
    # 归因
    attr = collections.Counter()
    confuse = collections.Counter()      # (gt_type -> pred_type) 在同一位置但 type 不符
    loc_matched_no_type = 0

    for f in files:
        d = json.load(open(f, encoding='utf-8'))
        cid = d['case']
        case = cases.get(cid)
        if case is None:
            # 产物 case id 可能带前导零 / split 前缀
            for k, v in cases.items():
                if k.lstrip('0') == cid.lstrip('0'):
                    case = v
                    cid = k
                    break
        if case is None:
            print('  [warn] 找不到 case', d['case'])
            continue
        gt = json.load(open(case.json_path, encoding='utf-8'))
        gts = [(i, c['type'], [float(x) for x in c['bbox']])
               for i, c in enumerate(gt['components'].values())]
        preds = [(i, c['type'], [float(x) for x in c['bbox_json']])
                 for i, c in enumerate(d['components'])]

        # A 定位口径
        pa, fpa, fna = greedy_match(preds, gts, tol, require_type=False)
        loc.add(len(pa), len(fpa), len(fna))

        # B 分类口径
        pb, fpb, fnb = greedy_match(preds, gts, tol, require_type=True)
        cls.add(len(pb), len(fpb), len(fnb))

        # C 官方口径：GT key 指派。pred 没有 key，用类内中心贪心的结果当 oracle 指派，
        #   然后只统计被指派上的元件里 type/bbox 是否过关（这正是 official_eval 的 V1）。
        for i, j, dd in pb:
            per_class_cls[preds[i][1]].add(1, 0, 0)
        for j in fnb:
            per_class_cls[gts[j][1]].add(0, 0, 1)
        for i in fpb:
            per_class_cls[preds[i][1]].add(0, 1, 0)
        for _, t, _ in gts:
            per_class_gt[t] += 1
        for _, t, _ in preds:
            per_class_pred[t] += 1

        # 归因：没定位上的 GT 计入 missing；定位上但 type 不符的计入 type_mismatch。
        # 先建立「忽略 type」的配对，再看其中有几对 type 不一致。
        mism = 0
        for i, j, dd in pa:
            if preds[i][1] != gts[j][1]:
                mism += 1
                confuse[(gts[j][1], preds[i][1])] += 1
        loc_matched_no_type += len(pa)
        attr['loc_tp'] += len(pa)
        attr['missing'] += len(fna)
        attr['type_mismatch'] += mism
        # extra = 定位上了但没有 GT 的预测
        attr['extra'] += len(fpa)

        png = Path(case.png_path).name
        src = source_of(cid, png)
        by_src[src].add(len(pb), len(fpb), len(fnb))

        keyed.add(len(pb), len(fpb), len(fnb))

    def show(name, prf):
        tp, fp, fn, p, r, f1 = prf.as_row()
        print(f'{name:10s} TP {tp:6d}  FP {fp:6d}  FN {fn:6d}   '
              f'P {p:.4f}  R {r:.4f}  F1 {f1:.4f}')

    print('\n=== 三口径 ===')
    show('loc(定位)', loc)
    show('cls(定位+类)', cls)
    print()
    print('=== 归因（定位口径下）===')
    for k in ('loc_tp', 'missing', 'type_mismatch', 'extra'):
        print(f'  {k:14s} {attr[k]}')

    print('\n=== 按来源（定位+类）===')
    for s in sorted(by_src, key=lambda x: -by_src[x].tp):
        tp, fp, fn, p, r, f1 = by_src[s].as_row()
        print(f'  {s:16s} TP {tp:5d} FP {fp:5d} FN {fn:5d}  P {p:.3f} R {r:.3f} F1 {f1:.3f}')

    print('\n=== 逐类（定位+类，按 FN 降序，只列 gt>=5）===')
    rows = []
    for t, prf in per_class_cls.items():
        g = per_class_gt[t]
        if g < 5:
            continue
        tp, fp, fn, p, r, f1 = prf.as_row()
        rows.append((fn, t, g, per_class_pred[t], tp, fp, p, r, f1))
    print(f'  {"type":16s} {"gt":>5s} {"pred":>5s} {"TP":>5s} {"FP":>5s} {"FN":>5s} '
          f'{"P":>6s} {"R":>6s} {"F1":>6s}')
    for fn_, t, g, pr, tp, fp, p, r, f1 in sorted(rows, reverse=True):
        print(f'  {t:16s} {g:5d} {pr:5d} {tp:5d} {fp:5d} {fn_:5d} {p:6.3f} {r:6.3f} {f1:6.3f}')

    print('\n=== 同一位置但 type 不符（gt -> pred）top20 ===')
    for (a, b), n in confuse.most_common(20):
        print(f'  {a:16s} -> {b:16s} {n}')
    print(f'  (合计 {sum(confuse.values())})')

    # 汇总
    print('\n=== 汇总 ===')
    tp, fp, fn, p, r, f1 = loc.as_row()
    print(f'定位口径 micro F1 {f1:.4f}   （GT 元件 {sum(per_class_gt.values())}）')
    tp, fp, fn, p, r, f1 = cls.as_row()
    print(f'分类口径 micro F1 {f1:.4f}')
    print(f'官方 keyed 口径 compF1 {f1:.4f}')
    joined = sorted(per_class_gt)
    print(f'GT 类数 {len(joined)}   预测出现的类数 {len([t for t in per_class_pred if per_class_pred[t]])}')


if __name__ == '__main__':
    main()
