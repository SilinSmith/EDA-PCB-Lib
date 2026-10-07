# -*- coding: utf-8 -*-
"""没配上的 GT IC：正确位号离候选表**有多远**（决定「模糊/家族修复」值不值得做）。

`_desigcand2.py` 已经量出：220 张里没配上的 GT IC 有 629 颗，其中
    · 418 颗 `desig_slots` **是空的**（stage2 没关联上任何位号文字）
    · 152 颗**有候选但全错**
    ·   9 颗正确位号**就在候选表里**（1.4% 元件 / 3.6% 引脚）—— 修选择的天花板
所以「修选择」这条路已经被否掉。剩下那条路是「**把错的候选修成对的**」，
它值得做的**前提**是：错的候选与正确位号**大部分只差一两个字符**（形近/漏字符），
而不是毫无关系。本脚本就量这个距离分布。

编辑距离用 `difflib.SequenceMatcher.ratio()` 的补（1 - ratio 归一化），
另外单独统计「只差一个字符」的常见形态（多一位 / 少一位 / 换一位 / 大小写）。

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_desigmatch.py
    python work\\tools\\diag\\_desigmatch.py --split 200_train_cases --top 15
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json          # noqa: E402
from schematic_parser.metrics import (EvalConfig, IC_TYPES,             # noqa: E402
                                      assign_components, canon_type, is_unkeyed)
import stage2_eval as SE                                                # noqa: E402
from fusion import claims as CL                                         # noqa: E402

RUN = ROOT / "work" / "_submit_out220r36" / "runs"
_PRE = re.compile(r"^([A-Za-z]+)")
_NUM = re.compile(r"(\d+)")


def edist(a, b):
    """真·Levenshtein（短串，直接 DP；位号最多 8 字符）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _kind(gk, cand):
    """把「最近的候选」与 GT 键的差异归个类，看有没有可复用的规律。"""
    gp = (_PRE.match(gk) or [None, ""])[1] if _PRE.match(gk) else ""
    cp = (_PRE.match(cand) or [None, ""])[1] if _PRE.match(cand) else ""
    gn = (_NUM.search(gk) or [None, ""])[1] if _NUM.search(gk) else ""
    cn = (_NUM.search(cand) or [None, ""])[1] if _NUM.search(cand) else ""
    if gp.upper() == cp.upper() and gn == cn:
        return "只差字母大小写"
    if gp.upper() == cp.upper() and gn and cn:
        if gn in cn:
            return "字母同、数字被多读（GN⊂CN）"
        if cn in gn:
            return "字母同、数字被少读（CN⊂GN）"
        return "字母同、数字不同"
    if gp.upper() != cp.upper():
        return "字母都不同"
    return "其它"


def _overlap(a, b):
    x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return x * y


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(RUN / "stage2"))
    ap.add_argument("--s1", default=str(RUN / "stage1"))
    ap.add_argument("--asg", default=str(RUN / "stage2" / "_assigned"))
    ap.add_argument("--split", default="")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args(argv)

    exact, digits = SE.build_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()
    files = [f for f in sorted(Path(a.pred).glob("*.json"))
             if not f.name.startswith("_")]

    st = collections.Counter()
    dist = collections.Counter()
    kinds = collections.Counter()
    samples = collections.defaultdict(list)

    for f in files:
        case, _how = SE.resolve(f.stem, exact, digits)
        if case is None or (a.split and case.split != a.split):
            continue
        gt = load_json(case.json_path)
        pr = json.loads(f.read_text(encoding="utf-8"))
        try:
            _f1, k2g = assign_components(gt, pr, cfg, {})
        except Exception:                                    # noqa: BLE001
            continue
        try:
            s1 = json.loads((Path(a.s1) / ("%s.stage1.json" % f.stem))
                            .read_text(encoding="utf-8"))
            asg = json.loads((Path(a.asg) / ("%s.json" % f.stem))
                             .read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            continue
        slots = asg.get("desig_slots") or {}
        matched_g = set(k2g.values())
        pcomps = pr.get("components") or {}
        free_p = [(k, v) for k, v in pcomps.items() if k not in k2g]

        for gk, gv in (gt.get("components") or {}).items():
            if canon_type((gv or {}).get("type")) not in IC_TYPES:
                continue
            if gk in matched_g or is_unkeyed(gk):
                continue
            gp = len((gt.get("pins") or {}).get(gk) or {})
            gb = gv.get("bbox") or [0, 0, 0, 0]
            best, bestk = 0.0, None
            for pk, pv in free_p:
                ov = _overlap(gb, pv.get("bbox") or [0, 0, 0, 0])
                if ov > best:
                    best, bestk = ov, pk
            # cid 用 stage1 的 bbox_json 反查（bbox_img 是另一个坐标系，踩过坑）
            bi, bo = None, 0.0
            for c in (s1.get("components") or []):
                ov = _overlap(gb, c.get("bbox_json") or c.get("bbox_img") or [0, 0, 0, 0])
                if ov > bo:
                    bo, bi = ov, c.get("id")
            cands = [str(c.get("canon") or c.get("text") or "")
                     for c in (slots.get(bi) or [])]
            cands = [c for c in cands if c]
            st["gt_unmatched"] += 1
            st["pins"] += gp
            if not cands:
                st["无候选"] += 1
                continue
            st["有候选"] += 1
            best_c = min(cands, key=lambda c: edist(gk, c))
            d = edist(gk, best_c)
            dist[min(d, 6)] += 1
            kinds[_kind(gk, best_c)] += 1
            # ★ 关键判别：这个「错候选」是不是**同一张图里别的 GT 元件的正确键**？
            #   是 ⇒ 我们的框吃掉/串了邻居的位号（**关联**错，几何可修）；
            #   否 ⇒ 图上那段字被读成了另一个形状（**识别**错，要换识别器）。
            gtkeys = set((gt.get("components") or {}).keys())
            if best_c in gtkeys and best_c != gk:
                st["候选=别的GT键"] += 1
                kinds["→ 且那个键是别的GT元件"] += 1
            else:
                st["候选不在任何GT键里"] += 1
            if d <= 2 and len(samples["near"]) < a.top:
                samples["near"].append((case.case_id, gk, gp, best_c, d,
                                        cands[:4]))
            if d >= 4 and len(samples["far"]) < a.top:
                samples["far"].append((case.case_id, gk, gp, best_c, d, cands[:4]))

    def pct(n, d):
        return "%6.1f%%" % (100.0 * n / max(1, d))

    print("== 没配上的 GT IC（非占位符） %d 颗 / %d 引脚 =="
          % (st["gt_unmatched"], st["pins"]))
    print("  候选表是空的        %5d  %s" % (st["无候选"], pct(st["无候选"], st["gt_unmatched"])))
    print("  有候选              %5d  %s" % (st["有候选"], pct(st["有候选"], st["gt_unmatched"])))
    print("\n== 「最近候选」与 GT 键的编辑距离分布（分母 = 有候选的 %d 颗） ==" % st["有候选"])
    for d in range(7):
        tag = "%d" % d if d < 6 else ">=6"
        print("  d=%-4s %5d  %s" % (tag, dist[d], pct(dist[d], st["有候选"])))
    print("\n  ★ 距离 ≤1 的 %d 颗（%s）、≤2 的 %d 颗（%s）—— 这是「模糊修复」的真天花板"
          % (dist[1], pct(dist[1], st["有候选"]),
             dist[1] + dist[2], pct(dist[1] + dist[2], st["有候选"])))
    print("\n== ★ 错候选是「别的 GT 元件的正确键」吗（关联错 vs 识别错） ==")
    print("  候选 = 同一张图里**另一个** GT 元件的键   %5d  %s"
          % (st["候选=别的GT键"], pct(st["候选=别的GT键"], st["有候选"])))
    print("  候选不在任何 GT 键里                     %5d  %s"
          % (st["候选不在任何GT键里"], pct(st["候选不在任何GT键里"], st["有候选"])))
    print("\n== 差异形态 ==")
    for k, n in kinds.most_common():
        print("  %-26s %5d  %s" % (k, n, pct(n, st["有候选"])))
    for key, hdr in (("near", "距离 ≤2 的样例（case, GT键, 引脚, 最近候选, 距离, 全部候选）"),
                     ("far", "距离 ≥4 的样例（case, GT键, 引脚, 最近候选, 距离, 全部候选）")):
        if samples[key]:
            print("\n== %s ==" % hdr)
            for s in samples[key]:
                print("   %s" % (s,))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
