# -*- coding: utf-8 -*-
"""有名字的 GT 元件里，我们没配上键的那些到底为什么丢 —— 分四类。

  A 压根没检出（附近没有任何预测元件框）
  B 检出了、但键读错（IoU>=0.5 的预测元件键 != GT 键）
  C 检出了、但被别的键占了（同键的预测元件离得更远）
  D 其它

用法：python work\tools\diag\_misswhy.py --pred work\stage2_v13
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eda-pcb"))
sys.path.insert(0, str(REPO / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json  # noqa: E402
from stage2_eval import build_index, resolve  # noqa: E402

UKEY = "\u2205"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def _ed(x, y):
    """编辑距离（短串够用）。"""
    n, m = len(x), len(y)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (x[i - 1] != y[j - 1]))
        prev = cur
    return prev[m]


def _fkey(s):
    return len(s) > 1 and s[0] == "p" and s[1:].isdigit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(REPO / "work" / "stage2_v13"))
    ap.add_argument("--samples", type=int, default=25)
    a = ap.parse_args()
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    cat = Counter()
    kc = Counter()
    tail = Counter()
    bytype = Counter()
    misskey = Counter()
    samples = []
    for f in sorted(Path(a.pred).glob("*.json")):
        if f.name.startswith("_"):
            continue
        case, _ = resolve(f.stem, exact, digits)
        if case is None:
            continue
        gt = load_json(case.json_path)
        pred = json.loads(f.read_text(encoding="utf-8"))
        gc, pc = gt.get("components") or {}, pred.get("components") or {}
        H = 0.0
        # GT 与预测同为左下角原点，直接比
        pboxes = [(k, v.get("bbox") or [0, 0, 0, 0]) for k, v in pc.items()]
        for k, v in gc.items():
            if k.startswith(UKEY):
                continue
            if k in pc:
                continue
            ty = (v.get("type") or "?")
            bytype[ty] += 1
            misskey[k] += 1
            gb = v.get("bbox") or [0, 0, 0, 0]
            best, bk = 0.0, None
            for pk, pb in pboxes:
                s = iou(gb, pb)
                if s > best:
                    best, bk = s, pk
            if best >= 0.5:
                cat["B_检出但键读错"] += 1
                # 键错法细分
                if not bk or bk.startswith("p") and bk[1:].isdigit():
                    kc["没读到位号（兜底键）"] += 1
                elif bk == k:
                    kc["同名？"] += 1
                elif len(bk) > len(k) and bk.startswith(k):
                    kc["GT键 + 多读尾巴"] += 1
                    tail[k[len(k):]] += 1
                elif len(bk) < len(k) and k.startswith(bk):
                    kc["少读一截"] += 1
                elif bk.upper() == k.upper():
                    kc["只差大小写"] += 1
                else:
                    d = _ed(k, bk)
                    kc["编辑距离 %d" % min(d, 4)] += 1
                if len(samples) < a.samples:
                    samples.append((f.stem, k, ty, bk, round(best, 2)))
            elif best >= 0.15:
                cat["C_框对不齐"] += 1
            else:
                cat["A_压根没检出"] += 1
    print("有名字但没配上键的 GT 元件 %d" % sum(bytype.values()))
    print("  分类:", json.dumps(cat, ensure_ascii=False))
    print("  键错法:", json.dumps(kc, ensure_ascii=False))
    print("  多读的尾巴:", tail.most_common(15))
    print("  按 type:", bytype.most_common())
    print("  最常丢的键:", misskey.most_common(25))
    print("\n样例（case, GT键, type, 同位置预测键, IoU）：")
    for s in samples:
        print("   %-8s %-8s %-8s -> %-10s %s" % s)


if __name__ == "__main__":
    main()
