# -*- coding: utf-8 -*-
"""A/B 功效分析（**只看 GT**，不跑模型）：主终点的那一格，现 ruler 到底能不能测出 +0.05？

主终点（§5.156 追加的人类裁定）= role+长度受控下 `value` 且 GT ≥5 字的 exact 增量 ≥ +0.05。
它的**可测性**只取决于 ruler 里有多少条这样的框，与模型无关 —— 所以可以、也应该在花 GPU 之前算清楚。

输出：
  1. 全集与逐 sheet 的 role/长度计数，以及现 ruler（每 k 张取 1）能拿到多少条目标框；
  2. 独立样本 MDE（alpha=0.05 双侧、power=0.80）；
  3. 配对 McNemar 的 MDE（按不协调率）——配对设计真正的样本量是**不协调对数**；
  4. 备选 ruler：按每张 sheet 的目标框数降序取前 k 张（GT-only 规则，模型输出零参与）。

用法（仓根）：python -X utf8 tools/diag/_abpower.py [--role value] [--min-len 5] [--ruler-every 6]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

EDA = Path(__file__).resolve().parents[2]
Z_A = 1.959963985     # alpha=0.05 双侧
Z_B = 0.8416212336    # power 0.80


def load(p: Path):
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    if k < 0:
        return 0.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, min(k, n) + 1))


def _binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def mde_independent(p: float, n: int) -> float:
    """独立样本、双侧 alpha=0.05、power=0.80 的最小可测差。"""
    q = p
    for _ in range(8):
        se = math.sqrt(p * (1 - p) / n + q * (1 - q) / n)
        q = min(0.999, p + (Z_A + Z_B) * se)
    return (Z_A + Z_B) * math.sqrt(p * (1 - p) / n + q * (1 - q) / n)


def mcnemar_reject(b: int, c: int) -> bool:
    """双侧精确 McNemar：b 胜 / c 负，H0 p=0.5。"""
    n = b + c
    if n == 0 or min(b, c) == n:
        return False
    return 2 * _binom_cdf(min(b, c), n) <= 0.05


def mde_mcnemar(n: int, disc: float) -> float:
    """配对 MDE：n 条框、不协调率 disc，80% power（固定不协调对数 nb=disc*n）。"""
    nb = max(0, round(n * disc))
    if nb < 2:
        return float("nan")
    rej = [mcnemar_reject(k, nb - k) for k in range(nb + 1)]
    for step in range(0, nb + 1, 2):
        b_true = (nb + step) // 2
        pi = b_true / nb
        power = sum(_binom_pmf(k, nb, pi) for k in range(nb + 1) if rej[k])
        if power >= 0.80:
            return step / n
    return float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(EDA / "textrec" / "data" / "manifest.jsonl"))
    ap.add_argument("--role", default="value")
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--ruler-every", type=int, default=6, help="现 ruler 规则：每 k 张 sheet 取 1")
    ap.add_argument("--baseline", type=float, default=0.1915,
                    help="该格在交付权重上的 exact（§5.162 冻结值），用于功效公式里的 p")
    a = ap.parse_args(argv)

    rows = [r for r in load(Path(a.manifest)) if r.get("split") == "real"]
    per_sheet: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_sheet[Path(r["image"]).name].append(r)
    sheets = sorted(per_sheet)

    def stat(rs):
        tgt = [r for r in rs if r["role"] == a.role and len(r["text"]) >= a.min_len]
        allg = [r for r in rs if len(r["text"]) >= a.min_len]
        return len(rs), len(tgt), len(allg)

    tot = stat(rows)
    print("[GT] %s：%d 张图 / %d 框；%s>=%d = **%d**；ALL>=%d = %d"
          % (a.manifest, len(sheets), tot[0], a.role, a.min_len, tot[1], a.min_len, tot[2]))

    rul = sum(stat(per_sheet[s])[1] for i, s in enumerate(sheets) if i % a.ruler_every == 0)
    print("\n现 ruler（每 %d 张取 1，%d 张）%s>=%d = **%d** 条"
          % (a.ruler_every, sum(1 for i in range(len(sheets)) if i % a.ruler_every == 0),
             a.role, a.min_len, rul))

    p = a.baseline
    print("\n[功效] 独立样本 MDE（p=%.4f）:" % p)
    for n in (rul, max(rul, 190), 300, 500, tot[1]):
        print("   n=%4d 条 -> MDE ≈ %.4f" % (n, mde_independent(p, n)))

    print("\n[功效] 配对 McNemar MDE（n=%d）:" % rul)
    for disc in (0.05, 0.10, 0.20, 0.30):
        print("   不协调率 %2.0f%% -> MDE ≈ %.4f" % (disc * 100, mde_mcnemar(rul, disc)))
    print("[功效] 配对 McNemar MDE（不协调率 20%%）随 n:")
    for n in (rul, 190, 300, tot[1]):
        print("   n=%4d -> MDE ≈ %.4f" % (n, mde_mcnemar(n, 0.20)))

    rank = sorted(((stat(per_sheet[s])[1], s) for s in sheets), reverse=True)
    print("\n[备选 ruler] 按每张 sheet 的 %s>=%d 降序取前 k 张（GT-only，模型输出零参与）:"
          % (a.role, a.min_len))
    cum = 0
    for k, (v, s) in enumerate(rank, 1):
        cum += v
        if k in (4, 6, 8, 10, 12, 16):
            print("   k=%2d -> 目标框 %3d 条：独立 MDE %.4f；配对(不协调 20%%) MDE %.4f   [%s]"
                  % (k, cum, mde_independent(p, cum), mde_mcnemar(cum, 0.20), s[-36:]))
    print("   注：ruler 越大，能用于训练的真实行越少（现 control 臂真实训练 2,573 行）——明确的取舍。")

    lo, hi = 1, 100000
    while lo < hi:
        mid = (lo + hi) // 2
        if mde_independent(p, mid) <= 0.05:
            hi = mid
        else:
            lo = mid + 1
    print("\n[需求] 独立样本 MDE<=0.05 需要目标框 %d 条，而全集只有 %d 条 ⇒ %s"
          % (lo, tot[1], "全集都不够" if lo > tot[1] else "全集够"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
