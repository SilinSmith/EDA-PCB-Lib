# -*- coding: utf-8 -*-
"""按**元件类型**量「键配上了没有」——决定 component / pin / net 三块权重的天花板。

为什么要这个：`component`（0.30）、`pin`（0.25）、`net`（0.35+0.10）都要先「键配上」
才可能拿到 tp。以前只知道总数（`component.missing` / `component.extra`），
不知道**这个缺口在哪一类元件上**，所以修起来没有方向。

口径（逐字照 `eda-pcb/schematic_parser/metrics.py` 的 `assign_components`）：
- 第 1 步只做 `set(gt_keys) & set(pred_keys)` 的**精确键匹配**；
- 第 2 步坐标匹配只在 `cfg.unkeyed_match == 'gt'`（默认、官方口径）时**只允许
  GT 键是 `∅NNN` / `pNNN`**（`is_unkeyed`）的元件被救回；
- 返回的 `km` 方向是 **预测键 → GT 键**，所以判「GT 键配上了没有」要查
  `gk in set(km.values())`，不是 `gk in km`。

用法：
    python work\\tools\\diag\\_keytype.py --pred work\\_submit_out220r33\\runs\\stage2
    python work\\tools\\diag\\_keytype.py --pred work\\_submit_out220r32\\runs\\stage3_vlm
    python work\\tools\\diag\\_keytype.py --pred <目录> --split 10GT
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                       # eval 等价层级：diag -> tools -> work -> EDA
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))

from schematic_parser.data import (                                          # noqa: E402
    DEFAULT_DATA_ROOT, SPLITS, discover, load_json, net_pin_set, parse_hypergraph,
)
from schematic_parser.metrics import (                                       # noqa: E402
    EvalConfig, assign_components, is_unkeyed,
)

sys.stdout.reconfigure(encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="按元件类型量键命中")
    ap.add_argument("--pred", required=True, help="预测产物目录（含 <case>.json）")
    ap.add_argument("--split", default="", help="只跑某个 split")
    ap.add_argument("--top", type=int, default=12, help="每类最多打几个未命中样例")
    a = ap.parse_args(argv)

    pred_dir = Path(a.pred)
    splits = (a.split,) if a.split else SPLITS
    cfg = EvalConfig()

    # 每类：GT 总数 / 键精确配上 / 坐标救回（GT 键是 unkeyed）/ 键没配上（有 IO 无 IO 分开）
    stat = defaultdict(Counter)
    miss_examples = defaultdict(list)

    for case in discover(DEFAULT_DATA_ROOT, splits=splits):
        pp = pred_dir / (case.case_id + ".json")
        if not pp.exists():
            continue
        gt = load_json(Path(case.json_path))
        pr = load_json(pp)
        gcs = gt.get("components") or {}
        pcs = pr.get("components") or {}
        exact = set(gcs) & set(pcs)
        _f1, km = assign_components(gt, pr, cfg)
        rescued = set(km.values()) - exact      # 只可能是 unkeyed 的 GT 键

        # 预测侧：哪些键是「兜底键」（该位置其实检出了元件，只是没位号）
        pk_unkeyed = {k for k in pcs if is_unkeyed(k)}

        for gk, gc in gcs.items():
            ct = str(gc.get("type") or "?")
            st = stat[ct]
            st["gt"] += 1
            if gk in exact:
                st["exact"] += 1
            elif gk in rescued:
                st["rescued"] += 1
            else:
                st["miss"] += 1
                if is_unkeyed(gk):
                    st["miss_unkeyed_gt"] += 1
                if len(miss_examples[ct]) < a.top:
                    miss_examples[ct].append(
                        f"{case.case_id}:{gk} Name={gc.get('Name')!r}")

        stat["__ALL__"]["pred"] += len(pcs)
        stat["__ALL__"]["pred_unkeyed"] += len(pk_unkeyed)
        stat["__ALL__"]["gt"] += len(gcs)
        stat["__ALL__"]["exact"] += len(exact)
        stat["__ALL__"]["rescued"] += len(rescued)
        stat["__ALL__"]["miss"] += len(gcs) - len(exact) - len(rescued)

    print(f"预测目录 {pred_dir}")
    print(f"{'类型':<16}{'GT数':>7}{'精确配上':>9}{'占比':>8}{'坐标救回':>9}"
          f"{'未配上':>8}{'其中GT是兜底键':>15}")
    print("-" * 74)
    order = sorted(stat, key=lambda k: -stat[k]["gt"])
    for ct in order:
        s = stat[ct]
        n = s["gt"]
        if not n:
            continue
        if ct.startswith("__"):
            continue
        pct = s["exact"] / n if n else 0.0
        print(f"{ct:<16}{n:>7}{s['exact']:>9}{pct:>8.1%}"
              f"{s['rescued']:>9}{s['miss']:>8}{s['miss_unkeyed_gt']:>15}")
    s = stat["__ALL__"]
    n = s["gt"]
    print("-" * 74)
    print(f"{'合计':<16}{n:>7}{s['exact']:>9}{s['exact']/n:>8.1%}"
          f"{s['rescued']:>9}{s['miss']:>8}{'-':>15}")
    print(f"预测元件总数 {s['pred']}，其中兜底键（`pNNN`/`∅NNN`）{s['pred_unkeyed']}"
          f" = {s['pred_unkeyed']/max(1,s['pred']):.1%}")

    print("\n★ 每类未命中样例（最多 %d 个）：" % a.top)
    for ct in sorted(miss_examples, key=lambda k: -stat[k]["miss"]):
        if not miss_examples[ct]:
            continue
        print(f"  [{ct}] 未命中 {stat[ct]['miss']} 个，样例：")
        for e in miss_examples[ct]:
            print(f"      {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
