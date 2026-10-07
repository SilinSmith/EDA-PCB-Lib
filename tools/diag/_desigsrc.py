"""未对上的 GT `box` 元件，按**失效模式 × 画图工具**分类：
「没读到」（占位符）/「读到了但 GT 里没这个键」/「前缀同、号读错」。

为什么要分这么细：`_boxkey.py` 证明了 A 类 100%（大 IC 全都检出了），
但「键对不上」有两种完全不同的原因，修法相反：

  * **我们没读出来**（键是占位符 `p000`/`∅001`）⇒ 修 stage2 的位号取景与读法；
  * **我们读出来了，但 GT 里根本没有这个键** ⇒ 很可能**这份 GT 的位号被重编过**。
    已实证：`200_train_cases\0024\0024-other_target_named.json` 把 33x33 的
    `Name="Key"` 四脚开关记成 `U5`，而图上画的是 `K11`，GT 里**根本没有 `K11`**
    （GT 键 = `C1,C2,R1..R8,U1..U16,∅337`）。这种失分**不是我们的**。

用法：
  python work\\tools\\diag\\_desigsrc.py --dir work\\_fusion_out220no4
  python work\\tools\\diag\\_desigsrc.py --dir work\\_fast_out --show 6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import data as D                                    # noqa: E402
from schematic_parser.metrics import (                                    # noqa: E402
    EvalConfig, assign_components, canon_type, normalise_doc,
)

PLACEHOLDER = re.compile(r"^(?:p\d{3,4}|\u2205\d+)$", re.I)
MODES = ("没检出", "占位符(没读到)", "我们的键在GT里(被抢/错位)",
         "前缀同(号读错)", "前缀不同(GT里没有这个键)")


def _bb(v):
    b = (v or {}).get("bbox") or (v or {}).get("box")
    if not b or len(b) < 4:
        return None
    return [float(x) for x in b[:4]]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    if i <= 0:
        return 0.0
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def find_pred(cid, pdir):
    for cand in (pdir / f"{cid}.json",
                 pdir / "final_results" / f"{cid}_Pred.json",
                 pdir / "runs" / "fusion" / f"{cid}.json"):
        if cand.exists():
            return cand
    return None


def source_of(stem: str) -> str:
    """与 `_pinloss.py --by-source` 同口径：文件名 `\\d+-<工具>_target_named`。"""
    return stem.split("-", 1)[1].rsplit("_target", 1)[0] if "-" in stem else "10GT(?)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "work" / "_fusion_out220no4"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--show", type=int, default=0, help="每个来源打印几个样例")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    want = {x.strip() for x in a.cases.split(",") if x.strip()} or None
    pdir = Path(a.dir)
    cfg = EvalConfig()
    agg = defaultdict(Counter)
    ex = defaultdict(list)
    ph_type = Counter()

    for c in D.discover(D.DEFAULT_DATA_ROOT):
        if not c.json_path or (want and c.case_id not in want):
            continue
        f = find_pred(c.case_id, pdir)
        if not f:
            continue
        s = source_of(Path(c.json_path).stem)
        gt = normalise_doc(D.load_json(c.json_path))
        pr = normalise_doc(json.loads(f.read_text(encoding="utf-8")))
        _f1, km = assign_components(gt, pr, cfg, {})
        matched = set(km.values())
        ours = {}
        otype = {}
        for k, v in pr["components"].items():
            b = _bb(v)
            if b:
                ours[k] = b
                otype[k] = canon_type((v or {}).get("type"))
        gk = {k.lower() for k in gt["components"]}
        agg[s]["case"] += 1
        for k, v in gt["components"].items():
            if canon_type((v or {}).get("type")) not in D.IC_TYPES:
                continue
            agg[s]["gtbox"] += 1
            if k in matched:
                agg[s]["对上"] += 1
                continue
            gb = _bb(v)
            hit = sorted(((round(_iou(gb, b), 3), pk) for pk, b in ours.items()
                          if _iou(gb, b) > 0.05), reverse=True)
            our = hit[0][1] if hit else None
            if our is None:
                mode = "没检出"
            elif PLACEHOLDER.match(our):
                mode = "占位符(没读到)"
                ph_type[otype.get(our) or "?"] += 1
            elif our.lower() in gk:
                mode = "我们的键在GT里(被抢/错位)"
            elif our[:1].upper() == k[:1].upper():
                mode = "前缀同(号读错)"
            else:
                mode = "前缀不同(GT里没有这个键)"
            agg[s][mode] += 1
            if len(ex[s]) < max(1, a.show):
                ex[s].append((c.case_id, k, our, mode, hit[:1]))

    print("=== 未对上的 GT box 元件：失效模式 × 画图工具（%s）===" % pdir)
    print("%-18s %5s %6s %6s | %s" % ("来源", "case", "gtbox", "对上", "  ".join(
        "%s" % m.split("(")[0] for m in MODES)))
    tot = Counter()
    for s, c in sorted(agg.items(), key=lambda x: -x[1]["gtbox"]):
        tot.update(c)
        print("%-18s %5d %6d %6d | %5d %5d %5d %5d %5d"
              % (s, c["case"], c["gtbox"], c["对上"], c["没检出"], c["占位符(没读到)"],
                 c["我们的键在GT里(被抢/错位)"], c["前缀同(号读错)"],
                 c["前缀不同(GT里没有这个键)"]))
    print("%-18s %5d %6d %6d | %5d %5d %5d %5d %5d"
          % ("合计", tot["case"], tot["gtbox"], tot["对上"], tot["没检出"],
             tot["占位符(没读到)"], tot["我们的键在GT里(被抢/错位)"],
             tot["前缀同(号读错)"], tot["前缀不同(GT里没有这个键)"]))
    if ph_type:
        print()
        print("=== 「占位符(没读到)」%d 个里，我们那个预测元件的**类型**分布 ===" % sum(ph_type.values()))
        iclike = sum(v for k, v in ph_type.items() if k in ("box", "ic", "boxic"))
        for k, v in ph_type.most_common():
            print("  %-22s %5d" % (k, v))
        print("  => 其中类型 ∈ IC(box/ic/boxic) 的 %d 个（这些**已经**读过位号却没读出来）；"
              "其余 %d 个类型不在 IC ⇒ 位号通道压根没为它们发起读取。" % (iclike, sum(ph_type.values()) - iclike))
    if a.show:
        print()
        for s in sorted(agg, key=lambda x: -agg[x]["gtbox"]):
            for row in ex[s]:
                print("  %-18s %-6s GT=%-10s 我们=%-10s %s %s" % (s, row[0], row[1], row[2], row[3], row[4]))


if __name__ == "__main__":
    main()
