# -*- coding: utf-8 -*-
"""量「轴 + 全局铺号」这一层到底能救回多少位号 —— 纯 CPU，不碰流水线。

口径：
- 输入 = `stage2/_assigned/<case>.json` 的 `desig_slots`（每个 cid 的文字候选框，
  `box` 是**图像空间**，和 `_assigned` 里的其它框一致）；
- 真值配对 = 文字框到最近 GT 元件框（`bbox` 翻成图像空间）距离 ≤ `--match`；
- 基线 = `comps[cid].designator` 给出的键（现在的生产行为）；
- 对照 = 轴解码（`fusion.axis`）给出的键；
- 只有当 cid 被配对到真值、且基线或对照至少有一个非空时，才计入分母。

用法：python work\tools\diag\_axiseval.py --tier quickset [--match 40] [--axis-min 2] [--show]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("Y:/workspace/EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser import data as D          # noqa: E402
from fusion import axis as AX                   # noqa: E402
from PIL import Image                           # noqa: E402

R = ROOT / "work/_submit_out220r36/runs"
TIERS = json.loads((ROOT / "work/tools/eval/tiers.json").read_text(encoding="utf-8"))
CASES = {c.case_id: c for c in D.discover()}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dist_box(a, b):
    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return (dx * dx + dy * dy) ** 0.5


def gt_index(case):
    """`{真值键: 图像空间框}` 与 `[(键, 框)]`。"""
    gt = load(CASES[case].json_path)
    H = float(Image.open(CASES[case].png_path).size[1])
    out = {}
    for k, v in (gt.get("components") or {}).items():
        bb = v.get("bbox")
        if bb:
            out[k] = [float(bb[0]), H - float(bb[3]), float(bb[2]), H - float(bb[1])]
    return gt, out


def build_items(asg):
    """`desig_slots` -> [(cid, Item)]，框用首选候选的框，reads 收全部候选。"""
    out = []
    for cid, slots in (asg.get("desig_slots") or {}).items():
        if not slots:
            continue
        reads, box = [], None
        for s in slots:
            t = (s.get("canon") or s.get("text") or "").strip()
            if t:
                reads.append((t.upper(), float(s.get("conf") or 0.0)))
            if box is None and s.get("box"):
                box = [float(v) for v in s["box"]]
        if box is None or not reads:
            continue
        best = max(reads, key=lambda x: x[1])
        out.append((cid, AX.Item(key=cid, box=box, text=best[0], score=best[1], reads=reads)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="quickset")
    ap.add_argument("--cases", default="")
    ap.add_argument("--match", type=float, default=40.0)
    ap.add_argument("--axis-min", type=int, default=2)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    names = ([c.strip() for c in a.cases.split(",") if c.strip()] if a.cases
             else TIERS.get(a.tier, []))
    base_ok = axis_ok = both_ok = 0
    n = 0
    fix = bade = spoke = 0
    rows = []
    for case in names:
        if case not in CASES:
            print("跳过不存在的 case %s" % case)
            continue
        gw = R / "stage2/_assigned" / (case + ".json")
        if not gw.exists():
            continue
        gt, gtb = gt_index(case)
        asg = load(gw)
        comps = asg.get("comps") or {}
        items = build_items(asg)
        if not items:
            continue
        # 真值配对
        pair = {}
        for cid, it in items:
            if not gtb:
                break
            d, k = min(((dist_box(it.box, box), k) for k, box in gtb.items()))
            if d <= a.match:
                pair[cid] = k
        # 按前缀聚轴 -> 全局解码
        bypre = defaultdict(list)
        for cid, it in items:
            pre, _num = AX.split_desig(it.text)
            if pre:
                bypre[pre].append(it)
        got = {}
        for pre, its in bypre.items():
            axes = AX.build_axes(its, min_members=a.axis_min)
            if not axes:
                continue
            plan = AX.decode_prefix(axes, topk=4)
            if plan is None:
                continue
            for ax, asg_i, lo in plan.per_axis:
                for i, dec in enumerate(asg_i.per):
                    if dec is None:
                        continue
                    got[ax.items[i].key] = "%s%d" % (pre, dec)
        for cid, it in items:
            if cid not in pair:
                continue
            gk = pair[cid]
            b = ((comps.get(cid) or {}).get("designator") or {}).get("text") or ""
            b = b.strip().upper() or None
            x = got.get(cid)
            n += 1
            if x is not None:
                spoke += 1
            bok = (b == gk)
            xok = (x == gk)
            base_ok += bok
            axis_ok += xok
            if bok and xok:
                both_ok += 1
            if x is not None and xok and not bok:
                fix += 1
                if a.show:
                    rows.append(("FIX ", case, cid, gk, b, x))
            # 只有「轴真的出声、且说错了」才算弄坏；沉默不是弄坏（本层是追加证据）
            if bok and x is not None and not xok:
                bade += 1
                if a.show:
                    rows.append(("BREAK", case, cid, gk, b, x))
    print("档 %s 共 %d 个「配对上的位号槽位」" % (a.tier, n))
    if not n:
        return
    print("  基线（stage2 现在给出的键） 正确 %d = %.4f" % (base_ok, base_ok / n))
    print("  轴 + 全局铺号 出声 %d 条、正确 %d = %.4f"
          % (spoke, axis_ok, axis_ok / n))
    print("  两边都对 %d｜轴救回 %d｜轴弄坏 %d" % (both_ok, fix, bade))
    print("  ★ 轴是**追加证据**不是替换：oracle 上限 (base+fix)/n = %.4f；"
          "若一律改用轴 (base+fix-bade)/n = %.4f"
          % ((base_ok + fix) / n, (base_ok + fix - bade) / n))
    if spoke:
        print("  在轴出声的 %d 条里：救回 %d、弄坏 %d ⇒ 精确率 %.4f"
              % (spoke, fix, bade, fix / float(spoke)))
    for r in rows[:60]:
        print("   %-5s %-8s %-8s GT=%-7s 基线=%-7s 轴=%s" % r)


if __name__ == "__main__":
    main()
