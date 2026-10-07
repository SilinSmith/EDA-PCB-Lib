"""量 GT 的「引脚点相对元件框」几何 —— 为「没有导线触到的引脚」找一条可用的规则。

stage1 目前**一个引脚点都不输出**，而 pin 占 0.25、net_hypergraph 占 0.35，合计
0.60 的分数完全拿不到。主线做法是几何：引脚画在框边上「有导线触到」的地方。
但没导线触到的引脚（图里只有一半引脚会接线）得靠类型先验补，先把这个先验量出来。

输出：按 type 汇总
  n_comp / n_pin / 每个元件的引脚数分布
  引脚到框的距离（0 = 正好在框边上，正 = 框外）
  引脚落在框的哪一侧（上下左右），以及在该侧上的相对位置（0=起点,1=终点）
  短边长度 / 长边长度的比值，看偏移是不是与框尺寸成比例

用法：python Y:\workspace\EDA\work\gt_pin_geom.py [--limit N]
"""
import argparse
import collections
import json
import sys

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

from schematic_parser.data import DEFAULT_DATA_ROOT, discover  # noqa: E402


def side_of(px, py, x1, y1, x2, y2):
    """引脚相对框在哪一侧；返回 (side, 沿该侧的相对位置)。

    判定用「到哪条边最近」，而不是「在框外哪个方向」—— 引脚常常正好贴在边上。
    """
    cands = [
        ("top", abs(py - y1), (px - x1) / max(1e-6, x2 - x1)),
        ("bottom", abs(py - y2), (px - x1) / max(1e-6, x2 - x1)),
        ("left", abs(px - x1), (py - y1) / max(1e-6, y2 - y1)),
        ("right", abs(px - x2), (py - y1) / max(1e-6, y2 - y1)),
    ]
    cands.sort(key=lambda t: t[1])
    return cands[0][0], cands[0][2]


def dist_to_box(px, py, x1, y1, x2, y2):
    """点到【实心矩形】的距离；框内为 0，框外为到最近边的距离。"""
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return (dx * dx + dy * dy) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cases = sorted(discover(DEFAULT_DATA_ROOT), key=lambda c: c.case_id)
    if a.limit:
        cases = cases[:a.limit]

    per_type = collections.defaultdict(lambda: dict(
        n_comp=0, n_pin=0, npins=collections.Counter(), dists=[], sides=collections.Counter(),
        fracs=collections.defaultdict(list), aspect=[], ratio=[]))
    overall = dict(n_comp=0, n_pin=0, dists=[], npins=collections.Counter(),
                   sides=collections.Counter())
    n_case = 0

    for case in cases:
        doc = json.load(open(case.json_path, encoding="utf-8"))
        comps = doc.get("components") or {}
        pins = doc.get("pins") or {}
        # **必须在同一套坐标里比**：GT 的 `bbox` 与 `point` 都是 y 向上的 JSON 坐标，
        # 直接把 point 转成图像 y 再和没转的 bbox 比，会得到「引脚离框 288px」这种
        # 荒唐数（第一版就是这么错的，整张表全废）。两边都用 JSON 坐标就不用管 H。
        n_case += 1
        for cid, c in comps.items():
            bb = c.get("bbox")
            if not bb or len(bb) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bb]
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            pl = pins.get(cid) or {}
            t = c.get("type") or "?"
            rec = per_type[t]
            rec["n_comp"] += 1
            rec["npins"][len(pl)] += 1
            overall["n_comp"] += 1
            overall["npins"][len(pl)] += 1
            rec["aspect"].append(min(w, h) / max(1e-6, max(w, h)))
            for pkey, pr in pl.items():
                pt = pr.get("point") or {}
                if pt.get("x") is None:
                    continue
                px = float(pt["x"])
                py = float(pt["y"])              # 与 bbox 同为 y 向上的 JSON 坐标
                d = dist_to_box(px, py, x1, y1, x2, y2)
                s, f = side_of(px, py, x1, y1, x2, y2)
                rec["n_pin"] += 1
                rec["dists"].append(d)
                rec["sides"][s] += 1
                rec["fracs"][s].append(f)
                overall["n_pin"] += 1
                overall["dists"].append(d)
                overall["sides"][s] += 1

    def pct(v, q):
        if not v:
            return 0.0
        v = sorted(v)
        i = min(len(v) - 1, max(0, int(round(q / 100.0 * (len(v) - 1)))))
        return v[i]

    print("cases %d / 元件 %d / 引脚 %d" % (n_case, overall["n_comp"], overall["n_pin"]))
    d = overall["dists"]
    print("引脚到框距离：p10 %.1f / p25 %.1f / p50 %.1f / p75 %.1f / p90 %.1f  (0=贴边)"
          % tuple(pct(d, q) for q in (10, 25, 50, 75, 90)))
    print("  贴边(<=1px) %.1f%% / <=5px %.1f%% / <=12px %.1f%% / >24px %.1f%%"
          % tuple(100.0 * sum(1 for x in d if x <= t) / max(1, len(d))
                  for t in (1, 5, 12, 24)))
    print("  在框内 %.1f%%" % (100.0 * sum(1 for x in d if x == 0) / max(1, len(d))))
    print("落边分布：", dict(overall["sides"].most_common()))
    print("每元件引脚数分布（前 12）：", dict(overall["npins"].most_common(12)))

    print("")
    print("| type | 元件 | 引脚 | 每元件引脚 p50 | 到框 p50 | <=5px%% | 框内%% | 短长比 p50 | 主要落边 |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = sorted(per_type.items(), key=lambda kv: -kv[1]["n_comp"])
    for t, r in rows[:26]:
        ds = r["dists"]
        npl = sorted(r["npins"].elements())
        print("| %s | %d | %d | %s | %.1f | %.0f%% | %.0f%% | %.2f | %s |"
              % (t, r["n_comp"], r["n_pin"], pct(npl, 50) if npl else "-",
                 pct(ds, 50), 100.0 * sum(1 for x in ds if x <= 5) / max(1, len(ds)),
                 100.0 * sum(1 for x in ds if x == 0) / max(1, len(ds)),
                 pct(r["aspect"], 50), dict(r["sides"].most_common(2))))

    print("")
    print("落边上的相对位置（0=上/左端, 0.5=中点, 1=下/右端）：")
    for t, r in rows[:14]:
        parts = []
        for s in ("top", "bottom", "left", "right"):
            f = r["fracs"].get(s)
            if f and len(f) >= 5:
                parts.append("%s p50=%.2f(n=%d)" % (s, pct(f, 50), len(f)))
        if parts:
            print("  %-14s %s" % (t, "  ".join(parts)))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
