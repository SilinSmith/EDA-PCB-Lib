# -*- coding: utf-8 -*-
"""**引脚在哪一段被丢掉**：GT → stage1 产物 → 最终提交文档，逐跳数引脚。

对每张 case：
  * GT 引脚数、stage1 吐了多少个点、其中多少个 GT 引脚能在 5px 内找到点；
  * 提交文档里有多少个引脚、多少个 GT 引脚能在 5px 内找到点（提交文档的
    `point` 是 JSON 空间，翻 y 后比）；
  * 顺带数「提交文档里的引脚点」有多少能在 stage1 的点集里找到（看有没有被搬走）。

    python work\\tools\\diag\\_pinchain.py [--sub DIR] [case ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
from schematic_parser import data as D                       # noqa: E402
from stage2.text import imread_unicode                       # noqa: E402

S1 = ROOT / "work" / "_submit_out220r36" / "runs" / "stage1"


def pts_of(p):
    """把各种写法的 `pins` 摊平成点列表。"""
    out = []

    def eat(v):
        if isinstance(v, dict):
            p2 = v.get("point") or v.get("pt")
            if isinstance(p2, dict):
                out.append((float(p2["x"]), float(p2["y"]))); return
            for vv in v.values():
                eat(vv)
        elif isinstance(v, (list, tuple)):
            if len(v) >= 2 and not isinstance(v[0], (list, tuple, dict)):
                out.append((float(v[0]), float(v[1])))
            else:
                for vv in v:
                    eat(vv)

    if p:
        eat(p)
    return out


def gtpins(gt, H):
    out = []
    for _k, gp in (gt.get("pins") or {}).items():
        for _pn, pv in gp.items():
            p = pv.get("point")
            if p:
                out.append((float(p["x"]), H - float(p["y"])))
    return out


def hit(gt, got, tol=5.0):
    if not got:
        return 0
    t2 = tol * tol
    return sum(1 for a, b in gt
               if any((a - x) ** 2 + (b - y) ** 2 <= t2 for x, y in got))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    sub = ROOT / "work" / "_fusion_out220no4"
    if a and a[0] == "--sub":
        sub = Path(a[1]); a = a[2:]
    # 提交文档可能是 `<case>.json` 也可能在 `_assigned/`
    cases = a or ["1", "2", "3", "4", "5", "7", "8", "9", "10", "0092", "0048",
                  "0019", "0024", "0075", "0086", "0002", "0001", "0042", "0103"]
    T = {"gt": 0, "s1": 0, "s1hit": 0, "sub": 0, "subhit": 0, "s1near": 0}
    print(f"提交文档目录：{sub}")
    print(f"{'case':>6} | {'GT':>5} {'stage1点':>8} {'s1 5px':>7} | {'提交点':>6} {'提交 5px':>8}")
    for cid in cases:
        case = D.find_case(cid)
        gt = json.loads(Path(case.json_path).read_text(encoding="utf-8"))
        bgr = imread_unicode(case.png_path)
        H = bgr.shape[0]
        g = gtpins(gt, H)
        s1p = pts_of(json.loads((S1 / f"{case.case_id}.stage1.json")
                                .read_text(encoding="utf-8")).get("pins"))
        sp = sub / f"{case.case_id}.json"
        subp = []
        if sp.exists():
            d = json.loads(sp.read_text(encoding="utf-8"))
            subp = [(x, H - y) for x, y in pts_of(d.get("pins"))]
        T["gt"] += len(g); T["s1"] += len(s1p); T["s1hit"] += hit(g, s1p)
        T["sub"] += len(subp); T["subhit"] += hit(g, subp)
        T["s1near"] += sum(1 for x, y in subp
                           if any((x - u) ** 2 + (y - v) ** 2 <= 25 for u, v in s1p))
        print(f"{cid:>6} | {len(g):5d} {len(s1p):8d} {hit(g, s1p):7d} | "
              f"{len(subp):6d} {hit(g, subp):8d}")
    print("\n合计：")
    print(f"  GT 引脚                        {T['gt']}")
    print(f"  stage1 吐点                    {T['s1']}"
          f"   其中命中 GT(<=5px) {T['s1hit']} = {T['s1hit']/max(1,T['gt']):.3f}"
          f"   精确率 {T['s1hit']/max(1,T['s1']):.3f}")
    print(f"  提交文档引脚                   {T['sub']}"
          f"   其中命中 GT(<=5px) {T['subhit']} = {T['subhit']/max(1,T['gt']):.3f}"
          f"   精确率 {T['subhit']/max(1,T['sub']):.3f}")
    print(f"  提交点里能在 stage1 点集 5px 内找到的 {T['s1near']}"
          f" = {T['s1near']/max(1,T['sub']):.3f}  （没找到的就是下游新造的）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
