"""_gndnet.py —— GND 互证先验（用户 m12359）到底有没有头寸？

问的是这一句：
    「引脚名是 `GND/AGND/PGND…` ⇒ 必落在接地网络上；反之检测出 GND 的导线也要
      连上对应引脚」

落成可量化的形式 = **同一张图里，所有引脚名是 `GND/AGND/PGND…` 的引脚，
在 GT 里是不是同一个 net？我们预测里是不是同一个 net？**

  python work\\tools\\diag\\_gndnet.py [case ...]     默认 10 1 0001
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                     # Y:\workspace\EDA
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json   # noqa: E402
from stage2_eval import build_index, resolve                     # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

OUT = ROOT / "work" / "_fast_out_base9"
GND = ("GND", "AGND", "PGND", "DGND", "EGND", "SGND", "EARTH", "GNDA", "GNDD",
       "VSS", "AGNDA")


def _ground(name) -> bool:
    s = str(name or "").strip().upper()
    return bool(s) and (s in GND or s.startswith("GND") or s.endswith("GND"))


def _pinnames(doc):
    """{(cid, pin_key): pinname} —— 兼容 GT 与我们预测的两种写法。

    ⚠️ 成员串写的是**引脚号**（`C8.1`），而 `pins` 字典的键是 **`pin_1`**。
    两个别名都要存，否则一条都对不上（踩过：合计 0）。
    """
    out = {}
    pins = doc.get("pins") or {}
    if isinstance(pins, dict):
        for cid, pv in pins.items():
            if isinstance(pv, dict):
                for pk, p in pv.items():
                    nm = (p.get("pinname") or p.get("name") or "") if isinstance(p, dict) else p
                    out[(cid, str(pk))] = nm
                    out[(cid, str(pk).replace("pin_", ""))] = nm
            else:
                out[(cid, "")] = pv
    elif isinstance(pins, list):
        for p in pins:
            cid = p.get("component") or p.get("cid") or p.get("id")
            pk = p.get("pin") or p.get("number") or p.get("pin_id")
            nm = p.get("pinname") or p.get("name") or ""
            out[(cid, str(pk))] = nm
            out[(cid, str(pk).replace("pin_", ""))] = nm
    return out


def _hyper(net):
    if isinstance(net, dict):
        hg = net.get("hyperGraph") or net.get("hypergraph") or ""
        if isinstance(hg, list):
            return hg
        return [t for t in str(hg).strip("()").split(",") if t]
    return []


def _netmap(doc):
    nets = doc.get("nets") or {}
    items = nets.values() if isinstance(nets, dict) else nets
    return [(_hyper(n), (n.get("name") if isinstance(n, dict) else None))
            for n in items]


def main(argv):
    cases = argv or ["10", "1", "0001"]
    exact, digits = build_index(DEFAULT_DATA_ROOT)
    for cid in cases:
        case, how = resolve(cid, exact, digits)
        if case is None:
            case, how = resolve(cid.lstrip("0") or "0", exact, digits)
        if case is None:
            print("! 找不到 case %s" % cid)
            continue
        gt = load_json(Path(case.json_path))
        pp = OUT / "final_results" / (case.case_id + "_Pred.json")
        pr = load_json(pp) if pp.exists() else {}
        print("=" * 78)
        print("case %s (%s)  GT nets=%d  pred nets=%d"
              % (case.case_id, case.split, len(gt.get("nets") or {}),
                 len(pr.get("nets") or {})))
        for tag, doc in (("GT", gt), ("PRED", pr)):
            pm = _pinnames(doc)
            nlist = _netmap(doc)
            # 每个 net 里「成员 id」有多少、其中名字是地的有几个
            per = []
            for i, (members, nm) in enumerate(nlist):
                g = [m for m in members if _ground(pm.get(_split_member(m), ""))]
                per.append((i, len(members), len(g), nm))
            tot_g = sum(x[2] for x in per)
            nets_with_g = [x for x in per if x[2] > 0]
            print("  [%s] 引脚名是地的成员合计 %d，散落在 %d 个 net 里：%s"
                  % (tag, tot_g, len(nets_with_g),
                     [(x[0], x[1], x[2]) for x in nets_with_g][:12]))
            # 有没有 net 名字本身叫 GND 的
            named = [(i, nm) for i, _, _, nm in per if nm]
            if named:
                print("      带名字的 net：%s" % named[:8])
        # 我们预测里**空 net**（一个成员都没有）
        pn = _netmap(pr)
        print("  PRED 空 net %d 个 / 共 %d"
              % (len([1 for m, _ in pn if not m]), len(pn)))


def _split_member(m: str):
    """`C8.1` / `U1.pin_7` -> (cid, pin_key)。返回的 key 要和 `_pinnames` 对上。"""
    s = str(m).strip().strip("()")
    if "." not in s:
        return (s, "")
    cid, pk = s.rsplit(".", 1)
    return (cid, pk)


if __name__ == "__main__":
    main(sys.argv[1:])
