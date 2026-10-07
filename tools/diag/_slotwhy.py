# -*- coding: utf-8 -*-
"""追一张图里「位号被改成错号」的现场：把该槽位的全部候选（值/分/来源）打出来。

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_slotwhy.py 0037 --bad C11 [--vlm work\\_fusion_vlm]

`--bad` 是在**融合产物**里看到的那个错号（例如 `C11`）。脚本按 bbox 把它对回
stage1 的 cid，然后重建候选池，打印 `comp.<cid>.key` 的候选表 —— 用来判断
「读数本来是对的、被先验盖掉了」还是「读数本身就是错的」。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "eda-pcb"))
sys.path.insert(0, "eda-pcb")

from fusion import claims as CL                     # noqa: E402
from fusion import priors as PR                     # noqa: E402

RUN = r"work\_submit_out220r36\runs"


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def s1c_ids(s1):
    cs = s1.get("components") or []
    if isinstance(cs, dict):
        return list(cs)
    return [v.get("id") for v in cs if v.get("id")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--bad", required=True, help="融合产物里那个错号")
    ap.add_argument("--out-dir", default=r"work\_fusion_out220no")
    ap.add_argument("--vlm", default=r"work\_fusion_vlm")
    ap.add_argument("--priors", action="store_true", help="先跑一遍先验，再看候选表")
    ap.add_argument("--all", action="store_true", help="打印该图全部 comp.*.key 槽位")
    a = ap.parse_args()
    c = a.case

    doc = json.load(open(os.path.join(a.out_dir, c + ".json"), encoding="utf-8"))
    comps = doc.get("components") or {}
    if a.bad not in comps:
        print("产物里没有键 %s；现有键：%s" % (a.bad, sorted(comps)[:40]))
        return 1
    bb = comps[a.bad]["bbox"]

    s1 = json.load(open(os.path.join(RUN, "stage1", c + ".stage1.json"), encoding="utf-8"))
    best, bs = None, -1.0
    _s1c = s1.get("components") or []
    if isinstance(_s1c, dict):                       # 两种形态都容忍
        _s1c = [dict(v, id=k) for k, v in _s1c.items()]
    for v in _s1c:
        s = _iou(bb, v.get("bbox_json") or v.get("bbox_img") or [])
        if s > bs:
            best, bs = v.get("id"), s
    print("错号 %s 的框 %s -> stage1 cid %s（IoU %.3f）" % (a.bad, bb, best, bs))

    asg = json.load(open(os.path.join(RUN, "stage2", "_assigned", c + ".json"),
                        encoding="utf-8"))
    vp = os.path.join(a.vlm, c + ".json")
    vdoc = json.load(open(vp, encoding="utf-8")) if os.path.exists(vp) else None
    pool = CL.from_stage2(c, s1, asg, bgr=None, vlm=vdoc)
    if a.priors:
        PR.Priors().annotate(pool)
        print("（已跑过先验）")

    ids = ([best] if not a.all
           else [k for k in sorted(s1c_ids(s1))])
    for cid in ids:
        sid = "comp.%s.key" % cid
        sl = pool.hy.slots.get(sid)
        print("\n== 槽位 %s" % sid)
        if sl is None:
            print("   池子里没有这个槽位")
            continue
        print("   question=%r" % sl.question)
        print("   chosen=%r decided_by=%r" % (sl.chosen, sl.decided_by))
        for al in sl.alts:
            print("     %-10s score=%9.4f  src=%-18s ev=%s" % (
                al.value, float(al.score), getattr(al, "source", "?"),
                json.dumps(getattr(al, "evidence", None), ensure_ascii=False)[:110]))
        for kk, vv in (sl.context or {}).items():
            print("     ctx %s = %s" % (kk, json.dumps(vv, ensure_ascii=False)[:160]))
        print("   取景框 region=%s" % json.dumps(
            (pool.regions.get(sid).to_dict() if pool.regions.get(sid) else None),
            ensure_ascii=False)[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
