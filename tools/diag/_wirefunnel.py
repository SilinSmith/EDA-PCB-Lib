# -*- coding: utf-8 -*-
"""_wirefunnel.py —— 逐 case 漏斗：stage1 线段 → stage1 网 → stage2 → fusion → 输出 net_line。

全是用现成产物数数：
  work\\_fast_out_reg9\\runs\\{stage1,stage2,stage3_vlm,fusion}   （官方 9 张）
  work\\_f220_A                                                  （派生 19 张的最终产物）
  work\\_full220\\runs\\{stage1,stage2,stage3_vlm}                 （220 张，取派生 19 张）
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
W = ROOT / "work"
GT9 = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
D19 = ["0092", "0048", "0019", "0024", "0086", "0042",
       "0075", "0083", "0090", "0057", "0062", "0032", "0028",
       "0175", "0198", "0105", "0106", "0036", "0120"]


def jload(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception as e:                                    # noqa: BLE001
        return {"__err__": str(e)}


def n_edges(d):
    return sum(len(n.get("edges") or {}) for n in (d.get("nets") or {}).values())


def n_net_no_edge(d):
    return sum(1 for n in (d.get("nets") or {}).values() if not (n.get("edges") or {}))


def s1_summary(p):
    d = jload(p)
    if "__err__" in d:
        return {"err": d["__err__"]}
    segs = d.get("segments") or []
    w = d.get("wires") or {}
    nets = d.get("nets") or []
    grp = d.get("seg_groups") or d.get("groups") or []
    return {
        "n_seg": len(segs),
        "wires_n_seg": w.get("n_segments"),
        "wires_n_ep": w.get("n_endpoints"),
        "wires_n_junc": w.get("n_junctions"),
        "wires_n_poly": w.get("n_polylines"),
        "wires_n_drop_comp": w.get("n_dropped_comp"),
        "wires_n_comp": w.get("n_components"),
        "wires_n_comp_cand": w.get("n_comp_cand"),
        "wires_n_seg_split": w.get("n_segments_split"),
        "wires_n_drop_poly": w.get("n_dropped_poly"),
        "n_net": len(nets),
        "n_net_nopin": sum(1 for n in nets if not (n.get("pins") or [])),
        "n_seg_in_nets": sum(len(n.get("seg_index") or []) for n in nets),
        "n_groups": len(grp),
        "timing": d.get("timing"),
        "errors": (d.get("errors") or [])[:3],
    }


rows = []
for grp, ids, s1dir, s2dir, s3dir, fusdir, outdir, outsuffix in (
        ("GT9", GT9, W / "_fast_out_reg9/runs/stage1", W / "_fast_out_reg9/runs/stage2",
         W / "_fast_out_reg9/runs/stage3_vlm", W / "_fast_out_reg9/runs/fusion",
         W / "_fast_out_reg9/final_results", "_Pred.json"),
        ("D19", D19, W / "_full220/runs/stage1", W / "_full220/runs/stage2",
         W / "_full220/runs/stage3_vlm", None, W / "_f220_A", ".json")):
    for i in ids:
        r = {"case": i, "grp": grp}
        r["s1"] = s1_summary(s1dir / ("%s.stage1.json" % i))
        for tag, d0 in (("s2", s2dir), ("s3vlm", s3dir)):
            p = d0 / ("%s.json" % i)
            d = jload(p)
            r[tag] = {"exist": p.exists(), "n_net": len(d.get("nets") or {}),
                      "n_edge": n_edges(d), "n_net_no_edge": n_net_no_edge(d)}
        p = outdir / ("%s%s" % (i, outsuffix))
        d = jload(p)
        r["out"] = {"exist": p.exists(), "n_net": len(d.get("nets") or {}),
                    "n_edge": n_edges(d), "n_net_no_edge": n_net_no_edge(d)}
        if fusdir:
            p2 = fusdir / ("%s.json" % i)
            d2 = jload(p2)
            r["fus"] = {"exist": p2.exists(), "n_net": len(d2.get("nets") or {}),
                        "n_edge": n_edges(d2)}
        rows.append(r)

lines = []
hdr = ("%-6s %-4s | %6s %6s %6s %6s | %5s %5s %6s | %5s %5s | %5s %5s | %5s %5s"
       % ("case", "grp", "n_seg", "nPoly", "nEndp", "nJunc",
          "s1net", "segIn", "nPin0", "s2net", "s2edge", "fnet", "fedge", "outN", "outE"))
lines.append(hdr)
lines.append("-" * len(hdr))
for r in rows:
    s1 = r["s1"]
    if "err" in s1:
        lines.append("%-6s %-4s  stage1 读取失败 %s" % (r["case"], r["grp"], s1["err"]))
        continue
    f = r.get("fus") or {}
    lines.append("%-6s %-4s | %6s %6s %6s %6s | %5d %5d %6d | %5d %5d | %5s %5s | %5d %5d"
                 % (r["case"], r["grp"], s1.get("wires_n_seg"), s1.get("wires_n_poly"),
                    s1.get("wires_n_ep"), s1.get("wires_n_junc"),
                    s1["n_net"], s1["n_seg_in_nets"], s1["n_net_nopin"],
                    r["s2"]["n_net"], r["s2"]["n_edge"],
                    f.get("n_net", "-"), f.get("n_edge", "-"),
                    r["out"]["n_net"], r["out"]["n_edge"]))
txt = "\n".join(lines)
(ROOT / "work" / "_wirefunnel_out.txt").write_text(txt, encoding="utf-8")
(ROOT / "work" / "_wirefunnel.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
print(txt)
print()
print("=== 逐 case stage1 顶部字段（含 timing / errors） ===")
for r in rows:
    s1 = r["s1"]
    if "err" in s1:
        continue
    print("%-6s %-4s timing=%s errors=%s drop_comp=%s comp=%s cand=%s split=%s drop_poly=%s"
          % (r["case"], r["grp"], json.dumps(s1["timing"], ensure_ascii=False),
             s1["errors"], s1["wires_n_drop_comp"], s1["wires_n_comp"],
             s1["wires_n_comp_cand"], s1["wires_n_seg_split"], s1["wires_n_drop_poly"]))
