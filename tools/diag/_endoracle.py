# -*- coding: utf-8 -*-
"""netL 端点定位的**上限**取证：拿 GT 的端点当神谕，修掉配对网里近失的边，看总分能到哪。

问题（`work/_tjsplit.py` 普查，当前交付 83.23 / 导线 F1 0.5674）：近失边
（最近 GT 边 ≤15px）fp 217 / fn 232，坏端 8px 内 **98% 有我方另一条边的端点**、
81% 有 GT 边端点、31% 有元件 bbox —— 全是 5~8px 的**纵向长短错**，且 over/short 相当
（sub 73 / super 81）⇒ 融合层再挤「均匀内收」已到拐点（§5.100：1px 最好、2/3px 更差）。

本探针不改交付，只回答一个问题：**「把配对网里近失边的端点钉到 GT 的位置」值多少分？**
- `--mode end` ：只把坏端挪到 GT 边最近的那个端点（长度修正，横向不动）——**可被证据规则实现**的上限；
- `--mode full`：整条边换成 GT 边（几何神谕，绝对上限）；
- `--mode none`：不改（自检，必须复现 83.23）。

产物写到 `work/_fast_out_<out>/final_results/`，再用
`python -X utf8 tools\\eval\\fast.py --score-only --cases 1,...,10 --out <out>` 打分。

用法：
  python -X utf8 work\\_endoracle.py --mode end  --out endorc_end
  python -X utf8 work\\_endoracle.py --mode full --out endorc_full
  python -X utf8 eda-pcb\\tools\\eval\\fast.py --score-only --cases 1,2,3,4,5,6,7,8,9,10 --out endorc_end
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # archived copy: work/ -> tools/diag/
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import metrics as M                      # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover   # noqa: E402

INF = float("inf")
RUNS = ROOT / "work"


def _xy(p):
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    return float(p[0]), float(p[1])


def _put(raw_pt, x, y):
    if isinstance(raw_pt, dict):
        return {"x": round(x, 2), "y": round(y, 2)}
    return [round(x, 2), round(y, 2)]


def _edges(net):
    out = []
    for k, v in (net.get("edges") or {}).items():
        e = M._edge_points(v)
        if e is not None:
            out.append((k, (e[0], e[1]), v))
    return out


def _pt_seg(pt, seg):
    (ax, ay), (bx, by) = seg
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 <= 1e-9 else ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / L2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    qx, qy = ax + dx * t, ay + dy * t
    return ((pt[0] - qx) ** 2 + (pt[1] - qy) ** 2) ** 0.5


def _d(pt, seg):
    return _pt_seg(pt, seg)


def _segd(pe, ge):
    return max(_d(pe[0], ge), _d(pe[1], ge))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUNS / "_fast_out_finalrd8"))
    ap.add_argument("--out", default="endorc")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--r", type=float, default=15.0, help="近失半径 px")
    ap.add_argument("--mode", default="end", choices=["end", "full", "none"])
    a = ap.parse_args()

    src = Path(a.run)
    dst = RUNS / ("_fast_out_" + a.out)
    if (dst / "final_results").exists():
        shutil.rmtree(dst / "final_results")
    (dst / "final_results").mkdir(parents=True, exist_ok=True)

    cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    data = {}
    for c in discover(DEFAULT_DATA_ROOT):
        cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
        if cid in cases:
            data[cid] = c
    cfg = M.EvalConfig()
    nfix = 0
    perfix = []
    for cid in cases:
        c = data.get(cid)
        p_in = src / "final_results" / ("%s_Pred.json" % cid)
        if c is None or not p_in.exists():
            print("! 缺 %s" % cid)
            continue
        raw = json.loads(p_in.read_text(encoding="utf-8"))
        gt = M.normalise_doc(json.loads(Path(c.json_path).read_text(encoding="utf-8")))
        pred = M.normalise_doc(json.loads(json.dumps(raw)))
        _comp, keymap = M.assign_components(gt, pred, cfg, {})
        pred2 = M.rewrite_keys(pred, keymap)
        _net, pairs = M.net_hypergraph_f1(gt, pred2, cfg, {})
        g_items = list(gt["nets"].items())
        p_items = list(pred2["nets"].items())
        fix = {}                                       # (pred_net_id, edge_key) -> (p0, p1)
        fix_geom = {}                                  # 兜底：按几何（pred2 的边几何 == 原始 doc）
        n = 0
        for i, j in pairs:
            if i >= len(p_items) or j >= len(g_items):
                continue
            pnet_id, pnet = p_items[i]
            gnet = g_items[j][1]
            pe_all = _edges(pnet)
            ge_all = _edges(gnet)
            if not pe_all or not ge_all:
                continue
            cost = [[(M._seg_distance(pe[1][:2], ge[1][:2], cfg.line_unordered)
                      if M._seg_distance(pe[1][:2], ge[1][:2], cfg.line_unordered)
                      <= cfg.line_tol else INF) for ge in ge_all] for pe in pe_all]
            got = M._assign(cost)
            gi = {b for _a, b in got}
            pi = {x for x, _b in got}
            for x, (k, pe, _raw_v) in enumerate(pe_all):
                if x in pi:
                    continue                            # 已配上的边不动
                best = (INF, None, None)
                for y, ge in enumerate(ge_all):
                    if y in gi and a.mode == "end":
                        continue                        # end 模式只借"没被用掉"的 GT 边
                    d = _segd(pe, ge[1])
                    if d < best[0]:
                        best = (d, ge[1], y)
                if best[1] is None or best[0] > a.r:
                    continue
                ge = best[1]
                if a.mode == "full":
                    new = (ge[0], ge[1])
                else:
                    d0, d1 = _d(pe[0], ge), _d(pe[1], ge)
                    bad, good = (0, 1) if d0 >= d1 else (1, 0)
                    tgt = ge[0] if _d(ge[0], pe) <= _d(ge[1], pe) else ge[1]
                    new = [pe[0], pe[1]]
                    new[bad] = tgt
                    new = (tuple(new[0]), tuple(new[1]))
                    if abs(new[0][0] - pe[0][0]) < 1e-6 and abs(new[0][1] - pe[0][1]) < 1e-6 \
                       and abs(new[1][0] - pe[1][0]) < 1e-6 and abs(new[1][1] - pe[1][1]) < 1e-6:
                        continue
                fix[(pnet_id, k)] = new
                fix_geom[(round(pe[0][0], 2), round(pe[0][1], 2),
                          round(pe[1][0], 2), round(pe[1][1], 2))] = new
                n += 1
                if len(perfix) < 12:
                    perfix.append((cid, k, pe, new))
        # ---- 把修改落到**原始** doc（键按 (net_id, edge_key) 对齐；几何一致才动）
        napply = 0
        for net_id, net in (raw.get("nets") or {}).items():
            for k, v in list((net.get("edges") or {}).items()):
                key = (net_id, k)
                e = M._edge_points(v)
                if e is None:
                    continue
                tgt = fix.get(key)
                if tgt is None:
                    tgt = fix_geom.get((round(e[0][0], 2), round(e[0][1], 2),
                                        round(e[1][0], 2), round(e[1][1], 2)))
                if tgt is None:
                    continue
                if a.mode == "end":                     # 只挪坏端（fix 里除被挪端外，另一端与原始几何相同）
                    cur = []
                    for i in (0, 1):
                        moved = (abs(e[i][0] - tgt[i][0]) > 1e-6
                                 or abs(e[i][1] - tgt[i][1]) > 1e-6)
                        cur.append(tgt[i] if moved else e[i])
                    tgt = (tuple(cur[0]), tuple(cur[1]))
                net["edges"][k] = [_put(v[0], tgt[0][0], tgt[0][1]),
                                   _put(v[1], tgt[1][0], tgt[1][1])]
                napply += 1
        nfix += n
        (dst / "final_results" / ("%s_Pred.json" % cid)).write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        print("case %-4s 判定要修 %d 条边，落到原始 doc %d 条" % (cid, n, napply))
    print("合计 %d 条（mode=%s r=%.1f）⇒ %s" % (nfix, a.mode, a.r, dst))
    for cid, k, pe, new in perfix:
        print("  %s %s  (%.0f,%.0f)-(%.0f,%.0f) → (%.0f,%.0f)-(%.0f,%.0f)"
              % (cid, k, pe[0][0], pe[0][1], pe[1][0], pe[1][1],
                 new[0][0], new[0][1], new[1][0], new[1][1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
