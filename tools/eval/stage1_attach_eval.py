"""第一阶段第三个交付物的验收：元件 ↔ 导线的几何附着关系。

判据（用 GT 当尺子）：
  1. 先用「同类 + 四角逐项差 <= 20px」把检出元件与 GT 元件贪心一对一配对。
  2. 对每个配上的 GT 元件，取其 GT 引脚点（`pins[key][pin].point`，y 翻成图像坐标）。
  3. 附着召回 = 「有引脚的已配对元件」中，附着点里存在一个落在某个 GT 引脚点 tol 内的比例。
  4. 附着精度 = 「已配对元件的所有附着点」中，落在某个 GT 引脚点 tol 内的比例。

用法: python work\\stage1_attach_eval.py [产物目录] [--tol 8,12,20]
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
from schematic_parser.data import discover, DEFAULT_DATA_ROOT  # noqa: E402

BBOX_TOL = 20.0


def corners_ok(ba, bb, tol=BBOX_TOL):
    return max(abs(ba[i] - bb[i]) for i in range(4)) <= tol


def flatten_at(v):
    """hits[i]['at'] 可能是嵌套点列，取最内层的点。"""
    while isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
        v = v[-1]
    return (float(v[0]), float(v[1]))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = args[0] if args else r"Y:\workspace\EDA\work\stage1_all6"
    tols = [8.0, 12.0, 20.0]
    for a in sys.argv[1:]:
        if a.startswith("--tol"):
            tols = [float(x) for x in a.split("=", 1)[1].split(",")]

    cases = {c.case_id: c for c in discover(DEFAULT_DATA_ROOT)}
    files = sorted(f for f in os.listdir(root) if f.endswith(".stage1.json"))
    print(f"stage1 产物 {len(files)} 份   root={root}")
    print(f"配对判据 同类 + 四角逐项差 <= {BBOX_TOL}px；附着 tol 扫描 {tols}\n")

    n_pair_with_pins = 0
    n_attach_pts = 0
    hit = {t: 0 for t in tols}
    phit = {t: 0 for t in tols}
    unmatched_cases = []
    per_src = {}

    for fn in files:
        cid = fn[: -len(".stage1.json")]
        case = cases.get(cid)
        if case is None:
            unmatched_cases.append(cid)
            continue
        d = json.load(open(os.path.join(root, fn), encoding="utf-8"))
        img = d.get("image") or {}
        H = img.get("h")
        if H is None:
            import cv2
            H = cv2.imread(case.png_path, cv2.IMREAD_GRAYSCALE).shape[0]
        gt = json.load(open(case.json_path, encoding="utf-8"))
        src = os.path.basename(case.png_path).split("-")[-1].replace(".png", "")
        st = per_src.setdefault(src, {"pins": 0, "pts": 0, **{f"h{t}": 0 for t in tols},
                                      **{f"p{t}": 0 for t in tols}})

        gt_items = [(k, v) for k, v in (gt.get("components") or {}).items()]
        preds = [c for c in d["components"]]
        attach = {a["id"]: a for a in d.get("attachments", [])}

        # 贪心配对：同类优先，四角逐项差最小
        cand = []
        for pi, p in enumerate(preds):
            pb = p["bbox_json"]
            for gi, (k, g) in enumerate(gt_items):
                if g.get("type") != p["type"]:
                    continue
                dmax = max(abs(pb[i] - float(g["bbox"][i])) for i in range(4))
                if dmax <= BBOX_TOL:
                    cand.append((dmax, pi, gi))
        cand.sort(key=lambda t: float(t[0]))
        used_p, used_g, pair = set(), set(), {}
        for _, pi, gi in cand:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi)
            used_g.add(gi)
            pair[gi] = pi

        for gi, pi in pair.items():
            gk = gt_items[gi][0]
            gpin = (gt.get("pins") or {}).get(gk) or {}
            gpts = []
            for pin in gpin.values():
                pt = pin.get("point") or {}
                if pt:
                    gpts.append((float(pt["x"]), H - float(pt["y"])))
            if not gpts:
                continue
            n_pair_with_pins += 1
            st["pins"] += 1
            pid = preds[pi]["id"]
            hits = attach.get(pid) or {}
            pts = [flatten_at(h["at"]) for h in (hits.get("hits") or [])]
            n_attach_pts += len(pts)
            st["pts"] += len(pts)
            for t in tols:
                if any(min(max(abs(px - gx), abs(py - gy)) for gx, gy in gpts) <= t
                       for px, py in pts):
                    hit[t] += 1
                    st[f"h{t}"] += 1
                good = sum(1 for px, py in pts
                           if min(max(abs(px - gx), abs(py - gy)) for gx, gy in gpts) <= t)
                phit[t] += good
                st[f"p{t}"] += good

    print("=== 附着 ===")
    print(f"已配对且有 GT 引脚的元件  {n_pair_with_pins}")
    print(f"附着点总数               {n_attach_pts}")
    print(f"  {'tol':>5s} {'命中元件':>9s} {'召回':>8s} {'命中点':>8s} {'精度':>8s}")
    for t in tols:
        r = hit[t] / max(1, n_pair_with_pins)
        p = phit[t] / max(1, n_attach_pts)
        print(f"  {t:5.0f} {hit[t]:9d} {r:8.4f} {phit[t]:8d} {p:8.4f}")

    print("\n=== 按来源（tol=12）===")
    t = 12.0 if 12.0 in tols else tols[0]
    for src, s in sorted(per_src.items(), key=lambda kv: -kv[1]["pins"]):
        r = s[f"h{t}"] / max(1, s["pins"])
        p = s[f"p{t}"] / max(1, s["pts"])
        print(f"  {src:22s} 元件 {s['pins']:5d} R {r:6.3f}   点 {s['pts']:6d} P {p:6.3f}")
    if unmatched_cases:
        print("\n未找到 case 的产物:", unmatched_cases[:10])


if __name__ == "__main__":
    main()
