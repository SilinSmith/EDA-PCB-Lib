# -*- coding: utf-8 -*-
"""核查引脚名救援发出去的名字到底对不对（对比 GT），并导出可用于离线挑门限的特征。

用法：python -X utf8 tools\diag\_pncheck.py --run <run> --split 10GT --n 10 [--dump <json>]
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "diag"))

from _pinledger import _find_assigned, _find_pred, _find_stage1         # noqa: E402
from schematic_parser import metrics as M                              # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json  # noqa: E402


def _cbox(b):
    """文字框（图像帧 [x0,y0,x1,y1]）中心。"""
    try:
        return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)
    except Exception:                                                       # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--dump", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = M.EvalConfig()
    tot = {"emit": 0, "right": 0, "num_not_in_gt": 0, "no_comp": 0,
           "true_tp": 0, "name_ok_pos_bad": 0, "neighbour": 0}
    recs = []
    for c in [x for x in discover(DEFAULT_DATA_ROOT) if x.split == a.split][:a.n]:
        raw = _find_pred(run, c.case_id)
        if raw is None:
            continue
        gt = M.normalise_doc(load_json(Path(c.json_path)))
        pred = M.normalise_doc(raw)
        _c, keymap = M.assign_components(gt, pred, cfg, {})
        pred2 = M.rewrite_keys(pred, keymap)
        asg = _find_assigned(run, c.case_id)
        if not asg:
            continue
        s1 = _find_stage1(run, c.case_id) or {}
        s1_pins = s1.get("pins") or {}
        s1_comps = {}
        for cc_ in (s1.get("components") or []):
            if isinstance(cc_, dict) and cc_.get("id"):
                s1_comps[cc_["id"]] = cc_
        for cid, cc in (asg.get("comps") or {}).items():
            key = ((cc.get("designator") or {}).get("text") or "").strip()
            # keymap: gt 键 -> pred 键
            gkey = None
            for k, v in keymap.items():
                if v == key:
                    gkey = k
            gpins = {M.pin_key_to_number(pk): v for pk, v in
                     ((gt.get("pins") or {}).get(gkey) or {}).items()}
            ppins = {M.pin_key_to_number(pk): v for pk, v in
                     ((pred2.get("pins") or {}).get(key) or {}).items()}
            pts = s1_pins.get(cid) or []
            ys = sorted(float(p[1]) for p in pts if len(p) >= 2)
            dd = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
            dd = [d for d in dd if d > 3]
            pitch = float(sorted(dd)[len(dd) // 2]) if dd else 20.0
            bb = (s1_comps.get(cid) or {}).get("bbox_img") or cc.get("bbox_img") or [0, 0, 0, 0]
            for k, rec in (cc.get("pins") or {}).items():
                nm = ((rec or {}).get("name") or {})
                if nm.get("src") != "rescue":
                    continue
                num = (((rec or {}).get("number") or {}).get("text") or "").strip()
                tot["emit"] += 1
                # ---- 发名时能得到、不依赖 GT 的特征 ----
                fe = {"case": c.case_id, "key": key, "cid": cid, "pin": k,
                      "text": str(nm.get("text") or "").strip(), "num": num,
                      "conf": float(nm.get("conf") or 0.0), "pitch": round(pitch, 2),
                      "n_pts": len(pts)}
                pt = None
                if str(k).isdigit() and int(k) < len(pts):
                    pt = (float(pts[int(k)][0]), float(pts[int(k)][1]))
                tb = nm.get("box")
                if pt and tb:
                    ctr = _cbox(tb)
                    fe["row_off"] = round(abs(ctr[1] - pt[1]), 2) if ctr else None
                    fe["row_off_rel"] = (round(abs(ctr[1] - pt[1]) / pitch, 3)
                                         if ctr and pitch > 0 else None)
                    fe["tb_h"] = round(float(tb[3]) - float(tb[1]), 1)
                nb = ((rec or {}).get("number") or {}).get("box")
                if pt and nb:
                    nctr = _cbox(nb)
                    if nctr:
                        fe["num_d"] = round(math.hypot(nctr[0] - pt[0], nctr[1] - pt[1]), 2)
                # 相对元件框的方位
                try:
                    x0, y0, x1, y1 = [float(v) for v in bb]
                    if pt:
                        dl, dr = abs(pt[0] - x0), abs(pt[0] - x1)
                        dt, db = abs(pt[1] - y0), abs(pt[1] - y1)
                        fe["side"] = "L" if dl == min(dl, dr, dt, db) else \
                                     "R" if dr == min(dl, dr, dt, db) else \
                                     "T" if dt == min(dl, dr, dt, db) else "B"
                except Exception:                                           # noqa: BLE001
                    pass
                # ---- GT 标签 ----
                if not gpins:
                    tot["no_comp"] += 1
                    fe["label"] = "no_gt_comp"
                    recs.append(fe)
                    continue
                if num not in gpins:
                    tot["num_not_in_gt"] += 1
                    fe["label"] = "num_not_in_gt"
                    if not a.quiet:
                        print("  %-3s %-5s %-4s %-8s -> 号 %-4s 不在 GT 键里"
                              % (c.case_id, key, k, nm.get("text"), num))
                    recs.append(fe)
                    continue
                want = (gpins[num] or {}).get("pinname") or ""
                got = str(nm.get("text") or "").strip()
                ok = got.lower() == str(want).strip().lower()
                if ok:
                    tot["right"] += 1
                else:
                    nb_ = []
                    for onum, orec in gpins.items():
                        onm = (orec or {}).get("pinname") or ""
                        if onm and str(onm).strip().lower() == got.lower() and str(onum) != str(num):
                            nb_.append(str(onum))
                    if nb_:
                        tot["neighbour"] += 1
                pd = None
                gp = M._pt((gpins[num] or {}).get("point")) if hasattr(M, "_pt") else None
                if gp is None:
                    sp = (gpins[num] or {}).get("point") or {}
                    gp = (float(sp.get("x", 0)), float(sp.get("y", 0))) if sp else None
                pp = ppins.get(num) or {}
                sp2 = pp.get("point") or {}
                if gp and sp2:
                    pd = math.hypot(float(sp2.get("x", 0)) - gp[0],
                                    float(sp2.get("y", 0)) - gp[1])
                pos_ok = pd is not None and pd <= cfg.pin_tol
                if ok and pos_ok:
                    tot["true_tp"] += 1
                    fe["label"] = "tp"
                elif ok and not pos_ok:
                    tot["name_ok_pos_bad"] += 1
                    fe["label"] = "name_ok_pos_bad"
                else:
                    fe["label"] = "name_bad"
                fe["gt_name"] = str(want).strip()
                fe["d"] = None if pd is None else round(pd, 2)
                recs.append(fe)
                if not a.quiet:
                    print("  %-3s %-5s pin%-4s %-8s -> GT %-8s %s  d=%s%s"
                          % (c.case_id, key, k, got, want,
                             ("✓tp" if (ok and pos_ok) else ("~名对位偏" if ok else "✗名错")),
                             ("%.1f" % pd) if pd is not None else "?",
                             ("  邻脚名=%s" % ",".join(sorted(nb_))) if (not ok and nb_) else ""))
    print("发出 %d  名对 %d（%.3f）其中真 tp %d / 名对位偏 %d；名错但=邻脚真名 %d；"
          "号不在 GT 键 %d；无 GT 元件 %d"
          % (tot["emit"], tot["right"], 1.0 * tot["right"] / max(1, tot["emit"]),
             tot["true_tp"], tot["name_ok_pos_bad"], tot["neighbour"],
             tot["num_not_in_gt"], tot["no_comp"]))
    # ---- 特征 × 标签 的小结：每个特征中位数的「对/错」分组 ----
    feats = ("conf", "row_off", "row_off_rel", "tb_h", "num_d", "pitch")
    for f in feats:
        good = [r[f] for r in recs if r.get("label") == "tp" and isinstance(r.get(f), (int, float))]
        bad = [r[f] for r in recs if r.get("label") in ("name_bad", "name_ok_pos_bad")
               and isinstance(r.get(f), (int, float))]
        def _md(v):
            if not v:
                return "-"
            v = sorted(v)
            return "%.2f" % v[len(v) // 2]
        print("  特征 %-12s tp 中位 %-7s | 名错/位偏 中位 %-7s | tp n=%d 错 n=%d"
              % (f, _md(good), _md(bad), len(good), len(bad)))
    if a.dump:
        Path(a.dump).write_text(json.dumps(recs, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        print("  明细已写 %s（%d 条）" % (a.dump, len(recs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
