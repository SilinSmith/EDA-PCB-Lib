# -*- coding: utf-8 -*-
"""轴层的**决策级**差分：轴开/关时，最终定下来的值到底哪里不一样、谁对谁错。

动机（owner m24624）：「因为一点点丢分就否定一整个链路」是不可取的 —— 所以要先看清
`FUSION_AXIS=1` 那 −0.16/−0.75 是**轴说错了**，还是**下游把它说对的话吃掉了**。

做法：同一张图建两次池（一次不调 `AX.inject`、一次调），各自跑完
`priors.annotate` → `solve` → `write_back`，再逐槽位比对最终值；
槽位通过 `stage1.bbox_json` 与 GT `bbox` 的 IoU≥0.5 配上 GT，于是每条差异都能判对错。

用法：
    python work\\tools\\diag\\_axisdiff.py 0001 0042
"""
import json
import os
import sys

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, r"Y:\workspace\EDA")
os.environ.setdefault("FUSION_STAGE1_DIR",
                      r"Y:\workspace\EDA\work\_submit_out220r36\runs\stage1")
os.environ.setdefault("FUSION_ASSIGNED_DIR",
                      r"Y:\workspace\EDA\work\_submit_out220r36\runs\stage2\_assigned")

from fusion import axis as AX          # noqa: E402
from fusion import claims as CL        # noqa: E402
from fusion import priors as PR        # noqa: E402
from fusion import solve as SV         # noqa: E402

VLM_DIR = r"Y:\workspace\EDA\work\_fusion_vlm"


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def gt_map(case):
    """cid -> GT 位号（JSON 空间 IoU≥0.5）。"""
    from schematic_parser import data as Sdata
    c = Sdata.find_case(case)
    gt = json.load(open(c.json_path, encoding="utf-8"))
    out = {}
    # GT 的 `components` 是 **dict，键 = 位号**（`{"U1": {"Name":…, "type":…, "bbox":[…]}}`），
    # 不是 list —— 照 list 遍历会拿到字符串键然后 `.get` 炸掉。
    for key, cv in (gt.get("components") or {}).items():
        if isinstance(cv, dict) and cv.get("bbox"):
            out[key] = cv["bbox"]
    s1 = load(os.path.join(os.environ["FUSION_STAGE1_DIR"], "%s.stage1.json" % case))
    m = {}
    for comp in s1.get("components") or []:
        bj = comp.get("bbox_json")
        if not bj:
            continue
        best, bk = 0.0, None
        for key, gb in out.items():
            v = iou(bj, gb)
            if v > best:
                best, bk = v, key
        if bk is not None and best >= 0.5:
            m[comp["id"]] = bk
    return m


def run(case, use_axis):
    doc = load(os.path.join(os.environ["FUSION_STAGE1_DIR"], "%s.stage1.json" % case))
    asg = load(os.path.join(os.environ["FUSION_ASSIGNED_DIR"], "%s.json" % case))
    vp = os.path.join(VLM_DIR, "%s.json" % case)
    vlm = load(vp) if os.path.exists(vp) else None
    pool = CL.from_stage2(case, doc, asg, bgr=CL.load_image(doc), vlm=vlm)
    st = {}
    mid = {}
    if use_axis:
        st = AX.inject(pool)
        # 注入之后、先验/求解之前，池子里的最优值是什么 —— 用来分清
        # 「轴没赢下池子」和「轴赢下池子但被 solve 的结构判据改回去了」。
        for sid in pool.slot_ids():
            b = pool.best(sid)
            mid[sid] = (str(b.value), str(b.source)) if b is not None else (None, None)
    PR.Priors().annotate(pool)
    sol = SV.solve(pool)
    SV.write_back(pool, sol)
    got = {}
    for sid in pool.slot_ids():
        b = pool.best(sid)
        got[sid] = (str(b.value), str(b.source)) if b is not None else (None, None)
    return got, st, mid


def main():
    cases = sys.argv[1:] or ["0001"]
    print("AX.ON(默认) =", AX.ON)
    for case in cases:
        gm = gt_map(case)
        sid2cid = {}
        off, _, _ = run(case, False)
        on, st, mid = run(case, True)
        print("\n===== case %s =====" % case)
        print("inject:", {k: v for k, v in st.items() if k != "plan"})
        # 轴自己的候选有没有在后验池里胜出？
        won_mid = [s for s in mid if mid[s][1] == "geo:axis"]
        print("轴候选在池子里当过最优的槽位：%d 个 %s" % (len(won_mid), won_mid[:8]))
        # 最终值与「轴注入后的池子最优」不同的槽位 = 被下游结构判据改回去的
        lost = [s for s in won_mid if on.get(s) != mid.get(s)]
        print("  其中被下游改回去的：%d 个" % len(lost))
        for s in lost[:8]:
            print("     %-22s 轴:%s → 最终:%s" % (s, mid[s], on.get(s)))
        diff = [(s, off.get(s), on.get(s)) for s in sorted(set(off) | set(on))
                if off.get(s) != on.get(s)]
        print("最终值不同的槽位：%d 个" % len(diff))
        n_off_ok = n_on_ok = n_both_bad = 0
        for sid, o, n in diff:
            cid = None
            for s2 in (sid,):
                pass
            # 从 `priors`/`claims` 命名里取 cid：comp.<cid>.key / comp.<cid>.value ...
            cid = sid.split(".")[1] if sid.startswith("comp.") and sid.count(".") >= 2 else None
            gk = gm.get(cid) if cid else None
            def is_ok(v):
                if not gk or v[0] is None:
                    return None
                return str(v[0]) == str(gk)
            ko, kn = is_ok(o), is_ok(n)
            if ko is True and kn is not True:
                n_off_ok += 1
            elif kn is True and ko is not True:
                n_on_ok += 1
            elif ko is False and kn is False:
                n_both_bad += 1
            tag = {True: "对", False: "错", None: "?"}
            print("  %-22s GT=%-6s 关:%-8s(%-12s %s)  开:%-8s(%-12s %s)"
                  % (sid, gk, o[0], o[1], tag[ko], n[0], n[1], tag[kn]))
        print("  >> 关对开错 %d ｜ 开对关错 %d ｜ 两边都错 %d" % (n_off_ok, n_on_ok, n_both_bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
