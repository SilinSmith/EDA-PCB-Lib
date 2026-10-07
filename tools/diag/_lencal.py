# -*- coding: utf-8 -*-
"""长度律的**域校准**：`fusion/charwidth.py` 是从**标注框**（manifest）拟合的，
而轴层吃的是 **stage1 检出的文字框** —— 两个域每字符宽并不一样，直接套用会系统性
高估框里应有的字数（stage1 框偏宽），于是把「本来就对的两字读数」判成「长度错」，
轴层就会去给它补字（实测 case 0001 的 `C4`→`C10`、`C8`→`C11` 就是这么来的，两条都错）。

做法：在**真的会被轴层看到的那些 item** 上，只取「读数 == GT 位号」的（**确认正确**
的读法不会缺字），量它们 `w/h - pred_wh_text(text)` 的分布 ⇒ 得到域偏移 `CAL`、
以及属于 stage1 域的残差 σ。

用法：python work\\tools\\diag\\_lencal.py [--cases 0001,1,...] [--all]
"""
import json
import os
import statistics as stx
import sys

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, r"Y:\workspace\EDA")
os.environ.setdefault("FUSION_STAGE1_DIR",
                      r"Y:\workspace\EDA\work\_submit_out220r36\runs\stage1")
os.environ.setdefault("FUSION_ASSIGNED_DIR",
                      r"Y:\workspace\EDA\work\_submit_out220r36\runs\stage2\_assigned")

from fusion import axis as AX          # noqa: E402
from fusion import claims as CL        # noqa: E402

VLM_DIR = r"Y:\workspace\EDA\work\_fusion_vlm"
FAST = "1 2 3 4 5 7 8 9 10 0092 0048 0019 0024 0075 0086 0002 0001 0042 0103".split()


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def gt_map(case, s1):
    from schematic_parser import data as Sdata
    c = Sdata.find_case(case)
    gt = json.load(open(c.json_path, encoding="utf-8"))
    gb = {}
    for key, cv in (gt.get("components") or {}).items():
        if isinstance(cv, dict) and cv.get("bbox"):
            gb[key] = cv["bbox"]
    m = {}
    for comp in s1.get("components") or []:
        bj = comp.get("bbox_json")
        if not bj:
            continue
        best, bk = 0.0, None
        for key, b in gb.items():
            v = iou(bj, b)
            if v > best:
                best, bk = v, key
        if bk is not None and best >= 0.5:
            m[comp["id"]] = bk
    return m


def main():
    cases = FAST
    if "--all" in sys.argv:
        cases = None
    for i, a in enumerate(sys.argv):
        if a == "--cases" and i + 1 < len(sys.argv):
            cases = sys.argv[i + 1].split(",")
    if cases is None:
        from schematic_parser import data as Sdata
        cases = [c.case_id for c in Sdata.discover()]

    rec = []          # (text, w, h, resid)
    n_item = n_conf = 0
    for case in cases:
        p1 = os.path.join(os.environ["FUSION_STAGE1_DIR"], "%s.stage1.json" % case)
        pa = os.path.join(os.environ["FUSION_ASSIGNED_DIR"], "%s.json" % case)
        if not (os.path.exists(p1) and os.path.exists(pa)):
            continue
        try:
            s1 = load(p1)
            asg = load(pa)
            vp = os.path.join(VLM_DIR, "%s.json" % case)
            vlm = load(vp) if os.path.exists(vp) else None
            pool = CL.from_stage2(case, s1, asg, bgr=CL.load_image(s1), vlm=vlm)
            gm = gt_map(case, s1)
            items = AX.items_from_pool(pool)
        except Exception as exc:                       # noqa: BLE001
            print("  跳过 %s：%s" % (case, exc))
            continue
        n_item += len(items)
        for it in items:
            cid = (pool.meta.get(it.key) or {}).get("cid")
            gk = gm.get(cid) if cid else None
            if not gk or str(it.text) != str(gk) or it.h <= 0:
                continue
            n_conf += 1
            rec.append((it.text, it.w, it.h,
                        it.w / it.h - AX.pred_wh_text(it.text)))
    print("扫了 %d 张，item 共 %d 个，其中「读数 == GT 位号」确认正确的 %d 个"
          % (len(cases), n_item, n_conf))
    if not rec:
        return 1
    r = sorted(x[3] for x in rec)
    n = len(r)
    med = r[n // 2]
    mean = sum(r) / n
    sd = stx.pstdev(r) if n > 1 else 0.0
    print("残差 `w/h - pred`：中位 %+.3f ｜ 均值 %+.3f ｜ 标准差 %.3f ｜ p10 %+.3f p90 %+.3f"
          % (med, mean, sd, r[int(0.10 * n)], r[int(0.90 * n)]))
    for lo, hi in ((-9, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.2)):
        k = sum(1 for x in r if lo <= x < hi)
        print("   [%+.1f,%+.1f) : %4d  (%.1f%%)" % (lo, hi, k, 100.0 * k / n))
    # 只按「确认正确」的读法重估：轴层该用的偏移与 σ
    print("\n建议：CAL = %+.3f（把域偏移加进 pred_wh_text），SIG = %.3f（stage1 域残差）"
          % (med, max(0.10, sd)))
    # 对照：套上 CAL 之后，「把 2 字读数当成 3 字框」这种误判会掉多少
    print("\n误判抽样（stage1 域，2 字读数）：")
    for t, w, h, _r in rec[:0]:
        pass
    two = [(t, w, h) for t, w, h, _ in rec if len(t) == 2][:6]
    for t, w, h in two:
        base = AX.len_prob_text(w, h, t)
        three = AX.len_prob_text(w, h, t + "5")
        print("   %-6s w/h=%.3f  原样 %.3f ｜ 补成 3 字 %.3f ｜ 差 %+.3f"
              % (t, w / h, base, three, three - base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
