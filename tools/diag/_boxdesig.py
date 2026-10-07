"""位号归因：GT 的大 IC（`box`）为什么**检出了却键对不上**。

`_boxkey.py` 已经证明 19 张里 **A 类 100% / B 类 0%** —— 大 IC 一个都没漏检
（IoU 0.97~0.99），死的全是位号。这个工具接着往下问三个问题：

  1. **真位号有没有被别的框抢走？** GT 键本身是不是出现在我们的某个预测键上
     （`0001 U1` vs 我们 `USB3` 是「读错」，而 `0001 U1` vs 另一个框叫 `U1` 是「抢走」，
     修法完全不同）。
  2. **我们的键和 GT 键差在哪？** 前缀 / 后缀 / 差一个字符 / 大小写
     （`0019 u1` → 我们 `U11`、`0013 U202` → 我们 `U22` 都属这类）。
  3. **真位号印在框里还是框外？** 从 `stage1` 的 `texts`（只有框、没有内容）看
     GT 框**内部**、**顶部带**、**底部带**、**左侧带**各有几个文字框 ——
     决定该把「读位号的取景窗」放在哪。
     裁图已确认三种画图工具都把位号画了（框内左上 / 框上 / 框下），
     位置随工具变，所以这一项必须量而不能猜。

用法：
  python work\\tools\\diag\\_boxdesig.py --dir work\\_fast_out --show 20
  python work\\tools\\diag\\_boxdesig.py --dir work\\_fusion_out220no4 --show 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser import data as D                                    # noqa: E402
from schematic_parser.metrics import (                                    # noqa: E402
    EvalConfig, assign_components, canon_type, normalise_doc, rewrite_keys,
)


def _bbox(v):
    b = (v or {}).get("bbox") or (v or {}).get("box")
    if not b or len(b) < 4:
        return None
    x1, y1, x2, y2 = [float(t) for t in b[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _lev(a: str, b: str) -> int:
    """编辑距离（都很短，直接 DP）。"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if not la:
        return lb
    if not lb:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb]


def relation(gt_key: str, our_key: str) -> str:
    """GT 键与我们键的关系（都小写化后比）。"""
    g, o = gt_key.lower(), our_key.lower()
    if g == o:
        return "相同"
    if o.startswith(g):
        return "我们多了后缀 +%s" % o[len(g):]
    if g.startswith(o):
        return "我们少了后缀 -%s" % g[len(o):]
    if g in o:
        return "GT 是我们键的子串"
    if o in g:
        return "我们是 GT 键的子串"
    d = _lev(g, o)
    if d <= 2:
        return "编辑距离 %d" % d
    return "无关"


def find_pred(cid, pdir):
    for cand in (pdir / f"{cid}.json",
                 pdir / "final_results" / f"{cid}_Pred.json",
                 pdir / "runs" / "fusion" / f"{cid}.json"):
        if cand.exists():
            return cand
    return None


def load_stage1(cid, pdir):
    f = pdir / "runs" / "stage1" / f"{cid}.stage1.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def _tbox(t):
    """**文字框用的是 `bbox_img`**（`bbox`/`box` 是元件字段）。
    第一版我复用了 `_bbox()` ⇒ 每个文字框都返回 None ⇒ 四条带全 0，
    而直接探针明明在 `0019 u1` 框内数到 81 个文字框。"""
    if not isinstance(t, dict):
        return None
    b = t.get("bbox_img") or t.get("bbox") or t.get("box")
    if not b or len(b) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in b[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def to_img(gb, h_img):
    """GT 框是 **JSON 空间（y 向上）**，`stage1.texts` 是 **图像空间（y 向下）**。
    不翻转就永远是「一个文字框都找不到」—— 我第一版就踩了这个坑。"""
    if not h_img:
        return gb
    x1, y1, x2, y2 = gb
    return x1, h_img - y2, x2, h_img - y1


def boxes_in(st1, gb, pad=2.0):
    """把 stage1 的 texts 按相对 GT 框（**已经是图像空间**）的位置分桶。"""
    x1, y1, x2, y2 = gb
    w, h = x2 - x1, y2 - y1
    bx = min(40.0, max(8.0, 0.12 * h))        # 顶/底带的厚度
    bl = min(60.0, max(14.0, 0.10 * w))       # 左/右带的宽度
    out = {"in": [], "top": [], "bot": [], "left": [], "right": []}
    for t in (st1.get("texts") or []):
        b = _tbox(t)
        if not b:
            continue
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        tw, th = b[2] - b[0], b[3] - b[1]
        if th > 0.6 * h or tw > 0.8 * w:      # 明显不是一个小位号
            continue
        if x1 - pad <= cx <= x2 + pad and y1 - pad <= cy <= y2 + pad:
            if x1 + pad < cx < x2 - pad and y1 + pad < cy < y2 - pad:
                out["in"].append((round(cx - x1), round(cy - y1), round(tw), round(th)))
            elif y1 - bx <= cy < y1 + pad and cx <= x1 + 0.5 * w:
                out["top"].append((round(cx - x1), round(cy - y1), round(tw), round(th)))
            elif y2 - pad < cy <= y2 + bx and cx <= x1 + 0.5 * w:
                out["bot"].append((round(cx - x1), round(cy - y1), round(tw), round(th)))
            elif x1 - bl <= cx <= x1 + pad:
                out["left"].append((round(cx - x1), round(cy - y1), round(tw), round(th)))
            elif x2 - pad <= cx <= x2 + bl:
                out["right"].append((round(cx - x1), round(cy - y1), round(tw), round(th)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "work" / "_fast_out"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--show", type=int, default=20)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    want = {x.strip() for x in a.cases.split(",") if x.strip()} or None
    cases = {c.case_id: c for c in D.discover(D.DEFAULT_DATA_ROOT)
             if c.json_path and (not want or c.case_id in want)}
    pdir = Path(a.dir)
    cfg = EvalConfig()

    tot = {"gt": 0, "un": 0, "stolen": 0, "near": 0, "st1": 0,
           "caseonly": 0, "gt_nonupper": 0, "ci_match": 0,
           "with_in": 0, "with_top": 0, "with_bot": 0, "with_lr": 0, "with_any": 0,
           "in_n": 0}
    rows = []
    for cid, case in sorted(cases.items()):
        f = find_pred(cid, pdir)
        if not f:
            continue
        st1 = load_stage1(cid, pdir)
        h_img = ((st1 or {}).get("image") or {}).get("h")
        gt_n = normalise_doc(D.load_json(case.json_path))
        pred_n = normalise_doc(json.loads(f.read_text(encoding="utf-8")))
        _c, keymap = assign_components(gt_n, pred_n, cfg, {})
        pred2 = rewrite_keys(pred_n, keymap)

        boxk = [k for k, v in gt_n["components"].items()
                if canon_type((v or {}).get("type")) in D.IC_TYPES]
        mapped = set(keymap.values())          # = 已被配上的 **GT** 键
        our_keys = {k: _bbox(v) for k, v in pred_n["components"].items()}
        pboxes = [(k, b) for k, b in our_keys.items() if b]
        our_lower = {k.lower(): k for k in our_keys}
        gt_lower_all = {k.lower() for k in gt_n["components"]}

        for k in boxk:
            tot["gt"] += 1
            if k != k.upper():
                tot["gt_nonupper"] += 1
            if k in mapped:
                continue
            # 大小写不敏感地在**我们自己的键**里能不能找到？
            if k.lower() in our_lower and our_lower[k.lower()] != k:
                tot["ci_match"] += 1
            gb = _bbox(gt_n["components"].get(k))
            if not gb:
                continue
            tot["un"] += 1
            hit = sorted(((round(_iou(gb, b), 3), pk) for pk, b in pboxes
                          if _iou(gb, b) > 0.05), reverse=True)
            our = hit[0][1] if hit else None
            if our and our.lower() == k.lower():
                tot["caseonly"] += 1
            # 1) 真位号被别的框抢走了吗？
            stolen = None
            for pk, b in pboxes:
                if pk != our and pk.lower() == k.lower():
                    stolen = (pk, round(_iou(gb, b), 3))
            if stolen:
                tot["stolen"] += 1
            # 2) 我们的键与 GT 键的关系
            rel = relation(k, our) if our else "（没重叠）"
            if our and rel not in ("无关", "相同"):
                tot["near"] += 1
            # 3) 文字框分布（GT 框翻到图像空间）
            bk = None
            if st1 and h_img:
                tot["st1"] += 1
                bk = boxes_in(st1, to_img(gb, h_img))
                n_in = len(bk["in"])
                tot["in_n"] += n_in
                if n_in:
                    tot["with_in"] += 1
                if bk["top"]:
                    tot["with_top"] += 1
                if bk["bot"]:
                    tot["with_bot"] += 1
                if bk["left"] or bk["right"]:
                    tot["with_lr"] += 1
                if any(bk[s] for s in bk):
                    tot["with_any"] += 1
            rows.append((cid, k, our, hit[:2], rel, stolen, bk))

    print("=== GT box 位号归因（%s）===" % pdir)
    print("GT box 元件 %d · 键没对上 %d" % (tot["gt"], tot["un"]))
    print("  ★ GT box 键里本来就非全大写 : %d" % tot["gt_nonupper"])
    print("  ★ 大小写不敏感能救回来的     : %d（纯大小写差 %d）"
          % (tot["ci_match"], tot["caseonly"]))
    print("  真位号被别的预测键抢走 : %d" % tot["stolen"])
    print("  我们的键与 GT 键是近邻   : %d" % tot["near"])
    if tot["st1"]:
        print("  看过 stage1 文字的 %d 个框里：" % tot["st1"])
        print("     框内(位号印在框里)有文字框 : %d 个框（共 %d 个文字框）"
              % (tot["with_in"], tot["in_n"]))
        print("     框顶带有文字框             : %d 个框" % tot["with_top"])
        print("     框底带有文字框             : %d 个框" % tot["with_bot"])
        print("     框左右带有文字框           : %d 个框" % tot["with_lr"])
        print("     四条带里至少有一条有       : %d 个框" % tot["with_any"])
    print()
    for cid, k, our, hit, rel, stolen, bk in rows[:max(0, a.show)]:
        print("  %-6s GT=%-10s 我们=%-10s %s" % (cid, k, our or "-", rel))
        if hit:
            print("        IoU: %s" % (hit,))
        if stolen:
            print("        ⚠ 真位号被 %s 抢走（IoU %.3f）" % stolen)
        if bk:
            for tag, label in (("in", "框内"), ("top", "框顶带"),
                               ("bot", "框底带"), ("left", "框左"), ("right", "框右")):
                if bk[tag]:
                    print("        %s %d 个 (dx,dy,w,h)=%s"
                          % (label, len(bk[tag]), bk[tag][:6]))


if __name__ == "__main__":
    main()
