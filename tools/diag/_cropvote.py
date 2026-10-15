# -*- coding: utf-8 -*-
"""同一检出框多裁片投票：量「TTA 投票」能不能超过**任何**单一裁片。

背景（`eda-pcb/README.md` §八.12）：把引脚号/名全读对的端到端上界只有 +4.60 分
（官方 9 张 58.6304 → 63.2307），而瓶颈在**裁剪分布**，不在权重 ——
真值紧框 exact 0.843 / stage1 紧框 0.197 / 外扩 +3px → 0.654。
如果不同外扩量读错的**方向不同**，那么「多裁片读同一框再投票」应当严格优于
其中任何一个单一裁片。本探针就是量这个，先量再决定要不要实装 TTA。

口径（沿用 `_textseg_eval.py`）：真值框来自 `*_CVJsonStd.json` 的 `texts[].location`
+ `content`；用 `cvparts.textseg`（当前推荐配置）检出；每个真值框取 IoU 最好的检出框
（IoU ≥ `--iou`，默认 0.05）作为「我方框」，再在所有裁片变体上读它。

    cd Y:\\workspace\\EDA\\eda-pcb
    python -X utf8 tools\\diag\\_cropvote.py --limit 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from cvparts import textseg                                        # noqa: E402
from stage2.text import (TextEngine, imread_unicode, pad_box,      # noqa: E402
                         tight_ink_box, TIGHT_ROLES)
from _textseg_eval import ROLE_OF, cases, iou_matrix               # noqa: E402

H_CFG = dict(ink_mode="max", min_dim=2, min_area=3, line_len=15,
             resid_min_h=4, resid_min_dim=1)
# 管线当前口径（`stage2/text.py` 的默认旋钮）
CUR = dict(pad_min=2.0, pad_frac=0.0, pad_min_y=3.0, pad_frac_y=0.20)


def gt_of(js):
    d = json.loads(Path(js).read_text(encoding="utf-8"))
    out = []
    for t in (d.get("texts") or {}).values():
        loc = t.get("location") or {}
        c = (t.get("content") or "").strip()
        role = ROLE_OF.get(str(t.get("details") or ""))
        if not role or not c or len(c) > 24:
            continue
        try:
            b = tuple(float(loc[k]) for k in ("x1", "y1", "x2", "y2"))
        except (TypeError, KeyError):
            continue
        if b[2] - b[0] < 2 or b[3] - b[1] < 2:
            continue
        out.append((b, c, role))
    return out


def pad(b, x, y):
    return (b[0] - x, b[1] - y, b[2] + x, b[3] + y)


def multi(b, wx=0.15, hy=0.43):
    w = max(1.0, b[2] - b[0])
    h = max(1.0, b[3] - b[1])
    return pad(b, wx * w, hy * h)


def cur_crop(bgr, b, role):
    """当前管线给这个角色的裁片。"""
    if role in TIGHT_ROLES:
        tb = tight_ink_box(bgr, b)
        if tb is not None:
            py = 0.0 if role == "pin_number" else CUR["pad_frac_y"]
            return pad_box(tb, CUR["pad_min"], CUR["pad_frac"],
                           pad_min_y=CUR["pad_min_y"], pad_frac_y=py)
    return pad_box(b, CUR["pad_min"], CUR["pad_frac"],
                   pad_min_y=CUR["pad_min_y"], pad_frac_y=CUR["pad_frac_y"])


def variants(bgr, b, role):
    """-> {名字: 裁片框}。前 6 个是**通用几何变体**（不依赖角色），
    `cur` 是管线现状（参考），`ink` 是管线用的切描边版（仅引脚角色非空）。"""
    out = {
        "p0": b,
        "p23": pad_box(b, 2, 0, pad_min_y=3, pad_frac_y=0.0),
        "p33": pad_box(b, 3, 0, pad_min_y=3, pad_frac_y=0.0),
        "p44": pad_box(b, 4, 0, pad_min_y=4, pad_frac_y=0.0),
        "p55": pad_box(b, 5, 0, pad_min_y=5, pad_frac_y=0.0),
        "multi": multi(b),
        "cur": cur_crop(bgr, b, role),
    }
    tb = tight_ink_box(bgr, b)
    if tb is not None:
        out["ink23"] = pad_box(tb, 2, 0, pad_min_y=3, pad_frac_y=0.0)
        out["ink33"] = pad_box(tb, 3, 0, pad_min_y=3, pad_frac_y=0.0)
    return out


def vote(items):
    """items = [(text, conf)] -> 多数串（票数并列时取 conf 最大者）。"""
    cnt = {}
    for t, _c in items:
        if t:
            cnt[t] = cnt.get(t, 0) + 1
    if not cnt:
        return ""
    top = max(cnt.values())
    cands = [t for t, n in cnt.items() if n == top]
    if len(cands) == 1:
        return cands[0]
    best, bc = cands[0], -1.0
    for t, c in items:
        if t in cands and c > bc:
            best, bc = t, c
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--iou", type=float, default=0.05)
    ap.add_argument("--roles", default="designator,value,pin_number,pin_name")
    args = ap.parse_args(argv)
    want_roles = {r.strip() for r in args.roles.split(",") if r.strip()}

    rr = TextEngine().recognizer()
    # arm -> {(ci,i): (text, conf)}；arm 名 = 变体名，外加 GT_raw/GT_pad 参考
    res = {}
    truth = {}
    role_of = {}
    n_gt = 0
    for ci, (png, js) in enumerate(cases(args.limit)):
        gts = gt_of(js)
        if not gts:
            continue
        bgr = imread_unicode(png)
        det = [tuple(float(v) for v in b) for b in textseg.detect_boxes(bgr, **H_CFG)]
        M = iou_matrix([g[0] for g in gts], det) if det else np.zeros((len(gts), 0))
        jobs = {}          # (arm, role) -> [(key, box)]
        for k, (b, c, role) in enumerate(gts):
            if role not in want_roles:
                continue
            key = (ci, k)
            truth[key] = c
            role_of[key] = role
            n_gt += 1
            row = M[k] if M.size else np.zeros(0)
            j = int(np.argmax(row)) if row.size else -1
            if j < 0 or float(row[j]) < args.iou:
                continue
            db = det[j]
            for arm, vbox in variants(bgr, db, role).items():
                jobs.setdefault((arm, role), []).append((key, vbox))
            jobs.setdefault(("GT_raw", role), []).append((key, b))
            jobs.setdefault(("GT_pad", role), []).append((key, pad_box(b)))
        for (arm, role), lst in jobs.items():
            boxes = [b for _k, b in lst]
            outs = rr.read_many(bgr, [(b, role) for b in boxes])
            d = res.setdefault(arm, {})
            for (key, _b), (txt, cf) in zip(lst, outs):
                d[key] = ((txt or "").strip(), float(cf))
        print("  case %-28s GT %3d  检出 %3d" % (png.parent.name, len(gts), len(det)))

    if not n_gt:
        print("没有真值")
        return 1

    base = set(res.get("cur", {}))          # 「命中集」= 管线现状能读到的那些框
    allk = set(truth)

    def exact(arm, keys):
        d = res.get(arm, {})
        return sum(1 for k in keys if d.get(k, ("", 0))[0] == truth[k])

    arms = [a for a in res if a != "cur"]
    print("\n真值（限 %s）%d 个；`cur` 命中 %d 个 = 公共集口径"
          % ("/".join(sorted(want_roles)), n_gt, len(base)))
    print("\n=== 单变体 exact（公共集 = cur 命中的框）===")
    print("  %-10s %8s %8s" % ("变体", "公共集", "端到端"))
    for a in sorted(arms):
        com = base
        print("  %-10s %8.4f %8.4f"
              % (a, exact(a, com) / max(1, len(com)), exact(a, allk) / max(1, n_gt)))
    print("  %-10s %8.4f %8.4f"
          % ("cur(参考)", exact("cur", base) / max(1, len(base)),
             exact("cur", allk) / max(1, n_gt)))

    print("\n=== 投票（公共集）===")
    geo = ["p0", "p23", "p33", "p44", "p55", "multi"]
    combos = [("p23", "p33"), ("p23", "p44"), ("p23", "multi"), ("p33", "p44"),
              ("p33", "multi"), ("p23", "p33", "p44"), ("p0", "p23", "p33"),
              ("p23", "p33", "p44", "p55"), ("p23", "p33", "p44", "multi"),
              ("p0", "p23", "p33", "p44", "multi")]
    rows = []
    for cb in combos:
        if any(c not in res for c in cb):
            continue
        ok = 0
        for k in base:
            items = [res[c].get(k, ("", 0.0)) for c in cb]
            if vote(items) == truth[k]:
                ok += 1
        rows.append((ok / max(1, len(base)), cb))
    for r, cb in sorted(rows, reverse=True):
        print("  %-34s %.4f" % ("+".join(cb), r))
    print("  %-34s %.4f" % ("cur(参考)", exact("cur", base) / max(1, len(base))))

    print("\n=== 按角色（公共集）===")
    print("  %-12s %6s %8s %8s" % ("角色", "n", "cur", "最佳投票"))
    byrole = {}
    for k in base:
        byrole.setdefault(role_of[k], []).append(k)
    for role, ks in sorted(byrole.items()):
        cur = exact("cur", set(ks)) / max(1, len(ks))
        best, bname = -1.0, ""
        for r, cb in rows:
            if any(c not in res for c in cb):
                continue
            ok = sum(1 for k in ks
                     if vote([res[c].get(k, ("", 0.0)) for c in cb]) == truth[k])
            v = ok / max(1, len(ks))
            if v > best:
                best, bname = v, "+".join(cb)
        print("  %-12s %6d %8.4f %8.4f  (%s)" % (role, len(ks), cur, best, bname))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
