# -*- coding: utf-8 -*-
"""「位号候选没产生」到底属于哪一种？—— 决定性归因，跑全量。

    & 'K:\python\311\python.exe' Y:\workspace\EDA\work\tools\diag\_des_why2.py --limit 220

对每一颗「最终 key 是兜底 `pNNN`、而 GT 的 Name 是可读位号（不是 ∅NNN）」的元件，
在**整张图**里找有没有哪个文字框真的读出了那个位号字符串，然后归类：

* `A_owned_by_other`  读出来了，但那个框按「最近元件」归属**判给了别的元件**
                      （= 归属/贪心问题，用户 m06973 说的「用贪心解决」正对症）
* `B_free_far`        读出来了，但那个框离所有元件都 > FAR，进了 `free`
                      （= 归属半径太小）
* `C_not_read`        整张图没有任何框读出这个字符串
                      （= 检测/切分/识别丢了这个框，改归属没用）
* `D_no_gt_box`       GT 里这颗元件压根没有 bbox 信息

同时记录三种「放宽」各能救回多少：
  +far   归属半径翻倍（FAR -> 2*FAR 且允许多个元件共享同一个框）
  +global 打开 `assign(..., global_match=True)` 的匈牙利匹配
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "viz"))

import comp_cases as CC                                            # noqa: E402

A = importlib.import_module("stage2.assign")
B = importlib.import_module("stage2.assemble")
T = importlib.import_module("stage2.text")

FALLBACK = re.compile(r"^p\d{3,4}$")
DIAM = re.compile(r"^\u2205\d+$")
NORM = re.compile(r"[^0-9A-Za-z]")


def norm(s: str) -> str:
    return NORM.sub("", (s or "")).upper()


def _edit(a: str, b: str) -> int:
    """Levenshtein。字符串都很短（位号 <= 8 字符），直接 DP。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def pair_greedy(gt_items, pred_items, thr=0.1):
    """GT↔预测 一对一贪心配对；返回 [(gt_key, gt_val, pred_key, pred_val, iou)]。"""
    cands = []
    for gi, (gk, gv) in enumerate(gt_items):
        gb = gv.get("bbox")
        if not gb or len(gb) != 4:
            continue
        gbox = [float(gb[0]), float(gb[1]), float(gb[2]), float(gb[3])]
        for pi, (pk, pv) in enumerate(pred_items):
            pb = pv.get("bbox")
            if not pb:
                continue
            v = iou(gbox, [float(x) for x in pb])
            if v >= thr:
                cands.append((v, gi, pi))
    cands.sort(key=lambda x: -x[0])
    ug, up, out = set(), set(), []
    for v, gi, pi in cands:
        if gi in ug or pi in up:
            continue
        ug.add(gi)
        up.add(pi)
        out.append((gt_items[gi][0], gt_items[gi][1], pred_items[pi][0],
                    pred_items[pi][1], v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=str(ROOT / "work" / "stage1_all14"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    s1dir = Path(args.stage1)
    idx = CC.gt_index(CC.DEFAULT_DATA_ROOT)
    engine = T.TextEngine()

    cnt = Counter()
    rescue = Counter()
    examples = []
    rows = []

    cases = sorted(p.name.split(".")[0] for p in s1dir.glob("*.stage1.json"))
    if args.limit:
        cases = cases[:args.limit]

    for n, case in enumerate(cases, 1):
        rec = idx.get(case)
        s1 = s1dir / ("%s.stage1.json" % case)
        if rec is None or not rec.json_path or not s1.exists():
            continue
        doc = json.loads(s1.read_text(encoding="utf-8"))
        gdoc = json.loads(Path(rec.json_path).read_text(encoding="utf-8"))
        bgr = T.imread_unicode(rec.png_path)
        boxes = T.read_stage1_texts(doc)
        reads = engine.read(bgr, boxes)

        comps = [c for c in (doc.get("components") or []) if c.get("bbox_img")]
        a = A.assign(doc.get("components") or [], doc.get("pins") or {}, reads)
        pred = B.build_doc(doc, a, case)
        pred_items = list((pred.get("components") or {}).items())
        # `build_doc` 的键顺序 == `comps` 顺序（跳过没有 bbox 的）
        if len(pred_items) != len(comps):
            print("!! %s 预测元件数 %d != comps %d" % (case, len(pred_items), len(comps)))
        # GT 的 bbox 是**左下角原点**，先翻成图像坐标再配对
        hb = float((doc.get("image") or {}).get("h") or 0)
        gt_items = []
        for k, v in (gdoc.get("components") or {}).items():
            gb = v.get("bbox")
            if not gb or len(gb) != 4:
                continue
            gt_items.append((k, dict(v, bbox=[float(gb[0]), hb - float(gb[3]),
                                              float(gb[2]), hb - float(gb[1])])))

        pairs = pair_greedy(gt_items, pred_items)

        # 整张图里「读出过每个字符串」的框（按归一化文本索引）
        read_at = {}
        for i, t in enumerate(reads):
            for _role, (txt, _c) in (t.get("reads") or {}).items():
                k = norm(txt)
                if k:
                    read_at.setdefault(k, []).append(i)
        # 归属（复刻 assign 的最近元件规则，用来判断「被谁拿走了」）
        own = {}
        for i, t in enumerate(reads):
            b = t["bbox"]
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            ds = [A.dist_to_rect(cx, cy, [float(v) for v in c["bbox_img"]])
                  for c in comps]
            if not ds:
                own[i] = (-1, 1e9)
                continue
            j = min(range(len(ds)), key=lambda k: ds[k])
            own[i] = (j, ds[j])

        for gk, gv, pk, pv, v in pairs:
            gname = str(gv.get("Name") or "")
            if not gname or DIAM.match(gname):
                continue
            if not FALLBACK.match(pk):
                continue
            # 找到这颗元件在我们这边的下标
            pi = [i for i, (k, _x) in enumerate(pred_items) if k == pk]
            pi = pi[0] if pi else -1
            gbox = gv.get("bbox")
            if pi < 0 or not gbox:
                cnt["D_no_gt_box"] += 1
                continue
            hb = float((doc.get("image") or {}).get("h") or 0)
            gbox_img = [float(gbox[0]), float(gbox[1]), float(gbox[2]), float(gbox[3])]
            target = norm(gname)
            hits = read_at.get(target) or []
            if not hits:
                cnt["C_not_read"] += 1
                # 这颗元件附近有没有「读得很像、但差一两个字符」的框？
                # 若有，说明框在、位置对，只是**识别读错**（用户 m06973：这类先不算错，
                # 后面 LLM 可修）；若没有，才是真的没检出。
                ed = 99
                near_i = -1
                for i, t in enumerate(reads):
                    b = t["bbox"]
                    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                    d = A.dist_to_rect(cx, cy, [float(x) for x in comps[pi]["bbox_img"]])
                    if d > A.FAR:
                        continue
                    for _role, (txt, _c) in (t.get("reads") or {}).items():
                        e = _edit(norm(txt), target)
                        if 0 < e < ed:
                            ed, near_i = e, i
                if ed <= 1:
                    cnt["C1_near_miss_ed<=1"] += 1
                elif ed <= 2:
                    cnt["C2_near_miss_ed==2"] += 1
                elif ed <= 3:
                    cnt["C3_near_miss_ed==3"] += 1
                else:
                    cnt["C9_no_similar_box_nearby"] += 1
                rows.append({"case": case, "gt": gname, "kind": "C_not_read",
                             "ed": ed})
                if ed <= 3 and len(examples) < 25:
                    examples.append({"case": case, "gt": gname, "pred_key": pk,
                                     "kind": "C_not_read(ed=%d)" % ed,
                                     "min_d": -1.0, "iou": round(v, 2)})
                continue
            kinds = set()
            best_d = 1e9
            other = None
            for bi in hits:
                b = reads[bi]["bbox"]
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                d = A.dist_to_rect(cx, cy, [float(x) for x in comps[pi]["bbox_img"]])
                best_d = min(best_d, d)
                oj, od = own.get(bi, (-1, 1e9))
                if oj == pi:
                    kinds.add("A_own")          # 已经是自己的框却仍是兜底 key
                elif od <= A.FAR:
                    kinds.add("A_owned_by_other")
                    if other is None:
                        other = {"stolen_by_cid": comps[oj]["id"], "d_to_target": round(d, 1)}
                else:
                    kinds.add("B_free_far")
            if "A_own" in kinds:
                k = "A_own_still_lost"
            elif "A_owned_by_other" in kinds:
                k = "A_owned_by_other"
            else:
                k = "B_free_far"
            cnt[k] += 1
            rescue["far2_min_d<=%d" % int(2 * A.FAR)] += int(best_d <= 2 * A.FAR)
            rescue["free_or_other_but_close"] += int(best_d <= 2 * A.FAR)
            if k != "A_own_still_lost" and len(examples) < 25:
                examples.append({"case": case, "gt": gname, "pred_key": pk,
                                 "kind": k, "min_d": round(best_d, 1),
                                 "iou": round(v, 2), **(other or {})})
            rows.append({"case": case, "gt": gname, "kind": k,
                         "pred_key": pk, "min_d": round(best_d, 1),
                         "iou": round(v, 2)})
        if n % 20 == 0:
            print("... %d/%d  %s" % (n, len(cases), dict(cnt)))
        sys.stdout.flush()

    print("\n==== 结果（%d 张）====" % len(cases))
    tot = sum(cnt.values())
    for k, v in cnt.most_common():
        print("  %-22s %5d  %5.1f%%" % (k, v, 100.0 * v / max(1, tot)))
    print("  合计 %d" % tot)
    print("\n能救回的（框在整张图里读对了，只是没归到这颗元件）：%d"
          % (cnt["A_owned_by_other"] + cnt["B_free_far"]))
    print("其中「离这颗元件 <= %d px」的：%d"
          % (int(2 * A.FAR), rescue["free_or_other_but_close"]))
    print("\n---- 样例 ----")
    for e in examples[:25]:
        print("  %s GT=%-12s pred=%-6s %-18s min_d=%5.1f iou=%.2f %s"
              % (e["case"], e["gt"], e["pred_key"], e["kind"], e["min_d"], e["iou"],
                 e.get("stolen_by_cid", "")))
    if args.dump:
        Path(args.dump).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print("\n明细写到 %s" % args.dump)


if __name__ == "__main__":
    main()
