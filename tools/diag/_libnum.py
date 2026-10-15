# -*- coding: utf-8 -*-
"""官方元器件库当「引脚号来源」的机会 —— 只针对**名读到了、号没有**的引脚。

背景（`_pinmiss.py`，半径 8 px；引脚列间距 ~20 px）：

    A3 名读到、桩在、号没读到   22 (40.0%)   ← 最大的一块
    C  附近无名                14 (25.5%)
    A2 名+号都读到、该处无桩      9 (16.4%)
    A1 名+号+桩都有             7 (12.7%)
    D  同位同名、号不同           2 (3.6%)
    A4 只有名                   1 (1.8%)

A3 的桩在、名字读到了，缺的**只是号**。`_pinlat.py` 证明号不能用几何插值补
（≤5 px 只 1/52）。那么唯一的外部号来源就是官方库：`cpntLibrary[型号].pins[]`
里有 `{name, number}` 对。

本工具量（官方 9 张、`--pred` 产物）：
  * **上界**：拿型号名 + GT 引脚名去库里查，库给出的 number 集合里有 GT 号吗；
  * **可执行规则**：该名字在库里**唯一映射到一个号**（`uniq`）时才补 —— 数它
    恰好等于 GT 号的比例（= 规则的精度）；
  * 按「我方 `_assigned` 里到底读没读到这个名 / stage1 有没有桩」四桶切开。

用法（从仓库根跑）：
    python -X utf8 work\\tools\\diag\\_libnum.py --pred work\\_fnew \\
        --assigned work\\_fnew\\_assigned --stage1 work\\_fast_out_reg9\\runs\\stage1 \\
        --cases 1,2,3,4,5,7,8,9,10 --radius 8
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "diag"))

from schematic_parser.metrics import (  # noqa: E402
    EvalConfig, assign_components, canon_type, rewrite_keys)
from schematic_parser.data import IC_TYPES, pin_key_to_number  # noqa: E402
from _libcov import Lib, load, norm  # noqa: E402

GT9 = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
LIB = ROOT / "work" / "data" / "cpntLibrary.json"
S1 = ROOT / "work" / "_fast_out_reg9" / "runs" / "stage1"


def _ctr(b):
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def _dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def _dist1(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(l)):
        if l[:i] + l[i + 1:] == s:
            return True
    return False


def nums_for_name(lib: Lib, key: str, gname: str) -> set:
    """库里该型号下，名字 ≈ gname 的所有引脚号。"""
    out = set()
    ng = norm(gname)
    if not ng or not key:
        return out
    for p in lib.raw.get(key, {}).get("pins", []) or []:
        nm = norm(str(p.get("name") or ""))
        if not nm:
            continue
        if nm == ng or _dist1(nm, ng) or nm in ng or ng in nm:
            num = str(p.get("number") or "").strip()
            if num:
                out.add(num)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--assigned", default="")
    ap.add_argument("--stage1", default=str(S1))
    ap.add_argument("--cases", default="1,2,3,4,5,7,8,9,10")
    ap.add_argument("--gt", default=str(GT9))
    ap.add_argument("--lib", default=str(LIB))
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--fuzzy", type=float, default=0.85)
    ap.add_argument("--verbose", type=int, default=0)
    a = ap.parse_args()

    lib = Lib(load(Path(a.lib)))
    gtroot, predroot = Path(a.gt), Path(a.pred)
    asgroot, s1root = Path(a.assigned), Path(a.stage1)
    cfg = EvalConfig()
    R = a.radius

    buck = Counter()
    libk = Counter()
    per_case = []
    detail = []

    for c in [x.strip() for x in a.cases.split(",") if x.strip()]:
        gtf = gtroot / c / ("%s.json" % c)
        pf = predroot / ("%s.json" % c)
        sf = s1root / ("%s.stage1.json" % c)
        af = asgroot / ("%s.json" % c)
        if not (gtf.exists() and pf.exists() and sf.exists()):
            print("  ! 跳过 case %s" % c)
            continue
        gt = json.loads(gtf.read_text(encoding="utf-8"))
        pred = json.loads(pf.read_text(encoding="utf-8"))
        s1 = json.loads(sf.read_text(encoding="utf-8"))
        asg = json.loads(af.read_text(encoding="utf-8")) if af.exists() else {}
        H = float(s1["image"]["h"])
        _f1, keymap = assign_components(gt, pred, cfg)
        pred2 = rewrite_keys(pred, keymap)

        s1comps = []
        for comp in s1.get("components") or []:
            b = comp.get("bbox_json") or []
            if len(b) == 4:
                s1comps.append((comp.get("id"), _ctr(b)))
        s1_pins = s1.get("pins") or {}
        cmps = asg.get("comps") or {}

        n_miss = n_hit = n_uniq = n_uniqhit = n_a3 = n_a3_hit = n_a3_uniqhit = 0
        for k, rec in (gt.get("components") or {}).items():
            if canon_type(rec.get("type")) not in IC_TYPES:
                continue
            if k not in (pred2.get("components") or {}):
                continue
            gtname = rec.get("Name") or ""
            pname = (pred2["components"].get(k) or {}).get("Name") or ""
            cand_gt = (lib.exact(gtname) or lib.normalized(gtname)
                       or [x[0] for x in lib.fuzzy(gtname, a.fuzzy)])
            cand_pr = (lib.exact(pname) or lib.normalized(pname)
                       or [x[0] for x in lib.fuzzy(pname, a.fuzzy)])
            k_gt = cand_gt[0] if cand_gt else None
            k_pr = cand_pr[0] if cand_pr else None
            if k_gt:
                libk["GT名命中库"] += 1
            if k_pr:
                libk["我方名命中库"] += 1
            src = k_gt or k_pr
            # ---- 这颗 GT 元件在 stage1 里的 cid 与读数 ----
            gb = rec.get("bbox") or []
            cid = None
            if len(gb) == 4 and s1comps:
                cid = min(s1comps, key=lambda t: _dist(t[1], _ctr(gb)))[0]
            raw_pins = ((cmps.get(cid) or {}).get("pins") or {}) if cid else {}
            nrec = []
            for _i, pr in raw_pins.items():
                nm = (pr.get("name") or {}).get("text")
                box = (pr.get("name") or {}).get("box") or []
                if nm and len(box) == 4:
                    nrec.append((nm, _ctr(box)))
            stubs = [tuple(q[:2]) for q in (s1_pins.get(cid) or []) if len(q) >= 2]
            for pk, pv in (gt.get("pins", {}).get(k) or {}).items():
                if k in (pred2.get("pins") or {}) and pk in (pred2["pins"][k] or {}):
                    continue                       # 已经发出来了
                gname = (pv or {}).get("pinname") or ""
                gnum = str(pin_key_to_number(pk) or "").strip()
                pt = (pv or {}).get("point") or {}
                if "x" not in pt or "y" not in pt:
                    continue
                p_img = (float(pt["x"]), H - float(pt["y"]))
                # ★ 关键：`_assigned` 的引脚索引 k 与 stage1 的桩点**同序**，
                # 所以「stage2 有没有把正确的名绑到 GT 这个位置的桩上」= 找最近的桩
                # 索引 j，再看 raw_pins[j].name。
                j, best = None, 1e9
                for idx, q in enumerate(stubs):
                    d = _dist(tuple(q[:2]), p_img)
                    if d < best:
                        best, j = d, idx
                stub_ok = j is not None and best <= R
                rec_j = {}
                if j is not None:
                    rec_j = raw_pins.get(str(j)) or raw_pins.get(j) or {}
                nm_txt = ((rec_j.get("name") or {}).get("text") or "")
                num_txt = ((rec_j.get("number") or {}).get("text") or "")
                read_ok = bool(norm(gname)) and norm(nm_txt) == norm(gname)
                b = "%s+%s" % ("名读到" if read_ok else "名没读到",
                               "桩在" if stub_ok else "无桩")
                nums = nums_for_name(lib, src, gname)
                hit = gnum in nums
                uniq = len(nums) == 1
                n_miss += 1
                buck[b + "|missing"] += 1
                if hit:
                    n_hit += 1
                    buck[b + "|库有号(hit)"] += 1
                if uniq:
                    n_uniq += 1
                if uniq and hit:
                    n_uniqhit += 1
                    buck[b + "|库唯一且对"] += 1
                if b == "名读到+桩在":
                    n_a3 += 1
                    n_a3_hit += 1 if hit else 0
                    n_a3_uniqhit += 1 if (uniq and hit) else 0
                if a.verbose:
                    detail.append("    %s %s.%s %-14r 号=%-4s 读=%s 桩=%s 库键=%s 库号=%s%s"
                                  % (c, k, pk, gname, gnum, "y" if read_ok else "n",
                                     "y" if stub_ok else "n", src or "-",
                                     sorted(nums)[:6], "  ★HIT" if hit else ""))
        per_case.append((c, n_miss, n_hit, n_uniq, n_uniqhit, n_a3, n_a3_hit, n_a3_uniqhit))
        print("  case %-3s 缺 %2d | 库有 GT 号 %2d | 库唯一映射 %2d（对 %2d）| "
              "名读到+桩在 %2d（库有号 %2d / 唯一且对 %2d）"
              % (c, n_miss, n_hit, n_uniq, n_uniqhit, n_a3, n_a3_hit, n_a3_uniqhit))

    if detail:
        print("\n".join(detail))
    s = lambda i: sum(x[i] for x in per_case)          # noqa: E731
    print("\n===== 官方库当「引脚号来源」的机会（官方 9 张，半径 %.0f px）=====" % R)
    print("  missing 引脚总数                  %4d" % s(1))
    print("  库里同型号+同名给出 GT 号（上界）    %4d" % s(2))
    print("  库对该名唯一映射到一个号（可执行）    %4d（其中恰好是 GT 号 %4d ⇒ 规则精度）"
          % (s(3), s(4)))
    print("  其中「名读到+桩在」（= A3 那一块）   %4d（库有号 %4d / 唯一且对 %4d）"
          % (s(5), s(6), s(7)))
    print("  库键来源：%s" % ", ".join("%s×%d" % kv for kv in libk.most_common()))
    print("  ---- 四桶明细 ----")
    for b in ("名读到+桩在", "名读到+无桩", "名没读到+桩在", "名没读到+无桩"):
        print("    %-12s missing %3d | 库有号 %3d | 库唯一且对 %3d"
              % (b, buck[b + "|missing"], buck[b + "|库有号(hit)"],
                 buck[b + "|库唯一且对"]))


if __name__ == "__main__":
    main()
