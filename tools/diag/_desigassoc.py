# -*- coding: utf-8 -*-
"""位号「关联错」还是「识别错」—— 决定「全局重归属」这条路的**天花板**。

`_desigmatch.py` 已量到：没配上的 629 颗 GT IC 里，有候选的 210 颗当中
**101 颗（48.1%）的候选恰好等于同一张图里另一个 GT 元件的正确键**。
字读对了，只是**挂到了错的元件框上**。

本脚本要判定：这 101 颗能不能在**融合层**救回来。判据是硬性的 ——

  对每一颗没配上的 GT IC，取其几何重叠的我方元件 `cid = X`（`bbox_json` 空间）。
  然后在**全部 cid 的 `desig_slots`** 里找 `canon == GT键` 的候选：
    (i)  出现在 X **自己**的表里      -> 这是「选择」问题（`_desigcand2` 量过，仅 9 颗）
    (ii) 出现在**别的** cid Y 的表里  -> 这是「关联」问题。**只有当 (ii) 的量足够大，
         且 Y 发出的键与 X 应得的键能互换，全局匈牙利重归属才有意义。**
    (iii) 全表都没有                  -> 这个字符串**根本没被读出来**（识别/检测问题）

另外对 (ii) 记录：候选文字框到 X 的框、到 Y 的框各自多远（`d`/`fit` 字段），
以及 Y 自己发出的键是什么 —— 如果 Y 发出的键正好是 X 的 GT 键，那就是一个
**纯粹的交换**（permutation），是一行代码能修的。

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_desigassoc.py
    python work\\tools\\diag\\_desigassoc.py --dump 0001
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json          # noqa: E402
from schematic_parser.metrics import (EvalConfig, IC_TYPES,             # noqa: E402
                                      assign_components, canon_type, is_unkeyed)
import stage2_eval as SE                                                # noqa: E402
from fusion import claims as CL                                         # noqa: E402

RUN = ROOT / "work" / "_submit_out220r36" / "runs"


def _overlap(a, b):
    x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return x * y


def _val(c):
    return str(c.get("canon") or c.get("text") or "")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(RUN / "stage2"))
    ap.add_argument("--s1", default=str(RUN / "stage1"))
    ap.add_argument("--asg", default=str(RUN / "stage2" / "_assigned"))
    ap.add_argument("--split", default="")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--dump", default="", help="打印这张图的一条原始候选 dict 看字段")
    a = ap.parse_args(argv)

    exact, digits = SE.build_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()
    files = [f for f in sorted(Path(a.pred).glob("*.json"))
             if not f.name.startswith("_")]

    st = collections.Counter()
    st_pin = collections.Counter()
    sam = collections.defaultdict(list)
    ykeys = collections.Counter()          # (ii) 里 Y 自己发出的键
    dumped = [False]

    for f in files:
        case, _how = SE.resolve(f.stem, exact, digits)
        if case is None or (a.split and case.split != a.split):
            continue
        gt = load_json(case.json_path)
        pr = json.loads(f.read_text(encoding="utf-8"))
        try:
            s1 = json.loads((Path(a.s1) / ("%s.stage1.json" % f.stem))
                            .read_text(encoding="utf-8"))
            asg = json.loads((Path(a.asg) / ("%s.json" % f.stem))
                             .read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            continue
        slots = asg.get("desig_slots") or {}
        comps = asg.get("comps") or {}

        # 全表倒排：canon -> [(cid, 候选dict), ...]
        inv = collections.defaultdict(list)
        for cid, cands in slots.items():
            for c in cands:
                v = _val(c)
                if v:
                    inv[v].append((cid, c))

        if a.dump and f.stem == a.dump and not dumped[0]:
            for cid, cands in slots.items():
                if cands:
                    print("== dump %s %s ==" % (f.stem, cid))
                    print(json.dumps(cands[0], ensure_ascii=False, indent=2))
                    print("   comps[%s] = %s" % (cid, json.dumps(
                        comps.get(cid), ensure_ascii=False)[:600]))
                    dumped[0] = True
                    break
            continue

        try:
            _f1, k2g = assign_components(gt, pr, cfg, {})
        except Exception:                                    # noqa: BLE001
            k2g = {}
        matched_g = set(k2g.values())
        pcomps = pr.get("components") or {}
        free_p = [(k, v) for k, v in pcomps.items() if k not in k2g]

        # 我方元件键 -> 它的 cid
        km = CL.vlm_keymap(s1, asg)

        for gk, gv in (gt.get("components") or {}).items():
            if canon_type((gv or {}).get("type")) not in IC_TYPES:
                continue
            gp = len((gt.get("pins") or {}).get(gk) or {})
            if gk in matched_g:
                continue
            st["unmatched"] += 1
            st_pin["unmatched"] += gp

            # X = 与这颗 GT IC 几何重叠最多的我方元件对应的 cid
            gb = gv.get("bbox") or [0, 0, 0, 0]
            bo, bi = 0.0, None
            for c in (s1.get("components") or []):
                ov = _overlap(gb, c.get("bbox_json") or c.get("bbox_img")
                              or [0, 0, 0, 0])
                if ov > bo:
                    bo, bi = ov, c.get("id")
            if bi is None:
                st["X未知"] += 1
                continue
            X = bi

            own = [c for c in (slots.get(X) or []) if _val(c) == str(gk)]
            others = [(cid_, c_) for cid_, c_ in inv.get(str(gk), []) if cid_ != X]

            if own:
                st["(i)在X自己表里"] += 1
                st_pin["(i)在X自己表里"] += gp
            elif others:
                st["(ii)在别的cid表里"] += 1
                st_pin["(ii)在别的cid表里"] += gp
                Y, c0 = others[0]
                yemit = ((comps.get(Y) or {}).get("designator") or {})
                ykey = yemit.get("canon") or yemit.get("text") or ""
                # 纯交换：Y 发出的键 == X 该有的键，且 X 发出的键 == Y 该有的键
                xemit = ((comps.get(X) or {}).get("designator") or {})
                xkey = xemit.get("canon") or xemit.get("text") or ""
                swap = bool(ykey) and str(ykey) == str(gk)
                st["··且Y发出的键正是GT键(纯交换)"] += 1 if swap else 0
                ykeys[str(ykey)] += 1
                if len(sam["ii"]) < a.top:
                    sam["ii"].append(
                        (case.case_id, gk, gp, X, xkey, Y, ykey, swap,
                         {k: c0.get(k) for k in ("text", "canon", "score",
                                                 "conf", "d", "fit")}))
            else:
                st["(iii)全表都没有"] += 1
                st_pin["(iii)全表都没有"] += gp

    if a.dump:
        return 0

    def pct(n, d):
        return "%6.1f%%" % (100.0 * n / max(1, d))

    print("== 没配上的 GT IC：正确位号在**全表**的哪个位置 ==")
    print("  没配上总数                  %5d   引脚 %5d"
          % (st["unmatched"], st_pin["unmatched"]))
    print("  (i)  在 X 自己的候选表里    %5d %s  引脚 %5d %s   <- 选择问题"
          % (st["(i)在X自己表里"], pct(st["(i)在X自己表里"], st["unmatched"]),
             st_pin["(i)在X自己表里"], pct(st_pin["(i)在X自己表里"], st_pin["unmatched"])))
    print("  (ii) 在**别的** cid 表里   %5d %s  引脚 %5d %s   <- 关联问题"
          % (st["(ii)在别的cid表里"], pct(st["(ii)在别的cid表里"], st["unmatched"]),
             st_pin["(ii)在别的cid表里"], pct(st_pin["(ii)在别的cid表里"], st_pin["unmatched"])))
    print("        ·其中纯交换（Y 发的键正是 GT 键） %5d %s"
          % (st["··且Y发出的键正是GT键(纯交换)"],
             pct(st["··且Y发出的键正是GT键(纯交换)"], max(1, st["(ii)在别的cid表里"]))))
    print("  (iii) 全表都没有            %5d %s  引脚 %5d %s   <- 根本没读出来"
          % (st["(iii)全表都没有"], pct(st["(iii)全表都没有"], st["unmatched"]),
             st_pin["(iii)全表都没有"], pct(st_pin["(iii)全表都没有"], st_pin["unmatched"])))
    print("  X 未知（没框住）            %5d" % st["X未知"])
    if ykeys:
        print("\n  (ii) 里 Y 自己发出的键（前 15）：", ykeys.most_common(15))
    if sam["ii"]:
        print("\n(ii) 样例  case / GT键 / GT引脚 / X / X发出 / Y / Y发出 / 纯交换 / 候选dict")
        for s in sam["ii"]:
            print("   %s" % (s,))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
