# -*- coding: utf-8 -*-
"""**引脚名字账本**：官方档 IC 引脚的 GT `pinname` vs 我方 `_assigned` 槽位里读到的
`name.text` —— 到底是「OCR 读错一两个字符」还是「绑到了别的文字」。

    python work/tools/diag/_pinnamepair.py --split 10GT --run work/_full220 --show 40
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))

from schematic_parser import metrics as M                                  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402


def ed(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "work" / "_full220"))
    ap.add_argument("--split", default="10GT")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)

    st = collections.Counter()
    pairs = collections.Counter()
    rows = []
    for c in [x for x in discover(DEFAULT_DATA_ROOT) if x.split == a.split][:a.n]:
        ap_ = run / "runs" / "stage2" / "_assigned" / (c.case_id + ".json")
        if not ap_.exists():
            print("!! 缺 _assigned %s" % c.case_id)
            continue
        asg = load_json(ap_)
        gt = M.normalise_doc(load_json(Path(c.json_path)))
        box = {k for k, v in (gt.get("components") or {}).items()
               if M.canon_type((v or {}).get("type")) in M.IC_TYPES}
        slots = {}
        for _cid, cc in (asg.get("comps") or {}).items():
            key = ((cc.get("designator") or {}).get("text") or "").strip()
            b = slots.setdefault(key, {})
            for _s, rec in (cc.get("pins") or {}).items():
                num = (((rec or {}).get("number") or {}).get("text") or "").strip()
                if num:
                    b.setdefault(num, rec)
        for comp in box:
            for pk, info in ((gt.get("pins") or {}).get(comp) or {}).items():
                gname = ((info or {}).get("pinname") or "")
                num = M.pin_key_to_number(pk)
                st["gt"] += 1
                rec = (slots.get(comp) or {}).get(str(num))
                if rec is None:
                    st["no_num_slot"] += 1
                    continue
                st["num_slot"] += 1
                ourname = (((rec.get("name") or {}).get("text")) or "").strip()
                if not ourname:
                    st["name_empty"] += 1
                    continue
                st["name_read"] += 1
                if ourname == gname:
                    st["name_eq"] += 1
                else:
                    st["name_ne"] += 1
                    n = len(gname)
                    st["ne_len%d" % min(n, 8)] += 1
                    pairs[("%s→%s" % (ourname, gname))] += 1
                    if a.show:
                        rows.append((c.case_id, comp, pk, gname, ourname,
                                     ed(ourname, gname), n))
    print("run=%s split=%s" % (run, a.split))
    print("GT IC 引脚 %d 根" % st["gt"])
    print("  其中「按号在 _assigned 里找到槽位」%d（%.1f%%）；没找到 %d"
          % (st["num_slot"], 100.0 * st["num_slot"] / max(st["gt"], 1), st["no_num_slot"]))
    print("  有槽位的里面：读到名字 %d，名字为空 %d"
          % (st["name_read"], st["name_empty"]))
    print("  **读到了名字的里面：与 GT 逐字符相等 %d（%.1f%%），不等 %d**"
          % (st["name_eq"], 100.0 * st["name_eq"] / max(st["name_read"], 1), st["name_ne"]))
    nb = st["name_ne"]
    if nb:
        d1 = sum(v for k, v in st.items() if k.startswith("ne_len") and k == "ne_len1")
        print("  不一致的里：编辑距离 1 %d（%.0f%%）、GT 名长度 1 的 %d"
              % (d1, 100.0 * d1 / nb, st["ne_len1"]))
        print("  GT 名长度分布（不一致样本）：",
              {k.split("ne_len")[1]: v for k, v in sorted(st.items()) if k.startswith("ne_len")})
        print()
        print("  不一致的字符串对（前 20）：")
        for k, v in pairs.most_common(20):
            print("     %-26s %d" % (k, v))
    if a.show and rows:
        print()
        rows.sort(key=lambda r: r[5])
        print("  样本（编辑距离升序）：")
        for cid, comp, pk, g, o, d, n in rows[:a.show]:
            print("     %-4s %-5s %-8s gt=%-10s our=%-10s 距离=%d" % (cid, comp, pk, g, o, d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
