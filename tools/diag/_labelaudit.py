# -*- coding: utf-8 -*-
"""标注审计：当前训练集（trainset_v3）里的标签**本身**有多可疑。

触发：owner 逐张肉眼核对后发现 `fn_r` 的红框全落在一张图的大锯齿（电感画法）上、
`fn_net_input` 的红框扣在普通文字和导线拐角上 —— 也就是说**那些框住的不是目标**。
于是必须回答：这些坏标签是不是**已经进了训练集**、甚至进了当前的 v3。

用法: python work\\tools\\diag\\_labelaudit.py [--root Y:\\workspace\\EDA\\trainset_v3]
"""
import argparse
import statistics as stx
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

SUSPECT = ["0050", "0027", "0011", "0064", "0047", "0190", "0002", "0101", "0127"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"Y:\workspace\EDA\trainset_v3")
    a = ap.parse_args()
    root = Path(a.root)
    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    print("root = %s" % root)
    print("类别数 %d ｜ r=%s c=%s l=%s v=%s gnd=%s testpoint=%s net_input=%s box=%s"
          % (len(names), names.index("r") if "r" in names else "?",
             names.index("c") if "c" in names else "?",
             names.index("l") if "l" in names else "?",
             names.index("v") if "v" in names else "?",
             names.index("gnd") if "gnd" in names else "?",
             names.index("testpoint") if "testpoint" in names else "?",
             names.index("net_input") if "net_input" in names else "?",
             names.index("box") if "box" in names else "?"))

    size_cache = {}

    def imsize(p):
        if p not in size_cache:
            try:
                size_cache[p] = Image.open(str(p)).size
            except Exception:                          # noqa: BLE001
                size_cache[p] = (0, 0)
        return size_cache[p]

    for split in ("train", "val", "test"):
        ldir = root / "labels" / split
        idir = root / "images" / split
        if not ldir.exists():
            continue
        lbls = sorted(ldir.glob("*.txt"))
        print("\n===== split=%s  %d 张图" % (split, len(lbls)))
        percls = defaultdict(list)             # cname -> [(w,h)]
        suspicious = defaultdict(int)          # (cname,w,h) 计数
        stems = defaultdict(Counter)           # 命中 SUSPECT 的图 -> 类直方图
        for lb in lbls:
            img = None
            for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                q = idir / (lb.stem + ext)
                if q.exists():
                    img = q
                    break
            if img is None:
                continue
            W, H = imsize(img)
            if W == 0:
                continue
            hit = [k for k in SUSPECT if k in lb.stem]
            for line in lb.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                ci = int(float(p[0]))
                cn = names[ci] if 0 <= ci < len(names) else "?%d" % ci
                bw = float(p[3]) * W
                bh = float(p[4]) * H
                percls[cn].append((bw, bh))
                if hit:
                    for k in hit:
                        stems["%s|%s" % (k, lb.stem)][cn] += 1
                # 可疑签名：极小框
                if cn == "r" and bw <= 11 and 16 <= bh <= 24:
                    suspicious[("r", round(bw), round(bh))] += 1
                if cn == "net_input" and bh <= 14:
                    suspicious[("net_input", round(bw), round(bh))] += 1
        tot = sum(len(v) for v in percls.values())
        print("总框数 %d" % tot)
        row = []
        for cn, v in sorted(percls.items(), key=lambda kv: -len(kv[1]))[:14]:
            mw = stx.median(x[0] for x in v)
            mh = stx.median(x[1] for x in v)
            row.append("%s:%d(%.0fx%.0f)" % (cn, len(v), mw, mh))
        print("  类分布 " + " ".join(row))
        print("  可疑签名 %s" % (sorted(suspicious.items(), key=lambda kv: -kv[1])[:8] or "无"))
        for k in sorted(stems):
            print("    [%s] %s" % (k, dict(stems[k])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
