# -*- coding: utf-8 -*-
"""骨架连通性体检：原始骨架的分支点/端点/连通分量 vs stage1 工件里的 junctions/nodes。

判据：若原始骨架有大量分支点而工件的 `junctions` 为 0，则丢失发生在
`pskel.build_graph`/`group_segments` 一侧；若原始骨架几乎没有分支点，
则交叉处的骨架本身是断开的（细笔画在交叉/框边断了），属图像层缺陷。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))

from pipeline import pskel  # noqa: E402
from schematic_parser.data import DEFAULT_DATA_ROOT, discover  # noqa: E402

K = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], np.uint8)


def cases_by_id():
    out = {}
    for c in discover(DEFAULT_DATA_ROOT):
        cid = str(getattr(c, "case_id", "") or Path(getattr(c, "json_path", "")).parent.name)
        out[cid] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="work/_fast_out_leadmove")
    ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
    a = ap.parse_args()
    idx = cases_by_id()
    for cid in [s.strip() for s in a.cases.split(",") if s.strip()]:
        c = idx.get(cid)
        if c is None:
            continue
        bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
        ex = pskel.WireExtractor(ink_mode="adaptive", thin_w=2.6, use_colour=True)
        thin, _i, _h = ex.foreground(bgr)
        skel = pskel.zhang_suen((thin > 0).astype(np.uint8) * 255) > 0
        nb = cv2.filter2D(skel.astype(np.uint8), -1, K, borderType=cv2.BORDER_CONSTANT)
        deg = np.where(skel, nb - 10, 0)
        n_branch = int((deg >= 3).sum())
        n_end = int((deg == 1).sum())
        ncomp = int(cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)[0] - 1)
        # 交叉处是否断开：统计「两条骨架线相距 <=3px 但互不相连」的点对（粗略 = 端点近邻非自连通）
        st = json.loads((ROOT / a.run / "runs" / "stage1" / ("%s.stage1.json" % cid)).read_text(encoding="utf-8"))
        w = st.get("wires") or {}
        wd = st.get("wire_detail") or {}
        print("case %-3s 骨架 px %6d  分支点 %4d  端点 %4d  连通分量 %4d | 工件 n_nodes %s junctions %s "
              "n_junction_nodes %s endpoints %s n_segments %s n_rays %s"
              % (cid, int(skel.sum()), n_branch, n_end, ncomp, w.get("n_nodes"), w.get("n_junctions"),
                 w.get("n_junction_nodes"), w.get("n_endpoints"), w.get("n_segments"), wd.get("n_rays")))


if __name__ == "__main__":
    main()
