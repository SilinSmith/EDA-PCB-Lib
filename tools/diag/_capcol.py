# -*- coding: utf-8 -*-
"""§5.116 探针：给队友树的彩色实测臂加「外伸量上限」，检验极低分是否来自
`apply_color_pins` 把引脚点放在红桩末端（实测 ~19px，而官方 GT 约定 ~12px）。

改动：`work/_wt_pr3/stage1/pipeline.py` 的 `apply_color_pins()` 内，在把
`r["point_img"]` 收进 `measured` 之前，若设了 `STAGE1_PIN_COLOR_CAP`（>0），
就沿「框边锚点 anchor_img → 实测点」方向把该点拉回到距锚点不超过 cap 像素处。

幂等：正文里带 `_CAP_MARK` 标记，已打过就跳过。

用法：
    python -X utf8 work\\_capcol.py --apply
    python -X utf8 work\\_capcol.py --revert
"""
from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

TARGET = Path(r"Y:\workspace\EDA\work\_wt_pr3\stage1\pipeline.py")
MARK = "_CAP_MARK_STAGE1_PIN_COLOR_CAP"

BLOCK = '''            # %s：把实测点沿「框边锚点→点」方向拉回到不超过 CAP 像素处。
            # 动机（官方 10GT 实测）：彩色实测点落在红桩末端（外距中位 19.3px），
            # 而官方 GT 的引脚点外距中位 12.0px（均值 12.5，6~22），pin_tol=5
            # ⇒ 整边被判 position_mismatch，引脚通道崩到 F1 0.13（我们树同理 65.14）。
            _cap = float(os.environ.get("STAGE1_PIN_COLOR_CAP", "0") or 0)
            if _cap > 0:
                _a = r.get("anchor_img")
                _p = r["point_img"]
                if isinstance(_a, (list, tuple)) and len(_a) == 2:
                    _dx, _dy = _p[0] - _a[0], _p[1] - _a[1]
                    _d = math.hypot(_dx, _dy)
                    if _d > _cap and _d > 1e-6:
                        r["point_img"] = [round(_a[0] + _dx / _d * _cap, 1),
                                          round(_a[1] + _dy / _d * _cap, 1)]
''' % MARK

ANCHOR = '            r["trust"] = "trusted"\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    src = io.open(str(TARGET), encoding="utf-8").read()
    if a.revert:
        if MARK not in src:
            print("未打补丁，无需还原")
            return
        out = re.sub(re.escape(BLOCK), "", src, count=1)
        io.open(str(TARGET), "w", encoding="utf-8", newline="\n").write(out)
        print("已还原：%s" % TARGET)
        return
    if MARK in src:
        print("已打过补丁（幂等跳过）")
        return
    if ANCHOR not in src:
        raise SystemExit("找不到插入锚点：%r" % ANCHOR)
    out = src.replace(ANCHOR, BLOCK + ANCHOR, 1)
    io.open(str(TARGET), "w", encoding="utf-8", newline="\n").write(out)
    print("已打补丁：%s（+%d 行）" % (TARGET, BLOCK.count("\n")))


if __name__ == "__main__":
    main()
