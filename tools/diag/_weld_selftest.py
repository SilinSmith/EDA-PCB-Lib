"""方向性桥接（stage1.labels._weld_unions / wire_groups 规则 3）的合成自检。

**为什么必须写这个**：本项目两次栽在连通图代码上 —— `stage1_conn_eval.build_groups`
把第 i 段与自己的序号后继并起来（整张图塌成一个分量），`stage1_filter_ablation` 把
`dist_pts_to_segs` 的 (M,) 当 (M,N) 用（所有引脚绑到第 0 条段）。两次都是**真实数据
上看不出来的**：真实数据只给你一个「R 有点低 / R 有点高」的数字，合成用例能给出
非黑即白的对错。所以每次动图代码，先跑这个。

用法：python Y:\workspace\EDA\work\_weld_selftest.py
"""
import sys

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "eval"))
import numpy as np

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

from stage1.labels import _extend_unions, _weld_unions, wire_groups  # noqa: E402

FAILS = []


def check(name, got, want):
    ok = got == want
    if not ok:
        FAILS.append(name)
    print("  [%s] %-44s got %-8s want %s" % ("OK" if ok else "FAIL", name, got, want))
    return ok


def ncomp(segs, **kw):
    return len(set(wire_groups(np.asarray(segs, np.float64).reshape(-1, 4), **kw)))


def weld_pairs(segs, weld, perp_tol=1.5, cos_tol=-0.75):
    """只要 _weld_unions 的判定，绕开规则 (1)(2) —— 否则测不出方向门控。"""
    arr = np.asarray(segs, np.float64).reshape(-1, 4)
    ends = np.stack([arr[:, :2], arr[:, 2:]], axis=1).reshape(-1, 2)
    own = np.repeat(np.arange(len(arr)), 2)
    return sorted((int(a), int(b))
                  for a, b in _weld_unions(arr, ends, own, weld, perp_tol, cos_tol))


def main():
    print("=== A. 只测 _weld_unions 的方向门控（绕开规则 1/2）===")
    print("A1) 共线、面对面、间隔 6px -> 应桥接")
    check("面对面共线", weld_pairs([[0, 0, 50, 0], [56, 0, 106, 0]], 8.0), [(0, 1)])
    print("A2) 平行、垂直偏移 6px -> 不得桥接（perp 门控）")
    check("平行偏移", weld_pairs([[0, 0, 50, 0], [0, 6, 50, 6]], 8.0), [])
    print("A3) 同向、间隔 6px（a 尾 +x 朝外，b 头也 +x 朝外）-> 不得桥接（cos 门控）")
    check("同向不相对", weld_pairs([[0, 0, 50, 0], [56, 0, 6, 0]], 8.0), [])
    print("A4) 对面但距离 20px -> 够不着")
    check("超出 weld", weld_pairs([[0, 0, 50, 0], [70, 0, 120, 0]], 8.0), [])
    print("A5) 45 度共线断口 -> 应桥接")
    check("斜线断口", weld_pairs([[0, 0, 35.355, 35.355], [39.6, 39.6, 74.9, 74.9]], 10.0), [(0, 1)])
    print("A6) 对面但横向错位 4px（沿着延长线）-> perp_tol=1.5 时不得桥接")
    check("横向错位", weld_pairs([[0, 0, 50, 0], [56, 4, 106, 4]], 8.0), [])

    print("")
    print("=== B. wire_groups 整体（含规则 1/2）===")
    print("B1) 相互远离的段不得合并（塌成一坨 = concatenate 那个 bug 回来了）")
    far = [[0, 0, 10, 0], [200, 0, 210, 0], [400, 0, 410, 0]]
    check("三段远离", ncomp(far, tol=0.5, weld=0.0), 3)
    check("三段远离 + weld 开着", ncomp(far, tol=0.5, weld=40.0), 3)
    print("B2) 共线断口")
    gap = [[0, 0, 50, 0], [56, 0, 106, 0]]
    check("weld=0 不桥接", ncomp(gap, tol=0.5, weld=0.0), 2)
    check("weld=4 够不着", ncomp(gap, tol=0.5, weld=4.0), 2)
    check("weld=8 桥接", ncomp(gap, tol=0.5, weld=8.0), 1)
    print("B3) T 型结点（规则 2，端点距线段 1px）")
    tee = [[0, 0, 100, 0], [50, 1, 50, 40]]
    check("tol=0.5 不合并", ncomp(tee, tol=0.5, weld=0.0), 2)
    check("tol=2 合并", ncomp(tee, tol=2.0, weld=0.0), 1)
    print("B4) 端点-端点共点（规则 1）")
    check("共点合并", ncomp([[0, 0, 50, 0], [50, 0, 50, 40]], tol=0.5, weld=0.0), 1)
    print("B5) 段索引配平：两个端点必须在 ends[2i]/ends[2i+1]（否则塌成一坨）")
    check("远离段 + 自动子索引", ncomp(far, tol=0.5, weld=0.0), 3)

    print("")
    print("=== C. 规则 4：端点沿外法线朝前延伸（_extend_unions）===")
    print("C1) 竖直引脚走线停在水平总线上方 10px —— 这是实测里最常见的断法")
    tee = [[0, 0, 100, 0], [50, -30, 50, -10]]
    check("ext=8 够不着", ncomp(tee, tol=0.5, weld=0.0, ext=8.0, ext_perp=2.0), 2)
    check("ext=16 接上", ncomp(tee, tol=0.5, weld=0.0, ext=16.0, ext_perp=2.0), 1)
    print("C2) 正前方没有东西（目标段不在射线上）-> 不得瞎接")
    miss = [[0, 0, 20, 0], [50, -30, 50, -10]]
    check("ext=32 仍为 2", ncomp(miss, tol=0.5, weld=0.0, ext=32.0, ext_perp=2.0), 2)
    print("C3) 平行偏移 6px -> 侧向门控拦住")
    par = [[0, 0, 50, 0], [0, -6, 50, -6]]
    check("平行偏移不接", ncomp(par, tol=0.5, weld=0.0, ext=16.0, ext_perp=2.0), 2)
    print("C4) 三段远离 + ext 开着 -> 不得塌成一坨")
    check("远离段 + ext", ncomp(far, tol=0.5, weld=0.0, ext=16.0, ext_perp=2.0), 3)

    print("")
    if FAILS:
        print("**有 %d 个 FAIL，禁止把结果当真**：%s" % (len(FAILS), FAILS))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
