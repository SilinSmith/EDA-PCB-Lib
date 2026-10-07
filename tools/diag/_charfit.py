# -*- coding: utf-8 -*-
"""从 38.8 万条真实「框 + 文本」标注里**拟合逐字符的宽度**，替换 axis.py 里那个
「所有字符一样宽」的粗糙模型。

为什么必须做（owner m24624 的批评落到具体代码）：
  `eda-pcb\\fusion\\axis.py:115 pred_wh(n) = CHAR_A * n + CHAR_B` 把每个字符都当成
  同样宽。于是「框里少了一个窄字符（`1`）」几乎不改变预测宽度 ⇒ `len_prob` 很少
  掉到 `LEN_CONFLICT = 0.30` 以下 ⇒ `inject` 的**强证据分支是死代码** ⇒ 整条轴链
  在决策层**从不生效**（19 张定点档实测：18 张最终决策一字不改）。

判据：拟合完后，(a) 残差应显著小于均匀模型；(b) `len_prob_text` 对
`('C15','C155')` 这类只差一个窄字符的对，必须能分开。

用法：python work\\tools\\diag\\_charfit.py [--write]
"""
import json
import math
import os
import sys
from collections import Counter

MAN = r"Y:\workspace\EDA\eda-pcb\textrec\data\manifest.jsonl"
OUT = r"Y:\workspace\EDA\eda-pcb\fusion\charwidth.py"

ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_./+"
MIN_COUNT = 300
MAXLEN = 12
RIDGE = 1e-3


def main():
    rows = []
    cnt = Counter()
    n_seen = n_skip = 0
    with open(MAN, encoding="utf-8") as fh:
        for line in fh:
            n_seen += 1
            try:
                d = json.loads(line)
            except Exception:
                n_skip += 1
                continue
            t = str(d.get("text") or "")
            bb = d.get("bbox")
            if not t or not bb or len(t) > MAXLEN:
                n_skip += 1
                continue
            w = float(bb[2]) - float(bb[0])
            h = float(bb[3]) - float(bb[1])
            if h <= 1.0 or w <= 0.5:
                n_skip += 1
                continue
            if any(ch not in ALPHA for ch in t):
                n_skip += 1
                continue
            rows.append((t, w / h))
            cnt.update(t)
    chars = sorted([c for c in cnt if cnt[c] >= MIN_COUNT])
    keep = set(chars)
    data = [(t, r) for t, r in rows if all(ch in keep for ch in t)]
    print("读入 %d 行，可用 %d 行（丢 %d），出现≥%d 次的字符 %d 个"
          % (n_seen, len(data), n_skip, MIN_COUNT, len(chars)))
    print("字符频次 top12:", cnt.most_common(12))
    if not data:
        print("没有可用数据")
        return 1

    # 最小二乘：r_i ≈ pad + Σ_c count_ic * adv_c
    # **必须带截距 `pad`**：这一版第一稿强行过原点，被长文本主导，斜率虚高到 0.85/字符；
    # 实际 `R1` 的 r=1.24、`EB+` 的 r=2.09 ⇒ 每个框有一份**与字符数无关的边距**。
    # 丢掉截距 = 把「框的边距」摊进「字符宽度」，那样算出来的字数会系统性偏多。
    import numpy as np
    idx = {c: i for i, c in enumerate(chars)}
    n = len(chars)
    A = np.zeros((len(data), n + 1))
    y = np.zeros(len(data))
    for i, (t, r) in enumerate(data):
        for ch in t:
            A[i, idx[ch]] += 1.0
        A[i, n] = 1.0                       # 截距列
        y[i] = r
    R = RIDGE * len(data) * np.eye(n + 1)
    R[n, n] = 0.0                           # 截距不加正则
    adv = np.linalg.solve(A.T @ A + R, A.T @ y)
    pad = float(adv[n])
    adv = adv[:n]
    pred = A[:, :n] @ adv + pad
    res = y - pred
    rmse = float(np.sqrt((res ** 2).mean()))
    # 对照一：均匀模型 a*n + b（旧 axis.CHAR_A/CHAR_B 的函数形态）
    cnts = A[:, :n].sum(axis=1)
    D = np.stack([cnts, np.ones_like(cnts)], axis=1)
    ab = np.linalg.lstsq(D, y, rcond=None)[0]
    rmse0 = float(np.sqrt(((y - D @ ab) ** 2).mean()))
    # 对照二：强行过原点的逐字符模型（第一稿的错误形态）
    P = A[:, :n]
    adv_o = np.linalg.solve(P.T @ P + RIDGE * len(data) * np.eye(n), P.T @ y)
    rmse_o = float(np.sqrt(((y - P @ adv_o) ** 2).mean()))
    print("均匀 a·n+b  (a=%.3f b=%.3f) rmse = %.4f" % (ab[0], ab[1], rmse0))
    print("逐字符过原点                   rmse = %.4f" % rmse_o)
    print("逐字符 + 截距 (pad=%.3f)      rmse = %.4f（比均匀降 %.1f%%）"
          % (pad, rmse, 100.0 * (1 - rmse / rmse0)))

    table = {c: round(float(adv[idx[c]]), 4) for c in chars}
    # 抽检：只看数字与常见字母，确认「1」确实最窄
    dig = sorted([c for c in chars if c.isdigit()], key=lambda c: table[c])
    print("数字宽度排序（窄→宽）:", [(c, table[c]) for c in dig])
    print("字母宽度（A/C/R/U）:", {c: table.get(c) for c in "ACRU" if c in table})

    # 判据：只差一个窄字符的一对，必须能被分开
    def pred_wh_text(t):
        return pad + sum(table.get(ch, ab[0]) for ch in t)
    for a, b in [("C15", "C155"), ("C5", "C15"), ("R1", "R11")]:
        pa, pb = pred_wh_text(a), pred_wh_text(b)
        print("  %-5s → 预测 w/h %.3f ｜ %-6s → %.3f（差 %.3f）" % (a, pa, b, pb, pb - pa))
    # 真实标注上核对：`R1`(1.24) / `EB+`(2.09) 这两个具体框
    print("  抽检: R1(实测 1.24) 预测 %.3f ｜ EB+(实测 2.09) 预测 %.3f"
          % (pred_wh_text("R1"), pred_wh_text("EB+")))

    if "--write" in sys.argv:
        with open(OUT, "w", encoding="utf-8") as fh:
            fh.write('# -*- coding: utf-8 -*-\n')
            fh.write('"""逐字符宽度表（**自动生成**，勿手改）。\n\n')
            fh.write("来源：`eda-pcb\\\\textrec\\\\data\\\\manifest.jsonl`（38.8 万条真实「框+文本」标注），\n")
            fh.write("最小二乘拟合 `w/h ≈ PAD + Σ_c count_c · ADV[c]`（**带截距**：每个框有一份\n")
            fh.write("与字符数无关的边距；第一稿强行过原点导致斜率虚高到 0.85/字符）。\n")
            fh.write("生成脚本：`work\\\\tools\\\\diag\\\\_charfit.py`。\n\n")
            fh.write("为什么需要它：`fusion/axis.py` 原来把所有字符当一样宽（`CHAR_A·n + CHAR_B`），\n")
            fh.write("而真实「每字符宽度」在数字之间只差 0.05（`1`=%.4f … `8`=%.4f），**并不窄**；\n"
                     % (table.get("1", 0.0), table.get("8", 0.0)))
            fh.write("真正决定字数的是**框的边距 + 累计字宽**，所以必须逐字符 + 截距。\n")
            fh.write("实测：均匀 a·n+b rmse %.4f（a=%.3f b=%.3f）｜本表 rmse %.4f（降 %.1f%%）。\n"
                     % (rmse0, ab[0], ab[1], rmse, 100.0 * (1 - rmse / rmse0)))
            fh.write('"""\n\n')
            fh.write("PAD = %.4f\n\n" % pad)
            fh.write("ADV = {\n")
            for c in sorted(chars):
                fh.write("    %r: %.4f,\n" % (c, table[c]))
            fh.write("}\n\n")
            fh.write("# 未在表中出现的字符（罕见/符号）退化到均匀字宽。\n")
            fh.write("FALLBACK = %.4f\n" % ab[0])
            fh.write("# 残差 rmse，`len_prob` 的 σ 由它定（**不要**再用手拍的两档散度）。\n")
            fh.write("RMSE = %.4f\n" % rmse)
        print("已写入", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
