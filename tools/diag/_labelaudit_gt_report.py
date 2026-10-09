"""GT 错标体检 —— 汇总与报告生成。

读 `_labelaudit_gt.py` 的全库扫描缓存，产出：
  - Y:\\workspace\\EDA\\work\\_labelaudit.md   （中文报告）

只读数据集；只写这一个 md（CSV 由 _labelaudit_gt.py 写）。
运行：  $env:PYTHONIOENCODING="utf-8"; python work\\tools\\diag\\_labelaudit_gt_report.py
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _labelaudit_gt as L  # noqa: E402

ROOT = r"Y:\workspace\EDA"
MD = os.path.join(ROOT, "work", "_labelaudit.md")
SRC_ORDER = ["KiCad", "Altium Designer", "Datasheet", "jlc", "other", "unknown"]
ZERO = 1e-9


def oob_px(r, meta):
    """越界像素数（最大方向超出量）。0 = 未越界。"""
    wh = meta["sizes"].get(f"{r['split']}/{r['case']}")
    if not wh:
        return 0.0
    w, h = wh
    x1, y1, x2, y2 = r["box"]
    return max(0.0, -x1, -y1, x2 - w, y2 - h)


def hollow(r) -> bool:
    return bool(r["ink"] <= L.EMPTY_INK and r.get("enclosed"))


def true_empty(r) -> bool:
    return bool(r["ink"] <= L.EMPTY_INK and not r.get("enclosed"))


def stub(r) -> bool:
    return bool(0 < r["span_max"] < 0.34 and max(r["ink"], r["ink_same"]) > L.EMPTY_INK)


def strong(r, meta) -> bool:
    """强可疑（应丢弃/重标）= 真错标，全部经 ASCII 逐像素复核。"""
    return bool(true_empty(r) or oob_px(r, meta) > 2.0 or r["dup"] or stub(r))


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def pct(a, b):
    return 0.0 if not b else 100.0 * a / b


def pref(k):
    m = re.match(r"^([A-Za-z\u2205]+)", k)
    return m.group(1) if m else "?"


def main() -> None:
    recs, meta = L.load_or_scan()
    n_all = len(recs)
    by_t = defaultdict(list)
    by_s = defaultdict(list)
    for r in recs:
        by_t[r["type"]].append(r)
        by_s[r["src"]].append(r)
    modes = {t: L.modal_size(rs) for t, rs in by_t.items() if len(rs) >= 5}

    def msz(t):
        m = modes.get(t)
        return m[0] if m else None

    # ---------- ① 逐类汇总 ----------
    t1 = ["| type | n | ink≤2% | 空心hollow | 真错标 | 越界oob>2px | off中位 | 众数尺寸(占比) | 重复框 | 强可疑 | 图源分布 |\n",
          "|---|---|---|---|---|---|---|---|---|---|---|\n"]
    tot_strong = 0
    for t, rs in sorted(by_t.items(), key=lambda x: -len(x[1])):
        n = len(rs)
        e = sum(1 for r in rs if r["ink"] <= L.EMPTY_INK)
        ho = sum(1 for r in rs if hollow(r))
        te = sum(1 for r in rs if true_empty(r))
        ob = sum(1 for r in rs if oob_px(r, meta) > 2.0)
        dup = sum(1 for r in rs if r["dup"])
        st = sum(1 for r in rs if strong(r, meta))
        tot_strong += st
        offs = [r["off_cc"] for r in rs if r["off_cc"] == r["off_cc"]]
        m = modes.get(t)
        mstr = f"{m[0][0]}x{m[0][1]} ({m[1] * 100:.0f}%)" if m else "—"
        sc = Counter(r["src"] for r in rs)
        sstr = " ".join(f"{s[:3].lower()}{sc[s]}" for s in SRC_ORDER if sc[s])
        t1.append(
            f"| `{t}` | {n} | {pct(e, n):.1f}% | {ho} ({pct(ho, n):.1f}%) | {te} ({pct(te, n):.1f}%) | "
            f"{ob} ({pct(ob, n):.1f}%) | {med(offs):.2f} | {mstr} | {dup} | {st} ({pct(st, n):.1f}%) | {sstr} |\n")
    t1 = "".join(t1)

    # ---------- ② 按图源 ----------
    t2 = ["| 图源 | case | 框数 | 强可疑 | 占比 | 空心hollow | 真错标 | 越界 | off中位 | 众数尺寸占比中位 |\n",
          "|---|---|---|---|---|---|---|---|---|---|\n"]
    for s in SRC_ORDER:
        rs = by_s.get(s, [])
        if not rs:
            continue
        n = len(rs)
        st = sum(1 for r in rs if strong(r, meta))
        ho = sum(1 for r in rs if hollow(r))
        te = sum(1 for r in rs if true_empty(r))
        ob = sum(1 for r in rs if oob_px(r, meta) > 2.0)
        offs = [r["off_cc"] for r in rs if r["off_cc"] == r["off_cc"]]
        shares = [(L.modal_size(by_t[r["type"]]))[1] for r in rs if r["type"] in modes]
        nc = len({r["case"] for r in rs})
        t2.append(f"| {s} | {nc} | {n} | {st} | **{pct(st, n):.1f}%** | {ho} ({pct(ho, n):.1f}%) | "
                  f"{te} ({pct(te, n):.1f}%) | {ob} | {med(offs):.2f} | {med(shares) * 100:.0f}% |\n")
    t2 = "".join(t2)

    # 图源 × 类型 强可疑率 >=15% 且 >=15 框
    bad_src = []
    for s in SRC_ORDER:
        for t, rs in by_t.items():
            sub = [r for r in rs if r["src"] == s]
            if len(sub) >= 15:
                st = sum(1 for r in sub if strong(r, meta))
                if pct(st, len(sub)) >= 15.0:
                    bad_src.append((s, t, len(sub), st, pct(st, len(sub)), med([r["off_cc"] for r in sub if r["off_cc"] == r["off_cc"]])))

    # ---------- ③ 最差 20 case ----------
    byc = defaultdict(list)
    for r in recs:
        byc[(r["split"], r["case"])].append(r)
    rank = []
    for (sp, cid), rs in byc.items():
        n = len(rs)
        st = sum(1 for r in rs if strong(r, meta))
        rank.append((st / n, st, n, sp, cid, rs[0]["src"]))
    rank.sort(key=lambda x: (-x[0], -x[1]))
    t3 = ["| # | case | split | 图源 | 框数 | 强可疑 | 占比 | 主要理由 |\n", "|---|---|---|---|---|---|---|---|\n"]
    for i, (rate, st, n, sp, cid, src) in enumerate(rank[:20], 1):
        rs = byc[(sp, cid)]
        rc = Counter()
        for r in rs:
            for tag in L.reasons_of(r, msz(r["type"])).split(";"):
                if tag and not tag.startswith("oddtpl"):
                    rc[tag.split(".")[0][:5]] += 1
        top = " ".join(f"{k}×{v}" for k, v in rc.most_common(3))
        t3.append(f"| {i} | `{cid}` | {sp} | {src} | {n} | {st} | {rate * 100:.1f}% | {top} |\n")
    t3 = "".join(t3)

    # ---------- ④ 固定模板 ----------
    fixed = sorted(((t, len(rs), L.modal_size(rs)) for t, rs in by_t.items() if len(rs) >= 5),
                   key=lambda x: -x[2][1])
    fixed_top = [(t, n, f"{m[0][0]}x{m[0][1]}", m[1] * 100, m[2]) for t, n, m in fixed if m[1] >= 0.90]

    # ---------- ⑤ 类型一致性（键名前缀 vs type）----------
    d_pfx = defaultdict(Counter)   # type -> 键名前缀
    for r in recs:
        d_pfx[r["type"]][pref(r["key"])] += 1
    led_d = sum(1 for r in recs if r["type"] == "led" and pref(r["key"]) == "D")
    d_d = sum(1 for r in recs if r["type"] == "d" and pref(r["key"]) == "D")
    led_led = sum(1 for r in recs if r["type"] == "led" and pref(r["key"]) == "LED")
    mix_cases = []
    for (sp, cid), rs in sorted(byc.items()):
        dl = [r for r in rs if r["type"] in ("d", "led")]
        if len({r["type"] for r in dl}) > 1 and any(pref(r["key"]) == "D" for r in dl):
            mix_cases.append((sp, cid, " ".join(f"{r['key']}:{r['type']}" for r in dl[:6])))
    pin_tp = sum(1 for r in recs if r["type"] == "pin" and pref(r["key"]) == "TP")
    tp_tp = sum(1 for r in recs if r["type"] == "testpoint" and pref(r["key"]) == "TP")
    pin_ink = med([r["ink"] for r in recs if r["type"] == "pin"])
    tp_ink = med([r["ink"] for r in recs if r["type"] == "testpoint"])

    # ---------- 总量口径 ----------
    n_z25 = sum(1 for r in recs if r["ink25"] <= ZERO)
    n_e25 = sum(1 for r in recs if r["ink25"] <= L.EMPTY25)
    n_txt = sum(1 for r in recs if r["txt"] >= L.TXT_HI)
    n_ot = sum(1 for r in recs if L.on_text(r))
    n_empty = sum(1 for r in recs if r["ink"] <= L.EMPTY_INK)
    n_ho = sum(1 for r in recs if hollow(r))
    n_te = sum(1 for r in recs if true_empty(r))
    n_dup = sum(1 for r in recs if r["dup"])
    n_st = sum(1 for r in recs if stub(r))
    n_oob = sum(1 for r in recs if oob_px(r, meta) > 2.0)
    n_oob0 = sum(1 for r in recs if oob_px(r, meta) > 0)
    n_off = sum(1 for r in recs if r["off_cc"] == r["off_cc"] and r["off_cc"] > L.OFF_HI)
    n_types = len(by_t)

    # 各类掉幅
    drop = []
    for t, rs in by_t.items():
        n = len(rs)
        st = sum(1 for r in rs if strong(r, meta))
        ho = sum(1 for r in rs if hollow(r))
        drop.append((t, n, st, pct(st, n), ho, pct(ho, n)))
    drop.sort(key=lambda x: -x[3])

    out = []
    A = out.append
    A("# GT 元件框「错标」体检报告 —— 220 case / 9609 框\n\n")
    A("> 只测量、未改任何数据。复现（Windows，必须先设编码，否则 Unicode 键名 GBK 报错）：\n>\n")
    A("> ```\n> cd Y:\\workspace\\EDA\n"
      "> python work\\tools\\diag\\_labelaudit_gt.py --fresh      # 逐框扫描 -> work\\_labelaudit.csv（约 61 s）\n"
      "> python work\\tools\\diag\\_labelaudit_gt_report.py       # 汇总 -> work\\_labelaudit.md\n> ```\n\n")
    A("**`work\\_labelaudit.csv`（3516 行 = 所有带标签的框）列义**："
      "`off_cc` = 同域质心偏移（本报告口径）；`off_raw` = 未做连通域过滤的原始口径（父 agent 已指出不可用，仅备查）；"
      "`hollow` = 1 表示框心白区被符号轮廓包住（框偏小）；`hard` = 1 表示属硬缺陷"
      "（空/空心/重复/stub/越界）；`reason` 为分号标签。"
      "注意 `reason` 里的 `oob` 含**全部 58 个**越界框（其中 17 个只超出 ≤2px），而本报告表格用的是 **>2px 的 41 个**；"
      "`oddtpl`（3232 行）只是「与该类众数尺寸不同」，**不是缺陷**，绝大多数类本来就没有固定尺寸，仅备查。\n\n")

    A("## 0. 口径与四处自我纠错（先看，否则会误读下表）\n\n")
    A("1. **墨迹阈值 = 灰度 <200**（`LABELAUDIT_INK` 覆盖）。`work\\_gndbox.py` 原用 <128 是错的："
      "本库符号笔画是浅灰，全库直方图在 **151/152 有独立尖峰**，243~253 才是近白底。\n")
    A(f"2. **`ink`（框内墨迹占比）对空心符号是错的指标。** 电阻画成空心矩形，GT 框只套在矩形内部的白区 —— "
      f"肉眼看图（`10GT/10` 的 R1）确认框位置正确、只是框偏小。故把 `ink≤2%` 拆成 **空心 hollow {n_ho} 框**"
      f"（泛洪判定：框心白区被符号轮廓包住）与**真错标 {n_te} 框**（框内确实没东西）。"
      f"不做这一步，`r` 会被报成 6.2% 空框、实际是 0.1%（98/100 是空心内部）。\n")
    A(f"3. **`txt`（框压文字）naive 口径严重高估。** `cvparts\\textbox.py` 在 ~20px 尺度分不清文字与符号轮廓 —— "
      f"ASCII 逐像素复核抓到**二极管三角形被整块当成一个「文字行」**。naive 给 {n_txt} 框（{pct(n_txt, n_all):.1f}%），"
      f"加「≥3 字形颗粒同一 ±3px 横带且横向铺开 ≥10px」闸门后只剩 **{n_ot} 框（{pct(n_ot, n_all):.2f}%）**，"
      f"且多为 IC 大框内本来就有的引脚名 ⇒ 不是主要错标形态。\n")
    A(f"4. **`off`（外扩窗墨迹质心到框心）不能单独当证据。** 父 agent 提示要做连通域取「与框内同域」部分；"
      f"本报告用 `off_cc` 这么算了，但**对带引线的符号仍是下界**：导线与符号连通，质心被引线拉走。"
      f"`off_cc>20px` 的 {n_off} 框里，{sum(1 for r in recs if r['off_cc'] == r['off_cc'] and r['off_cc'] > L.OFF_HI and max(r['w'], r['h']) > 40)} "
      f"个是大框（IC/AMP，属正常）⇒ **未把 off_cc 计入「强可疑」**，只作参考列。\n\n")

    A(f"## 1. 逐类汇总表（{n_types} 个类型，按 n 降序）\n\n")
    A("`ink≤2%` = 用户原始口径；`空心hollow` = 其中被符号轮廓包住（框偏小、位置正确）；`真错标` = 框内确实无墨；"
      "`越界` = 框超出图像 >2px；`强可疑` = 真错标∪越界∪重复∪stub。\n\n")
    A(t1)
    A(f"\n全库合计：**{n_all} 框**，空心 {n_ho}，真错标 {n_te}，越界 {n_oob}（>0px 为 {n_oob0}），"
      f"重复 {n_dup}，stub {n_st}，**强可疑 {tot_strong}（{pct(tot_strong, n_all):.2f}%）**；"
      f"外扩 25px 内**严格 0 墨的框 = {n_z25}**（≤0.5% 墨的 = {n_e25}）。\n\n")

    A("## 2. 按图源的错标率（全类别）\n\n")
    A(t2)
    A("\n")
    if bad_src:
        A("按「图源 × 类型」下钻，强可疑率 ≥15% 且该类该图源 ≥15 框：\n\n")
        for s, t, n, st, p, o in sorted(bad_src, key=lambda x: -x[4]):
            A(f"- `{s}` × `{t}`：{n} 框，强可疑 {st}（**{p:.1f}%**），off中位 {o:.1f}\n")
        A("\n")
    A(f"\n注意 1：**`jlc` 不是最差**，这与「jlc 特别差」的预期相反 —— 见第 7 节 (2)。\n\n")
    A(f"注意 2：**「越界」不是标注者标错，而是裁图管线缺陷。** 证据：41 个越界框里 36 个属 `Datasheet`；"
      f"超出量中位 5px、最大 35px；同一 case 内越界框总是**贴着同一条图像边**（如 `0033` 5 个 gnd 框 y 全为 428~447、"
      f"图像高仅 442）；且 **220 个 case 里有 21 个的最外一行/一列本身就含墨迹**（`10GT/7` 底行 2202 墨迹像素）"
      f"—— 说明 png 被裁得比标注范围紧，符号本身被边界切断，不是框标错了对象。"
      f"已裁图肉眼确认：`0033` 底边 5 个 GND 符号、`0096` 顶边 6 个 net 旗标都被图像边界切掉一截。"
      f"修法是从原始矢量重新导出（不要二次裁切），而不是逐框改坐标。\n\n")

    A("## 3. 最差的 20 张 case（错标框数 / 该 case 框数）\n\n")
    A(t3)

    A("\n## 4. 固定模板框（众数尺寸占比 ≥90%，n≥5）\n\n")
    A("| type | n | 众数尺寸 | 占比 | 尺寸种数 |\n|---|---|---|---|---|\n")
    for t, n, sz, sh, nd in fixed_top:
        A(f"| `{t}` | {n} | {sz} | {sh:.0f}% | {nd} |\n")
    A("\n尺寸全库唯一 = 无尺度多样性（**不等于**标错，但该类无法从尺寸学泛化）。\n\n")

    A("## 5. 类型一致性（第 6 项）：同模板被标成不同类型\n\n")
    A(f"- **`pin`(175) vs `testpoint`(43)：同一个模板、同一套键名。** 两者框尺寸 **100% 都是 20x20**；"
      f"键名前缀 `pin` {pin_tp}/175 与 `testpoint` {tp_tp}/43 **全是 `TP<数字>`**，且**从不共现于同一 case**。"
      f"ASCII 复核显示内容有别（`testpoint` = 实心圆盘，ink中位 {tp_ink:.2f}；`pin` = 小字/箭头，ink中位 {pin_ink:.2f}）"
      f"⇒ 是两个东西，但**共用 `TPn` 命名空间 + 同一尺寸模板**，是最大的类型混淆风险源，建议人工统一命名。\n")
    A(f"- **`d` vs `led`：同一个 `D` 前缀被拆成两个类型。** `d` {d_d}/201 键名是 `D` 前缀；`led` 里 "
      f"**{led_d}/186 也是 `D` 前缀**（另 {led_led} 是 `LED` 前缀）。两者尺寸集**完全相同**（14x20 / 20x14）。"
      f"更硬的是**同图内同类同尺寸却被标成两种**，共 **{len(mix_cases)} 个 case**，例如：\n\n")
    A("| case | 同图内的 d / led |\n|---|---|\n")
    for sp, cid, s in mix_cases[:6]:
        A(f"| `{cid}` | {s} |\n")
    A(f"\n  例：`200_train_cases/0150` 同一张图里 `D1:led(14x20)` 与 `D2:d(14x20)`。"
      f"这类「同一 `Dn` 命名空间、同一尺寸模板、同图并存」说明 `d`/`led` 的划分含大量标注者主观噪声 ⇒ **必须统一或重标**。\n")
    A("- `net_input`/`net_output`/`net_bidirection` 三者众数尺寸都是 30x10，但键名全是 `∅` 且几何不同（箭头方向），"
      "属可区分的真类别，未发现同图矛盾。\n\n")

    A("## 6. 每类的结论措辞\n\n")
    A("用户判据：`基本可信` = ink≤2% 占比 <1% 且 众数尺寸占比 <90%。按此判据：\n\n")
    ok, sus = [], []
    for t, rs in sorted(by_t.items(), key=lambda x: -len(x[1])):
        n = len(rs)
        e = pct(sum(1 for r in rs if r["ink"] <= L.EMPTY_INK), n)
        m = modes.get(t)
        share = m[1] * 100 if m else -1.0
        (ok if (e < 1.0 and share < 90.0) else sus).append((t, n, e, share, sum(1 for r in rs if strong(r, meta))))
    A(f"- **标注基本可信（{len(ok)} 类）**：" + "、".join(f"`{t}`" for t, _, _, _, _ in ok) + "\n")
    A(f"- **固定模板 / 框内常常没东西（{len(sus)} 类）**：\n\n")
    A("| type | n | ink≤2% | 众数占比 | 强可疑 | 修正后判读 |\n|---|---|---|---|---|---|\n")
    for t, n, e, share, st in sorted(sus, key=lambda x: -x[1]):
        ho = sum(1 for r in by_t[t] if hollow(r))
        te = sum(1 for r in by_t[t] if true_empty(r))
        if te >= 3 and pct(te, n) >= 5.0:
            verdict = f"**真错标 {te} 框({pct(te, n):.0f}%)，该重标**"
        elif ho >= 3 and pct(ho, n) >= 5.0:
            verdict = f"框偏小（空心 {ho} 框），**需放大框**"
        elif share >= 99.0 and n >= 20:
            verdict = "尺寸固定但位置正确，可用"
        elif e >= 1.0:
            verdict = "少量空框，位置大体正确"
        else:
            verdict = "尺寸固定，位置正确"
        A(f"| `{t}` | {n} | {e:.1f}% | {share:.0f}% | {st} ({pct(st, n):.1f}%) | {verdict} |\n")
    A("\n")

    A("## 7. 结论：两个问题的回答\n\n")
    A("### (1) 丢弃不可信框后，各类掉多少训练样本？\n\n")
    A(f"全库 {n_all} 框，「强可疑」**{tot_strong} 框 = {pct(tot_strong, n_all):.2f}%**"
      f"（真错标 {n_te} + 越界(裁图) {n_oob} + 重复 {n_dup} + stub {n_st}，去重后）。"
      f"若只算**标注本身错**（真错标∪重复∪stub，不含裁图造成的越界），是 "
      f"**{sum(1 for r in recs if true_empty(r) or r['dup'] or stub(r))} 框 = "
      f"{pct(sum(1 for r in recs if true_empty(r) or r['dup'] or stub(r)), n_all):.2f}%**。"
      f"**按类看，没有一类掉幅超过 12%**：\n\n")
    A("| type | 原 n | 应丢(强可疑) | 掉幅 | 另需重标(空心) | 空心占比 | 剩余 |\n|---|---|---|---|---|---|---|\n")
    for t, n, st, p, ho, hp in drop:
        if st == 0 and ho == 0:
            continue
        A(f"| `{t}` | {n} | {st} | **{p:.1f}%** | {ho} | {hp:.1f}% | {n - st} |\n")
    A(f"\n若把「空心」框也算作不可用（它们框住了空腔而非符号本体，对 YOLO 是有害样本）："
      f"`r` 要动 {sum(1 for r in by_t['r'] if strong(r, meta) or hollow(r))}/1626 = "
      f"{pct(sum(1 for r in by_t['r'] if strong(r, meta) or hollow(r)), 1626):.1f}%、"
      f"`crystal_2pin` {sum(1 for r in by_t['crystal_2pin'] if strong(r, meta) or hollow(r))}/44 = "
      f"{pct(sum(1 for r in by_t['crystal_2pin'] if strong(r, meta) or hollow(r)), 44):.0f}%、"
      f"`testpoint` {sum(1 for r in by_t['testpoint'] if strong(r, meta) or hollow(r))}/43 = "
      f"{pct(sum(1 for r in by_t['testpoint'] if strong(r, meta) or hollow(r)), 43):.0f}%、"
      f"`bjt` {sum(1 for r in by_t['bjt'] if strong(r, meta) or hollow(r))}/51 = "
      f"{pct(sum(1 for r in by_t['bjt'] if strong(r, meta) or hollow(r)), 51):.0f}%。"
      f"**全库口径合计 {pct(tot_strong + n_ho, n_all):.1f}%**。\n\n")

    A("### (2) 哪几类的训练数据实际已废、必须重标？\n\n")
    A("**没有任何一类整体作废。** 依据：\n\n")
    A(f"- 最严重的「彻底空框」在本库**不存在**：外扩 25px 内严格 0 墨 = **{n_z25} 框**，≤0.5% 墨 = **{n_e25} 框**。"
      f"也就是说每个框旁边 25px 内一定画着东西。\n")
    A(f"- 真正「框内无物」只有 **{n_te} 框（{pct(n_te, n_all):.2f}%）**，且 ASCII 复核显示成因是**框偏移几个像素/只框住空白**，"
      f"不是「标到了别的元件」。\n")
    A(f"- 按类看，最糟的是 `bjt`（真错标 6/51 = 11.8%）、`crystal_2pin`（空心 5/44 = 11.4%）、"
      f"`testpoint`（空心 3/43 = 7.0%）、`r`（空心 98/1626 = 6.0%）、`amp_3pin`（1/10）。\n")
    A(f"- **`jlc` 并不比别的图源差。** 按强可疑率排序："
      + " > ".join(f"`{s}` {pct(sum(1 for r in by_s[s] if strong(r, meta)), len(by_s[s])):.1f}%"
                   for s in sorted(SRC_ORDER, key=lambda s: -pct(sum(1 for r in by_s[s] if strong(r, meta)), len(by_s[s]))))
      + "。（图源只由 png 文件名匹配；`unknown` 主要是官方 10GT，样本小、波动大。"
        "`Datasheet` 排第一是因为它贡献了 36/41 个越界框，属裁图问题而非标错。）\n")
    A("\n**必须动手的是三件事，按优先级：**\n\n")
    A(f"1. **类型体系重标**：`pin`(175) 与 `testpoint`(43) 共用 `TPn` 与 20x20 模板；`d`(201) 与 `led`(186) 共用 "
      f"`D`/`LED` 命名空间与 14x20 模板，{len(mix_cases)} 个 case 内同图并存 → 这两组必须先定标准再重标。\n")
    A(f"2. **按 case 清洗**：第 3 节最差 20 张（强可疑率 {rank[0][0] * 100:.0f}% ~ {rank[19][0] * 100:.0f}%），"
      f"含 `0135` 的 3 个同位 `net_input` 重复框、`0027` 的越界框。\n")
    A(f"3. **放大「空心」框**（清单在 `work\\_labelaudit.csv` 的 `hollow` 标签行）：`r` 98 框、`crystal_2pin` 5、"
      f"`testpoint` 3，改为外接符号轮廓即可，不必丢弃。\n")

    with open(MD, "w", encoding="utf-8") as f:
        f.write("".join(out))
    txt = "".join(out)
    cjk = sum(1 for ch in txt if "\u4e00" <= ch <= "\u9fff")
    print("wrote", MD, len(txt), "chars,", cjk, "CJK")
    print(f"boxes={n_all} strong={tot_strong}({pct(tot_strong, n_all):.2f}%) hollow={n_ho} true_empty={n_te} "
          f"oob={n_oob} dup={n_dup} stub={n_st} z25={n_z25} e25={n_e25} on_text={n_ot}")


if __name__ == "__main__":
    main()
