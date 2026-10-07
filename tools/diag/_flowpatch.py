"""Patch work/flow/workflow.html's graphdata block with this round's measured findings.

幂等：重复运行不会重复追加（靠标记串判断）。
用法:  python work/tools/diag/_flowpatch.py
"""
import json
import re
from pathlib import Path

HTML = Path(r"Y:\workspace\EDA\work\flow\workflow.html")
PAT = re.compile(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', re.S)

MARK = "【尺度归一这一轮】"

TOP_ADD = """

【尺度归一这一轮】用户 m20796：『做可靠的尺度归一，然后再更新一下之前的 workflow.html 给我看一下』。
结论写在 `work\\reports\\尺度归一_诊断.md`，一句话：**「墨迹归一」不可靠（三版全败），可靠的是「定长外扩」**。
- 线上与训练的差距不在模型：同一份 796 条真实 val，直调 `Recognizer.read_many`（真值框）**exact 0.677**（designator 0.885），
  走线上 `TextEngine.read`（stage1 紧框）只有 **0.197**（designator 0.239）。**CRNN 本身是好的，坏在预处理。**
- 真凶是 `stage2\\text.py` 的 `pad_box` **垂直**外扩（与 `crop_gray` 自己的 `PAD_Y_FRAC=0.25` 叠成约 `0.70h+3`，
  裁剪高变 `2.4h+6`，而 `crop_gray` 按**裁剪高**缩放到 `IMG_H=32` ⇒ 字形只占 32px 里的 1/3，踩出训练分布）。
  在真值框上：垂直 0 ⇒ 0.588，全 0 ⇒ 0.661。
- **但不能直接把外扩关掉**：stage1 紧框系统性比标注框小（中位 IoU 0.36），关掉端到端会崩（48.60 → 24.54）。
- 已按端到端网格把默认改成 **水平 `+2` 定长 / 垂直 `max(3, 0.20h)`**：10GT(9) **+0.75**、19 张 **+0.48**。
  ⚠️ 增益小、单张会倒退、**还没过 220 张**。"""

EV_TEXT_ADD = """

**【尺度归一这一轮】上面「别再试」那句要分区看 —— 当时没有把水平/垂直拆开扫。**
- `stage2\\text.py:47-50` 记着定长外扩曲线（10 张 CVJsonStd / 1103 个配对）：`+0 0.1915 · +2 0.6345 · **+3 0.6544** · +4 0.6105 · +5 0.5076 · +6 0.2554 · +8 0.0527`（真值框原样 0.8430）⇒ **最优是定长 +3，不是按比例**；老默认 `max(2.0, 0.30h)` 在 `h≈19` 时给 5.7px，**已越过峰**。
- 端到端（`fast.py --no-vlm`，`STAGE1_WIRE_COLOUR=off`）扫出来的最优是**水平 `+2` 定长 / 垂直 `max(3, 0.20h)`**：10GT(9) **49.3516** vs 老默认 48.5981；19 张 **29.4111** vs 28.9338。
  垂直不能低于 3（`3/0 · 2/0` ⇒ 44.12，掉 5 分）；水平比例项在现有框尺寸下不起作用（`2/0.10` 与 `2/0` 同分）。
- ⚠️ 不是全面变好：19 张里 `case 1` 掉 4.56（被 `case 7` 的 +7.28 抹平）；两套样本都只有 9/19 张。
- 结论：**问题不在「要不要外扩」，在「水平与垂直要的量不一样」** —— 水平丢的是首/末字符（`IS`→`S`、`GND`→`D`），垂直多扩会把**邻行的字**拉进来。"""

EV_OCR_ADD = """

**【尺度归一这一轮】上面「CRNN 到顶了」要修正：不是模型到顶，是线上裁剪把它废掉了。**
- 同一份 796 条真实 val（`work\\ocr_real\\manifest.jsonl`）：直调 `Recognizer.read_many` **exact 0.677**（designator 0.885 / value 0.681 / pin_number 0.726 / pin_name 0.586 / net_label 0.509）；
  走线上 `TextEngine.read` **0.197**（designator 0.239 / value 0.117 / pin_number 0.332 / pin_name 0.084 / net_label 0.098）。
  **⇒ 线上路径把识别器砍掉 68%。**
- 真凶：`stage2\\text.py:80-94 pad_box` 的**垂直**外扩（默认 `max(2.0, 0.30h)` 两侧各一次）+ `crop_gray` 自己的 `PAD_Y_FRAC=0.25`。
  在真值框上只关垂直 ⇒ 0.588；全关 ⇒ 0.661（基线 0.197）。
- **墨迹尺度归一：做了三版，全败，默认关**（`textrec\\config.py` 的 `INK_*` + `textrec\\dataset.py` 的 `_ink_band`，`TEXTREC_INK_YNORM=1` 才开）。逐图诊断 `work\\tools\\diag\\_inkdiag.py`，对照图 `work\\preview\\_inkvis*.png`：
  | 判墨迹方法 | exact（真值框，基线 0.6815） |
  |---|---:|
  | ① `dark.any(axis=1)` 行投影 | 0.146（比全 0 的 0.197 还差） |
  | ② 行内最长暗游程 ≤ 0.5×框宽 | designator 0.84 → **0.36** |
  | ③ 形态学抠长直线 + 中心行扩张 | 最好 **0.378** |
  ①：导线是长直线，经过的每一行都「有墨」⇒ 整块裁剪被当成字。②：门限把**短标签自己的笔画**（`1` 的底衬线）判成线，把字裁掉一截，designator 全是 1–2 字符最惨。
  ③：抠线是对的（`_inkvis.png` 里 case 0059 绿线精确套住字），但输出图行距太密（`0112` 字高约 7px、行距约 10px），探测带够到邻行，同一 case 量到 `g` 从 7 到 17 都有而框高只有 14 ⇒ 取景忽大忽小。
  **更根本**：要从墨迹高反推标注框高，实测 `g/H` p5..p95 = **0.47..1.36**（中位 0.60）——**不是常数**，反推不成立。
- ⇒ **靠谱的路是「定长外扩」这条确定性的**（已改为默认），或者**连同模型一起重训**（把外扩抖动加进 `crop_gray` 的 `augment`，现在只抖 ±30%；训练/推理同用一个函数）。"""

EV_OCR_PARAMS = [
    ["解码", "按 w//8 截断"],
    ["CHUNK", "256"],
    ["外扩 水平", "+2 定长（PAD_MIN=2 / PAD_FRAC=0）"],
    ["外扩 垂直", "max(3, 0.20h)（PAD_MIN_Y=3 / PAD_FRAC_Y=0.20）"],
]


def main():
    s = HTML.read_text(encoding="utf-8")
    m = PAT.search(s)
    if not m:
        raise SystemExit("graphdata block not found")
    g = json.loads(m.group(2))
    g["name"] = "目标架构 · 证据层 → 约束求解 → 迭代重扫（对照：现管线 27.98 带 VLM / 25.99 stage2-only，均 220 张）"

    if MARK not in (g.get("note") or ""):
        g["note"] = (g.get("note") or "") + TOP_ADD

    touched = []
    for n in g["nodes"]:
        if n["id"] == "ev_text" and MARK not in (n.get("note") or ""):
            n["note"] = (n.get("note") or "") + EV_TEXT_ADD
            touched.append("ev_text")
        if n["id"] == "ev_ocr":
            if MARK not in (n.get("note") or ""):
                n["note"] = (n.get("note") or "") + EV_OCR_ADD
                touched.append("ev_ocr")
            n["params"] = EV_OCR_PARAMS

    # 沿用原文件的缩进风格，别把整块 JSON 重排
    orig = m.group(2)
    if "\n" not in orig:
        indent = None
        sep = (",", ":")
    else:
        nxt = orig[orig.index("\n") + 1:]
        indent = max(1, len(nxt) - len(nxt.lstrip(" ")))
        sep = None
    blob = json.dumps(g, ensure_ascii=False, indent=indent, separators=sep)
    print("indent=%r orig_len=%d new_len=%d" % (indent, len(orig), len(blob)))
    HTML.write_text(s[: m.start(2)] + blob + s[m.end(2):], encoding="utf-8")
    print("patched nodes:", touched)
    print("name:", g["name"])
    print("nodes=%d links=%d" % (len(g["nodes"]), len(g["links"])))


if __name__ == "__main__":
    main()
