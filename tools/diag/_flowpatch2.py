"""第二批 workflow.html 更新（用户要求「完全更新」）：
  ① 全文档数字一致性  ② 逐节点核对参数栏  ③ 给「要新写」的节点标进度  ④ 把尺度归一做成独立节点

幂等：靠 MARK 判断，重复运行不会重复插入/追加。
用法:  python work/tools/diag/_flowpatch2.py
"""
import json
import re
from pathlib import Path

HTML = Path(r"Y:\workspace\EDA\work\flow\workflow.html")
PAT = re.compile(r'(<script type="application/json" id="graphdata">)(.*?)(</script>)', re.S)

MARK = "【第二批】"
STATUS_MARK = "**进度："

LEGEND = """

【第二批】用户 2026-10：『你没有完全更新 html』⇒ 做四件事：**数字一致性 / 参数栏核对 / 目标节点标进度 / 尺度归一做独立节点**。
**进度图例**（每个待建节点开头都有一行 `**进度：…**`）：
`✅ 已落地`＝代码在跑｜`⚠ 未落地`＝文件不存在｜`⛔ 已证否`＝量过，这条路判负。
**现状盘点**：21 个节点里，`ev_*` 七个 + `assemble`/`export`/`metrics`/`input`/`batch` 是**已有的代码**；
`know`/`pool`/`solve`/`plausible`/`conflict`/`llm_judge`/`rescan`/`iterate` **八个全部没写**（`priors.py`/`claims.py`/`solve.py`/`plausible.py`/`llm_judge.py`/`rescan.py` 一个都不存在，已 `Get-ChildItem` 核对）；
`fixstr` 前提被测量否掉，**已证否**。
**水位口径**（凡数字必标是哪套 case 集；「外扩」= 裁剪外扩默认，见 `ev_crop`）：
| 集合 | 外扩改默认前 | 改后 |
|---|---:|---:|
| 220 张 stage2-only | — | **25.9861**（`work\\_submit_out220r36`） |
| 220 张 端到端带 VLM | **27.9843**（`work\\_submit_out220r32\\runs\\stage3_vlm`） | 未重跑 |
| 10GT（9 张）无 VLM | 48.5981 | **49.3516** |
| 19 张 无 VLM | 28.9338 | **29.4111** |
⇒ **两个 220 张的数字都是「改外扩默认之前」的**，别和新默认的 9/19 张结果混着看。"""

EV_CROP_NOTE = """**为什么单独拎出一个节点**：这一层原来藏在 `ev_ocr` 的预处理里，没人看得见，而它恰好是**线上 vs 离线差距的全部来源**。

同一份 796 条真实 val（`work\\ocr_real\\manifest.jsonl`），只换这一层：

| 走法 | exact | designator |
|---|---:|---:|
| 直调 `Recognizer.read_many`（喂真值框） | **0.677** | 0.885 |
| 走线上 `TextEngine.read`（喂 stage1 紧框） | **0.197** | 0.239 |

⇒ **线上这一层把识别器砍掉 68%。**

**真凶是「垂直」外扩**：`stage2\\text.py:80-94 pad_box`（默认 `max(2.0, 0.30h)`，上下各一次）叠上 `crop_gray` 自己的 `PAD_Y_FRAC=0.25`，垂直合计约 `0.70h+3`，裁剪高度变成 `2.4h+6`；而 `crop_gray` 按**裁剪高度**缩放到 `IMG_H=32` ⇒ 字形只占 32px 里的约 1/3，踩出训练分布。
在真值框上只关垂直 ⇒ 0.588；全关 ⇒ 0.661（基线 0.197）。

**⚠ 但不能直接把外扩关掉**：stage1 紧框系统性比标注框小（中位 IoU 0.36，`reports\\textseg_诊断.md`），关掉端到端会崩（10GT(9) 48.60 → 24.54）。

**定长曲线的峰**（`work/_boxgeom.py`，10 张 CVJsonStd / 1103 个配对）：`+0 0.1915 · +2 0.6345 · **+3 0.6544** · +4 0.6105 · +5 0.5076 · +6 0.2554 · +8 0.0527`（真值框原样 0.8430）
⇒ **最优是定长 +3，不是按比例**；老默认在 `h≈19` 时给 5.7px，**早就越过峰了**。

**端到端网格（`fast.py --no-vlm`，`STAGE1_WIRE_COLOUR=off`）** ⇒ 已把默认改成 **水平 `+2` 定长 / 垂直 `max(3, 0.20h)`**：
10GT(9) **49.3516** vs 48.5981｜19 张 **29.4111** vs 28.9338。
垂直不能低于 3（`3/0 · 2/0` ⇒ 44.12，掉 5 分）；水平比例项在现有框尺寸下不起作用（`2/0.10` 与 `2/0` 同分）。
⚠️ 增益小（相对 +1.6%）、**单张会倒退**（19 张里 case 1 掉 4.56，被 case 7 的 +7.28 抹平）、**还没过 220 张**。它不是精确最优点，只是 0.15–0.25 这段平台里量到最高的那个。

**墨迹尺度归一：做了三版，全败，默认关**（`TEXTREC_INK_YNORM`，代码留在 `textrec\\config.py` 的 `INK_*` + `textrec\\dataset.py` 的 `_ink_band`，留给将来重训）：
| 判墨迹方法 | exact（真值框，基线 0.6815） |
|---|---:|
| ① `dark.any(axis=1)` 行投影 | 0.146（比全 0 的 0.197 还差） |
| ② 行内最长暗游程 ≤ 0.5×框宽 | designator 0.84 → **0.36** |
| ③ 形态学抠长直线 + 中心行扩张 | 最好 **0.378** |
①：导线是长直线，经过的每一行都「有墨」⇒ 整块裁剪被当成字。②：门限把**短标签自己的笔画**（`1` 的底衬线）判成线，把字裁掉一截。③：抠线是对的（`_inkvis.png` 里 case 0059 绿线精确套住字），但输出图行距太密（`0112` 字高约 7px、行距约 10px），探测带够到邻行，同一 case 量到 `g` 从 7 到 17 都有而框高只有 14。
**更根本**：要从墨迹高反推标注框高，实测 `g/H` p5..p95 = **0.47..1.36**（中位 0.60）——**不是常数**，反推不成立。

⇒ **两条可靠的路**：①定长外扩（已落）；②**连同模型一起重训**——把外扩抖动加进 `crop_gray` 的 `augment`（现在只抖 ±30%），训练/推理同用这一个函数。
细节与逐图清单：`work\\reports\\尺度归一_诊断.md`；对照图 `work\\preview\\_padab.png`、`_inkvis*.png`。"""

EV_CROP_PARAMS = [
    ["水平", "+2 定长（PAD_MIN=2 / PAD_FRAC=0）"],
    ["垂直", "max(3, 0.20h)（PAD_MIN_Y=3 / PAD_FRAC_Y=0.20）"],
    ["上层边距", "crop_gray 再扩 PAD_X_FRAC=0.12 / PAD_Y_FRAC=0.25"],
    ["缩放到", "IMG_H=32、保宽高比（超过 MAX_W=384 才压）"],
    ["墨迹归一", "三版全败，INK_YNORM 默认关"],
    ["⚠ 未复核", "220 张还没跑过新默认"],
]

STATUS = {
    "know": "**进度：⚠ 未落地**（`priors.py` 不存在）。素材散落在四处：`stage1/labels.py`（网络标签候选）、"
            "`stage2/assign.py`（`TYPE_PREFIX` / `is_netname` / `_digitize` / `fit_pinnum`）、"
            "`stage2/structmatch.py`（沿边单调 / `_ladder` / `rebind`）、`stage2/expect.py`（由前缀反推「名字该多长」）。",
    "pool": "**进度：⚠ 未落地**（`claims.py` 不存在）。这是整张图里**最前置、也最缺**的一块。",
    "solve": "**进度：⚠ 未落地**（`solve.py` 不存在）。现在实际在解的是 `stage2/assign.py` 的局部贪心 "
             "+ `stage2/structmatch.py` 的单调约束与 `rebind` + `stage2/assemble.py` 的 `merge_shared_pins`（传递闭包）。",
    "plausible": "**进度：⚠ 未落地**（`plausible.py` 不存在）。",
    "fixstr": "**进度：⛔ 已证否**（`fixstr.py` 不存在，且**前提被测量否掉**，见下）。",
    "conflict": "**进度：⚠ 未落地**（计划写进 `solve.py`，目前不存在）。",
    "llm_judge": "**进度：⚠ 未落地**（`llm_judge.py` 不存在）。旧的全量仲裁 `stage3/arbitrate.py` **已停用**（默认 off）。",
    "rescan": "**进度：⚠ 未落地**（`rescan.py` 不存在）。",
    "iterate": "**进度：⚠ 未落地**（计划写进 `solve.py`，目前不存在）。",
}

METRICS_ADD = """

**【第二批】数字刷新（原「10GT 49.29 / 10_GTcase 29.83 / 200_train_cases 26.34 / 总分 27.33」是旧快照，作废）。**
现在的口径必须连**哪套 case 集**一起说，且要注明裁剪外扩改默认**之前还是之后**：
| 集合 | 外扩改默认前 | 改后 | 备注 |
|---|---:|---:|---|
| 220 张 stage2-only | — | **25.9861** | `work\\_submit_out220r36` |
| 220 张 端到端带 VLM | **27.9843** | 未重跑 | `work\\_submit_out220r32\\runs\\stage3_vlm` |
| 10GT（9 张）无 VLM | 48.5981 | **49.3516** | 带 VLM 是 59.35 |
| 19 张 无 VLM | 28.9338 | **29.4111** | 含 9 张 10GT + 10 张真实扫描 |
⚠️ **两个 220 张的数字都是改外扩之前跑的**；新默认只在 9 + 19 两套小样本上验过，**换生产默认前应复核 220 张**。"""


def main():
    s = HTML.read_text(encoding="utf-8")
    m = PAT.search(s)
    g = json.loads(m.group(2))
    if MARK in json.dumps(g, ensure_ascii=False):
        print("已经打过第二批补丁，退出（幂等）")
        return

    # ---- ① 顶部口径 ----
    g["name"] = ("目标架构 · 证据层 → 约束求解 → 迭代重扫"
                 "（对照：现管线 27.98 带 VLM / 25.99 stage2-only，均 220 张、均为改裁剪外扩默认之前）")
    g["note"] = (g.get("note") or "") + LEGEND

    # ---- ④ 插入「裁剪尺度归一」节点，并把 ev 组从 ev_ocr 起整体下移 ----
    if not any(n["id"] == "ev_crop" for n in g["nodes"]):
        for n in g["nodes"]:
            if n["id"] in ("ev_ocr", "ev_vlm", "ev_wire", "ev_pin", "ev_net"):
                n["y"] = int(n["y"]) + 200
        idx = next(i for i, n in enumerate(g["nodes"]) if n["id"] == "ev_ocr")
        g["nodes"].insert(idx, {
            "id": "ev_crop", "group": "ev", "kind": "exec",
            "title": "③ 裁剪尺度归一",
            "sub": "pad_box + crop_gray：把紧框外扩到与训练同尺度再送 CRNN",
            "x": 400, "y": 440, "w": 280,
            "file": "stage2/text.py（pad_box）+ textrec/dataset.py（crop_gray）",
            "ins": [{"id": "img", "label": "原图"}, {"id": "boxes", "label": "文字框"}],
            "outs": [{"id": "boxes", "label": "裁剪框"}],
            "params": EV_CROP_PARAMS,
            "note": EV_CROP_NOTE,
        })
        # 重新编号
        for nid, t in (("ev_ocr", "④ CRNN 识别"), ("ev_vlm", "⑤ LLM / VLM 读取"),
                       ("ev_wire", "⑥ 导线与结点"), ("ev_pin", "⑦ 引脚点"), ("ev_net", "⑧ 初始网络")):
            for n in g["nodes"]:
                if n["id"] == nid:
                    n["title"] = t
        # 改线：ev_text.boxes → ev_ocr.boxes  拆成两跳
        g["links"] = [l for l in g["links"]
                      if not (l.get("from") == "ev_text.boxes" and l.get("to") == "ev_ocr.boxes")]
        g["links"] += [
            {"from": "input.img", "to": "ev_crop.img", "kind": "data"},
            {"from": "ev_text.boxes", "to": "ev_crop.boxes", "kind": "data"},
            {"from": "ev_crop.boxes", "to": "ev_ocr.boxes", "kind": "data"},
        ]

    # ---- ② 参数栏核对（已逐条对代码验证）----
    for n in g["nodes"]:
        if n["id"] == "ev_vlm":
            ps = [p for p in n["params"] if not str(p[0]).startswith("引脚名裁片")]
            ps.append(["引脚名裁片 inner", "160.0（旧 78.0；19 张 A/B 总分 +5.27）"])
            n["params"] = ps
        if n["id"] == "ev_text":
            n["params"] = [["来源", "stage1 文字框 ∪ CV 现检（texts=both）"],
                           ["外扩归属", "不在本节点 —— 见 ③ 裁剪尺度归一"]]

    # ---- ③ 目标节点标进度 ----
    for n in g["nodes"]:
        if n["id"] in STATUS and STATUS_MARK not in (n.get("note") or ""):
            n["note"] = STATUS[n["id"]] + "\n\n" + (n.get("note") or "")
        if n["id"] == "fixstr":
            n["file"] = "（已证否）fixstr.py"

    # ---- ① metrics 数字刷新 ----
    for n in g["nodes"]:
        if n["id"] == "metrics" and MARK not in (n.get("note") or ""):
            n["note"] = (n.get("note") or "") + METRICS_ADD

    blob = json.dumps(g, ensure_ascii=False, indent=1)
    HTML.write_text(s[: m.start(2)] + blob + s[m.end(2):], encoding="utf-8")
    print("nodes=%d links=%d" % (len(g["nodes"]), len(g["links"])))


if __name__ == "__main__":
    main()
