# -*- coding: utf-8 -*-
"""第二阶段端到端 CLI：stage1 产物 -> 文字识别 -> 角色分配 -> 官方格式 JSON。

    cd Y:\\workspace\\EDA\\eda-pcb
    python -m stage2.run --stage1 Y:\\workspace\\EDA\\work\\stage1_all14 ^
                         --out Y:\\workspace\\EDA\\work\\stage2_all --limit 10

产物：`<out>\\<case>.json`（官方 components/pins/nets 格式，左下角原点）
      `<out>\\_stage2_report.json`（每个 case 的统计与耗时）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .text import DEFAULT_CKPT, TextEngine, detect_texts, imread_unicode, read_stage1_texts
from .assign import assign, dist_to_rect
from .assemble import build_doc
from .expect import merge_runs, score_expect, should_grow
from .growbox import grow_reading

# 「不可分就默认是长的」时，长出来的新读法要过这道门才允许进候选池（用户 m06734：
# 「截止条件定的高一点」）。0.9 = 必须落在它自己前缀所期望的长度区间里。
GROW_MIN_SCORE = float(os.environ.get("STAGE2_GROW_MIN", "0.9"))
# 每颗元件最多为几个框做生长（控成本）
GROW_PER_COMP = int(os.environ.get("STAGE2_GROW_PER_COMP", "4"))
# ★★ 把 `stage1/pincol.py` 算好的引脚号窗口当文字框送进 OCR（默认开）。
# 只取 `out_win`（框外的**号**）：名字在框内，检测器本来就检得到，而号漏检最狠。
# 想连名字一起读就设 `STAGE2_PINCOL_WIN_KEYS=out_win,in_win`。
PINCOL_WIN_TEXTS = os.environ.get("STAGE2_PINCOL_WIN", "1").strip().lower() not in (
    "", "0", "false", "off", "none")
PINCOL_WIN_KEYS = tuple(k.strip() for k in
                        os.environ.get("STAGE2_PINCOL_WIN_KEYS", "out_win").split(",")
                        if k.strip())
# 这些是「参数」类元件，它们旁边没有长器件名，不参与生长
_VALUE_TYPES = {"r", "c", "l", "fuse", "crystal", "crystal_2pin",
                "crystal_3pin", "crystal_4pin"}


def _longest_read(reads) -> str:
    best = ""
    for _role, (t, _c) in (reads or {}).items():
        if t and len(t) > len(best):
            best = t
    return best


def recut_candidates(doc, reads, engine, bgr, max_new: int = 400):
    """**期望驱动的重切**（用户 m06470）：把同一行里相邻的**碎片框**拼成更长的框再读一遍。

    背景：`work\\reports\\错例集_导线与元件.md` 里 7645 条 `text_mismatch` 全是
    「长器件名被切碎」（GT `RP2350B` → 我们 `GP/2`）。`expect.merge_runs` 用
    「读出来很短 / 没小写 / 带分隔符」判碎片、用「拼起来更像个器件名」判该不该拼，
    这里把它接到管线上：拼出来的框补一次 OCR，再整体重新分配。
    """
    comps = [c for c in (doc.get("components") or []) if c.get("bbox_img")]
    if not comps:
        return []
    groups = {}
    for r in reads:
        b = r["bbox"]
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        j = min(range(len(comps)),
                key=lambda i: dist_to_rect(cx, cy, comps[i]["bbox_img"]))
        groups.setdefault(j, []).append(r)
    seen = {tuple(round(float(v), 3) for v in r["bbox"]) for r in reads}
    boxes_new = []
    for j, rs in groups.items():
        if len(rs) < 2:
            continue
        boxes = [list(r["bbox"]) for r in rs]
        texts = [_longest_read(r.get("reads")) for r in rs]
        hs = sorted(max(1.0, b[3] - b[1]) for b in boxes)
        for cand in merge_runs(boxes, texts, char_h=hs[len(hs) // 2]):
            key = tuple(round(float(v), 3) for v in cand["bbox"])
            if key in seen:
                continue
            seen.add(key)
            boxes_new.append(cand["bbox"])
            if len(boxes_new) >= max_new:
                break
    if not boxes_new:
        return []
    return engine.read(bgr, boxes_new)


def grow_candidates(doc, reads, engine, bgr, max_new: int = 600):
    """**沿阅读方向再框几个字**（用户 m06734）。

    与 `recut_candidates` 的区别很重要：那个是「拼**相邻的检测框**」，而真实情况是
    旁边**根本没有第二个框** —— 切分器只给出盖住名字中段的一个框（GT `RP2350B`
    读成 `GP/2`）。所以这里直接看像素：`growbox.grow_reading` 沿同一行往左右扩，
    停在「贯穿式竖线」（导线/符号边框）或空列预算用完的地方。

    两个「高」门槛：①只有 `expect.should_grow` 认为「这不像一个干净位号」的框才动
    ②长出来的新读法必须 `score_expect >= GROW_MIN_SCORE`（默认 0.9）才进候选池。
    """
    comps = [c for c in (doc.get("components") or []) if c.get("bbox_img")]
    if not comps:
        return []
    groups: dict = {}
    for r in reads:
        b = r["bbox"]
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        j = min(range(len(comps)),
                key=lambda i: dist_to_rect(cx, cy, comps[i]["bbox_img"]))
        groups.setdefault(j, []).append((dist_to_rect(cx, cy, comps[j]["bbox_img"]), r))
    seen = {tuple(round(float(v), 3) for v in r["bbox"]) for r in reads}
    boxes_new = []
    for j, rs in groups.items():
        if (comps[j].get("type") or "") in _VALUE_TYPES:
            continue                      # 参数类元件旁边没有长器件名
        rs.sort(key=lambda x: x[0])
        n = 0
        for _d, r in rs:
            if n >= GROW_PER_COMP:
                break
            if not should_grow(_longest_read(r.get("reads"))):
                continue
            g = grow_reading(bgr, r["bbox"])
            if not g:
                continue
            key = tuple(round(float(v), 3) for v in g)
            if key in seen:
                continue
            seen.add(key)
            boxes_new.append(g)
            n += 1
            if len(boxes_new) >= max_new:
                break
    if not boxes_new:
        return []
    out = []
    for r in engine.read(bgr, boxes_new):
        best = max((score_expect(t) for t, _c in (r.get("reads") or {}).values()),
                   default=0.0)
        if best >= GROW_MIN_SCORE:
            out.append(r)
    return out


def process(case: str, stage1_dir: Path, out_dir: Path, engine: TextEngine,
            roles=None, do_ocr: bool = True, min_conf: float = 0.0,
            texts: str = "both", recut: str = "off") -> dict:
    t0 = time.time()
    doc = json.loads((stage1_dir / ("%s.stage1.json" % case)).read_text(encoding="utf-8"))
    png = (doc.get("image") or {}).get("path")
    bgr = imread_unicode(png)

    # 文字框的来源：`stage1` = 复用 stage1 产物里的 `texts`（`cvparts.textbox` 出的，
    # 召回高、框偏小）；`detect` = 用 `cvparts.textseg` 当场重切（框紧，但少切 ~8%
    # 的框）；`union` = 以 stage1 框保底、能在 textseg 框里找到对应就把框收紧；
    # `both` = `union` 的结果再补回被换掉的那些 stage1 框（两个切法都留着）；
    # `auto` = 有 stage1 产物就复用，否则当场重切。
    #
    # **默认值是 `both`（2025-10-04 改）**。这个决定量过三次：
    #   旧管线（总分 ~6.7、`PIN_STAMP` 之前）：union 6.6592 < stage1 6.7307。
    #   中管线（总分 ~15.95）220 张：union **15.9525** | stage1 15.5567 |
    #     auto 同 stage1 | detect 15.4180。
    #   官方口径管线（2025-10-04）220 张四臂：
    #     both   **23.3815**（compF1 0.5144 tp 5158 fp 4470 fn 4451 netH 0.1940）
    #     union   22.9068（0.5077 / 5077 / 4551 / 4532 / 0.1862）
    #     stage1  22.7717（0.5005 / 5054 / 4574 / 4555 / 0.1889）
    #     auto    22.7717（同 stage1）
    #     detect  22.4571（0.4930 / 4923 / 4705 / 4686 / 0.1876）
    #   ⇒ union 一直是「只留 textseg 那一版」里最好的；`both`（两版都留）再 +0.475。
    #   别再来回翻。代价：0.74 s/张 → 0.87 s/张。
    #
    # `both` 的动机：`union` 是**替换**，同一个位置只留 textseg 那一版，可 textseg
    # 的切法在这一版 CRNN 上是**逐框随机**地更好或更差 —— `0001 c0052`（`STAGE2_DBG_CID`
    # 实测）位置上有 `[603,856,620,864]`（stage1 框，`pin_name` 读得 `TP1` 正确）
    # 和 `[602,855,619,863]`（textseg 框，只差 1px，五个角色全读成 `11`/`O`/`C1`）。
    # 一刀切就把好的那版丢了；两个都交给 `assign` 打分去选反而涨分。
    boxes, src = [], texts
    if texts in ("stage1", "auto"):
        boxes = read_stage1_texts(doc)
        src = "stage1"
    elif texts in ("union", "both", "multi"):
        from .text import merge_boxes, merge_boxes_both, merge_boxes_multi
        base = read_stage1_texts(doc)
        det = detect_texts(bgr)
        if texts == "both":
            boxes = merge_boxes_both(base, det)
        elif texts == "multi":
            boxes = merge_boxes_multi(base, det)
        else:
            boxes = merge_boxes(base, det)
        src = texts
    if not boxes:
        boxes = detect_texts(bgr)
        src = "detect"

    # ★★ **pincol 的引脚号窗口直接切、直接读**（`stage1/pincol.py`，默认开）。
    #
    # 引脚号是 6x8 px 的数字，`cvparts.textbox` 对它的召回很低（0143 的 U1：GT 64 个
    # 号只检出约 33 个），检不到的号就没有候选、`assign` 也就无从归位。而 pincol
    # 已经把每个号的窗口算好了 ⇒ 不必等检测器，按窗口切开收紧（定长 +3px 纵向外扩，
    # 曲线见 `work\reports\尺度归一_诊断.md`）再交给同一个 CRNN 读。
    if PINCOL_WIN_TEXTS:
        try:
            from stage1 import pincol as _pc
            _wb = _pc.win_boxes(bgr, doc.get("pincol") or {}, keys=PINCOL_WIN_KEYS)
        except Exception:
            _wb = []
        if _wb:
            boxes = list(boxes) + [b for b, _c, _k in _wb]
            src += "+pincolwin"

    t1 = time.time()
    if do_ocr:
        reads = engine.read(bgr, boxes) if roles is None else engine.read(bgr, boxes, roles)
    else:
        reads = [{"bbox": b, "reads": {}} for b in boxes]
    t2 = time.time()

    for r in reads:
        for k, (txt, conf) in list(r["reads"].items()):
            if conf < min_conf:
                r["reads"][k] = ("", conf)

    a = assign(doc.get("components") or [], doc.get("pins") or {}, reads,
               pincol=doc.get("pincol"))
    if recut in ("expect", "grow", "both") and do_ocr:
        extra = []
        if recut in ("expect", "both"):
            extra += recut_candidates(doc, reads, engine, bgr)
        if recut in ("grow", "both"):
            extra += grow_candidates(doc, reads, engine, bgr)
        if extra:
            for r in extra:
                for k, (txt, conf) in list(r["reads"].items()):
                    if conf < min_conf:
                        r["reads"][k] = ("", conf)
            reads = list(reads) + list(extra)
            a = assign(doc.get("components") or [], doc.get("pins") or {}, reads,
                       pincol=doc.get("pincol"))
    # ★ 引脚名救援（`stage2/pinname.py`，`STAGE2_PINNAME_RESCUE` 默认**开**，2026-10-16）：
    #   「号读对了、名字没读到」的 box 脚以前会被 `assemble.py` 的 NEED_NAME 闸门
    #   整条丢掉（官方 9 张：缺号 51 里有 23 根 = 45%）。这里从脚点往体内切带、
    #   剔掉体描边、按行分组只取离脚最近的一行，读一次；过闸门才写回名字。
    #   必须在 `build_doc` **之前**（名字要进交付文档），且只动 `a` 不删任何东西。
    #   默认值依据：官方 10 张 --no-vlm 60.5854 → 61.0000（+0.4146/张，3 涨 0 跌）、
    #   派生 val 19 张 17.0047 → 17.0322（+0.0274，4 涨 0 跌）；置 0 时逐位不变。
    try:
        from . import pinname as _pinname
        _pn = _pinname.rescue(bgr, doc, a, engine)
    except Exception as _e:                      # noqa: BLE001
        _pn = {"on": True, "error": "%s: %s" % (type(_e).__name__, _e)}
    # ★ 竖直脚旋转重读（`stage2/pinrot.py`，`STAGE2_PINROT` 默认关）：上下边的脚
    #   文字是旋转 90° 印的（官方 10GT：case 6 有 6 根、case 10 有 3 根、case 7 有 2 根），
    #   轴对齐裁片只能读成 'L'/'33' 这类碎片 —— case 6 引脚通道整张 0 分。这里把整图
    #   转 90°（顺/逆各一轮）、在旋转帧里取体内/体外条带重读号与名，过闸门才覆盖。
    try:
        from . import pinrot as _pinrot
        _pr = _pinrot.fix(bgr, doc, a, engine)
    except Exception as _e:                      # noqa: BLE001
        _pr = {"on": True, "error": "%s: %s" % (type(_e).__name__, _e)}
    pred = build_doc(doc, a, case)
    pin_index = pred.pop("__pin_index", None)
    t3 = time.time()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("%s.json" % case)).write_text(
        json.dumps(pred, ensure_ascii=False), encoding="utf-8")
    # ★ 旁路文件：引脚索引 → 交付文档里的 `pin_<N>` 键。融合层要改引脚号/名就
    # 必须知道这个映射，而它是 `build_doc` 内部「结构期望重排」的产物，外部
    # 反推不出来（实测三种反推法覆盖率 0.560 / 0.462 / 0.318）。
    # ⚠ 必须 `pop` 后再写 `<case>.json`：交付文档的字节不能有任何变化。
    if pin_index is not None:
        (out_dir / ("%s.pinindex.json" % case)).write_text(
            json.dumps(pin_index, ensure_ascii=False), encoding="utf-8")
    # 原始分配结果单独存一份：阶段三要拿它里的 `desig_slots` 做 LLM 仲裁，
    # 再把选定值回填后重新 `build_doc`（改名会牵动 `pins` 的键和 net 的 hyperGraph，
    # 所以只能重装配，不能事后改字符串）。
    (out_dir / "_assigned").mkdir(exist_ok=True)
    # ★ 「不删除，只削弱」账本（m40034 第四项）：随 `_assigned` 旁路落盘，**不进交付物**
    #   （`pred` 已在上面 `build_doc` 时定稿，这里只动旁路 dict `a`）。
    #   内容 = `cvparts.hypo.LEDGER`（改写/截断/丢弃事件 + 降权分）+ `structmatch`
    #   的降权候选与保留池（`HIST_RESERVE` = 重扫队列的输入）。
    try:
        from cvparts.hypo import LEDGER as _LG
        from . import structmatch as _sm
        # ⚠ 不能写 `from . import assign as _as`：`stage2/__init__` 里有
        #   `from .assign import assign`，包属性 `stage2.assign` 已被**函数**遮蔽，
        #   那样拿到的是函数（实测 `AttributeError: 'function' object has no
        #   attribute 'LAST_CONFORM'`）。直接按名字取模块级列表对象即可。
        from .assign import LAST_CONFORM as _lc
        from .assign import LAST_CONFORM_EXTRA as _le
        _nd = _LG.dump()
        _nd["demotions_pin"] = len(_sm.HIST_DEMOTED)
        _nd["reserve_pin"] = len(_sm.HIST_RESERVE)
        _nd["demoted"] = _sm.HIST_DEMOTED[-60:]
        _nd["reserve"] = _sm.HIST_RESERVE[-60:]
        _nd["residual"] = list(_sm.LAST_RESIDUAL)[-40:]
        # ★ 内涵「遗漏检测」（m48732，`STAGE2_CONFORM_AUDIT` 默认开）：本来按内涵
        #   规则（位号前缀 ⇄ 元件类型互证、削开头胶连）能拿到键、却因为判型/取证
        #   不对而最终没挂上键的元件。官方 10 张上它恰好点出 4 颗被丢掉的 LED
        #   （case 6 ×2、case 7、case 10）。**只写旁路报告**，不进交付物。
        _nd["conform_audit"] = list(_lc)
        # ★ 内涵「多余检测」（同一开关）：我方发出去的、内涵上必然判错的元件
        #   （`d`/`led` 的 Name 槽在官方 10GT 里是型号，我方 `name = key`）。
        _nd["conform_extra"] = list(_le)
        # ★ 大一统交叉校验层（用户 m50175，`CV_UNIFIED` 默认 0=关）：把已有的各种
        #   候选池/降权痕迹**只读**投影成统一 `Registry`（5 种并行候选表示 → `Claim`），
        #   跑内置校验（几何判据走 `GridIndex`/`AlongIndex`），产出 `Verdict` 台账。
        #   **只往旁路 dict `a` 里加 `unified` 键**：`pred` 早在 `:263 build_doc`
        #   定稿，交付 `<case>.json` 的字节不受影响；自带 try，坏了不牵连 `no_delete`。
        try:
            if os.environ.get("CV_UNIFIED", "0").strip().lower() not in (
                    "", "0", "false", "off", "none"):
                from cvparts import unified as _uf
                _reg = _uf.Registry("case%s" % case)
                _reg.add_many(_uf.project_stage1(doc))
                _reg.add_many(_uf.project_assigned(a))
                _reg.register(*_uf.default_checks())
                _reg.run()
                _nd["unified"] = _reg.to_json()
        except Exception as _ue:                  # noqa: BLE001
            _nd["unified"] = {"error": "%s: %s" % (type(_ue).__name__, _ue)}
        a["no_delete"] = _nd
    except Exception as _e:
        a["no_delete"] = {"error": "%s: %s" % (type(_e).__name__, _e)}
    (out_dir / "_assigned" / ("%s.json" % case)).write_text(
        json.dumps(a, ensure_ascii=False), encoding="utf-8")

    n_des = sum(1 for v in a["comps"].values() if v.get("designator"))
    n_val = sum(1 for v in a["comps"].values()
                if v.get("value") or v.get("partname"))
    n_pn = sum(1 for v in a["comps"].values()
               for p in (v.get("pins") or {}).values() if p.get("name"))
    n_pnum = sum(1 for v in a["comps"].values()
                 for p in (v.get("pins") or {}).values() if p.get("number"))
    n_pins = sum(len(v.get("pins") or {}) for v in a["comps"].values())
    return {
        "case": case, "n_components": len(doc.get("components") or []),
        "n_texts": len(boxes), "texts_src": src,
        "n_designator": n_des, "n_value_name": n_val,
        "n_pin_name": n_pn, "n_pin_number": n_pnum, "n_pins": n_pins,
        "n_net_labels": len(a["net_labels"]), "n_nets": len(pred["nets"]),
        "labels_merged": sum(1 for v in pred["nets"].values() if v.get("label")),
        "pinname_rescue": _pn,
        "pinrot": _pr,
        "t_read": round(t2 - t1, 2), "t_assign": round(t3 - t2, 2),
        "t_total": round(t3 - t0, 2),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="第二阶段：文字识别 + 组装")
    ap.add_argument("--stage1", required=True, help="stage1 产物目录")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                    help="CRNN 权重（默认取 stage2.text.DEFAULT_CKPT）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 case")
    ap.add_argument("--cases", default="", help="逗号分隔的 case id")
    ap.add_argument("--no-ocr", action="store_true", help="只做定位不做识别（对照臂）")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--texts", default="both", choices=("stage1", "detect", "union", "both", "multi", "auto"),
                    help="文字框来源（默认 both）：stage1=复用 stage1 产物（召回高、框偏小）；"
                         "detect=用 cvparts.textseg 当场重切（框紧）；union=以 stage1 框保底、"
                         "能用 textseg 框收紧就收紧；both=union 再补回被换掉的 stage1 框；"
                         "multi=both 再补上每个检出框的「外扩版」（**实测是负的**：23.0447 vs both 23.5009，留着只作对照）"
                         "（实测最好，+0.475）；auto=有 stage1 产物就复用")
    ap.add_argument("--recut", default="off", choices=("off", "expect", "grow", "both"),
                    help="expect=拼相邻碎片框再读一遍；grow=沿阅读方向把邻近笔画框进来再读；"
                         "both=两者都做（默认 off）")
    args = ap.parse_args(argv)

    s1 = Path(args.stage1)
    out = Path(args.out)
    files = sorted(s1.glob("*.stage1.json"))
    cases = [f.name[: -len(".stage1.json")] for f in files]
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c in want]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("没有可跑的 case（--stage1 指向的目录里没有 *.stage1.json）")
        return 1

    engine = TextEngine(ckpt=args.ckpt, device=args.device)
    print("[stage2] %d 个 case  识别器: %s" % (len(cases), engine.describe()[11:]))

    rows, failed, t0 = [], [], time.time()
    for i, c in enumerate(cases, 1):
        try:
            r = process(c, s1, out, engine, do_ocr=not args.no_ocr,
                        min_conf=args.min_conf, texts=args.texts, recut=args.recut)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print("  [%d/%d] %s 失败: %r" % (i, len(cases), c, exc))
            failed.append({"case": str(c), "error": "%s: %s" % (type(exc).__name__, exc)})
            continue
        rows.append(r)
        print("  [%d/%d] %s  元件 %d  文字 %d  位号 %d  值/名 %d  引脚名 %d  "
              "引脚号 %d  标签 %d  网络 %d  %.1fs"
              % (i, len(cases), c, r["n_components"], r["n_texts"], r["n_designator"],
                 r["n_value_name"], r["n_pin_name"], r["n_pin_number"],
                 r["n_net_labels"], r["n_nets"], r["t_total"]))

    wall = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    (out / "_stage2_report.json").write_text(
        json.dumps({"cases": rows, "wall_s": round(wall, 1),
                    "n_cases": len(rows), "failed": failed},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")
    if rows:
        n = len(rows)
        # 2026-07 审计修复：这两个「赋值率」原来的分母是 `n_pin_name + n_pin_number`
        # —— 那是**成功数**之和，不是引脚总数 ⇒ 名字和号都读全时两个率都显示 0.5，
        # 任何调参都看不出真实的 0.9x 或 0.1x。改用 process() 新返回的 `n_pins`。
        _npins = max(1, sum(r.get("n_pins", 0) for r in rows))
        print("\n合计 %d 张，墙钟 %.1fs（%.2fs/张）；位号赋值率 %.3f，"
              "引脚名赋值率 %.3f，引脚号赋值率 %.3f"
              % (n, wall, wall / n,
                 sum(r["n_designator"] for r in rows) / max(1, sum(r["n_components"] for r in rows)),
                 sum(r["n_pin_name"] for r in rows) / _npins,
                 sum(r["n_pin_number"] for r in rows) / _npins))
    if failed:
        print("\n⚠️ 有 %d/%d 张 case 抛异常、**没有产物**（列出前 20）：%s"
              % (len(failed), len(cases), failed[:20]))
    print("产物 -> %s" % out)
    # 2026-07 审计修复：原来无论多少 case 失败都返回 0，「跑完了」≠「220 张都在」，
    # 缺文件在提交里只能表现为掉分。现在有失败就以非 0 退出，调用方（run.py -workers）
    # 才能把这一层报上去。
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
