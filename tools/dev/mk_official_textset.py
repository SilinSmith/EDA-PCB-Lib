# -*- coding: utf-8 -*-
"""从**官方公开数据**（10GT + 200_train_cases）造一份「域内」文字识别训练集。

## 为什么需要它（证据）

`work\\ocr_mix\\runs\\best.pt` 的训练语料是：合成 382K 条 + `Dataset300` 的 6.6K 条
真实裁片（`textrec\\data\\manifest.jsonl` 的 `split` 列），**里面没有一条来自官方
公开数据**。而官方那 10 张 GT 图用的是另一种渲染器/字体，实测：

* 在 `Dataset300` 的真实裁片上，模型对 `nF`/`uF` 是 **25/25 + 25/25 全对**；
* 官方 case 1 的 5 个电容值（栅格上清清楚楚是 `10nF`）被读成 `10uF`，而且
  `u` 的对数概率是 −0.92、**`n` 连前三名都进不去**（`work\\tools\\diag\\_valbeam.py`）
  —— 不是「拿不准」，是没学过这个字形。

代价是直接的：官方 9 张里 component 的 7 对 `text_mismatch` **全是 r/c 的 `value`**，
全修上界 +1.02 分（b897）。而隐藏测试集与这 10 张同源，所以域内适配的收益不止公开这 9 张。

## 做法（标签从 GT 来，框从我们自己的检测来）

对每个 case：
1. GT（`<case>.json`）给出 `components[设计号] = {Name, type, value, bbox}` 与
   `pins[设计号][pin_K] = {pinname, point}`；
2. `work\\_full220\\runs\\stage1\\<case>.stage1.json` 给出我们检测到的元件框
   （`components[i].bbox_img`，**图像坐标**，与 GT 的 `bbox` 同一坐标系）与引脚端点
   （`pins[cid] = [[x, y], ...]`）；
3. `work\\_f220_A\\_assigned\\<case>.json` 给出每个元件/引脚上**实际被裁去识别的那块
   文字的框**（`comps[cid].designator/value/partname.box`、`comps[cid].pins[j].name/number.box`）。

于是「框」来自我们自己的检测，「标签」来自 GT：

| 字段 | 标签 | role |
|---|---|---|
| `designator` | GT 的设计号（键形如 `C152`/`U1`，`∅NNN`/`net_N` 跳过） | `designator` |
| `value` | GT 的 `value` | `value` |
| `partname` | GT 的 `Name`（等于键、或 `∅`/`net_` 开头则跳过） | `value` |
| 引脚名 | GT 的 `pinname` | `pin_name` |
| 引脚号 | GT 的 `pin_K` 里的 `K`（`pin_A5` -> `A5`） | `pin_number` |

元件用 IoU 配对，引脚用端点距离配对（都要**唯一**，配不上就丢），宁可少要也不喂脏标签。

用法：
    python -X utf8 work\\tools\\dev\\mk_official_textset.py --out work\\official_text --limit 5 --sheet
    python -X utf8 work\\tools\\dev\\mk_official_textset.py --out work\\official_text
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EDA = Path(r"Y:\workspace\EDA\eda-pcb")
WORK = Path(r"Y:\workspace\EDA\work")
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(WORK / "tools" / "eval"))

import stage2_eval as SE       # noqa: E402

DESIG = re.compile(r"^[A-Za-z]{1,4}\d{1,4}$")
PINNUM = re.compile(r"^[A-Za-z]{0,2}\d{1,3}$")


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-6, aa + bb - inter)


def cdist(p, q) -> float:
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def match_comps(gt_comps, s1_comps, min_iou=0.35, max_cd=10.0):
    """GT 元件 <-> stage1 元件：IoU 优先，退而求中心距离。返回 {cid: gt_key}。

    **坐标系**：GT 的 `bbox` 用的是官方（JSON）约定，`y` 从图底往上数；stage1 同时存了
    `bbox_img`（图像坐标，左上原点）与 `bbox_json`（官方约定，= `[x1, H-y2, x2, H-y1]`）。
    这里必须拿 **`bbox_json`** 去配 GT —— 实测 case 1 的 `c0001.bbox_json`
    `[185.24,1033.97,201.24,1053.97]` 对上 GT `C152` 的 `[187,1033,203,1053]`，
    而拿 `bbox_img`（`y≈148`）去配就只剩 3/22 配得上（踩过一次，记在这里）。
    """
    out, used = {}, set()
    cand = []
    for cid, c in s1_comps.items():
        cb = c.get("bbox_json") or c.get("bbox_img")
        if not cb:
            continue
        for gk, g in gt_comps.items():
            gb = g.get("bbox")
            if not gb:
                continue
            v = iou(cb, gb)
            cd = cdist(((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2),
                       ((gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2))
            if v >= min_iou or cd <= max_cd:
                cand.append((v, -cd, cid, gk))
    cand.sort(reverse=True)
    for v, ncd, cid, gk in cand:
        if cid in out or gk in used:
            continue
        out[cid] = gk
        used.add(gk)
    return out


def match_pins(ours, gtpins, img_h):
    """ours: [[x,y],...]（stage1 端点，**图像坐标**）; gtpins: {pin_key: point} -> {our_idx: pin_key}。

    GT 的 `point` 是官方约定（y 从图底数），先把我们的端点翻过去再比：
    实测 case 1 `c0001` 的第 0 个端点 `[194.5,138.0]` 翻成 `y=1064.0`，正对 GT `C152.pin_2`
    的 `(195.0, 1063.0)`（差 1 px）。"""
    cand = []
    for i, p in enumerate(ours):
        py = img_h - p[1]
        for pk, gp in gtpins.items():
            pt = gp.get("point") or {}
            if "x" not in pt:
                continue
            d = cdist((p[0], py), (pt["x"], pt["y"]))
            if d <= 12.0:
                cand.append((d, i, pk))
    cand.sort()
    out, ui, ug = {}, set(), set()
    for d, i, pk in cand:
        if i in ui or pk in ug:
            continue
        out[i] = pk
        ui.add(i)
        ug.add(pk)
    return out


def edit_dist(a: str, b: str, cap: int = 4) -> int:
    """小写化的编辑距离；超过 cap 直接返回 cap+1（省时间）。"""
    a, b = (a or "").lower(), (b or "").lower()
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return min(prev[-1], cap + 1)


def junk_key(k: str) -> bool:
    """`∅NNN`（电源/地/网络伪元件）与 `net_N`：它们的 pin number 是随机生成的
    （见 `docs\\QA_2025.txt`），Name 也不是印在图上的文字，一律不要。"""
    k = str(k)
    return k.startswith("∅") or k.startswith("net_") or not k


def tol_edit(t: str) -> int:
    """多长的串允许多大的编辑距离。短串（`1`/`C1`）几乎没有鉴别力，给 1；长串给 2。"""
    n = len(t)
    if n <= 3:
        return 1
    return 2


def ocr_by(e: dict) -> bool:
    """这个槽位是不是「OCR 在那块框上读出来的」。

    踩过的坑：`_assigned` 里 `_by="solve:vlm:enrich"` 的槽位（器件名）**box 是继承来的**，
    跟 text 不对应 —— 实测有一条 `partname.text='RP2350B'` 而按它的 box 裁出来是 `GPIO20`。
    这类槽位一律不要（`vlm` 字样即弃；没有 `_by` 的是 assign 的引脚槽，保留）。
    """
    return "vlm" not in str((e or {}).get("_by") or "").lower()


def pad_box(box, pad_min=2.0, pad_frac=0.0, pad_min_y=3.0, pad_frac_y=0.20,
            pad_max=200.0):
    """与 `stage2/text.py:164 pad_box` 同一算法（默认值也对齐 `stage2/text.py:69-80`）。

    语料必须按**管线真正喂给识别器的裁片**来做：stage2 运行时先外扩再裁，
    上一次微调用的是紧框裁片，于是同一个框在管线里读出 `C15S2`（见
    `eda-pcb/docs/DIAGNOSTICS.md` 的「域内微调权重不能直接换进 stage2」）。
    """
    h = max(1.0, float(box[3]) - float(box[1]))
    px = min(pad_max, max(pad_min, pad_frac * h))
    py = min(pad_max, max(pad_min_y, pad_frac_y * h))
    return (float(box[0]) - max(0.0, px), float(box[1]) - max(0.0, py),
            float(box[2]) + max(0.0, px), float(box[3]) + max(0.0, py))


def tight_ink_box(img_rgb, box, sat=60.0, dark=160.0):
    """`stage2/text.py:133 tight_ink_box` 的等价实现（PIL RGB -> numpy）。

    管线对 `pin_name` / `pin_number` 用的是「先切掉低饱和暗墨之外的彩色描边、再外扩」
    的裁片（`stage2/text.py:404-415`，默认 `TIGHT_ROLES=pin_number,pin_name`）。
    语料要跟管线完全同口径，就得照做一遍。返回 `None` = 框里没有这类像素（管线此时
    退回原框）。
    """
    import numpy as np
    a = np.asarray(img_rgb, dtype=np.int16)
    H, W = a.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    sub = a[y1:y2, x1:x2]
    mx = sub.max(2)
    mn = sub.min(2)
    g = sub.mean(2)
    m = ((mx - mn) < sat) & (g < dark)
    on = np.nonzero(m.sum(0) > 0)[0]
    if len(on) == 0:
        return None
    oy = np.nonzero(m[:, int(on[0]):int(on[-1]) + 1].sum(1) > 0)[0]
    if len(oy) == 0:
        return None
    return (float(x1 + on[0] - 1), float(y1 + oy[0] - 1),
            float(x1 + on[-1] + 2), float(y1 + oy[-1] + 2))


def verify_rows(rows, ckpt, device="cpu"):
    """重读一遍：要求「这次读出来的」与当初 `_assigned` 里的读数差 ≤ 1 个编辑距离。

    这是唯一能排除「box 与 text 根本不是一回事」的检查（另一个来源是上面 `ocr_by`）。
    按图分组批读，CPU 上 2 万条几分钟。
    注意：校验一律用 `bbox_tight`（当初读那个框时的口径），训练用的 `bbox`
    可能是外扩过的。
    """
    try:
        from PIL import Image
        from textrec.infer import Recognizer
    except Exception as e:                                   # pragma: no cover
        print("[warn] 没有 textrec，跳过重读校验: %s" % e)
        return rows, 0
    rec = Recognizer.load(ckpt, device)
    by_img = defaultdict(list)
    for i, r in enumerate(rows):
        by_img[r["image"]].append(i)
    keep, drop = [], 0
    for p, idxs in by_img.items():
        try:
            gray = Image.open(p).convert("L")
        except Exception:
            drop += len(idxs)
            continue
        items = [(rows[i].get("bbox_tight") or rows[i]["bbox"], rows[i]["role"])
                 for i in idxs]
        try:
            got = rec.read_many(gray, items)
        except Exception:
            drop += len(idxs)
            continue
        for i, (txt, conf) in zip(idxs, got):
            r = rows[i]
            if edit_dist(txt, str(r.get("was") or ""), 3) <= 1:
                r["reread"] = txt
                r["rconf"] = round(float(conf), 4)
                keep.append(r)
            else:
                drop += 1
    return keep, drop


def pin_number(pin_key: str):
    m = re.match(r"^pin_(.+)$", pin_key)
    if not m:
        return None
    k = m.group(1)
    return k if PINNUM.match(k) else None


def build_index():
    idx, digits = SE.build_index(SE.DEFAULT_DATA_ROOT)
    return idx, digits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WORK / "official_text"))
    ap.add_argument("--cases-file", default=str(WORK / "_cases220.txt"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stage1", default=str(WORK / "_full220" / "runs" / "stage1"))
    ap.add_argument("--assigned", default=str(WORK / "_f220_A" / "_assigned"))
    ap.add_argument("--no-pins", action="store_true")
    ap.add_argument("--ckpt", default=str(WORK / "ocr_mix" / "runs" / "best.pt"))
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过「重读一遍确认 box 与 text 对得上」的校验（默认会做）")
    ap.add_argument("--maxedit", type=int, default=2,
                    help="只在「我们原来的读数」与 GT 足够像（编辑距离 ≤ 此值）时才收这一条"
                         "——框来自检测、标签来自 GT，文本门槛是用来挡错配的")
    ap.add_argument("--sheet", action="store_true", help="抽样拼一张对照图并打印标签")
    ap.add_argument("--no-pad", action="store_true",
                    help="不按 stage2 运行时口径外扩裁片（默认外扩，见 pad_box）")
    ap.add_argument("--no-pipe-crops", action="store_true",
                    help="对 pin_name/pin_number 不做 tight_ink_box（默认做，与管线同口径）")
    ap.add_argument("--tight-roles", default="pin_number,pin_name",
                    help="与 stage2/text.py:128 TIGHT_ROLES 对齐")
    ap.add_argument("--pad-min", type=float, default=2.0)
    ap.add_argument("--pad-frac", type=float, default=0.0)
    ap.add_argument("--pad-min-y", type=float, default=3.0)
    ap.add_argument("--pad-frac-y", type=float, default=0.20)
    ap.add_argument("--pinnum-frac-y", type=float, default=0.0,
                    help="引脚号角色的垂直比例外扩（stage2 `PINNUM_PAD_FRAC_Y`=0）")
    ap.add_argument("--sheet-only-diff", action="store_true",
                    help="sheet 只放「我们读错的那些」（GT != 原读数）——用来核标签真假")
    ap.add_argument("--sheet-n", type=int, default=40)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s1dir, asdir = Path(a.stage1), Path(a.assigned)

    names = [s.strip() for s in Path(a.cases_file).read_text(
        encoding="utf-8").strip().split(",") if s.strip()]
    if a.limit:
        names = names[:a.limit]

    idx, digits = build_index()
    img_cache = {}

    def _rgb(path):
        if path not in img_cache:
            from PIL import Image as _I
            img_cache[path] = _I.open(path).convert("RGB")
        return img_cache[path]

    tight_roles = tuple(r.strip() for r in a.tight_roles.split(",") if r.strip())

    def row_of(case_name, role, text, box, was, by, img):
        bx = [float(v) for v in box]
        base = bx
        fy = a.pinnum_frac_y if role == "pin_number" else a.pad_frac_y
        # 管线对 `pin_name`/`pin_number` 先切彩色描边再外扩（stage2/text.py:404-415）
        if (not a.no_pipe_crops) and role in tight_roles and img:
            try:
                tb = tight_ink_box(_rgb(img), bx)
            except Exception:
                tb = None
            if tb is not None:
                base = list(tb)
        pb = list(base) if a.no_pad else list(
            pad_box(base, a.pad_min, a.pad_frac, a.pad_min_y, fy))
        return {"image": img, "case": case_name, "role": role, "text": str(text),
                "bbox": pb, "bbox_tight": bx, "src": "official",
                "was": was, "_by": by}

    rows, stats, skip = [], Counter(), Counter()
    for name in names:
        s1p = s1dir / ("%s.stage1.json" % name)
        asp = asdir / ("%s.json" % name)
        if not s1p.is_file() or not asp.is_file():
            skip["缺产物"] += 1
            continue
        case, _how = SE.resolve(name, idx, digits)
        if case is None:
            skip["缺 GT"] += 1
            continue
        gt = json.loads(Path(case.json_path).read_text(encoding="utf-8"))
        s1 = json.loads(s1p.read_text(encoding="utf-8"))
        asg = json.loads(asp.read_text(encoding="utf-8"))
        img = str(case.png_path) if getattr(case, "png_path", None) else None
        if not img:
            skip["缺图"] += 1
            continue
        gt_comps = gt.get("components") or {}
        gt_pins = gt.get("pins") or {}
        try:
            from PIL import Image as _I
            img_h = _I.open(img).size[1]
        except Exception:
            skip["读不到图尺寸"] += 1
            continue
        s1c = {c["id"]: c for c in (s1.get("components") or []) if c.get("id")}
        cmap = match_comps(gt_comps, s1c)
        for cid, gk in cmap.items():
            if junk_key(gk):
                continue
            g = gt_comps[gk] or {}
            ac = (asg.get("comps") or {}).get(cid) or {}
            # --- 位号 / 值 / 器件名：同一个元件上的三个文字槽
            specs = [("designator", gk if DESIG.match(gk) else None, "designator")]
            specs.append(("value", g.get("value"), "value"))
            gn = g.get("Name")
            if gn and gn != gk and not junk_key(gn) and not DESIG.match(str(gn)):
                specs.append(("partname", gn, "value"))
            for slot, gt_text, role in specs:
                e = ac.get(slot) or {}
                if not gt_text or not e.get("box") or not ocr_by(e):
                    if e.get("box") and not ocr_by(e):
                        skip["槽位来自 VLM（box 不可信）"] += 1
                    continue
                # 位号槽的错配检测：这块框读出来的字符串如果**正好是同一个 case 里另一个
                # 元件的设计号**，那它显然属于那个元件（实测 `GT='R15' was='R13'` 的裁片
                # 上印的就是 `R13`）——这种行会把「别的元件的文字」标成这个元件的标签。
                if slot == "designator" and str(e.get("text")) in gt_comps \
                        and str(e.get("text")) != gk:
                    skip["位号框属于另一个元件（错配）"] += 1
                    continue
                thr = tol_edit(str(gt_text)) if a.maxedit < 0 else a.maxedit
                if edit_dist(e.get("text"), gt_text) > thr:
                    skip["原读数与 GT 差太远"] += 1
                    continue
                rows.append(row_of(name, role, gt_text, e["box"],
                                   e.get("text"), e.get("_by"), img))
                stats[role if slot != "partname" else "partname"] += 1
            # --- 引脚
            if a.no_pins:
                continue
            ours = (s1.get("pins") or {}).get(cid) or []
            gps = gt_pins.get(gk) or {}
            pmap = match_pins(ours, gps, img_h)
            apins = ac.get("pins") or {}
            for j, pk in pmap.items():
                ae = apins.get(str(j)) or {}
                gp = gps.get(pk) or {}
                nm, num = gp.get("pinname"), pin_number(pk)
                en = ae.get("name") or {}
                if nm and en.get("box") and ocr_by(en) and not junk_key(nm) \
                        and not junk_key(str(nm)) and len(str(nm)) >= 3 \
                        and edit_dist(en.get("text"), nm) <= tol_edit(str(nm)):
                    rows.append(row_of(name, "pin_name", nm, en["box"],
                                       en.get("text"), en.get("_by"), img))
                    stats["pin_name"] += 1
                en = ae.get("number") or {}
                # 引脚号很短（`1`/`2`），编辑距离没有鉴别力：只收**完全相同**的读数，
                # 而且这个数必须在本 case 的引脚号里只出现一次（否则无从判重）。
                if num and en.get("box") and ocr_by(en) and str(en.get("text")) == str(num) \
                        and len(str(num)) >= 2:
                    rows.append(row_of(name, "pin_number", num, en["box"],
                                       en.get("text"), en.get("_by"), img))
                    stats["pin_number"] += 1

    if not a.no_verify and rows:
        n0 = len(rows)
        rows, drop = verify_rows(rows, a.ckpt)
        print("重读校验: 保留 %d / 丢 %d（共 %d）" % (len(rows), drop, n0))

    man = out / "manifest.jsonl"
    for r in rows:
        r["dist"] = edit_dist(r.get("was") or "", r["text"], 9)
    with man.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("cases=%d  rows=%d  -> %s" % (len(names), len(rows), man))
    print("按 role:", dict(stats))
    if skip:
        print("跳过:", dict(skip))
    # 「模型原来读成什么」的分布：这就是域内适配要修的东西
    diff = Counter()
    for r in rows:
        if r.get("was") is not None and r["was"] != r["text"]:
            diff[r["role"]] += 1
    print("模型原读数与 GT 不同的行数:", dict(diff))
    print("编辑距离分布:", dict(sorted(Counter(r["dist"] for r in rows).items())))

    if a.sheet:
        pool = rows
        if a.sheet_only_diff:
            pool = [r for r in rows if r.get("was") is not None and r["was"] != r["text"]]
        make_sheet(pool, out, a.sheet_n)
    return 0


def make_sheet(rows, out: Path, n: int):
    try:
        from PIL import Image, ImageDraw
    except Exception as e:                                  # pragma: no cover
        print("[warn] 没有 PIL，跳过 sheet: %s" % e)
        return
    import random
    random.seed(1)
    by_role = defaultdict(list)
    for r in rows:
        by_role[r["role"]].append(r)
    pick = []
    for role, rs in sorted(by_role.items()):
        pick += random.sample(rs, min(len(rs), max(4, n // max(1, len(by_role)))))
    pick = pick[:n]
    crops, cache = [], {}
    for r in pick:
        p = r["image"]
        if p not in cache:
            cache[p] = Image.open(p).convert("L")
        im = cache[p]
        b = [int(round(v)) for v in r["bbox"]]
        c = im.crop((b[0], b[1], b[2], b[3]))
        sc = max(1, min(6, 240 // max(1, c.height)))
        c = c.resize((max(1, c.width * sc), max(1, c.height * sc)), Image.LANCZOS)
        crops.append((r, c))
    W = max(c.width for _, c in crops) + 430
    H = sum(c.height + 8 for _, c in crops) + 8
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    y = 4
    for r, c in crops:
        sheet.paste(c, (430, y))
        d.rectangle([430, y, 430 + c.width, y + c.height], outline=0)
        tag = "%s | GT=%r was=%r" % (r["role"], r["text"], r.get("was"))
        d.text((4, y + max(0, c.height // 2 - 6)), tag, fill=0)
        y += c.height + 8
    sp = out / "sheet.png"
    sheet.save(sp)
    print("sheet -> %s (%d 条)" % (sp, len(crops)))
    for r, _ in crops[:n]:
        print("   %-11s GT=%-14r was=%r" % (r["role"], r["text"], r.get("was")))


if __name__ == "__main__":
    raise SystemExit(main())
