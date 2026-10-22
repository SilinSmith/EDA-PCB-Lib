# -*- coding: utf-8 -*-
"""GT 页面 -> CRNN 训练清单（弱配对）。

为什么需要它
-----------
`textrec/manifest.py` 有两个来源：合成语料（自带逐条 bbox+text，干净但域差大）
和 Dataset300 的 `*_CVJsonStd.json`（自带 `texts[].location`，也是逐条 bbox）。
用户明天要给的「新正确标注数据集」两边都不是：它是**页面级 GT**
（`components[<位号>] = {bbox, value, ...}` + `pins[<位号>][pin_N] = {point}`），
只有「字符串」和「元件的框/引脚的点」，**没有每个字符串自己的框**。

所以这里做的事是：先生成候选文字框（`stage2.text.detect_texts`，就是线上那个
检测器），再把每个 GT 字符串**唯一地**配到一个候选框上；配不上就丢。

配对规则（宁缺毋滥，绝不编标签）
------------------------------
每个 GT 字符串有自己的**锚点**：
* 位号 / 值 —— 元件 `bbox` 的中心（GT 里没有更细的位置了）；
* 引脚号 —— 引脚 `point`（这是引脚的连接点，不是数字的框）。

一对 (GT 字符串 g, 候选框 b) 只有在**同时**满足下面四条时才被接受：
1. `距离(g.锚点, b.中心) <= --radius`（默认 40 px）；
2. `b` 是 g 的**最近**候选框；
3. 次近候选框比它远**超过 4 px**（否则并列 ⇒ 丢）；
4. `g` 是 `b` 的**唯一**最近 GT 字符串（否则并列 ⇒ 丢）。

四条一起保证「一对一」：一个框最多一个标签，一个字符串最多一个框。
第 4 条会连带丢掉**同元件位号与值**的配对——两者的锚点是同一个 bbox 中心，
到同一批框的距离完全相等，谁对谁无法判定。这是页面级 GT 的固有损失，
也正是「请标框级标签」的理由（见 `docs/RETRAIN.md` §1）。

产物格式与 `manifest.build_real` **逐字段一致**：
    {"image","case","src","split","role","role_id","text","bbox"}
`role` 用 `textrec/config.py::ROLES` 的词表（`designator`/`value`/`pin_number`），
`role_id` 就是它在 ROLES 里的下标——注意**不能用** `desig`/`pin_num` 这类简写，
它们不在表里，会静默落到兜底 role 上，把角色嵌入训歪。

用法
----
    cd Y:\\workspace\\EDA
    python -X utf8 eda-pcb\\tools\\dev\\build_manifest_from_gt.py ^
        --gt "eda-pcb\\yolo_part\\Dataset(waitforpreprocess)\\10GT" --stats
    python -X utf8 eda-pcb\\tools\\dev\\build_manifest_from_gt.py ^
        --gt <新目录> --out work\\_man_new.jsonl
    # 厂商给的是框级 JSONL 时（首选，见 RETRAIN.md）
    python -X utf8 eda-pcb\\tools\\dev\\build_manifest_from_gt.py ^
        --boxes <vendor.jsonl> --gt <图片根目录> --out work\\_man_v.jsonl

只读 GT 与图片，只写 `--out`；纯 CPU（不 import torch）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EDA = Path(__file__).resolve().parents[2]       # eda-pcb/
if str(EDA) not in sys.path:
    sys.path.insert(0, str(EDA))

from stage2 import text as T                     # noqa: E402  检测器（imread_unicode / detect_texts）
from textrec import config as C                  # noqa: E402
from textrec.manifest import _ok, _ok_text, write  # noqa: E402  复用同一套判定与落盘

# 位号形状。比任务书里的 `^[A-Z]{1,3}\d+$` 宽一点：允许 1-3 个字母、大小写不限、
# 前面至多两个非字母数字前缀。实测（`10GT/1/1.json` 严格式命中 14/22，池 `0001` 46/91）：
#   * 唯一被放宽规则救回来的是池子里的**小写键** `u2`（KiCad 原样导出）；
#   * `∅202` 这类电源符号键严格式与放宽式**都不匹配**——它不是位号，是内部网络名；
#     它们的 `value`（`GND`/`+5V`/`3.3V`）照常按 value 角色取用。
DESIG_RE = re.compile(r"^[^A-Za-z0-9]{0,2}[A-Za-z]{1,3}\d+$")
TIE_TOL = 4.0                 # 次近框比最近框远多少才算「不并列」
VAL_EVERY_DEFAULT = 5         # 按 case 哈希每 N 个抽一个进 val（10 案 ⇒ 约 2 案）
DEFAULT_OUT = "work/_manifest_gt.jsonl"

# GT 侧的角色名 -> 训练词表（config.ROLES）。`other`/`note` 走词表里的兜底槽。
ROLE_CANON = {
    "desig": "designator", "designator": "designator", "designatorname": "designator",
    "value": "value", "param": "value", "params": "value",
    "pin_num": "pin_number", "pin_number": "pin_number", "pinnumber": "pin_number",
    "pin_name": "pin_name", "pinname": "pin_name",
    "net_label": "net_label", "netlabel": "net_label",
    "other": "note", "note": "note",
}


def canon_role(name: str) -> str:
    key = str(name or "").strip().lower().replace("_", "")
    return ROLE_CANON.get(key, "note")


# --------------------------------------------------------------------------
# 发现 case
# --------------------------------------------------------------------------
def find_cases(root: Path) -> List[Path]:
    """`--gt` 下面每个子目录是一个 case；也允许直接把某个 case 目录传进来。"""
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir()] if root.is_dir() else []
    dirs = [p for p in dirs if pick_image(p) is not None]
    if dirs:
        return dirs
    return [root] if pick_image(root) is not None else []


def pick_image(case_dir: Path) -> "Path | None":
    """case 目录里的图片。**不要求**文件名等于目录名——池子里是 `0001/0001-KiCad.png`。
    优先级：`<目录名>.<ext>` -> 唯一一个以目录名开头的 -> 目录里唯一一张图。"""
    exts = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")
    imgs = [p for p in sorted(case_dir.iterdir()) if p.is_file() and p.suffix in exts]
    if not imgs:
        return None
    for p in imgs:
        if p.stem == case_dir.name:
            return p
    pref = [p for p in imgs if p.stem.startswith(case_dir.name)]
    if len(pref) == 1:
        return pref[0]
    return imgs[0] if len(imgs) == 1 else None


def pick_label(case_dir: Path) -> "Path | None":
    """标签文件：先找 `*target_named.json`，再找 `<case>.json`（`*.json` 排除在最后）。"""
    named = sorted(case_dir.glob("*target_named.json"))
    if named:
        return named[0]
    guess = case_dir / (case_dir.name + ".json")
    if guess.is_file():
        return guess
    js = sorted(p for p in case_dir.glob("*.json"))
    return js[0] if len(js) == 1 else None


# --------------------------------------------------------------------------
# 从 GT 里取字符串 + 锚点
# --------------------------------------------------------------------------
def _centre(bbox) -> Tuple[float, float]:
    return (0.5 * (float(bbox[0]) + float(bbox[2])),
            0.5 * (float(bbox[1]) + float(bbox[3])))


def gt_items(doc: dict) -> List[dict]:
    """-> [{"role","text","anchor","rect"}...]，只取位号 / 值 / 引脚号三类。

    `rect` 是这条 GT 字符串所属元件的框（引脚号用该脚所在元件的框）；配对时的距离
    用它算「框到文字框的中心距」，而不是「元件中心到文字框」——大型 IC 的中心离
    自己的位号可能有 200 px，用中心做锚点会把绝大多数正确配对判成「半径内无框」
    （`work/_diag_zoom_5.png` 的复核即此）。
    """
    out: List[dict] = []
    comps = doc.get("components") or {}
    for key, comp in comps.items():
        if not isinstance(comp, dict):
            continue
        bbox = comp.get("bbox")
        if not _ok(bbox):
            continue
        anchor = _centre(bbox)
        desig = None
        for cand in (comp.get("desig"), key, comp.get("Name"), comp.get("name")):
            if cand and DESIG_RE.match(str(cand).strip()):
                desig = str(cand).strip()
                break
        if _ok_text(desig):
            out.append({"role": "designator", "text": desig, "anchor": anchor,
                        "metric": "rect", "rect": tuple(float(v) for v in bbox)})
        val = comp.get("value")
        if val is not None and _ok_text(str(val).strip()):
            out.append({"role": "value", "text": str(val).strip(), "anchor": anchor,
                        "metric": "rect", "rect": tuple(float(v) for v in bbox)})

    pins = doc.get("pins") or {}
    for ckey, plist in pins.items():
        if not isinstance(plist, dict):
            continue
        cbox = None
        comp = comps.get(ckey)
        if isinstance(comp, dict) and _ok(comp.get("bbox")):
            cbox = tuple(float(v) for v in comp["bbox"])
        for pname, pin in plist.items():
            if not isinstance(pin, dict):
                continue
            pt = pin.get("point") or {}
            try:
                anchor = (float(pt["x"]), float(pt["y"]))
            except (KeyError, TypeError, ValueError):
                continue
            num = str(pname)
            if num.startswith("pin_"):
                num = num[len("pin_"):]
            num = num.strip()
            if _ok_text(num):
                out.append({"role": "pin_number", "text": num, "anchor": anchor,
                            "metric": "point", "rect": cbox})
    return out


def _rect_dist(rect, pt) -> float:
    """点到矩形的欧氏距离（点在框内为 0）。`rect is None` 时退化为点距。"""
    if not rect:
        return 0.0
    dx = max(float(rect[0]) - pt[0], 0.0, pt[0] - float(rect[2]))
    dy = max(float(rect[1]) - pt[1], 0.0, pt[1] - float(rect[3]))
    return math.hypot(dx, dy)



# --------------------------------------------------------------------------
# 弱配对
# --------------------------------------------------------------------------
def pair(items: Sequence[dict], boxes: Sequence[Sequence[float]],
         radius: float = 40.0, tol: float = TIE_TOL) -> Tuple[List[Tuple[int, int, float]], Counter]:
    """见模块头四条规则。返回 ([(item_idx, box_idx, dist)], drop 计数)。"""
    stat: Counter = Counter()
    if not boxes:
        stat["drop_no_box"] = len(items)
        return [], stat
    centres = [(0.5 * (float(b[0]) + float(b[2])), 0.5 * (float(b[1]) + float(b[3])))
               for b in boxes]
    # 距离口径：位号/值 = 文字框中心到**元件框矩形**的距离（字形贴在框外 20~40 px）；
    # 引脚号 = 文字框中心到**引脚点**的距离（号就印在脚点旁边）。
    dist = []
    for it in items:
        if it.get("metric") == "point":
            ax, ay = it["anchor"]
            dist.append([math.hypot(ax - c[0], ay - c[1]) for c in centres])
        else:
            dist.append([_rect_dist(it.get("rect"), c) for c in centres])

    # 先算「每个框的唯一最近 GT」；并列则记为 None。
    box_arg: List["int | None"] = []
    for j in range(len(centres)):
        col = [row[j] for row in dist]
        m = min(col)
        win = [i for i, v in enumerate(col) if v <= m + 1e-9]
        box_arg.append(win[0] if len(win) == 1 else None)

    out: List[Tuple[int, int, float]] = []
    for i, _it in enumerate(items):
        row = dist[i]
        order = sorted(range(len(row)), key=lambda j: row[j])
        j, d = order[0], row[order[0]]
        if d > radius:
            stat["drop_no_box"] += 1
            continue
        if len(order) > 1 and row[order[1]] - d <= tol:
            stat["drop_tie_box"] += 1
            continue
        if box_arg[j] != i:
            stat["drop_tie_gt"] += 1
            continue
        out.append((i, j, d))
    return out, stat


# --------------------------------------------------------------------------
# 划分
# --------------------------------------------------------------------------
def _hash(case: str) -> int:
    return int(hashlib.md5(case.encode("utf-8")).hexdigest()[:8], 16)


def case_split(case: str, val_every: int) -> str:
    """按 case 哈希抽 val——与 `manifest._synth_split` 同精神（按 case，不按裁剪），
    只是基数是哈希而不是 case 序号，这样新数据集的目录名不要求是数字。"""
    return "val" if _hash(case) % max(1, val_every) == 0 else "train"


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def build_from_gt(roots: Sequence[Path], radius: float, val_every: int, split_mode: str,
                  cases_filter: Sequence[str] = (), limit_cases: int = 0,
                  quiet: bool = False) -> List[dict]:
    recs: List[dict] = []
    n_case = n_skip = 0
    gt_total: Counter = Counter()
    drops: Counter = Counter()
    splits: Dict[str, str] = {}
    case_list: List[Tuple[str, Path, Path, Path]] = []

    for root in roots:
        for cdir in find_cases(root):
            name = cdir.name
            if cases_filter and name not in cases_filter:
                continue
            img = pick_image(cdir)
            lab = pick_label(cdir)
            if img is None or lab is None:
                print("  ! 跳过 %s：%s" % (cdir, "缺图片" if img is None else "缺 GT json"))
                n_skip += 1
                continue
            case_list.append((name, cdir, img, lab))
            if limit_cases and len(case_list) >= limit_cases:
                break

    for name, cdir, img, lab in case_list:
        try:
            doc = json.loads(lab.read_text(encoding="utf-8-sig"))
        except Exception as exc:                                   # noqa: BLE001
            print("  ! 跳过 %s：GT 读不动（%s）" % (name, exc))
            n_skip += 1
            continue
        if not isinstance(doc, dict) or not (doc.get("components") or doc.get("pins")):
            print("  ! 跳过 %s：GT 里没有 components/pins（%s）" % (name, lab.name))
            n_skip += 1
            continue
        try:
            bgr = T.imread_unicode(img)
            boxes = T.detect_texts(bgr)
        except Exception as exc:                                   # noqa: BLE001
            print("  ! 跳过 %s：读图/检测失败（%s）" % (name, exc))
            n_skip += 1
            continue

        items = gt_items(doc)
        for it in items:
            gt_total[it["role"]] += 1
        # 分角色配对：位号、值、脚号的候选框互不竞争（同页 60+ 个脚号会把位号的框抢走）。
        # 一个文字框最终仍只保留一条标签，取距离最近的那条。
        pairs = []
        _gidx = {}
        for _i, _it in enumerate(items):
            _gidx.setdefault(_it["role"], []).append(_i)
        for _role, _idxs in _gidx.items():
            _sub = [items[_i] for _i in _idxs]
            _subp, _substat = pair(_sub, boxes, radius)
            drops.update(_substat)
            for _i, _j, _d in _subp:
                pairs.append((_idxs[_i], _j, _d))
        _best = {}
        for _i, _j, _d in pairs:
            if _j not in _best or _d < _best[_j][2]:
                _best[_j] = (_i, _j, _d)
        pairs = sorted(_best.values(), key=lambda t: t[1])
        n_case += 1
        split = "real" if split_mode == "real" else case_split(name, val_every)
        splits[name] = split
        for i, j, _d in pairs:
            it = items[i]
            text = it["text"]
            if not _ok_text(text) or not _ok(boxes[j]):
                continue
            recs.append({
                "image": str(img.resolve()),
                "case": name,
                "src": "real",
                "split": split,
                "role": it["role"],
                "role_id": C.ROLE_ID.get(it["role"], C.N_ROLES - 1),
                "text": text,
                "bbox": [round(float(v), 2) for v in boxes[j]],
            })
        if not quiet:
            _dropped = len(items) - len(pairs)
            print("  %-8s 框 %3d  GT %3d  配对 %3d  丢 %3d"
                  % (name, len(boxes), len(items), len(pairs), _dropped))

    # 只有 1 个 case 时切不出 val；此时如实告诉用户，不要静默空跑。
    if split_mode != "real" and case_list and "val" not in splits.values():
        if len(case_list) >= 2:
            first = min(splits, key=_hash)
            splits[first] = "val"
            for r in recs:
                if r["case"] == first:
                    r["split"] = "val"
            print("[split] 哈希没抽中 val，按最小哈希把 case `%s` 划进 val" % first)
        else:
            print("!! 只有 1 个 case，切不出 val；train.py 需要 train 与 val 都非空（请再给几页）")

    recs.sort(key=lambda r: (r["case"], r["bbox"][1], r["bbox"][0]))
    return recs, dict(gt_total=gt_total, drops=drops, n_case=n_case, n_skip=n_skip,
                      splits=splits)


def build_from_boxes(paths: Sequence[Path], roots: Sequence[Path]):
    """框级 JSONL（厂商首选格式）：
    {"case","bbox":[x0,y0,x1,y1],"text","role"}，`image` 可选（缺省 `<case>.png`）。"""
    idx: Dict[str, Path] = {}
    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                idx.setdefault(p.stem, p)
    recs: List[dict] = []
    n_bad = 0
    n_box = 0
    for p in paths:
        with p.open(encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_box += 1
                r = json.loads(line)
                case = str(r.get("case") or "").strip()
                text = r.get("text")
                bbox = r.get("bbox")
                if not case or not _ok_text(text) or not _ok(bbox):
                    n_bad += 1
                    continue
                img = r.get("image")
                ipath = Path(img) if img else idx.get(case)
                if ipath is None or not Path(ipath).is_file():
                    n_bad += 1
                    continue
                role = canon_role(r.get("role"))
                recs.append({
                    "image": str(Path(ipath).resolve()),
                    "case": case,
                    "src": "real",
                    "split": "train",
                    "role": role,
                    "role_id": C.ROLE_ID.get(role, C.N_ROLES - 1),
                    "text": str(text),
                    "bbox": [round(float(v), 2) for v in bbox],
                })
    if n_bad:
        print("[boxes] 跳过无效/找不到图的行 %d 条" % n_bad)
    return recs, n_box


def summarise(recs: List[dict], info: dict) -> None:
    print("\n[gt] case 成功 %d / 跳过 %d" % (info.get("n_case", 0), info.get("n_skip", 0)))
    print("[gt] split: %s" % dict(Counter(r["split"] for r in recs)))
    gt_total: Counter = info.get("gt_total") or Counter()
    per_role: Dict[str, List[float]] = {}
    for r in recs:
        per_role.setdefault(r["role"], []).append(float(r["bbox"][3]) - float(r["bbox"][1]))
    print("  %-12s %8s %8s %10s" % ("role", "gt", "paired", "med_h px"))
    for role in ("designator", "value", "pin_number"):
        hs = sorted(per_role.get(role, []))
        med = hs[len(hs) // 2] if hs else float("nan")
        print("  %-12s %8d %8d %10.1f" % (role, gt_total.get(role, 0), len(hs), med))
    drops: Counter = info.get("drops") or Counter()
    print("[gt] 丢弃：无框在半径内 %d / 框侧并列 %d / GT 侧并列 %d"
          % (drops.get("drop_no_box", 0), drops.get("drop_tie_box", 0),
             drops.get("drop_tie_gt", 0)))
    # `--boxes` 模式下 `gt_total` 恒为 0（不走弱配对），百分比没有意义，改报框级命中率。
    if info.get("boxes_mode"):
        n_box = int(info.get("n_box", 0))
        print("[gt] 记录 %d 条（框级 %d 行里保留 %.1f%%）"
              % (len(recs), n_box, 100.0 * len(recs) / max(1, n_box)))
    else:
        print("[gt] 记录 %d 条（%.1f%% 的 GT 字符串被配上）"
              % (len(recs), 100.0 * len(recs) / max(1, sum(gt_total.values()))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GT 页面 -> CRNN 清单（弱配对）")
    ap.add_argument("--gt", nargs="+", default=[], help="GT 根目录（可给多个；每个 case 是子目录）")
    ap.add_argument("--boxes", nargs="+", default=[], help="框级 JSONL（厂商格式，给了就不走弱配对）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--stats", action="store_true", help="只打印统计，不落盘")
    ap.add_argument("--radius", type=float, default=40.0, help="GT 锚点到框中心的最大距离（px）")
    ap.add_argument("--cases", default="", help='只跑这些 case，如 "1,2,10"')
    ap.add_argument("--limit-cases", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=VAL_EVERY_DEFAULT,
                    help="按 case 哈希每 N 个抽一个进 val（默认 %d）" % VAL_EVERY_DEFAULT)
    ap.add_argument("--split-mode", choices=("hash", "real"), default="hash",
                    help="hash=按 case 哈希切 train/val（可训练）；real=全部 split='real'"
                         "（= build_real 口径，只用于 eval，train.py 不会吃）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    roots = [Path(p) for p in args.gt]
    for r in roots:
        if args.gt and not r.is_dir():
            print("  ! --gt 不存在或不是目录：%s" % r)
    if args.boxes:
        recs, n_box = build_from_boxes([Path(p) for p in args.boxes], roots or [Path(".")])
        recs.sort(key=lambda r: (r["case"], r["bbox"][1], r["bbox"][0]))
        info = dict(gt_total=Counter(), drops=Counter(), n_case=len({r["case"] for r in recs}),
                    boxes_mode=True, n_box=n_box)
    else:
        if not roots:
            ap.error("要么给 --gt，要么给 --boxes")
        wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
        recs, info = build_from_gt(roots, args.radius, args.val_every, args.split_mode,
                                   wanted, args.limit_cases, args.quiet)

    summarise(recs, info)
    if args.stats:
        print("[stats] 只统计，未落盘（去掉 --stats 才写 --out）")
        return 0
    write(recs, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
