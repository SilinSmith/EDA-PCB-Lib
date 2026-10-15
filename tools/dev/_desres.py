# -*- coding: utf-8 -*-
"""**位号重解 pass（原型）**：把「没挂上位号」的元件，用附近**空闲**文字框里读出的位号补上。

动机与读数：`_desceil.py` 量到派生档上「键不对」的元件里，只用空闲框、R=40 时
**15.1%（39/258）** 能被附近一个框救回（允许抢占用框时 24%）。本脚本把「救」做出来：

  1. 候选框 = stage1 `texts` 里尺寸像文字、且**还没被任何元件当位号用**（`_assigned` 的
     `designator.box_i`）的框；用域内微调后的 CRNN 按 `designator` 角色读一遍；
  2. 只收「读出来像位号」（`^[A-Za-z]{1,4}\\d{1,4}[A-Za-z]?$`）且置信度 ≥ `--min-conf` 的；
  3. 几何 + 类型前缀（`stage2.assign.TYPE_PREFIX`）都相容的 (元件, 框) 对，按「中心到元件框
     的距离」贪心做**全局一对一**（一个框只给一颗元件）；
  4. 重写最终 JSON：`components` 的键、`pins` 的键、`nets[*].hyperGraph` 里 `老键.` 的 token；
     IC 类的 `Name` 若为空/等于旧键也一起改。**没救到的保持原样。**

坐标系：pred/GT 的 `bbox` 是 **JSON 约定（y 从图底往上）**；stage1 `texts[].bbox_img` 与
`_assigned` 槽位的 `box` 是**图像坐标**。翻转 `y_img = H - y_json`（H 取 stage1 `image.h`）。

只写新目录，不动原产物。用法：
    python -X utf8 work\\tools\\dev\\_desres.py --pred work\\_ro220b --out work\\_ro220b_dr \\
        --cases-file work\\_tmp_cases20.txt --radius 40
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import re
import sys
from pathlib import Path

WORK = Path(r"Y:\workspace\EDA\work")
EDA = Path(r"Y:\workspace\EDA\eda-pcb")
PSEUDO = re.compile(r"^(?:∅\d+|p\d{3,4})$")
DESIG = re.compile(r"^[A-Za-z]{1,4}\d{1,4}[A-Za-z]?$")
EMPTY = ("", None)
sys.path.insert(0, str(EDA))


def dist_to_rect(cx: float, cy: float, b) -> float:
    dx = max(b[0] - cx, 0.0, cx - b[2])
    dy = max(b[1] - cy, 0.0, cy - b[3])
    return (dx * dx + dy * dy) ** 0.5


def norm(t: str):
    a = (t or "").strip().upper()
    return a, re.sub(r"[^0-9A-Za-z]", "", a)


def rewrite(doc: dict, renames: dict) -> int:
    """把 components/pins 的键与 nets 的 hyperGraph token 一起改名。"""
    if not renames:
        return 0
    comps = doc.get("components") or {}
    for old, new in renames.items():
        if old not in comps:
            continue
        v = comps.pop(old)
        if not str(v.get("Name") or "").strip() or PSEUDO.match(str(v.get("Name") or "")) \
                or str(v.get("Name")) == old:
            v["Name"] = new
        v["_rename_from"] = old
        comps[new] = v
    pins = doc.get("pins")
    if isinstance(pins, dict):
        for old, new in renames.items():
            if old in pins:
                pins[new] = pins.pop(old)
    for net in (doc.get("nets") or {}).values():
        hg = net.get("hyperGraph")
        if isinstance(hg, str):
            for old, new in renames.items():
                hg = hg.replace("(%s." % old, "(%s." % new).replace(",%s." % old, ",%s." % new)
            net["hyperGraph"] = hg
    return len(renames)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(WORK / "_ro220b"))
    ap.add_argument("--out", default=str(WORK / "_ro220b_dr"))
    ap.add_argument("--assigned", default=None, help="默认 <pred>\\_assigned")
    ap.add_argument("--stage1", default=str(WORK / "_full220" / "runs" / "stage1"))
    ap.add_argument("--cases-file", default=str(WORK / "_cases220.txt"))
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--radius", type=float, default=40.0)
    ap.add_argument("--max-dist", type=float, default=30.0, help="框中心到元件框的最大距离（像素）")
    ap.add_argument("--max-box-h", type=float, default=24.0)
    ap.add_argument("--max-box-w", type=float, default=90.0)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--prefix", choices=("strict", "off"), default="strict")
    ap.add_argument("--steal-real", action="store_true",
                    help="已经挂了真位号的元件也允许改挂到空闲框读出的另一个位号上（默认只补空/伪键）")
    ap.add_argument("--ckpt", default=str(WORK / "ocr_official_v1" / "best.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-print", type=int, default=10)
    a = ap.parse_args()

    from stage2.assign import TYPE_PREFIX

    # 只在**会印位号**的元件类型上做重解（`gnd`/`v`/`net_*`/`other` 在 GT 里是 ∅NNN 伪键，
    # 补了反而变成 fp+fn）。前缀表在 `stage2.assign.TYPE_PREFIX` 基础上补几个常见类型。
    DR_TYPES = dict(TYPE_PREFIX)
    DR_TYPES.pop("v", None)                  # v = 电源符号，GT 里没有位号
    DR_TYPES.pop("pin", None)
    DR_TYPES.pop("testpoint", None)
    DR_TYPES.pop("tp", None)
    DR_TYPES.update({"motor": ("M",), "battary": ("BT", "B"), "buzzer": ("LS", "BZ", "U"),
                     "jumper": ("J", "JP", "P"), "transfomer": ("T", "TR"), "ant": ("ANT", "E"),
                     "block": ("U", "J"), "net_input": (), "net_output": (), "net_bidirection": (),
                     "gnd": (), "other": (), "v": (), "net_bus": ()})

    def prefix_ok(ctype: str, text: str) -> bool:
        if a.prefix == "off":
            return True
        want = DR_TYPES.get((ctype or "").lower())
        if not want:
            return False
        tx = text.upper()
        return any(tx.startswith(str(w).upper()) for w in want)

    raw = Path(a.cases_file).read_text(encoding="utf-8-sig")
    cases, seen = [], set()
    for tok in re.split(r"[,\s]+", raw):
        if not re.match(r"^[0-9A-Za-z_]+$", tok or "") or tok in seen:
            continue
        seen.add(tok)
        cases.append(tok)
    cases = cases[: a.limit]

    from PIL import Image
    from textrec.infer import Recognizer
    rec = Recognizer.load(a.ckpt, device=a.device)
    outp = Path(a.out)
    outp.mkdir(parents=True, exist_ok=True)
    print("[ckpt] %s\n[pred] %s -> %s  cases=%d radius=%.0f steal_real=%s"
          % (a.ckpt, a.pred, a.out, len(cases), a.radius, a.steal_real))

    st = collections.Counter()
    examples = []
    for c in cases:
        fp = Path(a.pred) / ("%s.json" % c)
        s1p = Path(a.stage1) / ("%s.stage1.json" % c)
        adp = Path(a.assigned or (Path(a.pred) / "_assigned")) / ("%s.json" % c)
        if not fp.is_file() or not s1p.is_file():
            st["缺文件"] += 1
            continue
        doc = json.loads(fp.read_text(encoding="utf-8"))
        s1 = json.loads(s1p.read_text(encoding="utf-8"))
        H = float((s1.get("image") or {}).get("h") or 0) or None
        if H is None:
            st["缺图高"] += 1
            continue
        texts = s1.get("texts") or []
        ad = json.loads(adp.read_text(encoding="utf-8")) if adp.is_file() else {}
        used_i, cur_box = set(), {}
        for cid, cv in (ad.get("comps") or {}).items():
            ds = cv.get("designator")
            if isinstance(ds, dict) and ds.get("box_i") is not None:
                used_i.add(int(ds["box_i"]))
                cur_box[str(cid)] = int(ds["box_i"])

        comps = doc.get("components") or {}
        # 元件 id -> 键（pins/hyperGraph 用的是键；`_assigned` 用 stage1 id）
        key_of = {str(v.get("_id") or k): k for k, v in comps.items()}
        # 需要补位号的元件：伪键（或空）
        want, all_comps = [], []
        for k, v in comps.items():
            bb = v.get("bbox")
            if not bb or len(bb) != 4:
                continue
            bi = [bb[0], H - bb[3], bb[2], H - bb[1]]
            rec_ = {"key": k, "type": str(v.get("type") or "other"), "box_img": bi,
                    "pseudo": bool(PSEUDO.match(str(k)))}
            all_comps.append(rec_)
            if rec_["pseudo"] or a.steal_real:
                want.append(rec_)
        if not want:
            st["无需补"] += 1

        # 候选框：尺寸像文字、且没被任何元件占用
        cand = []
        for t in texts:
            ti = int(str(t.get("id") or "t-1").lstrip("t") or -1)
            bb = t.get("bbox_img") or [0, 0, 0, 0]
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            if h > a.max_box_h or w > a.max_box_w or w <= 1 or h <= 1 or ti in used_i:
                continue
            cand.append((ti, bb))
        if not cand:
            st["无空闲框"] += 1
            (outp / fp.name).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            continue

        img = Image.open((s1.get("image") or {}).get("path")).convert("L")
        got = rec.read_many(img, [(bb, "designator") for _ti, bb in cand])

        pairs = []          # (dist, -conf, comp_idx, cand_idx, text)
        for ci, cr in enumerate(want):
            cx = (cr["box_img"][0] + cr["box_img"][2]) / 2.0
            cy = (cr["box_img"][1] + cr["box_img"][3]) / 2.0
            ex = [cr["box_img"][0] - a.radius, cr["box_img"][1] - a.radius,
                  cr["box_img"][2] + a.radius, cr["box_img"][3] + a.radius]
            for xi, (ti, bb) in enumerate(cand):
                bx, by = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
                if not (ex[0] <= bx <= ex[2] and ex[1] <= by <= ex[3]):
                    continue
                txt, cf = got[xi]
                aa, _bb2 = norm(txt)
                if not DESIG.match(aa) or cf < a.min_conf:
                    continue
                if not prefix_ok(cr["type"], aa):
                    st["前缀不符"] += 1
                    continue
                d = dist_to_rect(bx, by, cr["box_img"])
                if d > a.max_dist:
                    st["太远"] += 1
                    continue
                pairs.append((d, -float(cf), ci, xi, aa))
        pairs.sort()
        have = set(str(k) for k in comps)
        used_c, used_x, take = set(), set(), {}
        for d, ncf, ci, xi, txt in pairs:
            if ci in used_c or xi in used_x:
                continue
            cr = want[ci]
            # 位号唯一：不能撞上任何**现有**键，也不能在这一趟里重复
            if txt in have and txt != cr["key"]:
                st["撞已有键"] += 1
                continue
            used_c.add(ci)
            used_x.add(xi)
            have.add(txt)
            take[ci] = (txt, cand[xi][0], d)
        renames = {}
        for ci, (txt, ti, d) in take.items():
            cr = want[ci]
            if txt == cr["key"]:
                st["本来就对"] += 1
                continue
            renames[cr["key"]] = txt
            st["补键"] += 1
            if len(examples) < a.max_print:
                examples.append((c, cr["key"], cr["type"], txt, round(d, 1), ti))
        rewrite(doc, renames)
        (outp / fp.name).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    print("== 位号重解 %d 张 ==" % len(cases))
    for k in sorted(st):
        print("  %-16s %6d" % (k, st[k]))
    for e in examples:
        print("   例: case=%s %s(%s) -> %s  d=%.1f box_i=%s" % e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
