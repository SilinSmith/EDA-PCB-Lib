# -*- coding: utf-8 -*-
"""**不改管线**的模拟：把「官方库给出的引脚号」直接补进交付 JSON，再打分。

`_libnum.py` 量到（官方 9 张、半径 8 px、55 个 missing 引脚）：
  * 库里同型号+同名能给出 GT 号（上界）40；
  * 该名字在库里**唯一映射到一个号**的 20 个，且 **20/20 都是 GT 号** ⇒ 规则精度 100%；
  * 其中「名读到（且绑到了 GT 那个位置的桩上）+ 桩在」这一块（= A3）14 个，
    库里能给对号 10、唯一且对 **6**。

本工具把那 6（或更多）个号**直接写进 pred 的 pins**（pinname 用我方已读到的名、
point 用 stage1 的桩点），产出到 `--out`，再用 `stage2_eval.py` 打分对比。
⚠️ 补进去的引脚**不进任何 net**（netH 不变），所以这测的是 **pinF1 单项的收益**。

用法（从仓库根跑）：
    python -X utf8 work\\tools\\diag\\_libsim.py --pred work\\_fnew \\
        --assigned work\\_fnew\\_assigned --stage1 work\\_fast_out_reg9\\runs\\stage1 \\
        --out work\\_libsim_uniq --mode uniq --cases 1,2,3,4,5,7,8,9,10
然后：
    python -X utf8 work\\tools\\eval\\stage2_eval.py work\\_libsim_uniq --quiet
    python -X utf8 work\\tools\\eval\\stage2_eval.py work\\_fnew     --quiet   # 基线
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "diag"))

from schematic_parser.metrics import (  # noqa: E402
    EvalConfig, assign_components, canon_type, rewrite_keys)
from schematic_parser.data import IC_TYPES, pin_key_to_number  # noqa: E402
from _libcov import Lib, load, norm  # noqa: E402

GT9 = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
LIB = ROOT / "work" / "data" / "cpntLibrary.json"
S1 = ROOT / "work" / "_fast_out_reg9" / "runs" / "stage1"


def _ctr(b):
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def _dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def _dist1(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(l)):
        if l[:i] + l[i + 1:] == s:
            return True
    return False


def nums_for_name(lib: Lib, key: str, gname: str) -> list:
    out = []
    ng = norm(gname)
    if not ng or not key:
        return out
    for p in lib.raw.get(key, {}).get("pins", []) or []:
        nm = norm(str(p.get("name") or ""))
        if not nm:
            continue
        if nm == ng or _dist1(nm, ng) or nm in ng or ng in nm:
            num = str(p.get("number") or "").strip()
            if num and num not in out:
                out.append(num)
    return out


def names_for_number(lib: Lib, key: str, num: str) -> list:
    """库里 (型号, 号) → 名字的**唯一**候选；多个不同名字时返回空（安全规则）。"""
    out = []
    nn = norm(num)
    if not nn or not key:
        return out
    for p in lib.raw.get(key, {}).get("pins", []) or []:
        if norm(str(p.get("number") or "")) != nn:
            continue
        nm = str(p.get("name") or "").strip()
        if nm and norm(nm) not in [norm(x) for x in out]:
            out.append(nm)
    return out if len(out) == 1 else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--assigned", default="")
    ap.add_argument("--stage1", default=str(S1))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="1,2,3,4,5,7,8,9,10")
    ap.add_argument("--gt", default=str(GT9))
    ap.add_argument("--lib", default=str(LIB))
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--fuzzy", type=float, default=0.85)
    ap.add_argument("--mode", default="uniq",
                    choices=["uniq", "pickfree"],
                    help="uniq=只在该名字唯一映射一个号时补；pickfree=多个号时挑没被本元件用过的")
    ap.add_argument("--inner-only", action="store_true",
                    help="只补「名读到+桩在」（A3）那一块")
    ap.add_argument("--ungated", action="store_true",
                    help="**管线等价档**：不要求桩上的名字 == GT 名（管线里没有 GT 可看，"
                         "闸门是「桩在半径内 + 名非空 + 库里唯一」，见 --radius/--minname）")
    ap.add_argument("--minname", type=int, default=2,
                    help="--ungated 时「名非空」的最短长度（默认 2，滤掉单个字符噪声）")
    ap.add_argument("--fixname", action="store_true",
                    help="第二通道：号读对但名读错/没读到 ⇒ 用库的 (型号,号)→名 唯一候选改写")
    ap.add_argument("--verbose", type=int, default=0)
    a = ap.parse_args()

    lib = Lib(load(Path(a.lib)))
    gtroot, predroot = Path(a.gt), Path(a.pred)
    asgroot, s1root = Path(a.assigned), Path(a.stage1)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cfg = EvalConfig()
    R = a.radius

    tot_add = tot_right = tot_wrong = 0
    tot_fix = tot_fix_ok = 0
    for c in [x.strip() for x in a.cases.split(",") if x.strip()]:
        gtf = gtroot / c / ("%s.json" % c)
        pf = predroot / ("%s.json" % c)
        sf = s1root / ("%s.stage1.json" % c)
        af = asgroot / ("%s.json" % c)
        if not (gtf.exists() and pf.exists() and sf.exists()):
            print("  ! 跳过 case %s" % c)
            continue
        gt = json.loads(gtf.read_text(encoding="utf-8"))
        pred = json.loads(pf.read_text(encoding="utf-8"))
        s1 = json.loads(sf.read_text(encoding="utf-8"))
        asg = json.loads(af.read_text(encoding="utf-8")) if af.exists() else {}
        H = float(s1["image"]["h"])
        _f1, keymap = assign_components(gt, pred, cfg)
        pred2 = rewrite_keys(pred, keymap)
        s1comps = []
        for comp in s1.get("components") or []:
            b = comp.get("bbox_json") or []
            if len(b) == 4:
                s1comps.append((comp.get("id"), _ctr(b)))
        s1_pins = s1.get("pins") or {}
        cmps = asg.get("comps") or {}
        n_add = n_ok = 0
        nfix = nfix_ok = 0
        for k, rec in (gt.get("components") or {}).items():
            if canon_type(rec.get("type")) not in IC_TYPES:
                continue
            if k not in (pred2.get("components") or {}):
                continue
            gtname = rec.get("Name") or ""
            pname = (pred2["components"].get(k) or {}).get("Name") or ""
            k_gt = (lib.exact(gtname) or lib.normalized(gtname)
                    or [x[0] for x in lib.fuzzy(gtname, a.fuzzy)])
            k_pr = (lib.exact(pname) or lib.normalized(pname)
                    or [x[0] for x in lib.fuzzy(pname, a.fuzzy)])
            src = (k_gt or k_pr or [None])[0]
            gb = rec.get("bbox") or []
            cid = None
            if len(gb) == 4 and s1comps:
                cid = min(s1comps, key=lambda t: _dist(t[1], _ctr(gb)))[0]
            raw_pins = ((cmps.get(cid) or {}).get("pins") or {}) if cid else {}
            stubs = [tuple(q[:2]) for q in (s1_pins.get(cid) or []) if len(q) >= 2]
            have = pred2.setdefault("pins", {}).setdefault(k, {})
            used = {str(pin_key_to_number(pk)) for pk in have}
            for pk, pv in (gt.get("pins", {}).get(k) or {}).items():
                if pk in have:
                    continue
                gname = (pv or {}).get("pinname") or ""
                gnum = str(pin_key_to_number(pk) or "").strip()
                pt = (pv or {}).get("point") or {}
                if "x" not in pt or "y" not in pt:
                    continue
                p_img = (float(pt["x"]), H - float(pt["y"]))
                j, best = None, 1e9
                for idx, q in enumerate(stubs):
                    d = _dist(tuple(q[:2]), p_img)
                    if d < best:
                        best, j = d, idx
                if j is None or best > R:
                    continue
                rec_j = raw_pins.get(str(j)) or raw_pins.get(j) or {}
                nm_txt = ((rec_j.get("name") or {}).get("text") or "").strip()
                if a.ungated:
                    # 管线等价档：不看 GT 名，只看「名非空 + 库唯一」（管线的真实闸门）。
                    if len(norm(nm_txt)) >= a.minname:
                        cand_name = nm_txt
                    else:
                        continue
                else:
                    if not nm_txt or norm(nm_txt) != norm(gname):
                        continue                  # 只有「名读到」的才谈得上补号
                    cand_name = gname
                if a.inner_only and not nm_txt:
                    continue
                nums = nums_for_name(lib, src, cand_name)
                if not nums:
                    continue
                if a.mode == "uniq":
                    if len(nums) != 1:
                        continue
                    pick = nums[0]
                else:
                    free = [x for x in nums if x not in used]
                    if not free:
                        continue
                    pick = free[0]
                if pick in used:
                    continue
                # 用我方的桩点（json 坐标）当引脚点
                q = stubs[j]
                pk0 = "pin_%s" % pick
                if pk0 in have:
                    continue
                have[pk0] = {"pinname": nm_txt,
                             "point": {"x": round(float(q[0]), 2),
                                       "y": round(H - float(q[1]), 2)}}
                used.add(pick)
                n_add += 1
                if pick == gnum:
                    n_ok += 1
                if a.verbose:
                    print("    [add] %s %s 名=%-10r 号 %s%s" % (
                        c, k, nm_txt, pick, "" if pick == gnum else "（≠GT %s）" % gnum))
            if a.fixname:
                for pk, pv in (gt.get("pins", {}).get(k) or {}).items():
                    pt = (pv or {}).get("point") or {}
                    if "x" not in pt or "y" not in pt:
                        continue
                    p_img = (float(pt["x"]), H - float(pt["y"]))
                    pk2, best2 = None, 1e9
                    for pk_o, pv_o in have.items():
                        po = (pv_o or {}).get("point") or {}
                        if "x" not in po or "y" not in po:
                            continue
                        d = _dist((float(po["x"]), H - float(po["y"])), p_img)
                        if d < best2:
                            best2, pk2 = d, pk_o
                    if pk2 is None or best2 > R:
                        continue
                    gname = (pv or {}).get("pinname") or ""
                    ours = (have[pk2] or {}).get("pinname") or ""
                    if not gname or norm(ours) == norm(gname):
                        continue
                    num_o = str(pin_key_to_number(pk2) or "").strip()
                    cand = names_for_number(lib, src, num_o)
                    if not cand:
                        continue
                    have[pk2]["pinname"] = cand[0]
                    nfix += 1
                    if norm(cand[0]) == norm(gname):
                        nfix_ok += 1
                    elif a.verbose:
                        print("    [name] %s %s.%s 号=%s 名 %r→%r（GT %r）"
                              % (c, k, pk2, num_o, ours, cand[0], gname))
        tot_add += n_add
        tot_right += n_ok
        tot_wrong += n_add - n_ok
        tot_fix += nfix
        tot_fix_ok += nfix_ok
        (out / ("%s.json" % c)).write_text(
            json.dumps(pred2, ensure_ascii=False), encoding="utf-8")
        print("  case %-3s 补 %2d 个引脚（其中号对 %2d）%s"
              % (c, n_add, n_ok,
                 "" if not a.fixname else "｜改名 %d（对 %d）" % (nfix, nfix_ok)))

    print("\n===== 模拟补号（mode=%s, radius=%.0f）=====" % (a.mode, R))
    print("  共补入 %d 个引脚（号正确 %d / 错 %d）⇒ 规则精度 %.1f%%"
          % (tot_add, tot_right, tot_wrong,
             100.0 * tot_right / tot_add if tot_add else 0.0))
    if a.fixname:
        print("  库改名通道：%d 处（改对 %d / 改错 %d）⇒ 精度 %.1f%%"
              % (tot_fix, tot_fix_ok, tot_fix - tot_fix_ok,
                 100.0 * tot_fix_ok / tot_fix if tot_fix else 0.0))
    print("  产物写到 %s ⇒ 打分：" % out)
    print("    python -X utf8 work\\tools\\eval\\stage2_eval.py %s --quiet" % out)
    print("    python -X utf8 work\\tools\\eval\\stage2_eval.py %s --quiet  # 基线"
          % a.pred)


if __name__ == "__main__":
    main()
