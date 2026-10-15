# -*- coding: utf-8 -*-
"""建「第二把尺子」：**交付案池**（214 case 官方文字 GT）的 textrec ruler，GT-only、零 GPU、零训练。

动机（RESULTS §5.174 第 2 条 / §5.175 功效提醒 / §5.176）：
* 现 ruler（`work/fullpop_r8/ruler.jsonl`）只有 8 张 sheet、目标框 389 条，配对 MDE 0.0668，
  而判据要 Δ≥+0.05 ⇒ 功效不足；且主终点只有 15.2% 的行有重叠，「重叠分解」在该尺子上测不出来。
* 交付案池（本轮源：`work/official_text_pad/manifest.jsonl`，214 case / 4,974 条）与 32 张真实
  sheet 只共享 29 张图 ⇒ 其余 case 对 textrec（合成 + 32 张真实）**完全没训过**，对三臂 A/B 也是
  留出的 ⇒ 可当第二把尺子，且**不需要重训任何臂**。

口径（全 GT-only，模型输出不参与建尺子）：
* 标签来自官方 `<case>.json`（`tools/dev/mk_official_textset.py` 头注：标签从 GT 来、框由我们的
  检测/复读来）⇒ 标签不循环；`was` = 我们当初的读数，`dist` = 两者编辑距离（`textrec_manifest.jsonl`
  的官方块**已按 dist<=2 过滤**）。
* 裁剪框：`bbox` = 交付外扩框（交付口径）；`bbox_tight` = 紧框（训练侧口径，§5.174 的口径错配）。
* 重叠桶沿用 `textrec.eval_real`：与同图其它框的最大「交/较小框面积」≥0.10 mild / ≥0.30 strong。
* 污染：case 的图在 32 张真实语料图里 = 现交付模型训过 ⇒ 只进 `ruler_all`，不进干净尺子。

产出 `Y:\\workspace\\EDA\\work\\pool_r1\\`：
* `ruler_clean.jsonl`        干净图 + 交付外扩框（4,068 行 / 184 case）—— 交付口径主尺
* `ruler_clean_tight.jsonl`  与上者**逐行同标签**，只把 bbox 换成 `bbox_tight` —— 口径对照尺
* `ruler_all.jsonl`          交付池全量（4,974 行 / 214 case，含 29 张训过的图）—— 只作对照
* `ruler_big.jsonl`          官方块 ∖ 32 张真实图、紧框、dist<=2（20,343 行 / 184 case）—— n 最大的尺

喂法：
    cd eda-pcb
    python -X utf8 -m textrec.eval_real --ckpt weights\\textrec_best.pt \\
        --manifest Y:\\workspace\\EDA\\work\\pool_r1\\ruler_clean.jsonl --by-len --no-ov \\
        --dump Y:\\workspace\\EDA\\work\\pool_r1\\baseline_errors.jsonl
（基线读数见 RESULTS §5.176；`--no-ov` 只影响表的呈现，不影响 exact。）
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

EDA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EDA))
from textrec import config as C  # noqa: E402

WORK = Path(r"Y:\workspace\EDA\work")
POOL = WORK / "official_text_pad" / "manifest.jsonl"   # 已验证（reread ×2/×3）的交付池文字 GT
BIG = WORK / "official_text" / "textrec_manifest.jsonl"  # 官方块全量（已按 dist<=2 过滤）
EVAL_MAN = EDA / "textrec" / "data" / "manifest.jsonl"  # 32 张真实 sheet（= 三臂训练池）
OLD_REAL = WORK / "ocr_real" / "manifest.jsonl"
OUT = WORK / "pool_r1"

OV_THR, OV_STRONG = 0.10, 0.30


def load_jsonl(p: Path) -> list:
    return [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()]


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return 0.0 if (x1 <= x0 or y1 <= y0) else (x1 - x0) * (y1 - y0)


def ov_labels(rows: list, key: str = "bbox") -> None:
    """把重叠度写回每行（只用 GT 框，模型输出不参与）。"""
    by_img = collections.defaultdict(list)
    for r in rows:
        by_img[r["image"]].append(r)
    for _img, rs in by_img.items():
        ar = [_area(r[key]) for r in rs]
        for i, r in enumerate(rs):
            best = 0.0
            for j, s in enumerate(rs):
                if i == j or ar[i] <= 0 or ar[j] <= 0:
                    continue
                iv = _inter(r[key], s[key])
                if iv > 0:
                    best = max(best, iv / min(ar[i], ar[j]))
            r["ov"] = best
            r["ovb"] = ("clean" if best < OV_THR else
                        "mild" if best < OV_STRONG else "strong")


def bucket(t: str) -> str:
    n = len(t)
    return ("1-2" if n <= 2 else "3-4" if n <= 4 else
            "5-7" if n <= 7 else "8-11" if n <= 11 else "12+")


def report(name: str, rows: list) -> None:
    cases = {r["case"] for r in rows}
    imgs = {r["image"] for r in rows}
    print("\n== %s: %d case / %d 图 / %d 条" % (name, len(cases), len(imgs), len(rows)))
    print("   role:", dict(collections.Counter(r["role"] for r in rows)))
    for role in ("value", "designator", "pin_name", "pin_number"):
        sub = [r for r in rows if r["role"] == role]
        if not sub:
            continue
        print("   %-11s n=%-6d 长度 %s  >=5=%-5d >=3=%d" % (
            role, len(sub), dict(collections.Counter(bucket(r["text"]) for r in sub)),
            sum(1 for r in sub if len(r["text"]) >= 5),
            sum(1 for r in sub if len(r["text"]) >= 3)))
    print("   ALL>=5=%d  重叠桶（ALL）: %s" % (
        sum(1 for r in rows if len(r["text"]) >= 5),
        dict(collections.Counter(r["ovb"] for r in rows))))
    v5 = [r for r in rows if r["role"] == "value" and len(r["text"]) >= 5]
    if v5:
        print("   value>=5 n=%d 重叠桶: %s" % (
            len(v5), dict(collections.Counter(r["ovb"] for r in v5))))


def write_rows(path: Path, rows: list) -> None:
    rows.sort(key=lambda r: (r["case"], r["bbox"][1], r["bbox"][0]))
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("-> %s (%d 行)" % (path, len(rows)))


def build_small() -> tuple:
    """交付池（已验证）-> ruler_clean（外扩框） + ruler_all。"""
    pool = load_jsonl(POOL)
    real_img = {os.path.basename(r["image"]) for r in load_jsonl(EVAL_MAN)
                if r.get("split") == "real"}
    old_img = {os.path.basename(r["image"]) for r in load_jsonl(OLD_REAL)}
    print("池 %d 条 / %d case；32 张真实语料 %d 图；旧清单 %d 图"
          % (len(pool), len({r["case"] for r in pool}), len(real_img), len(old_img)))
    print("旧训练图 ⊆ 32 张真实图:", old_img <= real_img,
          "（差集 %s）" % sorted(old_img - real_img))

    rows = []
    for r in pool:
        if r.get("role") not in C.ROLE_ID or not Path(r["image"]).is_file():
            continue
        b = [round(float(v), 2) for v in r["bbox"]]
        if not (b[2] > b[0] and b[3] > b[1]):
            continue
        rows.append({
            "image": r["image"], "case": r["case"], "src": "official_pool", "split": "real",
            "role": r["role"], "role_id": C.ROLE_ID[r["role"]], "text": str(r["text"]),
            "bbox": b,
            "bbox_tight": [round(float(v), 2) for v in (r.get("bbox_tight") or b)],
            "rconf": r.get("rconf"), "_by": r.get("_by"), "was": r.get("was"),
        })
    ov_labels(rows, "bbox")

    clean = [r for r in rows if os.path.basename(r["image"]) not in real_img]
    contam = [r for r in rows if os.path.basename(r["image"]) in real_img]
    print("池内图 %d 张：与 32 张真实语料重叠 %d 张 -> clean %d 条；contam %d 条"
          % (len({r["image"] for r in rows}), len({r["image"] for r in rows} & real_img),
             len(clean), len(contam)))
    assert not ({os.path.basename(r["image"]) for r in clean} & real_img), "clean 与真实语料图重叠"
    assert not ({os.path.basename(r["image"]) for r in clean} & old_img), "clean 与旧训练图重叠"

    write_rows(OUT / "ruler_clean.jsonl", clean)
    write_rows(OUT / "ruler_all.jsonl", list(rows))
    return clean, rows


def build_clean_tight(clean: list) -> list:
    """与 ruler_clean 逐行同标签，只把 bbox 换成 bbox_tight（退化则回退外扩框）。"""
    tight, same, fallback = [], 0, 0
    for r in clean:
        t = [round(float(v), 2) for v in (r.get("bbox_tight") or r["bbox"])]
        if t[2] - t[0] < 2 or t[3] - t[1] < 2:
            fallback += 1
            t = list(r["bbox"])
        if t == list(r["bbox"]):
            same += 1
        q = dict(r)
        q["bbox"] = t
        q["bbox_pad"] = list(r["bbox"])
        tight.append(q)
    ov_labels(tight, "bbox")
    print("\n紧框尺：tight==pad %d 行；退化回退 %d 行" % (same, fallback))
    write_rows(OUT / "ruler_clean_tight.jsonl", tight)
    return tight


def build_big() -> list:
    """官方块全量（dist<=2）∖ 32 张真实图 -> ruler_big（紧框）。"""
    real_img = {os.path.basename(r["image"]) for r in load_jsonl(EVAL_MAN)
                if r.get("split") == "real"}
    rows = []
    for r in load_jsonl(BIG):
        if r.get("src") != "official" or r.get("role") not in C.ROLE_ID:
            continue
        if not Path(r["image"]).is_file():
            continue
        b = [round(float(v), 2) for v in r["bbox"]]
        if not (b[2] > b[0] and b[3] > b[1]):
            continue
        rows.append({"image": r["image"], "case": r["case"], "src": "official_pool",
                     "split": "real", "role": r["role"], "role_id": C.ROLE_ID[r["role"]],
                     "text": str(r["text"]), "bbox": b, "was": r.get("was"),
                     "dist": r.get("dist")})
    all_img = {r["image"] for r in rows}
    keep = all_img - {i for i in all_img if os.path.basename(i) in real_img}
    rows = [r for r in rows if r["image"] in keep]
    print("\n官方块 ∖ 32 张真实图: %d 行 / %d case / %d 图；dist %s"
          % (len(rows), len({r["case"] for r in rows}), len(all_img & keep),
             dict(collections.Counter(r["dist"] for r in rows))))
    ov_labels(rows, "bbox")
    write_rows(OUT / "ruler_big.jsonl", rows)
    return rows


def cross_check(clean, tight, small, big) -> None:
    """三条不变量：同标签换口径、紧框尺 ⊆ 大尺子、三把尺子与训练图零交集。"""
    # 行键 (image, role, text) 在同图内可能重复（同字面标签出现多次），故不能断言唯一；
    # 改成按「(key, bbox_tight)」配对：clean 的 bbox_tight 必须逐条等于 tight 的 bbox，
    # 且配对后标签逐条相同 —— 这才证明两把尺子是同一批行的同标签换口径。
    cs = sorted(clean, key=lambda r: (r["image"], r["role"], r["text"], tuple(r["bbox_tight"])))
    ts = sorted(tight, key=lambda r: (r["image"], r["role"], r["text"], tuple(r["bbox"])))
    assert len(cs) == len(ts), "两把尺子行数不同"
    assert all((a["image"], a["role"], a["text"]) == (b["image"], b["role"], b["text"])
               for a, b in zip(cs, ts)), "配对后标签不一致"
    assert all(tuple(a["bbox_tight"]) == tuple(b["bbox"]) for a, b in zip(cs, ts)), \
        "配对后紧框几何不一致"
    ndiff = sum(1 for a, b in zip(cs, ts) if tuple(a["bbox"]) != tuple(b["bbox"]))
    print("\n不变量：同标签配对 %d 行成功；其中 %d 行紧框≠外扩框（%d 行两者相同）"
          % (len(cs), ndiff, len(cs) - ndiff))
    s = {(r["image"], r["role"], r["text"], tuple(r["bbox"])) for r in tight}
    b = {(r["image"], r["role"], r["text"], tuple(r["bbox"])) for r in big}
    print("\n不变量：clean 的 bbox_tight == tight 的 bbox 逐条成立；"
          "tight ⊆ big: %s（差 %d 行 = %.1f%%）"
          % (s <= b, len(s - b), 100.0 * len(s - b) / max(1, len(s))))
    print("   （两源非嵌套：已验证/复读版与官方原始 dist<=2 版在少数行上标签或框有差异，"
          "互为交叉验证 ⇒ 两把都留）")
    real_img = {os.path.basename(r["image"]) for r in load_jsonl(EVAL_MAN)
                if r.get("split") == "real"}
    for nm, rows in (("clean", clean), ("tight", tight), ("big", big)):
        ov = {os.path.basename(r["image"]) for r in rows} & real_img
        print("   %-6s 与 32 张真实图交集: %d" % (nm, len(ov)))
        assert not ov, "%s 与训练图重叠" % nm


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    clean, allrows = build_small()
    tight = build_clean_tight(clean)
    big = build_big()
    cross_check(clean, tight, clean, big)
    report("ruler_clean（交付口径主尺）", clean)
    report("ruler_clean_tight（同标签、紧框口径）", tight)
    report("ruler_all（交付池全量，含 29 张训过的图）", allrows)
    report("ruler_big（紧框、dist<=2、n 最大）", big)
    print("\n喂法：cd eda-pcb; python -X utf8 -m textrec.eval_real "
          "--ckpt weights\\textrec_best.pt --manifest %s --by-len --no-ov "
          "--dump %s" % (OUT / "ruler_clean.jsonl", OUT / "baseline_errors.jsonl"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
