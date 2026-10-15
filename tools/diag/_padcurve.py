# -*- coding: utf-8 -*-
"""§5.177 预备：把 §5.176 C 的「外扩 vs 紧框」拆成单因子扫参（零 GPU、零训练）。

背景（`tools/dev/mk_official_textset.py:342-358`）：
* `bbox_tight`(bx) = 我们检测/adapter 给出的原始框；
* `pin_name`/`pin_number` 先用 `tight_ink_box` 切彩色描边 -> `base`；
* `bbox` = `pad_box(base, pad_min=2, pad_frac=0, pad_min_y=3, pad_frac_y=0.20)` = stage2 运行时
  真正喂给识别器的裁片（交付口径）。

本脚本从 `ruler_clean.jsonl`（交付口径）**反解**出 `base`，再按 pad 缩放因子 s 生成
`px = pad_min*s`、`py = max(pad_min_y*s, pad_frac_y*s*h)` 的变体：
    s=0    -> base（无外扩；pin 角色仍是 ink 再裁后的框）
    s=0.5  -> 半个外扩
    s=1    -> 与 ruler_clean 完全一致（自检：必须复现 §5.176 的基线数）
    s=2    -> 双倍外扩
产出 `work/pool_r1/pad/ruler_ps{0,05,1,2}.jsonl`，样本与标签逐条相同（只改 bbox）。
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(r"Y:\workspace\EDA\work\pool_r1")
OUT = R / "pad"

PAD_MIN, PAD_FRAC = 2.0, 0.0
PAD_MIN_Y, PAD_FRAC_Y = 3.0, 0.20


def unpad(box):
    """pad_box 的逆：由交付框反解 base 框（返回 (base, py, px)）。

    pad_frac=0 ⇒ px 恒 = PAD_MIN = 2；py = max(3, 0.20*h)，H = h + 2*py：
      h >= 15  ⇔  py = 0.2h  ⇒  H = 1.4h   （等价 H >= 21）
      h <  15  ⇔  py = 3     ⇒  H = h + 6  （等价 H <  21）
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    H = y2 - y1
    h = H / 1.4 if H >= 21.0 else H - 6.0
    h = max(1.0, h)
    py = max(PAD_MIN_Y, PAD_FRAC_Y * h)
    px = max(PAD_MIN, PAD_FRAC * H)
    return [x1 + px, y1 + py, x2 - px, y2 - py], py, px


def repad(base, s):
    x1, y1, x2, y2 = base
    h = max(1.0, y2 - y1)
    px = PAD_MIN * s
    py = max(PAD_MIN_Y * s, PAD_FRAC_Y * s * h)
    return [round(x1 - px, 2), round(y1 - py, 2), round(x2 + px, 2), round(y2 + py, 2)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (R / "ruler_clean.jsonl").open(encoding="utf-8")]
    scales = (0.0, 0.5, 1.0, 2.0)
    files = {}
    for s in scales:
        tag = ("ps%g" % s).replace(".", "")
        files[s] = (OUT / ("ruler_%s.jsonl" % tag)).open("w", encoding="utf-8")

    ident = 0
    for r in rows:
        base, py, px = unpad(r["bbox"])
        for s in scales:
            b = repad(base, s)
            if s == 1.0 and b == [round(float(v), 2) for v in r["bbox"]]:
                ident += 1
            q = dict(r)
            q["bbox"] = b
            q["bbox_tight"] = b
            files[s].write(json.dumps(q, ensure_ascii=False) + "\n")
    for fh in files.values():
        fh.close()
    print("行 %d；s=1 与交付框逐条完全相同 %d (%.4f)" % (len(rows), ident, ident / len(rows)))
    for s in scales:
        tag = ("ps%g" % s).replace(".", "")
        print("-> %s" % (OUT / ("ruler_%s.jsonl" % tag)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
