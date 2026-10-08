# -*- coding: utf-8 -*-
"""隔离 A/B：对同一份 stage2 预测跑 VLM 补强的几种开关组合，直接打分对比。

    python work\\tools\\diag\\_vlmab2.py --in work\\stage2_v17 --out work\\stage2_v18
    python work\\tools\\diag\\_vlmab2.py --arms value,all,none
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

from schematic_parser.data import discover  # noqa: E402
from vlm import enrich as vlm_enrich  # noqa: E402

ARMS = {
    "none":  dict(designator=False, value=False),
    "desig": dict(designator=True, value=False),
    "pins":  dict(designator=True, value=False),
    "value": dict(designator=False, value=True),
    "all":   dict(designator=True, value=True),
    # 位号读取扩到**所有带位号的元件**（r/c/l/d/led/switch/…），不只 IC
    "desigall": dict(designator=True, desig_all=True, value=False),
    # 只量「值」的干净对照：两边都跳过引脚网格，只有 value 开关不同
    "valbase": dict(designator=False, value=False, min_pins=10 ** 9),
    "valonly": dict(designator=False, value=True, min_pins=10 ** 9),
    # 用 `_vlmval3.py` 里读数最好的那组参数（margin 80 / 放大 4×）再试一次值覆盖：
    # `_vlmval3` 同参数下 VLM 0.7050 vs 管线 0.5025，而 `valonly`（margin 90 / 不放大）却是负的。
    "valexp": dict(designator=False, value=True, min_pins=10 ** 9,
                   value_margin=80.0, value_scale=4),
    # **带「位号一致」门控 + 放大 4× + margin 80**：离线复算 0.7950 vs 管线 0.5025
    "valgate": dict(designator=False, value=True, min_pins=10 ** 9,
                    value_margin=80.0, value_scale=4, value_key_gate=True),
    # 同样门控，但保持旧参数（margin 90 / 不放大），用来分离「门控」与「放大」的贡献
    "valgate0": dict(designator=False, value=True, min_pins=10 ** 9,
                     value_margin=90.0, value_scale=1, value_key_gate=True),
    # 只开「读 IC 器件型号 -> 写 Name」：官方 PDF 第 8 页 ic 比 Name，
    # box 现在 996 个里只有 2 个 tp（我们的 Name 全是引脚名/器件值）。
    "nameonly": dict(designator=False, value=False, name=True, min_pins=10 ** 9),
    # **完整臂**：位号 + 引脚 + 带门控的值（margin 80 / 4×），对照 b294 的 desig+引脚 24.5211
    "allgate": dict(designator=True, value=True, value_margin=80.0, value_scale=4,
                    value_key_gate=True),
    # nameonly 单独 +0.541（23.9559 -> 24.4969），allgate 单独 25.4226。
    # 这一臂把两者合起来，预期 ≈ 25.96。
    "full": dict(designator=True, value=True, value_margin=80.0, value_scale=4,
                 value_key_gate=True, name=True),
    # full + 「值通道顺手捡位号」（只对 pNNN 兜底键生效，零额外 VLM 调用）
    "fullk": dict(designator=True, value=True, value_margin=80.0, value_scale=4,
                  value_key_gate=True, name=True, value_use_key=True),
    # **兜底键补读位号**：只对键是 `pNNN` 的元件跑 VLM 位号读取（2601 个，其中
    # 92% 在 10px 内就有文字框却拿不到位号）。相对 fullk 只多这一项。
    "fullfb": dict(designator=True, value=True, value_margin=80.0, value_scale=4,
                   value_key_gate=True, name=True, value_use_key=True,
                   desig_fallback=True),
    # 只量「兜底键补读」的干净对照：两臂都跳过引脚/值/型号，只有 desig_fallback 不同。
    "fb1": dict(designator=True, value=False, min_pins=10 ** 9),
    "fb2": dict(designator=True, value=False, min_pins=10 ** 9, desig_fallback=True),
    # 「所有元件都重读一遍位号」（不只是兜底键）+ 分立元件专用提示词。
    # 目标是 W 桶那 768 个「有键但读错了」的（`0001` 的 `C6` 被 CRNN 读成 `C1`）。
    "dall": dict(designator=True, value=False, min_pins=10 ** 9, desig_all=True),
    "dallfb": dict(designator=True, value=False, min_pins=10 ** 9,
                   desig_all=True, desig_fallback=True),
    # ★ 2026-10 补：**引脚通道**。原来那条 `"pins"` 臂只设了 designator、`pin` 根本没开
    #   ⇒ 「量引脚通道」从来没被真正执行过（等于拿基线比基线）。
    #   `enrich_doc` 里引脚通道的门是 `opts.get("pin", PIN_CHANNEL)`；merge 门是
    #   `opts.get("pin_merge")`（`eda-pcb\vlm\enrich.py:689` / `:717`）。
    #   merge=True：号用我们自己的（几何/CRNN），只把模型读到的名并进来；
    #   merge=False：VLM 读到的号直接替换键（旧实测三网格全负）。
    "pinmerge": dict(pin=True, pin_merge=True, designator=False, value=False),
    "pinrepl":  dict(pin=True, pin_merge=False, designator=False, value=False),
    # 推荐组合 (a)：位号通道 + 引脚 merge（位号那条是唯一被证明大赚的 VLM 通道）
    "pinmrgd":  dict(pin=True, pin_merge=True, designator=True, value=False),
    # 位号通道单独一臂，作为 pinmrgd 的对照
    "desigonly": dict(designator=True, value=False, pin=False),
    # ★ 分离 `VLM_PIN_REBUILD`：替换引脚键后**不**重写 nets[*].hyperGraph（旧行为）
    "pinrepl0": dict(pin=True, pin_merge=False, designator=False, value=False,
                     pin_rebuild=False),
    # 推荐组合 (a) 的完整版：位号 + 引脚替换（含 hyperGraph 重建）
    "pinfix":   dict(pin=True, pin_merge=False, designator=True, value=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default=str(ROOT / "work" / "stage2_v17"))
    ap.add_argument("--out", dest="outdir", default="")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", default="", help="逗号分隔的 case 白名单（空=全部）")
    ap.add_argument("--no-desig-pins", action="store_true",
                    help="value 臂里连引脚网格也一起跑（默认 value 臂只读值）")
    a = ap.parse_args()

    png_of = {c.case_id: c.png_path for c in discover() if c.png_path}
    for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
        opts = dict(ARMS[arm])
        if arm == "value":
            opts["designator"] = False
            opts["value"] = True
            opts["min_pins"] = 10 ** 9          # 跳过引脚网格
        out = Path(a.outdir) if a.outdir else (ROOT / "work" / ("_vlm_" + arm))
        t0 = time.time()
        st = vlm_enrich.enrich_dir(a.indir, out, png_of, opts=opts,
                                   workers=a.workers, limit=a.limit,
                                   cases=[c.strip() for c in a.cases.split(",")
                                          if c.strip()])
        dt = time.time() - t0
        print("[%s] %s  %.0fs  %s" % (arm, out, dt, st))
        r = subprocess.run([sys.executable,
                            str(ROOT / "work" / "tools" / "eval" / "stage2_eval.py"),
                            str(out), "--quiet"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        print("\n".join(r.stdout.strip().splitlines()[:6]))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
