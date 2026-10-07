# -*- coding: utf-8 -*-
"""把 `work\\_recover\\recovered_*.py` 写回源文件，并**正确地**重做「假值词表统一」。

已核验：recovered 与当前（被 'i'->'f' / 'B'->'I' 污染过的）文件在字符替换下**逐字相等**，
唯一的差异是本次要补的那一行 `from envflag import env_on`。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

R = Path(r"Y:\workspace\EDA\work\_recover")
S1 = Path(r"Y:\workspace\EDA\eda-pcb\stage1")

IMPORT_LINE = "from envflag import env_on      # 「假值词表」唯一实现，见 eda-pcb/envflag.py"

JOBS = {
    # 文件: (import 插到哪一行之后, [(旧, 新), ...])
    "wires.py": ("from pathlib import Path", [
        ('if os.environ.get("STAGE1_SPLIT_JUNCTIONS", "1") not in ("0", "false", "False"):',
         'if env_on("STAGE1_SPLIT_JUNCTIONS", "1"):'),
    ]),
    "detect.py": ("from pathlib import Path", []),
    "pipeline.py": ("from pathlib import Path", [
        ('BIGIC_ON = os.environ.get("STAGE1_BIGIC", "1") not in ("0", "off", "false")',
         'BIGIC_ON = env_on("STAGE1_BIGIC", "1")'),
    ]),
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for base, (anchor, subs) in JOBS.items():
        rec = R / ("recovered_" + base)
        dst = S1 / base
        text = rec.read_text(encoding="utf-8")

        if IMPORT_LINE not in text:
            lines = text.split("\n")
            for i, l in enumerate(lines):
                if l == anchor:
                    lines.insert(i + 1, IMPORT_LINE)
                    break
            else:
                raise SystemExit("找不到 import 锚点：%s" % base)
            text = "\n".join(lines)

        for old, new in subs:
            if old not in text:
                print("  !! %s 里没找到：%s" % (base, old[:60]))
                continue
            text = text.replace(old, new)

        bak = dst.with_suffix(dst.suffix + ".corrupt")
        if not bak.exists():
            shutil.copy2(dst, bak)
        dst.write_text(text, encoding="utf-8")
        print("restored %-12s %d 行  (备份 %s)" % (base, text.count("\n") + 1, bak.name))


if __name__ == "__main__":
    main()
