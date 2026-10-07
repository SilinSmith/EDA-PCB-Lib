# -*- coding: utf-8 -*-
"""可视化前的探针：确认 GT 图 / 预测图 / stage1 图像尺寸三者一致，且 PIL 可用。

用法（workdir = Y:\\workspace\\EDA\\eda-pcb 或 EDA）：
    python work\\tools\\diag\\_vizprobe.py [case ...]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))

from schematic_parser.data import DEFAULT_DATA_ROOT, SPLITS, discover, find_case  # noqa: E402

RUN = ROOT / "work" / "_submit_out220r36" / "runs"
IN = ROOT / "work" / "_submit_in220"

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception as exc:                                        # noqa: BLE001
    PIL_OK = False
    print("PIL 不可用：%s" % exc)

FONT_CAND = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
             r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\arial.ttf"]


def main():
    cases = sys.argv[1:] or ["1", "0001", "8", "0048"]
    print("PIL = %s" % PIL_OK)
    for f in FONT_CAND:
        print("  font %-32s %s" % (f, Path(f).exists()))
    print("DEFAULT_DATA_ROOT = %s  exists=%s" % (DEFAULT_DATA_ROOT, Path(DEFAULT_DATA_ROOT).is_dir()))
    allc = discover(DEFAULT_DATA_ROOT, SPLITS)
    print("数据集 case 数 = %d；前 3 = %s" % (len(allc), [c.key for c in allc[:3]]))
    for cid in cases:
        c = find_case(cid, DEFAULT_DATA_ROOT)
        s1p = RUN / "stage1" / ("%s.stage1.json" % cid)
        inp = IN / ("%s.png" % cid)
        print("\n== %s" % cid)
        print("   GT  case  = %s" % (c.key if c else None))
        if c:
            gp = Path(c.png_path)
            print("   GT  png   = %s  %s" % (gp.name, (gp.stat().st_size if gp.exists() else "缺")))
            print("   GT  json  = %s" % Path(c.json_path).name)
        print("   预测输入  = %s  %s" % (inp.name, (inp.stat().st_size if inp.exists() else "缺")))
        if PIL_OK and c and Path(c.png_path).exists() and inp.exists():
            a = Image.open(c.png_path)
            b = Image.open(inp)
            print("   PIL 尺寸  GT=%s  pred=%s  相同=%s" % (a.size, b.size, a.size == b.size))
        if s1p.exists():
            d = json.loads(s1p.read_text(encoding="utf-8"))
            im = d.get("image") or {}
            print("   stage1    image=%s  texts=%d comps=%d pins_keys=%d"
                  % ({k: im.get(k) for k in ("w", "h")}, len(d.get("texts") or []),
                     len(d.get("components") or []), len(d.get("pins") or {})))
            print("   stage1    image.path=%s" % im.get("path"))
        else:
            print("   stage1    缺 %s" % s1p)


if __name__ == "__main__":
    main()
