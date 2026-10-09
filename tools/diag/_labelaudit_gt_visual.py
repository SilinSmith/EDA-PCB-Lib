"""肉眼复核工具：把可疑 GT 框渲染成 ASCII 字符网格（或放大裁图）。

**为什么必须有这个**：报告里凡是要下"空框/错标"结论的框，都必须先肉眼看一眼。
`_gndbox.py` 的 <128 阈值就是靠这步才发现是假警报的。ASCII 比裁图好：约 20px 的框
放大到 110px 仍然看不清，而 ASCII 一格一像素级判读、且几乎不耗资源。

用法：
  $env:PYTHONIOENCODING="utf-8"; cd Y:\workspace\EDA
  python work\tools\diag\_labelaudit_gt_visual.py empty 12      # 最空的 12 个框
  python work\tools\diag\_labelaudit_gt_visual.py type r 8      # r 类里 ink 最低的 8 个
  python work\tools\diag\_labelaudit_gt_visual.py case 2 6      # case=2 里最可疑的 6 个
  python work\tools\diag\_labelaudit_gt_visual.py png empty 6   # 同时存放大裁图到 %TEMP%\labelaudit_crops
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _labelaudit_gt as L  # noqa: E402

PAD = 6      # 框外再留几像素，方便看"旁边有没有导线/文字"
SCALE = 1    # 每格像素数


def _img(path):
    return np.asarray(Image.open(path).convert("L"))


def render(r, cache, cols=None):
    """把一个框渲染成 ASCII。'#'=墨迹(<INK_THR)，'o'=文字行框内，'+-|'=GT 框边。"""
    key = (r["split"], r["case"])
    if key not in cache:
        cache[key] = _img(_png_of(r))
    gray = cache[key]
    h, w = gray.shape
    x1, y1, x2, y2 = r["box"]
    a, b = max(0, int(y1) - PAD), min(h, int(round(y2)) + PAD)
    cc, dd = max(0, int(x1) - PAD), min(w, int(round(x2)) + PAD)
    sub = gray[a:b, cc:dd]
    ink = sub < L.INK_THR
    chars = np.where(ink, "#", ".")
    # GT 框边
    iy1, iy2 = int(y1) - a, min(sub.shape[0] - 1, int(round(y2)) - a - 1)
    ix1, ix2 = int(x1) - cc, min(sub.shape[1] - 1, int(round(x2)) - cc - 1)
    for i in range(max(0, iy1), min(sub.shape[0], iy2 + 1)):
        if 0 <= ix1 < chars.shape[1]:
            chars[i, ix1] = "|"
        if 0 <= ix2 < chars.shape[1]:
            chars[i, ix2] = "|"
    for j in range(max(0, ix1), min(sub.shape[1], ix2 + 1)):
        if 0 <= iy1 < chars.shape[0]:
            chars[iy1, j] = "-"
        if 0 <= iy2 < chars.shape[0]:
            chars[iy2, j] = "-"
    head = ("%s/%s key=%s type=%s box=%.0f,%.0f,%.0f,%.0f (%.0fx%.0f) "
            "ink=%.3f ink25=%.3f ink_same=%.3f dom_in=%.2f off_cc=%.1f "
            "nglyph=%d nrow=%d txt=%.2f"
            % (r["split"], r["case"], r["key"], r["type"], x1, y1, x2, y2,
               r["w"], r["h"], r["ink"], r["ink25"], r["ink_same"], r["dom_in"],
               r["off_cc"], r["nglyph"], r["nrow"], r["txt"]))
    body = "\n".join("".join(row) for row in chars)
    return head, body


def _png_of(r):
    import schematic_parser.data as S
    for c in S.discover():
        if c.split == r["split"] and c.case_id == r["case"]:
            return str(c.png_path)
    raise KeyError(r["case"])


def save_crop(r, outdir):
    """放大裁图（含外扩 25px），GT 框描红，存 png 供 read_image 肉眼复核。"""
    from PIL import ImageDraw
    gray = _img(_png_of(r))
    h, w = gray.shape
    x1, y1, x2, y2 = r["box"]
    g = 25
    a, b = max(0, int(y1) - g), min(h, int(round(y2)) + g)
    cc, dd = max(0, int(x1) - g), min(w, int(round(x2)) + g)
    sub = gray[a:b, cc:dd]
    k = 6
    im = Image.fromarray(sub).resize(((dd - cc) * k, (b - a) * k),
                                     Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([(x1 - cc) * k, (y1 - a) * k, (x2 - cc) * k - 1, (y2 - a) * k - 1],
                outline=(255, 0, 0))
    fn = os.path.join(outdir, "%s_%s_%s.png" % (r["split"], r["case"], r["key"]))
    im.save(fn)
    return fn


def main() -> None:
    args = sys.argv[1:]
    want_png = False
    if args and args[0] == "png":
        want_png = True
        args = args[1:]
    mode = args[0] if args else "empty"
    n = int(args[2]) if len(args) > 2 else 10
    recs, _ = L.load_or_scan()
    if mode == "empty":
        cand = sorted(recs, key=lambda r: (r["ink"], r["ink_same"]))
    elif mode == "type":
        t = args[1]
        cand = sorted([r for r in recs if r["type"] == t],
                      key=lambda r: (r["ink"], r["ink_same"]))
    elif mode == "case":
        c = args[1]
        cand = sorted([r for r in recs if r["case"] == c],
                      key=lambda r: (r["ink"], r["ink_same"]))
    elif mode == "off":
        cand = sorted([r for r in recs if r["off_cc"] == r["off_cc"]],
                      key=lambda r: -r["off_cc"])
    elif mode == "domout":
        cand = sorted([r for r in recs if r["dom_in"] == r["dom_in"]],
                      key=lambda r: r["dom_in"])
    else:
        raise SystemExit("mode? empty|type|case|off|domout")
    outdir = os.path.join(os.environ.get("TEMP", "."), "labelaudit_crops")
    if want_png:
        os.makedirs(outdir, exist_ok=True)
    cache = {}
    for r in cand[:n]:
        head, body = render(r, cache)
        print("=" * 78)
        print(head)
        print(body)
        if want_png:
            print("crop:", save_crop(r, outdir))
    print("=" * 78)


if __name__ == "__main__":
    main()
