# -*- coding: utf-8 -*-
"""把 owner 7 条判断对应的证据图拼成一张总览（每格带标题）。"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")
EV = Path(r"Y:\workspace\EDA\work\preview")
OUT = Path(r"Y:\workspace\EDA\work\preview")

ROWS = [
    ("证据1 fn_r（8格，全部来自 200_train_cases-0050__full）"
     " -> 红框是 r 8x20，落在 R14..R25 电阻阵列的锯齿体空白段上；模型预测 r 但角点差 >20px",
     "fn_r.png"),
    ("证据2 原图区域 (150,2500)-(900,2620) 1.6x：R14 1.13k / R15 1.87k ... 0603 / 1%  ==> 这批器件是电阻（有阻值/封装/精度），不是电感",
     "region_200_train_cases-0050__full_150_2500.png"),
    ("证据3 fp_r：模型把同一颗电阻重复检测两次（19x7 与 14x6），GT 只有一个 (20,8) -> 贪心配对领走一个，另一个记 FP",
     "ev_FN_r_DUP.png"),
    ("证据4 fn_r 的 CLASSCONF 格 0011__full：红色 GT 8x20 落在电感（zigzag + 位号 + 100k 阻值?）笔画上，模型预测成 other",
     "ev_FN_r_CLASSCONF.png"),
    ("证据5 原图区域 (250,250)-(560,420) 3x：电阻正身是小矩形内嵌 33Ω，蓝色 33Ω 是网络名；GT 的 (20,8) 框系统性地落在蓝色文字上",
     "region_200_train_cases-0190__full_250_250.png"),
    ("证据6 fn_testpoint（20格全部来自 200_train_cases-0011__*）：红框确实扣在真实 TP 焊盘（蓝色同心圆环）上 -> 真漏检",
     "ev_FN_testpoint_DEADGT.png"),
    ("证据7 fn_net_input：0011__full 红框扣在普通文字 RESET/VOUT 上；0023__full 扣在纯黑噪声块上；0027__* 扣在反相器输入引脚箭头上",
     "ev_FN_net_input_DEADGT.png"),
    ("证据8 反证：0050__t00_00 的 2 个 r GT 框准确落在两条竖直锯齿电阻笔画上，模型预测 r 20x8 角点<=20 -> 标注和模型都对，是匹配判据把它记成 FN/FP",
     "verify_0050t00_boxes.png"),
]


def main():
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    except Exception:
        font = ImageFont.load_default()
    tiles = []
    for cap, name in ROWS:
        p = EV / name
        if not p.exists():
            print("[skip missing]", name)
            continue
        with Image.open(p) as im:
            im = im.convert("RGB")
            w = 1180
            h = max(1, int(im.height * w / im.width))
            tiles.append((cap, im.resize((w, h), Image.LANCZOS)))
    head, gap = 52, 10
    total = sum(t.height + head + gap for _, t in tiles) + 20
    sheet = Image.new("RGB", (1200, total), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    y = 10
    for cap, t in tiles:
        d.text((10, y + 4), cap[:118], fill=(0, 0, 0), font=font)
        y += head
        sheet.paste(t, (10, y))
        d.rectangle((10, y, 10 + t.width, y + t.height), outline=(200, 0, 0), width=2)
        y += t.height + gap
    out = OUT / "evidence_overview.png"
    sheet.save(out)
    print("[saved]", out, sheet.size)


if __name__ == "__main__":
    raise SystemExit(main())
