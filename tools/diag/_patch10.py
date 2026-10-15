"""把「第十批」的结论按字面替换写进 _wfbullets.py（work\\flow\\ 与 eda-pcb\\tools\\diag\\ 两份）。

本脚本自身的长文本用三引号双引号字符串，所以插进去的文本里
**不能出现连续三个双引号**；列表元素本身用双引号是没问题的。

用法：
  python work\\tools\\diag\\_patch10.py                 # dry-run
  python work\\tools\\diag\\_patch10.py --write
  python work\\tools\\diag\\_patch10.py --write --sync   # 再覆盖 eda-pcb 那份
"""
import argparse
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent            # work\
EDA = ROOT.parent / "eda-pcb"
LIVE = ROOT / "flow" / "_wfbullets.py"
SNAP = EDA / "tools" / "diag" / "_wfbullets.py"

PATCHES = []

# ---- 1. ev_pin 结尾追加本轮 bullets ----------------------------------------
PATCHES.append((
    '],\n"ev_net": [',
    '\n'.join([
        ' "★ 官方档与派生档的**引脚键形状不同**（`_gtpin10.py`）：10GT 的 **136 根 GT IC 引脚键 100% 是纯数字**（`pin_7`/`pin_8`…）、`pinname` 全是**真信号名** ⇒ 官方档的引脚号**是可赢的**；「`pin_N` 是内部计数器、不可赢」只对派生档成立",',
        ' "★ **引脚通道三分账**（官方 9 张交付产物，`_pinledger.py`）：GT 的 IC(box) 引脚 **120 根我们只发 75**；缺的 45 = **号已经读对、但名字没读到 ⇒ `stage2/assemble.py:759` 的 `BOX_PIN_NEED_NAME`（默认开）整条不发 23（51%）** + 号读错 22 + 附近 45px 没槽位 6",',
        ' "★ **按位置对账「读法」，不要按号**（`_ocrstat.py`）：45px 内有槽位 **135/136 = 99%** ⇒ 几何不缺；号 完全对 86 / 整串读错 26 / 空 14 / 截断 7；名 完全对 53 / 空 40 / 整串读错 25 / 截断 15（按号查只能查到 94/136，会把「读错」混成「没槽位」）",',
        ' "★ **截断是 CRNN 少读字符，不是检测切字**：`39→3` 的框宽 **24px**、`113→13` **34px**、`115→15` **35px**（都够装），框里确实有两个/三个字形；另一小撮 `GND→D`(10px) / `OUT→O`(10) / `IS→S`(8) / `LED_P→D`(8) 才是检测框只框住一个字符",',
        ' "★★ **「读法全对」的写实上界 = +4.60 分**（`_ocroracle.py`，官方 9 张）：`58.6304 → 61.2888（名全对）→ 63.2307（号+名全对）`，引脚 0.3920→**0.5515**、tp 引脚 37→52 ⇒ **动权重/重训之前先看这个数**，识别这条路全部能值的就是它",',
        ' "★ 识别器的瓶颈在**裁剪分布**不在权重：真值紧框 **0.843** / stage1 紧框 **0.197** / 外扩 +3 → **0.654**（`vlm/client.py:4`、`_flowpatch.py:20`、`cropfix.py:10`）；训练集 38.85 万条 = 合成 **381876** + 真实 **6642**（`split=real` 只验证不训练，混训要显式 `--include-real`）",',
        ' "⚠ **`VLM_PIN_CHANNEL` 默认关**，且**替换模式**三种网格全判负（不加 39.06 / 8×4 35.47 / 4×2 36.41 / 2×1 35.47）；但 `_pinsweep` 在**正确的引脚点**上读号+名 **0.826**（`inner=160`），而 **merge 模式从未真正 A/B 过**（`_vlmab2.py` 的 `pins` 臂 `pin=True` 根本没设 = 空臂）",',
        ' "⛔ **「把 box 引脚点吸附回元件框边」在官方档也判负**（`_pinsnap.py --split 10GT`：T0 原样 28 → T1/T2/T4 = 0、T3 外扩 8 才 15、T5 最优 (1,1) = 29）；派生档同结论（T0 178 / T1 8 / T2 5 / T5 (0,0) = 178）",',
        ' "⛔ **「GT 点 = 引脚名/号文字框心」「GT 点在 bbox 边线上」都被证伪**（`_pintext.py`：号框心 7.4% / 名框心 **0.0%** / 中点 1.6%；落在边线 **0%**、到最近边线 p50 **12px**）。真实形态是**框边往外每边一个固定桩距**（case 1 `U1` 左 x0−15 / 右 x1+20），而我们的点只差 **8px** 量级 ⇒ 点不是主因（IC 预测引脚 75 根里点已对 54 = 72%）",',
        '],\n"ev_net": [',
    ]),
))

# ---- 2. SUB 里 ev_pin 的一行摘要 -------------------------------------------
PATCHES.append((
    '"ev_pin": "引脚：pin+netH = 60% 权重同一根因；上帝键证明不是键的问题 → 修名字与产出率",',
    '"ev_pin": "引脚：pin+netH = 60% 权重同一根因；★ 三分账 = 不产出 23 / 读错 22 / 没检测 6；读法全对的上界 +4.60",',
))


def apply(path, write):
    txt = path.read_text(encoding="utf-8")
    orig = txt
    hits = []
    for i, (old, new) in enumerate(PATCHES, 1):
        n = txt.count(old)
        hits.append(n)
        if n == 1:
            txt = txt.replace(old, new)
    print("%s  命中 %s" % (path, hits))
    if all(h == 1 for h in hits):
        if write:
            if orig != txt:
                shutil.copy2(path, str(path) + ".bak10")
                path.write_text(txt, encoding="utf-8")
                print("  已写入（备份 .bak10）")
            else:
                print("  内容无变化")
        else:
            print("  dry-run：未写入")
        return True
    print("  ！有不命中/重复，跳过")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--sync", action="store_true", help="把 work 那份覆盖到 eda-pcb")
    a = ap.parse_args()
    ok = apply(LIVE, a.write)
    if a.sync:
        if not ok:
            print("work 那份没改成功，不 sync")
            return 1
        shutil.copy2(LIVE, SNAP)
        print("已同步 -> %s" % SNAP)
    else:
        if SNAP.exists():
            apply(SNAP, a.write)
    return 0


if __name__ == "__main__":
    sys.exit(main())
