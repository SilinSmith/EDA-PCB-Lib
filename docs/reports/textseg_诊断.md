# 文字框切分诊断（第二阶段 OCR）—— 结论与证据

## 一句话

**CRNN 识别器不是瓶颈，文字框才是。** 同一批文字，用真值框原样读是 **0.83**，
用我们切出来的框原样读只有 **0.15**，把框外扩一点能救到 **0.47** —— 然后就见顶了。
所以「切出紧框再外扩」这条路已经走到头，必须直接切出**单行、干净**的框。

## 口径

真值 = `Dataset300我不知道干嘛用的先放置\<id>\<id>-<source>_CVJsonStd.json` 里的
`texts[T_n] = {tid, state, category, categoryId, content, details, location{x1,y1,x2,y2}}`。
`location` 框与图像像素严丝合缝，`content` 就是画在纸上的字
（证据图 `work\preview\cvjson_align_0059.png`）。

**注意：官方 220 张训练图里没有 CVJsonStd 真值文字框**；同时有 PNG 和 CVJsonStd 的只有 29 个 case：
`0059 0061 0066 0092 0095 0112 0113 0114 0115 0122 0123 0124 0125 0140 0141 0146 0147 0148 0149 0170 0171 0176 0177 0178 0179 0182 0183 0184 0185`。

评测脚本 `Y:\workspace\EDA\work\_textseg_common.py`（唯一公平口径：按角色分组批量读，
同时报「命中集 exact」「端到端 exact」「**公共子集(∩old) exact**」）。

## 决定性数字（10 张图 / 1287 个真值框）

| 臂 | 命中 | 命中集 exact | 端到端 | 公共(∩old) |
|---|---|---|---|---|
| **GT_raw（真值紧框原样读）** | 1287 | **0.8430** | 0.8430 | 1049/1248 = **0.8405** |
| GT_pad（真值紧框 + `pad_box`） | 1287 | 0.1958 | 0.1958 | 233/1248 = 0.1867 |
| old（`cvparts\textbox.py`） | 1248 | 0.0954 | 0.0925 | 0.0954 |
| old_pad | 1248 | 0.4022 | 0.3901 | 0.4022 |
| new textseg `H_CFG` | 1152 | 0.1259 | 0.1127 | 0.1261 |
| new textseg `H_CFG` + pad | 1152 | 0.4288 | 0.3838 | 0.4321 |

在**同一批 1103 个配对**上直接读三份（`work\_boxgeom.py`）：
**GT原样 0.8314 / pad(det) 0.4760 / det原样 0.1451**。

## 固定像素外扩扫描（同一批 1103 个配对）

| pad | 全部 exact | 干净子集 exact（IoU>=0.40 且 高比 0.5~1.5，n=326） |
|---|---|---|
| +3 | 0.4705 | **0.7485** |
| +4 | 0.4370 | 0.6503 |
| +5 | 0.3781 | 0.5153 |
| +6 | 0.2303 | 0.2975 |
| +8 | 0.0671 | 0.1166 |
| +12 | 0.0281 | 0.0429 |
| +16 | 0.0136 | 0.0031 |
| +20 | 0.0100 | 0.0031 |

- 外扩在 **+3px 见顶（0.4705）**，与现行 `pad_box = max(2, 0.3*h)` 的整体 0.4760 基本持平
  ⇒ **外扩这条杠杆已经用尽**，再加就是把邻行/邻列的字拉进裁片（CTC 多读字符 ⇒ exact 归零）。
- 但**干净子集 +3px 能到 0.7485** ⇒ **损失集中在「脏框」上：干净框只占 326/1103 = 29.6%**。

## 检出框几何（`work\_boxgeom.py`）

配置 `ink_mode="fixed", ink=225, min_dim=2, min_area=3, line_len=15, resid_min_h=4, resid_min_dim=1`。

```
designator 'R1'  gt=(168,38,204,67)  36x29  det=(174,44,199,62)  25x18  IoU 0.43
pin_name   'VDD' gt=(313,70,370,95)  57x25  det=(318,74,367,92)  49x18  IoU 0.62
pin_number 'C1'  gt=(230,96,267,126) 37x30  det=(236,103,261,122) 25x19 IoU 0.43
pin_name   'VC'  gt=(313,168,354,197) 41x29 det=(318,174,350,193) 32x19 IoU 0.51
pin_name   'DO'  gt=(365,277,408,305) 43x28 det=(371,283,404,302) 33x19 IoU 0.52
  宽 det/gt  p10 0.50 p25 0.67 p50 0.79 p75 0.87 p90 1.21
  高 det/gt  p10 0.47 p25 0.50 p50 0.62 p75 1.93 p90 6.40
  中心 dx/gt宽 p50 0.02   dy/gt高 p50 0.06（近乎同心）
  IoU p10 0.11 p25 0.21 p50 0.37 p75 0.42 p90 0.49
  宽比<0.8 占 0.528（<0.6 占 0.175）；高比<0.8 占 0.653（<0.6 占 0.482，>1.3 占 0.276）
pad(det) 与 GT 的 IoU: p10 0.026 p50 0.646 p90 0.822
```

两种失败模式**并存**：

1. **偏小** —— 高比中位 0.62，48% 低于 0.6（框只裹住笔画核心，真值框自带约 5px 边距）；
2. **吞了邻行 / 并与邻列** —— 高比 >1.3 占 **27.6%**，p90 高达 **6.40**（把好几行并成一个框）。

`pad_box` **救①杀②**，净效果就是卡在 0.47。

## 一个具体的根因（0113 召回暴跌）

`_probe3.py 0113 299,89,312,102`：

```
分量15: box=(115,87,433,228) 318x141 area=4942 has_line=True 结构=True glyph_ok=False 框内像素=73
  残留碎片: (301,92) 4x7 area=13 ok=False   (306,92) 4x7 area=12 ok=False
```

引脚号 "24" 的两个数字与**整块 IC 符号（318×141，area=4942）是同一个 8 连通分量**（字贴着 IC 外框），
整块被判结构分量丢弃；回收流确实捞到两个 4×7 的碎片，但 **`h=7 < RESID_MIN_H=10` 被拒**。
把 `resid_min_h` 降到 4（即 `H_CFG`）后，0113 的命中从 149 回到 1152 那一档。

## 已建/已改的文件

- `Y:\workspace\EDA\eda-pcb\cvparts\textseg.py`（新建；旧的 `cvparts\textbox.py` 不动，stage1 导线掩膜继续用它）
  - 判别：**结构分量 = 「含有长直线像素」且「外接框最长边 >= `STRUCT_DIM`」的原始墨迹 8 连通分量，整块丢掉**；
    回收流把残留的「高个子紧凑块」（`h>=RESID_MIN_H`、短边>=`RESID_MIN_DIM`）捞回来，再 `_stitch` 缝被切断的同一个字。
  - 常量：`INK=170`、`LINE_LEN=25`、`STRUCT_DIM=36`、`RESID_MIN_H=10`、`RESID_MIN_DIM=2`、`STITCH_GAP=16`、
    `STITCH_OV=0.5`、`MIN_DIM=3`、`MIN_H,MAX_H=5,48`、`MIN_W,MAX_W=1,80`、`MIN_AREA=6`、
    `FILL_LO,FILL_HI=0.08,0.95`、`GAP_FRAC=0.45`、`V_OVERLAP=0.40`、`H_RATIO=2.2`、`MAX_LINE_W=400.0`。
  - 自适应墨迹阈值：`INK_MODE="max"`、`INK_CEIL=235`、`otsu_thr(gray)`、
    `ink_mask(bgr, ink=INK, mode=INK_MODE)` 支持 `fixed` / `max`(= `max(ink, Otsu)`) / `otsu`，统一 `thr=min(thr, INK_CEIL)`；
    `glyphs(..., ink_mode=INK_MODE)` 透传。
  - `detect_boxes(bgr, **kw)` 先把 `GROUP_KEYS = ("gap_frac","v_overlap","h_ratio","max_w")` 分给 `group_lines`，其余给 `glyphs`。
  - 推荐配置常量 `H_CFG = dict(ink_mode="max", min_dim=2, min_area=3, line_len=15, resid_min_h=4, resid_min_dim=1)`。
  - `python eda-pcb\cvparts\textseg.py` → 自检 7 项全过。
- `Y:\workspace\EDA\eda-pcb\stage2\text.py`：`PAD_MIN=2.0`、`PAD_FRAC=0.30`、`pad_box(box, pad_min, pad_frac)`
  （`p = max(pad_min, pad_frac*h)`，四边各外扩）；`TextEngine.read()` 只对裁剪外扩，`reads[i]["bbox"]` 仍是原框。
- 诊断脚本（都在 `Y:\workspace\EDA\work\`）：`_textseg_eval.py`、`_textseg_sweep.py`、`_textseg_common.py`、
  `_textseg_group.py`、`_seg_cmp.py`（四联对照图，`--crop x1,y1,x2,y2 --zoom N`）、`_seg_miss.py`、
  `_crop_cmp.py`（裁片对照表，`--n` `--h`）、`_boxgeom.py`、`_probe0095.py`、`_probe2.py`、`_probe3.py`。
- 对照图：`work\preview\segcmp_0059.png`、`segcmp_0095.png`、`segcmp_0113.png`、
  `segcmp_0113_crop_150_85.png`、`cropcmp_0113.png`。

## 教训

- **不要用「命中集 exact」比较不同召回率的检出器** —— 漏检多的臂会靠「排除最难的图」刷出高分
  （曾有一个臂刷到 0.5369，看公共子集就露馅）。必须比**公共子集**。
- 「检出框再外扩」是治标：它把偏小的检出框救到 0.47，却把本来正确的真值框毁到 0.196。
- `_textseg_eval.py` 早期把两个 `--crop` 的产物写成同一个文件名互相覆盖，已改成带坐标后缀。

## 下一步

1. 把框做成**单行、紧贴**：①用全图规律（并排同型元件簇的字符分布、符号位置、导线间距）推断正确的文字行几何；
   ②修「字贴 IC 外框被整块吞掉」的分量切分。
2. 目标是把「干净框」占比从 **29.6%** 抬上去 —— 干净框 +3px 已经是 **0.7485**，接近真值框的 **0.8314**。
3. 尚未做：并排同型元件簇的全图专门优化、典型问题图例套图、整理项目文件夹。
