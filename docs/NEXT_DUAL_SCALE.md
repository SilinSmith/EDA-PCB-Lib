# 下一轮第一件事：真双尺度（元件框 1600 / 导线几何 2560）

## 为什么是它

§5.280 实测（官方 10、`--no-vlm`、基线 `ps0` = 80.2976）：把整条管线换到 `-imgsz 2560` 后
`pin` tp 94→**97**、`netLine` tp 385→**390**，但 `component` tp 233→**228**，净 −0.06。
降 `SUBMISSION_CONF`（0.25→0.15→0.10）只捞回 1 个元件（228→229）⇒ **掉召回是尺度问题，不是阈值问题**。
`wires.extract_wires` 本来按图像尺度、以元件框作 `occlude` 独立提线 ⇒ 两件事可以解耦。

## 改法（三步，全部默认关）

1. `stage1/pipeline.py:418` 的 `d.detect(...)` 保持 1600，产出 `comps/hyps/cboxes`（召回不变）。
2. 在 `stage1/pipeline.py:514` 的 `wires_mod.extract_wires(...)` 之前，用 `STAGE1_DUAL_SCALE`（缺省 0，
   例如 2560）把图放大到该尺度、以 1600 的 `cboxes * (s/1600)` 作 `occlude` 再提一次线；
   取回的 `segments/endpoints/junctions` 坐标 **除以 s** 缩回原尺度后替换。
3. 电路语义（pin 生成、组网）不动，只换几何来源。

## 验收

- `--cases 1..10 --engine protocol2 --no-vlm`，与 `ps0` 比：目标 `component tp 233 不变`、
  `pin ≥ 97`、`netLine ≥ 390`；`work/_axes.py --a ps0 --b <新tag>`。
- 若 `component` 也掉，说明框被高分辨率线结果反噬，回到「只替换 `segments`、不替换 `endpoints`」再试一次。
- 成立才写台账 §5.281；不成立按判负记录，开关留 0。

## 之后：stage3 扫线

先量官方 10 里「GT 有、我们没画」的 22 段（`work/_lineloss.py --out <tag>` 的 `absent` 类）：
是被 `occlude` 的框遮住、还是被阈值/降采样吃掉（`stage1/wires.py` 的 `DEFAULT_MAX_PIXELS` 与二值化阈值），
再决定动哪个参数——不要凭感觉调阈值。
