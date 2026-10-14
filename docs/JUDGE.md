# 判题机 · 本地自评工具（`schematic_parser`）

> **一句话**：把你的产物目录交给它，它按官方四项指标给分。不需要 GPU、不需要图片、不需要模型权重。
>
> ```powershell
> python -X utf8 -m schematic_parser.evaluate --pred <你的输出>\final_results --split 10GT
> ```

本文件是「怎么用我们的判分器」的说明书。队友 clone 下来照第四节敲两条命令就能跑。

---

## 一、它是什么

`schematic_parser/` 是**官方四项指标的逆向实现**：官方标尺尚未发布（题面说「随代码一起提供」），
我们按题面公式 + 已公布 Q&A 逐条对齐，并把每一处有歧义的地方做成 `EvalConfig` 里的开关。

```
Score_Accuracy_SingleCase = 100 × ( 0.30·ComponentF1 + 0.25·PinF1 + 0.35·NetHypergraphF1 + 0.10·NetLineF1 )
```

| 文件 | 作用 |
|---|---|
| `schematic_parser/data.py` | 数据集发现、配对、`normalise_doc`、超图解析 |
| `schematic_parser/metrics.py` | 四个指标的实现 + `EvalConfig`（所有口径开关都在这里） |
| `schematic_parser/evaluate.py` | **命令行入口**（本文件的主角） |
| `schematic_parser/official.py` | 备用口径（`fast.py --engine harness`），早期按官方报告反推的连通性实现 |

> 逆向的强度：连通性通道（`net_soft=True`）在 2026-10-14 用**官方判分机的真实产物**标定过 ——
> case 1/7/2/3 的 `soft_tp` 与 `f1` 与官方报告**逐位相同**（见 `docs/RESULTS.md` §5.21）。
> 同一批标定还证明：官方**不**把非 box 元件的脚号折成通配 `*`（折算会让官方 10 张虚高 7.4 分，
> 所以 `net_pin_collapse_nonbox` 默认 **False**）。

**它不在流水线里，只是尺子。** 任何来源的产物（我们的、你们的、手写的）都能拿来打分。

---

## 二、依赖

```bash
python -m pip install numpy scipy      # 判题机只需要这两个第三方包
```

Python ≥ 3.9（我们用 3.11 / 3.12 跑过）。仓库根的 `requirements.txt` 是**整条流水线**的依赖
（torch / opencv / ultralytics …），判题机**不需要**它们。

---

## 三、GT 数据在哪

GT 标注**随仓库提交**（只有 `*.json`，没有 PNG —— 判分不读图）：

```
yolo_part/Dataset(waitforpreprocess)/
  10GT/<case>/<case>.json                  # 10 张官方公开 GT —— 评测基准
  10_GTcase/<id>-<source>/<id>-<source>_target.json
  200_train_cases/<id>/<id>-<source>_target_named.json
  10GT/type_list.md                        # 34 个 type 的权威表
```

`--data-root <目录>` 可以指向你自己的数据根，**布局要一样**（`<split>/<case>/<case>.json`）。

---

## 四、怎么跑

```powershell
# 1) 官方 10 张 —— 我们平时说的「分数」默认就是这一行
python -X utf8 -m schematic_parser.evaluate --pred submission\final_results --split 10GT

# 2) 三套 case 集一起（10GT + 10_GTcase + 200_train_cases）
python -X utf8 -m schematic_parser.evaluate --pred <目录>

# 3) 逐案结果存成 JSON（给脚本消费）
python -X utf8 -m schematic_parser.evaluate --pred <目录> --split 10GT --out scores.json

# 4) 自检：不需要任何预测，验证判题机本身
python -X utf8 -m schematic_parser.evaluate --self-test
```

`--pred` 支持两种布局，自动识别：

* **扁平**：`<pred_root>/<case_id>.json` —— 例如 `final_results/1_Pred.json`、`0092.json`；
* **子目录**：`<pred_root>/<case_id>/<?>.json`。

case id 取文件名里的**第一个数字串**，所以 `1_Pred.json` → GT case `1`，`0092_Pred.json` → `92`。

输出 = 逐案表 + 汇总 JSON：

```
case                                  score   compF1    pinF1   netHF1   lineF1
------------------------------------------------------------------------------
10GT/1                                86.10   1.0000   0.7308   0.8711   0.7343
...
10GT/8                                70.87   1.0000   0.3030   0.7989   0.5333
------------------------------------------------------------------------------
{
  "cases": 10,
  "mean_score": 84.945,
  "component_f1": 1.0,
  "pin_f1": 0.816599,
  "net_hypergraph_f1": 0.820979,
  "net_line_f1": 0.579579,
  "missing_predictions": 0
}
```

`mean_score` 就是那 10 张的平均总分（不是加权平均，是逐案总分的算术平均 —— 官方口径如此）。

---

## 五、四个指标怎么判（会踩的坑）

默认口径（`EvalConfig` 的真值，全部有 Q&A 依据）：

| 开关 | 默认 | 含义 |
|---|---|---|
| `bbox_tol` | `20.0` | 元件框容差（px），题面「以 20px 为准」 |
| `pin_tol` | `5.0` | 脚点容差（px） |
| `line_tol` | `5.0` | 导线端点容差（px） |
| `bbox_mode` | `'corners'` | 四个角点都在容差内才算框对 |
| `pin_box_only` | `True` | **引脚只统计 GT 里 `type ∈ {box, ic, boxic}` 的元件**，非 box 元件的脚号不参与 |
| `net_mode` / `net_soft` | `'micro'` / `True` | 连通性按官方软 tp 口径（见第一节） |
| `line_unordered` | `True` | 导线端点匹配不看方向 |
| `unkeyed_match` | `'gt'` | GT 里 `∅NNN` 这类无名元件的**键不比较**，改按坐标匹配（评测 Q4/Q5） |
| `coord_gate` | `40.0` | 坐标匹配的最大中心距（px） |
| `net_pin_collapse_nonbox` | `False` | 官方**不**折算非 box 脚号 |
| `canon_types` | `True` | `type` 先按 `type_list.md` 归一化（`ic`/`boxic` → `box`，`battery` → `battary`…） |

三个容易吃亏的判定：

1. **元件：一笔亏三处。** `type` / `bbox` / `text`（IC 的 `Name`、R/C 的 `value`）任一处错，
   该元件都是 **tp 不加、fp +1、fn +1**。
2. **引脚：键 = (元件, 脚号)。** 脚号对、点错 ⇒ `pin.position_mismatch`（fp+fn）；
   点对、脚名不对 ⇒ `pin.name_mismatch`（fp+fn）。所以「名字读出来但读错」比「没读出来」不亏。
3. **导线：只比较每条边的头两个点。** 折线中间点不影响判定，**端点是命门**。

---

## 六、已知性质（诚实清单）

1. **空产物在「GT 里没有 IC 元件」的 case 上会拿到引脚 F1 = 1.0** ⇒ 那一案白拿 25 分。
   220 个 case 里有 7 个这样的 case，**全部落在 `10_GTcase` / `200_train_cases`，官方 10 张一个都没有**
   （官方 10 张每张都有 box 元件）。这是「空集合 F1 约定」的副产物，不是可薅的分。
2. **GT 自评 216/220 满分**；4 个派生 case 掉 0.3~0.6 分（网络超图的 fp=1/fn=1，属 GT 自身表示与解析口径的边界）。
3. **官方标尺未公布。** 2026-10-13 的新 Q&A 还说：`10GT` 会**再更新**（引脚坐标改成「红绿导线交接处」）、
   最终评分**不设黑白图**、平台的 `∅` 键名 bug「评测代码会更新」⇒ 本地分数是**本地口径**，
   跟主办方最终的绝对分可能不同（相对排序应该稳）。
4. 所有模棱两可处都在 `EvalConfig` 里带注释和实测依据（`schematic_parser/metrics.py:84-160`）。
   要改口径，请 diff 那个 dataclass，别在算法里偷偷改。

---

## 七、我们这棵树的参考分（2026-10-14）

| 跑法 | 官方 10 张 | 元件 | 引脚 | 网络 | 导线 |
|---|---:|---:|---:|---:|---:|
| **交付口径**（VLM 位号/值/名开，`-vlm-pin off`） | **84.95** | 1.0000 | 0.8166 | 0.8210 | 0.5796 |
| 确定性口径（全部 LLM 关，`-offline -vlm-pin off`） | 80.48 | 0.9355 | 0.7385 | 0.8051 | 0.5780 |
| 池子 10 张（**非官方**，交付口径） | 18.59 | 0.3948 | 0.0570 | 0.1371 | 0.0521 |

> ⚠️ **跨口径比数字会凭空造出 4.47 分的「回归」** —— 我们自己踩过一次。报分数请连**口径**一起报：
> 哪套 case 集 + 哪套跑法（交付 / 确定性）+ 哪个提交号。

---

## 八、配套工具（都在仓库里）

| 工具 | 用途 |
|---|---|
| `tools/eval/fast.py` | 端到端「跑 + 判分」：`--cases 1,8`、`--save <名>`（归档到 `work/_fast_base/`）、`--score-only`（只重算分）、`--engine harness`（换备用口径） |
| `tools/eval/gt_retest.py` | **一键复测矩阵**：6 条臂 × 2 口径，并打印 GT 逐案 sha1 指纹；`--fingerprint-only` 用来查主办方是否换了 GT |
| `tools/dev/pinread_contract.py` | 引脚读数器接口契约检查（16 个入口，当前 16/16 PASS） |
| `docs/KNOBS.md` | **458 个环境开关**全量表（生成物，别手改） |
| `docs/TOOLS.md` | 720 个脚本索引（生成物） |
| `docs/RESULTS.md` | 全部实验账本（§0 起，逐条带数字与复现命令） |

---

## 九、给队友的最短路径

```powershell
git clone https://github.com/SilinSmith/eda-pcb.git
cd eda-pcb
python -m pip install numpy scipy

# 把你的产物放成 <你的输出>/final_results/<case>_Pred.json，然后：
python -X utf8 -m schematic_parser.evaluate --pred <你的输出>\final_results --split 10GT
```

想只看某一案：`--split 10GT` 会把十张都打出来；单案筛选用 `--out` 拿到 JSON 再取那一行，
或者用 `tools/eval/fast.py --score-only --cases 8 --run <产物目录>`。

---

## 十、互通约定（避免白干）

1. **报分数必须带口径**：case 集 + 跑法 + 提交号。否则数字不可比。
2. **判分器有任何不一致**：请提 issue / PR，附上 GT 与产物两个文件的最小复现 —— 我们对齐它比争论它快。
3. **`10GT` 更新后**：先跑 `python -X utf8 tools/eval/gt_retest.py --fingerprint-only`，
   确认主办方是否已经换 GT；换了就以新基线重新定标（我们已把这个坑写进 `work/NEXT.md` 的排期）。
