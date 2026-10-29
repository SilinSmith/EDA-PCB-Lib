# -*- coding: utf-8 -*-
"""融合与仲裁的总装 —— workflow.html 右侧那一列的入口。

**为什么要有这个文件。** 现状是：`stage1\\detect.py` 写 `components[i].type`、
`cvparts\\textbox.py` 写 `texts[j].bbox`、`textrec\\infer.py` 写 `texts[j].reads[role]`、
`vlm\\enrich.py` **直接改** `assigned.comps[cid].designator.text`、`stage1\\pintrace.py`
写 `pins[cid]`、`stage1\\nets.py` 写 `nets[k].pins` —— 每个读取器都**直接覆盖结论**，
于是没有任何地方能回答「这条网络该不该合并」「这个引脚号该挂在哪个桩上」。
后果是实测出来的：VLM 把 `component.text_mismatch` 从 915 降到 575（读对了），
但 `component.bbox_mismatch` 从 281 涨到 **420** —— 新读法挤掉了原本正确的框归属，
而本该拦住它的合理性检查不存在。

**这一层的形状**（照 workflow.html）：

    stage1 + stage2 `_assigned`
        → claims    （把已有结论降级成候选，建证据池）
        → priors    （常识：连号 / 词表 / 类型 —— 会自己造候选）
        → solve     （全局自洽：唯一性 + 连号 + 形状）
        → plausible（讲不讲得通：电源地短接 / 参数离群 / 重号）
        → llm_judge（把判不出来的切图去问视觉模型）
        → rescan    （带上上下文定向重读，新证据回池）
        → solve     （再解一次）
        → apply     （选定值回写 `assigned`）
        → doc       （**打补丁**：以 VLM 补强后的文档为底，只覆盖裁决过的字段）

**⚠️ 最后一步不是 `build_doc` 重装配。** 一开始我用的是「`pool.apply()` 回写 `assigned`
后 `assemble.build_doc(stage1, assigned)` 重建」，实测**在 VLM 本来就强的地方掉到**
（9 张 GT：`0001 31.83→29.40`、`0005 13.14→10.43`、`0009 31.29→28.05`、`0010 29.20→27.58`；
总分 22.5020→22.3306）。原因不是「解错了」，而是**重建一次就丢东西** —— `vlm/enrich.py`
对 `pins` 的补号/补名、对 `nets.hyperGraph` 的改名、以及 VLM **新增的元件**
在 `assigned` 里根本没有，`build_doc` 自然造不出来。判据很简单：**若某个槽位融合层没有
比现状更好的证据，它就不该被写。** 详见 `fusion/doc.py`。

**不推倒已有管线**：本模块消费 stage2 的 `_assigned` 与 stage1 文档，产物格式与现管线
**逐字节同构**，可以直接用 `work\\tools\\eval\\stage2_eval.py` 打分对比。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
_PCB = _HERE.parent                                   # eda-pcb
if str(_PCB) not in sys.path:
    sys.path.insert(0, str(_PCB))

from fusion import audit as AU                          # noqa: E402
from fusion import axis as AX                           # noqa: E402
from fusion import claims as CL                        # noqa: E402
from fusion import cropfix as CF                       # noqa: E402
from fusion import doc as DC                           # noqa: E402
from fusion import iterate as IT                        # noqa: E402
from fusion import llm_judge as LJ                     # noqa: E402
from fusion import modelset as MS                      # noqa: E402
from fusion import namefix as NS                       # noqa: E402
from fusion import pinfix as PF                         # noqa: E402
from fusion import valfix as VF                        # noqa: E402
from fusion import plausible as PL                     # noqa: E402
from fusion import priors as PR                        # noqa: E402
from fusion import rescan as RS                        # noqa: E402
from fusion import solve as SV                         # noqa: E402
from stage2.assemble import build_doc                  # noqa: E402
from stage2 import librefine as LR                      # noqa: E402
from stage2 import pinlib as PB                         # noqa: E402

ROOT = Path(__file__).resolve().parents[2]      # `fusion/run.py` -> `eda-pcb/` -> 仓库根
RUN_DIR = Path(os.environ.get("FUSION_RUN_DIR",
                              str(ROOT / "work" / "_submit_out220r36" / "runs")))
OUT_DIR = Path(os.environ.get("FUSION_OUT_DIR",
                              str(ROOT / "work" / "_fusion_out")))
# VLM 补强后的文档目录（`vlm/enrich.py` 的产物）。它必须来自**同一条链**
# （同一份 stage1/stage2），否则两边的元件键对不上。产出方式见
# `work\tools\diag\_fzenrich.py`。
VLM_DIR = Path(os.environ.get("FUSION_VLM_DIR", str(ROOT / "work" / "_fusion_vlm")))
# 最多跑几轮「判 → 重扫 → 再解」（workflow.html 里画的是最多 3 轮）
MAX_ROUNDS = int(os.environ.get("FUSION_ROUNDS", "2"))
from envflag import env_on                       # noqa: E402  (唯一权威假值词表)
# ★★ 「关不掉」的经典形态：这里原本手写了 `not in ("", "0", "false", "no")` ——
#   于是 `FUSION_LLM=off` / `=none` / `=disable` 全被当成**开**，想关 LLM 轮的人
#   越写越用力却关不掉（`work\tools\diag\_vlmab2.py` 那类对照实验最容易被它坑）。
#   统一走 `eda-pcb\envflag.py` 的 `FALSE_WORDS`（8 个词，`.lower()` 后比较）。
#
# ★ 默认值是 **"1"（开）**，而 `submission\run.py` 的 `-fusion` 默认也是 `on`
#   ⇒ **不设任何环境变量直接跑，LLM 轮就是开的**（`-fusion off` 才整层关）。
#   成本实测（10-08，case 1+10）：融合层墙钟 **关 3.5 s → 开 62.7 s**（+29.6 s/张，
#   220 张约 +1.8 h），而那两张图上两臂的 `*_Pred.json` **MD5 逐字节相同**。
#   改默认值前先读 `submission\run.py` 里 owner m25661 那段决定。
DO_LLM = env_on("FUSION_LLM", "1")
# ★ 收敛判据用「本轮结束时的题面」还是「本轮开始时的题面」（见下面 round loop 里
#   的长注释）。`0` = 老行为（判据慢一轮，几乎总会跑满 `MAX_ROUNDS`），保留它只是
#   为了让那次 A/B 的两条臂都能逐字节复现。
CONV_EXACT = env_on("FUSION_CONV_EXACT", "1")
# ★★ 交付文档里**不能有悬空引用**（指着一个本文件没发出的引脚）。开关名：`FUSION_REF_PRUNE`。
#
# 病（`work/_fast_out_trimdef2` 那类 VLM 臂）：兜底键元件（`pNNN`）的引脚被
# `STAGE2_PIN_DROP_LAUNDERED` / `STAGE2_PIN_DROP_UNKEYED` 摘掉之后，**网络侧还留着
# `p012.1` 这样的引用** —— 它指向一个文档里根本不存在的引脚。判题机报告里
# `connectivity.diagnostics.unmapped_pred_pin_references` 数得就是这类东西
# （第 6 次提交 case 1 有 18 条，因此 `conn` 少了一截）。
#
# 实测（`work/_undroppins.py --prune-refs`，判题机复刻尺 `work/_offval.py`）：
#   · val 19 张：悬空引用 **30 条**（0121=11 / 0151=4 / 0163=1 / 0166=14，其余 0），
#     摘掉后 **19.3963 → 19.4065（+0.0102）**、protocol2 **31.25 → 31.26（+0.01）**；
#   · 官方 10 张：本来就 0 条 ⇒ 逐位不变（±0.0000）。
# 只在「**兜底键形状**（`pNNN`）**且该键在本文件里没有引脚**」时才摘 —— 这条判据
# 正好是引脚闸门的补集，所以它动不了 `STAGE2_NET_DROP_UNKEYED=0` 那条臂
# （∅ 符号的键是 `∅NNN`，不匹配 `FKEY_RE`，那条臂要留的引用原样保留）。
#
# 按 owner m40034 ④「不许静默丢弃」的规矩：每条被摘掉的引用都进
# `cvparts.hypo.LEDGER`（`prune_ref` 事件），并汇总进本 case 的报告 `ref_prune`。
REF_PRUNE = env_on("FUSION_REF_PRUNE", "1")
REF_PRUNE_DESIG = env_on("FUSION_REF_PRUNE_DESIG", "1")   # 具名键的悬空引用也摘（见 `_prune_orphan_refs`）
_FKEY_RE = re.compile(r"^p\d{3,4}$")          # 形状同 `stage2/assemble.py:305`


def _prune_orphan_refs(doc: dict) -> dict:
    """摘掉「指向本文件没有发出的引脚」的兜底键引用（见上面 `REF_PRUNE` 的记录）。"""
    if not REF_PRUNE:
        return {"on": False}
    comps = doc.get("components") or {}
    pins = doc.get("pins") or {}

    def _has_pin(key: str, num: str) -> bool:
        grp = pins.get(key) or {}
        if not grp:
            return False
        if ("pin_%s" % num) in grp:
            return True
        # 少数臂把 `pins[k]` 的键写成脚名，兜一层
        return any(str((v or {}).get("pinname")) == str(num) for v in grp.values())

    try:
        from cvparts.hypo import LEDGER                  # 进程内默认账本
    except Exception:                                    # noqa: BLE001
        LEDGER = None

    n = 0
    ex: List[str] = []
    for nid, nv in (doc.get("nets") or {}).items():
        if not isinstance(nv, dict):
            continue
        hg = nv.get("hyperGraph")
        if isinstance(hg, str) and hg.strip():
            toks = [t.strip() for t in hg.strip().strip("()").split(",") if t.strip()]
            keep = []
            for t in toks:
                key, _, num = t.partition(".")
                if num == "*" and key in comps:
                    # `*` = 「不指具体引脚」的通配成员：只有没有引脚的电源/地符号会这么
                    # 写（`assemble.STAGE2_SYM_VERTEX`）。判题机对非 box 类型按
                    # `net_pin_collapse_nonbox` 折叠成元件身份 ⇒ 脚号不参与比较，
                    # 这条仍然满足「成员必须指向存在的元件」这个不变式。
                    keep.append(t)
                    continue
                if key and num and _FKEY_RE.match(key) and key not in comps:
                    # 键都没有 ⇒ 整条引用都是悬空的
                    n += 1
                    if len(ex) < 5:
                        ex.append("%s:%s" % (nid, t))
                    if LEDGER is not None:
                        LEDGER.record("prune_ref", nid, before=[t],
                                      reason="兜底键元件不在本文件里",
                                      by="fusion.ref_prune")
                    continue
                if key and num and _FKEY_RE.match(key) and not _has_pin(key, num):
                    n += 1
                    if len(ex) < 5:
                        ex.append("%s:%s" % (nid, t))
                    if LEDGER is not None:
                        LEDGER.record("prune_ref", nid, before=[t],
                                      reason="引脚的产出被兜底键闸门摘掉 ⇒ 引用会悬空",
                                      by="fusion.ref_prune")
                    continue
                if REF_PRUNE_DESIG and key and num and key in comps and not _has_pin(key, num):
                    # ★ 具名键也会悬空：200 案 × 10 轮抓到 1 例（`0019` 的 `net_9` 引用
                    #   `U0.3`，而 `pins["U0"]` 只剩 `pin_1`）。原来是只认兜底键 `p###`，
                    #   具名键漏网 ⇒ 判题机看到一个指向不存在引脚的成员。
                    n += 1
                    if len(ex) < 5:
                        ex.append("%s:%s" % (nid, t))
                    if LEDGER is not None:
                        LEDGER.record("prune_ref", nid, before=[t],
                                      reason="具名键的引脚不在本文件里 ⇒ 引用悬空",
                                      by="fusion.ref_prune")
                    continue
                keep.append(t)
            if len(keep) != len(toks):
                nv["hyperGraph"] = "(" + ",".join(keep) + ")"
        pl = nv.get("pins")
        if isinstance(pl, list) and pl:
            nv["pins"] = [t for t in pl if not (
                isinstance(t, str) and "." in t
                and _FKEY_RE.match(t.split(".", 1)[0])
                and not _has_pin(t.split(".", 1)[0], t.split(".", 1)[1]))]
    return {"on": True, "n_refs": n, "examples": ex}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assemble(s1_doc: dict, assigned: dict, case: str, vlm_doc: Optional[dict],
              sol: dict, assigned0: Optional[dict] = None,
              pin_index: Optional[dict] = None) -> dict:
    """产出最终文档：**以 VLM 补强后的文档为底打补丁**（没有 VLM 文档才重装配）。

    `vlm_doc` 为真时它就是同一条链上 `vlm/enrich.py` 的产物（含 VLM 补的引脚号/名、
    改名后的 `nets.hyperGraph`、以及 VLM 新增的元件）；重建一次这些全丢，实测 9 张
    GT 上总分 `22.5020 → 22.3306`。

    ⚠️ `assigned0` 必须是**未被 `pool.apply()` 改过**的那份：装配键是从
    `assigned[comps][cid].designator` 推的，若已经回写成新位号，键表就会指到
    VLM 文档里不存在的键上。

    `pin_index` 是 `stage2` 落下的旁路文件 `<case>.pinindex.json`：引脚索引 `k` →
    交付文档里的键 `pin_<号>`。没有它，引脚裁决就只能落在旁路产物里（`doc.apply`
    的 `pin.` 分支会整条跳过）。
    """
    # ★★ 装配底稿用谁？（2025-1x 定，可用 `FUSION_BASE=vlm` 退回老行为）
    #
    # 老行为是 `vlm_doc if vlm_doc else build_doc(...)`：只要开了 VLM，就直接拿
    # **VLM 补强后的整份文档**当底稿。看似省事，实测是**净负** —— 因为 VLM 那份文档
    # 的引脚表是「stage2 原样 + 只改了几个」，而 fusion 的 `build_doc` 路径本来会把
    # 引脚**重建成完备的**。0143 实测：
    #   `-vlm off`：`build_doc` 底稿 → `U1` **45** 个脚（`pin_1 VBAT / pin_7 NRST /
    #               pin_8 PC0 / pin_9 PC1 …` 全对），总分 **39.06**
    #   `-vlm on` ：`vlm_doc` 底稿 → `U1` 只剩 **13** 个脚，总分 **36.88**
    # 而 VLM 真正值钱的是**位号改名**（`component` F1 0.5833 → 0.8750），那一条根本
    # 不靠底稿 —— 它是 `pool` 里的 claim，`priors.write_back` 已经把它写进 `assigned`，
    # 而 `build_doc(s1_doc, assigned, case)` 正是从 `assigned` 取元件键的。
    # ⇒ 底稿永远用我们自己的 `build_doc`，VLM 的结论走 claim 通道进来。
    if os.environ.get("FUSION_DBG_NET"):
        try:
            _sn = s1_doc.get("nets") or {}
            _items = _sn.values() if isinstance(_sn, dict) else list(_sn)
            _nref = sum(len((n or {}).get("seg_index") or []) for n in _items)
            _wd = (s1_doc.get("wire_detail") or {})
            _seg = _wd.get("segments") if isinstance(_wd, dict) else None
            _ac = (assigned or {}).get("comps") if isinstance(assigned, dict) else None
            _msg = "input s1nets=%d segrefs=%d wire_segs=%s assigned_comps=%s" % (
                len(_items), _nref,
                len(_seg) if isinstance(_seg, (list, dict)) else _seg,
                len(_ac) if isinstance(_ac, (list, dict)) else _ac)
        except Exception as _e:                        # 只诊断，不许改动管线行为
            _msg = "input dump failed: %r" % (_e,)
        try:
            with open(os.environ["FUSION_DBG_NET"], "a", encoding="utf-8") as _fh:
                _fh.write("[dbg net] %s %s\n" % (case, _msg))
        except Exception:
            pass
    base = build_doc(s1_doc, assigned, case)
    if os.environ.get("FUSION_DBG_DUMP"):
        # ★ 只诊断：把 `_assemble` 手里的输入原样落盘，和磁盘上那份对比，
        #   用来判断「哪一步在装配前改了 s1_doc / assigned」（§5.459）。
        try:
            import json as _js
            from pathlib import Path as _P
            _d = _P(os.environ["FUSION_DBG_DUMP"])
            _d.mkdir(parents=True, exist_ok=True)
            (_d / ("%s.mem.s1.json" % case)).write_text(
                _js.dumps(s1_doc, ensure_ascii=False), encoding="utf-8")
            (_d / ("%s.mem.as.json" % case)).write_text(
                _js.dumps(assigned, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    if os.environ.get("FUSION_DBG_NET"):
        try:
            import json as _js
            from pathlib import Path as _P
            _rd = _P(os.environ["FUSION_RUN_DIR"])
            _f1 = _rd / "stage1" / ("%s.stage1.json" % case)
            _f2 = _rd / "stage2" / "_assigned" / ("%s.json" % case)
            _b2 = build_doc(_js.loads(_f1.read_text(encoding="utf-8")),
                            _js.loads(_f2.read_text(encoding="utf-8")), case)
            _n2 = _b2.get("nets") or {}
            _it2 = _n2.values() if isinstance(_n2, dict) else list(_n2)
            with open(os.environ["FUSION_DBG_NET"], "a", encoding="utf-8") as _fh:
                _fh.write("[dbg net] %s rebuild-from-disk nets=%d edges=%d\n"
                          % (case, len(_it2),
                             sum(len((x or {}).get("edges") or {}) for x in _it2)))
        except Exception as _e:
            with open(os.environ["FUSION_DBG_NET"], "a", encoding="utf-8") as _fh:
                _fh.write("[dbg net] %s rebuild-from-disk failed: %r\n" % (case, _e))
    if os.environ.get("FUSION_DBG_NET"):
        _nm = base.get("nets") or {}
        with open(os.environ["FUSION_DBG_NET"], "a", encoding="utf-8") as _fh:
            _fh.write("[dbg net] %s base nets=%d edges=%d\n"
                      % (case, len(_nm),
                         sum(len((n or {}).get("edges") or {}) for n in _nm.values())))
    _mode = os.environ.get("FUSION_BASE", "ours").strip().lower()
    if vlm_doc and _mode == "vlm":
        base = vlm_doc
    elif vlm_doc and _mode == "merge":
        base = _merge_vlm(base, vlm_doc)
    out, _rep = DC.apply(base, s1_doc, assigned0 or assigned, sol,
                         pin_index=pin_index)
    if os.environ.get("FUSION_DBG_NET"):
        _nm = out.get("nets") or {}
        with open(os.environ["FUSION_DBG_NET"], "a", encoding="utf-8") as _fh:
            _fh.write("[dbg net] %s applied nets=%d edges=%d\n"
                      % (case, len(_nm),
                         sum(len((n or {}).get("edges") or {}) for n in _nm.values())))
    return out


def _merge_vlm(ours: dict, vlm: dict) -> dict:
    """以我们的 `build_doc` 为底，**按元件**挑更好的那份引脚表。

    依据（19 张定点档，`work\\_fast_base\\vlm_ours.json` vs `vlm_oldbase.json`）：
    两种底稿**各有各赢的地方，没有一款通吃**：

      - 10GT（9 张）**老底稿赢**：`vlm = 51.91` / `ours = 43.46`，差全在
        **多余的脚**上 —— case 1 `pin` 的 fp 25→6、fn 26→18；case 2 `netH` 的 fp 16→0。
        那些图上的元件是两脚/三脚的小件，`build_doc` 从 stage1 重建出来的引脚表
        比 VLM 文档里那份**松得多**（多出来的脚既不是 GT 的号、名字也不对 ⇒ 每个 fp+fn）。
      - `200_train_cases`（10 张）**我们的底稿赢**：`ours = 16.04` / `vlm = 13.39`，
        个案 `0002 +11.92 / 0024 +6.79 / 0086 +2.31 / 0001 +2.23 / 0019 +1.99`。
        这批是 jlc 大 IC，pincol 能枚举出**完备**的引脚列（0143 的 U1：我们 45 个脚、
        VLM 文档只有 13 个）。

    ⇒ 判据用**引脚数**：`len(ours_pins) >= MERGE_PIN_MIN` 就信我们的，否则信 VLM 的。
    大 IC 的引脚数天然很多（几十个），小件最多 2~3 个，这条线分得很开。
    元件表本身取并集（我们的为主，VLM 独有的补进来）。
    """
    n = int(os.environ.get("FUSION_MERGE_PIN_MIN", "8"))
    op, vp = ours.get("pins") or {}, vlm.get("pins") or {}
    pins = {k: v for k, v in op.items()}
    n_use_vlm = 0
    for k, v in vp.items():
        if len(list(op.get(k) or {})) >= n:
            continue                                   # 大 IC：我们的更全
        if v:
            pins[k] = v
            n_use_vlm += 1
    for k, v in vp.items():
        if k not in pins and v:
            pins[k] = v
    comps = dict(vlm.get("components") or {})
    comps.update(ours.get("components") or {})         # 我们的为准
    out = dict(ours)
    out["components"] = comps
    out["pins"] = pins
    if not out.get("nets"):
        out["nets"] = vlm.get("nets") or out.get("nets")
    return out


def run_case(case: str, stage1_dir: Optional[Path] = None,
             assigned_dir: Optional[Path] = None,
             do_llm: bool = DO_LLM, rounds: int = MAX_ROUNDS,
             log=None) -> dict:
    """跑一张图，返回 report，并把产物写进 `OUT_DIR`。"""
    s1d = Path(stage1_dir or os.environ.get("FUSION_STAGE1_DIR",
                                            str(RUN_DIR / "stage1")))
    asd = Path(assigned_dir or os.environ.get("FUSION_ASSIGNED_DIR",
                                              str(RUN_DIR / "stage2" / "_assigned")))
    doc = _load(s1d / ("%s.stage1.json" % case))
    assigned = _load(asd / ("%s.json" % case))
    # ---- ⓪.92 型号名「用引脚名集合反推」（`FUSION_MODEL_SET`，默认 0=关）----
    # 比 ⓪.93 更狠的一步：`namefix` 只能救编辑距离 ≤2 的近失串，救不了「把框内引脚功能名
    # 当型号」这类（`VREFA+` / `NRESET` / `TPSW1240DR`）。这里拿元件框里**读到的引脚名
    # 集合**当指纹去库里找唯一最优的键（离线 10 张缺的 38 根 box 引脚里 18 根「只差型号名」）。
    # 同样必须在建池之前，理由见下一段。病/药/守卫见 `fusion/modelset.py`。
    rep_ms = {}
    try:
        if MS.ON:
            rep_ms = MS.snap_assigned(assigned, doc, log=log)
    except Exception as exc:                               # noqa: BLE001
        rep_ms = {"on": bool(MS.ON), "err": str(exc)}
        if log:
            log("  [model_set] 跳过：%s" % exc)
    # ---- ⓪.93 IC 型号名吸附到官方符号库闭词表（`FUSION_NAME_SNAP`，默认 0=关）----
    # ★ 位置很关键：**必须在 `CL.from_stage2` 建池之前**（下面几行）。第一版把它放在
    #   `rep["iterate"]` 之后，结果只修好了库通道（`pin_libseq`/`librefine`），**导出的
    #   `Name` 一个字没变** —— 因为 `comp.<cid>.partname` 这条 claim 的候选值是在
    #   `claims.from_stage2(pool)` 里从 `assigned` 抄的，`DC.apply` 最后又拿 claim 把
    #   `build_doc` 写好的 `Name` 覆盖回去了（离线档实测：pin_libseq 0→1 但 comp 与
    #   `Name` 逐字不变）。提前到建池之前，池里的候选值就是吸附后的库键。
    # 病与药、以及「只动 box / 已是库键不动 / 距离与裕度双守卫」见 `fusion/namefix.py`。
    rep_ns = {}
    try:
        if NS.ON:
            rep_ns = NS.snap_assigned(assigned, doc, log=log)
    except Exception as exc:                               # noqa: BLE001
        rep_ns = {"on": bool(NS.ON), "err": str(exc)}
        if log:
            log("  [name_snap] 跳过：%s" % exc)
    # ---- ⓪.94 元件「值」的类型内规范化（`FUSION_VALUE_FIX`，默认 0=关）----
    # 与 ⓪.93 同理，**必须在建池之前**（`value` 那条 claim 的候选值也是从这里抄的）。
    # 只做两条物理约束内的机械规则：电感/电容的单位家族互斥（`l` 只能亨、`c` 只能法）、
    # 以及数字尾零规范化（`3.30uH`→`3.3uH`）。见 `fusion/valfix.py`。
    rep_vf = {}
    try:
        if VF.ON:
            rep_vf = VF.fix_assigned(assigned, doc, log=log)
    except Exception as exc:                               # noqa: BLE001
        rep_vf = {"on": bool(VF.ON), "err": str(exc)}
        if log:
            log("  [value_fix] 跳过：%s" % exc)
    assigned0 = json.loads(json.dumps(assigned))       # `pool.apply()` 会就地改它
    # ★ 引脚索引 `k` → 文档键 `pin_<号>`：`build_doc` 的产物，`stage2.run` 落到
    # 旁路文件 `<case>.pinindex.json`。**没有它就无法把引脚裁决写回文档** ——
    # 从 `_assigned` 反推的三种办法覆盖率只有 0.560 / 0.462 / 0.318
    # （`work\tools\diag\_pinmap.py`），改错键比不改更糟。
    pin_index = None
    _pp = asd.parent / ("%s.pinindex.json" % case)
    if _pp.exists():
        try:
            pin_index = _load(_pp)
        except Exception:                                 # noqa: BLE001
            pin_index = None
    vlm_doc = None
    vp = VLM_DIR / ("%s.json" % case)
    if vp.exists():
        try:
            vlm_doc = _load(vp)
        except Exception:                                 # noqa: BLE001
            vlm_doc = None

    t0 = time.time()
    pool = CL.from_stage2(case, doc, assigned, bgr=CL.load_image(doc), vlm=vlm_doc)

    # ---- ⓪ 裁切方向反馈：先修**读数**，再谈先验与解 ------------------------
    # 顺序不能颠倒：`cropfix` 靠文本判据（边缘杂符 / 带空格 / 孤立窄笔画 / 前缀
    # 与类型不符）筛可疑槽位，先验一进来 `best` 就会被先验造的数顶掉，筛子失效。
    rep_cf = {"n_sus": 0, "n_fix": 0, "n_gain": 0, "sec": 0.0}
    if CF.ON and pool.bgr is not None:
        try:
            rep_cf = CF.run(pool, log=log)
        except Exception as exc:                           # noqa: BLE001
            if log:
                log("  [cropfix] 跳过：%s" % exc)

    # ---- ⓪.5 高维对齐：把「排」当一条轴，用框宽反推被吞掉的窄 `1` ----------
    # 必须在 `priors.annotate` **之前**：先验要看的 `best` 得先是**读数**，
    # 否则轴层和先验层会互相顶。本层只**追加候选**，不删不改（见 `fusion/axis.py`）。
    rep_ax = {}
    try:
        if AX.ON:
            rep_ax = AX.inject(pool, log=log)
    except Exception as exc:                               # noqa: BLE001
        if log:
            log("  [axis] 跳过：%s" % exc)

    pri = PR.Priors()
    pri_st = pri.annotate(pool)
    sol = SV.solve(pool)
    SV.write_back(pool, sol)
    t_prior = time.time() - t0

    rep = {"case": case, "n_slots": len(pool.slot_ids()),
           "vlm_doc": bool(vlm_doc),
           # ★ `pool.notes` = 建池阶段（`claims.build` / `priors` / `axis` / `cropfix`）
           #   留下的**计数笔记**（`shape_gate` / `boxscan` / `run_window` …）。以前只在
           #   `claims.py` 自己的 `__main__` 里才看得到，跑分路径上完全不可见 ——
           #   「boxscan 到底建了几个槽位」这种问题只能靠猜。原样搬进报告。
           "notes": list(pool.notes),
           "priors": pri_st, "rounds": [], "sec_prior": round(t_prior, 1),
           "cropfix": {k: v for k, v in rep_cf.items() if k != "rows"},
           "axis": {k: v for k, v in rep_ax.items() if k != "plan"},
           "score_before": _summary(pool, sol)}

    q_first = SV.open_questions(pool, sol["assign"])
    for rnd in range(1, max(0, int(rounds)) + 1):
        if not do_llm:
            break
        r = {"round": rnd}
        if rnd == 1:
            # ★★ 这里原本**没有 try**：`LJ.judge` 一抛异常，整个 case 就一份融合产物
            #   都没有（对照 `AU.run` 在下面就是包了 try 的）。LLM 是**增强**，
            #   不是前置条件 —— 它挂了也要把「先验 + 求解」这一层的结果交出去。
            try:
                j = LJ.judge(pool, q_first, log=log)
                r["judge"] = {k: v for k, v in j.items() if k != "rows"}
                r["judge_rows"] = j.get("rows")
            except Exception as exc:                       # noqa: BLE001
                r["judge"] = {"err": str(exc)}
                if log:
                    log("  [judge] 跳过：%s" % exc)
        # 先装配一次，`plausible` 的网络类检查需要网络已经成型
        pool.apply()
        pred = _assemble(doc, assigned, case, vlm_doc, sol, assigned0,
                         pin_index=pin_index)
        sus = PL.check(pool, pred)
        tasks = RS.plan(pool, sus)
        r["plausible"] = [{"kind": s["kind"], "severity": s["severity"],
                           "why": s["why"]} for s in sus[:20]]
        r["n_susp"] = len(sus)
        r["n_task"] = len(tasks)
        # ---- 按**现实规律**判断合理性：把疑点交给模型当硬件工程师判 ----------
        # 它只能提候选（只有 `key_duplicate` 收），所以不会变成第四个幻觉来源。
        try:
            au = AU.run(pool, sus, log=log)
            r["audit"] = {k: v for k, v in au.items() if k != "rows"}
            r["audit_rows"] = au.get("rows")
        except Exception as exc:                           # noqa: BLE001
            r["audit"] = {"err": str(exc)}
            if log:
                log("  [audit] 跳过：%s" % exc)
        try:
            rs = RS.run(pool, tasks, log=log)
            r["rescan"] = {k: v for k, v in rs.items() if k != "rows"}
            r["rescan_rows"] = rs.get("rows")
        except Exception as exc:                           # noqa: BLE001
            r["rescan"] = {"err": str(exc)}
            if log:
                log("  [rescan] 跳过：%s" % exc)
        sol = SV.solve(pool)
        SV.write_back(pool, sol)
        r["score_after"] = _summary(pool, sol)
        rep["rounds"].append(r)
        # ★★ 收敛判据：**必须用这一轮结束时的题面**。
        #
        #   原来这里是 `if not tasks and not q_hard: break`，而 `q_hard` 是从
        #   **循环开始时**的 `q_first` 算的（`q_first` 只在循环体最后才重新赋值）——
        #   于是判据永远慢一轮：第 1 轮把题全解掉了，第 2 轮照样空跑（而第 2 轮
        #   其实只跑 plausible/audit/rescan，`judge` 只在 rnd==1 调）。
        #   代价是每张图多一次装配 + 一次 `plausible` + 一次 VLM 审计。
        # ★ boxscan 的自由读数题（`q["free"]`，零候选槽位）**不参与收敛判据**：它是
        #   「补问一遍」，不是「还没定」。否则每张图都会因为多了几道零候选题而永远
        #   跑满 `MAX_ROUNDS`，把「多跑一轮」的成本算到 boxscan 头上。
        q_next = SV.open_questions(pool, sol["assign"])
        q_hard = [q for q in (q_next if CONV_EXACT else q_first)
                  if not q.get("free")]
        r["n_q_next"] = len(q_next)
        if not tasks and not q_hard:
            break
        q_first = q_next

    pool.apply()
    # ---- ⓪.9 定向回传：按 pincol 的格点窗问一次 VLM 的**引脚名** ------------
    # 必须赶在 `_assemble` **之前**：`stage2.assemble` 的 `BOX_PIN_NEED_NAME`（默认开）
    # 会把没有名字的 box 引脚整条丢掉，装配之后再补名字就补不回来了。
    # 见 `fusion/iterate.py` 顶部：0143 的 U1 用 pincol 的名字窗问 VLM 是 **54/54 全中**，
    # 而 `vlm/enrich.py:read_pins` 用自己那套窗口只回 13 个。
    rep_it = {}
    try:
        rep_it = IT.pin_fields(assigned, doc, pool.bgr, log=log)
    except Exception as exc:                               # noqa: BLE001
        rep_it = {"err": str(exc)}
        if log:
            log("  [iterate] 跳过：%s" % exc)
    rep["iterate"] = {k: v for k, v in rep_it.items() if k != "rows"}
    # ---- ⓪.92 型号名「引脚名集合反推」的读数（改动在建池之前，见 `run_case` 顶部）----
    rep["model_set"] = {k: v for k, v in rep_ms.items() if k != "rows"}
    # ---- ⓪.93 IC 型号名吸附的读数（真正的改动在建池之前就做完了，见 `run_case` 顶部）----
    rep["name_snap"] = {k: v for k, v in rep_ns.items() if k != "rows"}
    # ---- ⓪.94 值规范化（同上：真正的改动在建池之前）----
    rep["value_fix"] = {k: v for k, v in rep_vf.items() if k != "rows"}
    # ---- ⓪.95 槽位级「库给多值、图给顺序」回填（`FUSION_PIN_LIBSEQ`，默认 1=开；
    #          `stage2/pinlib.py:105` 是唯一权威：0=关 / 1=只在组内读数可疑时补纠（默认档，
    #          2026-10-15 转正）/ 2=只要脚数相等就整组按库顺序重写）--
    # 病：CRNN 把 `114/113/115` 的**前导 1 吃掉**、`48/87/68` 读成 `46/67/66`、
    # `90` 读成 `S0`；而 `BOX_PIN_DROP_UNREAD`（默认开，`stage2/assemble.py:1141`）
    # 会把「没读到合法号」的桩**整条丢掉** ⇒ 这些脚不是「名不对」，是**根本没进
    # 文档**；`libpins.complete_pins` 只做「名→唯一号」，`VDD`(14)/`GND`(6)/
    # `VDDC`(2) 一律放弃（`:1106-1129`）。
    # 药：官方库里同名脚的**号是升序**，图上同名脚**自上而下**也是升序（官方 10 张
    # 94 个同名组 88 个逐脚复现 GT = 93.6%，`work\_liborder.py`）⇒ 拿库的号集合 +
    # 槽位 y 序，把「读数不在集合里」的槽补上/纠正。三道守卫（乱序 / 组外占用 /
    # 撞号终检）在 `stage2/pinlib.py` 顶部有记。
    # ★ 必须赶在 `_assemble` **之前**：`build_doc` 才是丢桩的那一步。
    rep_pb = {}
    try:
        rep_pb = PB.repair_slots(assigned, log=log)
    except Exception as exc:                               # noqa: BLE001
        rep_pb = {"err": str(exc)}
        if log:
            log("  [pinlib] 跳过：%s" % exc)
    rep["pin_libseq"] = {k: v for k, v in rep_pb.items()
                         if k not in ("rows", "skip", "blocked")}
    pred = _assemble(doc, assigned, case, vlm_doc, sol, assigned0,
                     pin_index=pin_index)
    # ---- ⓪.98 行内位号重绑（`STAGE2_ROW_BIND`，默认 0）：落点必须在装配之后 --
    # 位号印在符号**右侧** ⇒ 逐颗捡最近文字框在等距排里必然右偏一颗（§5.423）。
    # 最终位号是 `DC.apply(base, s1_doc, assigned0, sol, …)` 按 claim 解写回的：
    # `claims.from_assigned` 把 stage2 的位号全部降级成候选、由 `solve` 全局重裁，
    # 而重裁不看「这个文字框印在哪个符号旁边」——实测 case1 栏内被整体错排
    # （body 1799 的框里印着 `R20`，解却给它 `R4`）。装配前改什么都不留痕
    # （`work/_o13rb6` 74.93 逐位不动），所以这里改成：拿 `assigned0`（`pool.apply()`
    # 之前那份，位号仍与文字框同源）跑 `rowbind.fix` 做行内对齐，再把
    # **最终文档**的键按 cid 改名，`components` / `pins` / `nets.hyperGraph` 一起改。
    if os.environ.get("STAGE2_ROW_BIND", "1").strip() not in ("", "0"):
        rep_rb = {}
        try:
            import copy as _copy
            from stage2 import rowbind as _rb
            a2 = _copy.deepcopy(assigned0 or assigned)
            rep_rb = _rb.fix(doc, a2)
            _mp, _vals = {}, {}
            # 交付键的来源改成 `claim` 解自己记的 `comp.<cid>.key`（§5.455）：旧写法拿
            # `assigned[cid].designator` 的**文本**当键，`value`/引脚读数一改就可能让候选
            # 打分变、解出来的位号文本跟着变，文本桥一断整行就不修（实测 case1 的
            # `renamed` 从 8 掉到 5）。用解里的键只依赖「哪个 cid 拿到哪个 key」。
            _sel = (sol or {}).get("assign") or {}

            def _solkey(_c):
                _v = _sel.get("comp.%s.key" % _c)
                if isinstance(_v, (list, tuple)) and _v:
                    return str(_v[0])
                return _v if isinstance(_v, str) else ""

            for _cid, _rec in (assigned.get("comps") or {}).items():
                _o = _solkey(_cid) or str(((_rec.get("designator") or {}).get("canon")
                                           or (_rec.get("designator") or {}).get("text") or ""))
                _r0 = ((a2.get("comps") or {}).get(_cid) or {})
                _n = str((_r0.get("designator") or {}).get("canon")
                         or (_r0.get("designator") or {}).get("text") or "")
                if _o and _n and _o != _n:
                    _mp[_o] = _n
                    _v = (_r0.get("value") or {})
                    _vt = str(_v.get("canon") or _v.get("text") or "")
                    if _vt:
                        _vals[_n] = _vt
            if _mp:
                pred = _rb.rename(pred, _mp)
                rep_rb["renamed"] = len(_mp)
            # 值也一起搬：同一行里位号与值印在同一块文字区，错位是同时发生的。
            _pc = pred.get("components") or {}
            for _k, _v2 in _vals.items():
                if _k in _pc and isinstance(_pc[_k], dict):
                    _pc[_k]["value"] = _v2
            if os.environ.get("STAGE2_ROW_BIND_DBG"):
                print("[rb] %s moved=%s renamed=%d map=%s"
                      % (case, rep_rb.get("moved"), len(_mp), _mp))
        except Exception as exc:                           # noqa: BLE001
            rep_rb = {"err": str(exc)}
            if log:
                log("  [rowbind] 跳过：%s" % exc)
        rep["row_bind"] = {k: v for k, v in rep_rb.items() if k != "rows"}
    # ---- ⑩ 库名回填：用官方库按 `(型号, 号) → 库名` 校准已 emit 的引脚名 --------
    # 必须在这之后：`_assemble` 才把 VLM 的引脚读数落进 `pred`，而 VLM 会覆盖
    # `build_doc` 里库规则 B 已经写好的名字（见 `stage2/librefine.py` 顶部）。
    # 默认开（2026-10 转正）：官方 10 张 66.4498 → 66.7807、val 19 张 19.2974 → 19.3987。
    try:
        rep["librefine"] = LR.refine(pred, log=log)
    except Exception as exc:                               # noqa: BLE001
        rep["librefine"] = {"err": str(exc)}
    # ---- ⑩.5 库权威的号/名联修（`FUSION_PIN_LIBFIX`，§5.49）--------------------
    # `librefine` 只管「库对该号有**唯一**名」，而官方库里一号多名极常见
    # （`AMS1117-3.3` 的 1 号有 ADJ / ADJ(GND) / GND … 六种写法 ⇒ 它整条放弃），
    # 且**号本身读错**时它也不适用（`pin_33` / `pin_24` 不在库号集合里）。
    # 必须放在 ⑪ `ref_prune` **之前**：改脚键会动 `hyperGraph` 里的 `元件.号` 引用，
    # 得让收尾那一步看到新键（顺序同 `librefine`：`_assemble` 之后才是出门的那一份）。
    try:
        rep_pf = PF.fix(pred, log=log)
    except Exception as exc:                               # noqa: BLE001
        rep_pf = {"err": str(exc)}
        if log:
            log("  [pinfix] 跳过：%s" % exc)
    rep["pin_libfix"] = {k: v for k, v in rep_pf.items() if k != "rows"}
    # ---- ⑪ 引用收尾：交付文档里不能有悬空引用（`FUSION_REF_PRUNE`，见函数上方长注释）----
    # 必须放在**最后**（`librefine` 之后、写盘之前）：只有这里的 `pred` 才是出门的那一份，
    # 前面 `build_doc` / `weld_nets` / `_merge_nets` 都会重建 `hyperGraph`，
    # 在中间任何一步摘引用都可能被下一步的并集又带回来（`stage2/assemble.py:1583` 的
    # 兜底键过滤就是这么被 `:1876` 的重建绕过的）。
    try:
        rep["ref_prune"] = _prune_orphan_refs(pred)
    except Exception as exc:                               # noqa: BLE001
        rep["ref_prune"] = {"err": str(exc)}
    try:
        rep["redetect"] = IT.redetect_hints(pred, doc)
    except Exception:                                      # noqa: BLE001
        rep["redetect"] = []
    rep["score_final"] = _summary(pool, sol)
    rep["sec"] = round(time.time() - t0, 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / ("%s.json" % case)).write_text(
        json.dumps(pred, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "_assigned").mkdir(exist_ok=True)
    (OUT_DIR / "_assigned" / ("%s.json" % case)).write_text(
        json.dumps(assigned, ensure_ascii=False), encoding="utf-8")
    return rep


def _summary(pool, sol) -> dict:
    """一眼看得懂的池子状态：多少槽位、多少带先验、多少还悬着。"""
    from collections import Counter
    src = Counter()
    for sid in pool.slot_ids():
        for a in pool.alts(sid):
            src[str(a.source).split(":")[0]] += 1
    st = dict(sol.get("stats") or {})
    return {"n_slots": len(pool.slot_ids()),
            "n_assigned": st.get("n_assigned"),
            "n_open": st.get("n_open"),
            "n_lost": st.get("n_lost"),
            "sources": dict(src)}


def cases_of(split: str = "") -> List[str]:
    """列出 `stage1` 目录里已有的 case（可用 `FUSION_CASES` 覆盖）。"""
    want = os.environ.get("FUSION_CASES", "").strip()
    if want:
        out = []
        for part in want.replace(",", " ").split():
            if "-" in part:
                a, b = part.split("-", 1)
                out += ["%04d" % i for i in range(int(a), int(b) + 1)]
            else:
                out.append(part.zfill(4))
        return out
    s1d = Path(os.environ.get("FUSION_STAGE1_DIR", str(RUN_DIR / "stage1")))
    names = sorted(p.name.split(".")[0] for p in s1d.glob("*.stage1.json"))
    return names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="融合与仲裁总装")
    ap.add_argument("cases", nargs="*", help="case 名（默认全部）")
    ap.add_argument("--no-llm", action="store_true", help="只跑先验 + 解 + 合理性检查")
    ap.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("FUSION_WORKERS", "1")),
                    help="并发张数（LM Studio 侧够用；产物按 case 名写，互不冲突）")
    a = ap.parse_args(argv)
    log = None if a.quiet else (lambda s: print(s, flush=True))

    # ⚠ 不能无脑 `zfill(4)`：数据集里 `1`..`10` 与 `0001`..`0010` 是**不同的图**
    # （`1` 有 22 个元件、`0001` 有 83 个）。补零只在「补了之后确实存在、而原名
    # 不存在」时才做。
    def _fix(c: str) -> str:
        if c in _all_cases:
            return c
        z = c.zfill(4)
        return z if z in _all_cases else c

    _all_cases = set(cases_of())
    names = [_fix(c) for c in a.cases] or sorted(_all_cases)
    rows = []
    t0 = time.time()

    def _one(c: str) -> dict:
        try:
            # `--no-llm` 与 `FUSION_LLM=0` 是「或」的关系：两者都能关掉 LLM 轮。
            # 之前这里写成 `not a.no_llm`，会把 `FUSION_LLM=0` 整个绕过 —— 于是
            # 一批以 `FUSION_LLM=0` 名义跑出来的产物其实带着完整 LLM 轮。
            return run_case(c, do_llm=(not a.no_llm) and DO_LLM,
                            rounds=a.rounds, log=log)
        except Exception as e:                        # noqa: BLE001
            return {"case": c, "error": "%s: %s" % (type(e).__name__, e)}

    nw = max(1, int(a.workers))
    if nw == 1:
        for i, c in enumerate(names, 1):
            if log:
                log("== [%d/%d] %s" % (i, len(names), c))
            r = _one(c)
            if r.get("error") and log:
                log("   !! %s" % r["error"])
            rows.append(r)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        done = 0
        with ThreadPoolExecutor(max_workers=nw) as ex:
            futs = [ex.submit(_one, c) for c in names]
            for fu in as_completed(futs):
                r = fu.result()
                done += 1
                rows.append(r)
                print("[%d/%d] %s %s" % (done, len(names), r.get("case"),
                                         ("ERR " + str(r.get("error"))) if r.get("error") else ""),
                      flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ★ **合并**进已有报告，不要覆盖：逐 case 分批跑（后台作业、重跑单张）时
    #   覆盖会让 `_fzrep.py` 只剩最后一次的那一张 —— 现场踩过。
    rp = OUT_DIR / "_fusion_report.json"
    old = []
    if rp.exists():
        try:
            prev = json.loads(rp.read_text(encoding="utf-8"))
            old = [x for x in (prev.get("cases") or [])
                   if x.get("case") not in {r.get("case") for r in rows}]
        except Exception:                                  # noqa: BLE001
            old = []
    merged = sorted(old + rows, key=lambda x: str(x.get("case") or ""))
    rp.write_text(
        json.dumps({"cases": merged, "wall_s": round(time.time() - t0, 1),
                    "run_dir": str(RUN_DIR)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("完成 %d 张（报告现共 %d 张），用时 %.1f s，产物 %s"
          % (len(names), len(merged), time.time() - t0, OUT_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
