# -*- coding: utf-8 -*-
"""单样例调试器（IDE debug 模式）。

    # 一个 case
    python work\\tools\\debug\\dbg.py --case 0001

    # 指定产物目录（默认取 work\\_submit_out220z）
    python work\\tools\\debug\\dbg.py --case 0001 ^
        --pred work\\_submit_out220z\\final_results ^
        --runs work\\_submit_out220z\\runs ^
        --log  work\\_submit_out220z\\decision_log.json

    # 全部 220 张 + 总览页（按丢分排序，找最值得看的图）
    python work\\tools\\debug\\dbg.py --all

输出 `work\\debug\\<case>.html`（单文件、图内嵌）与 `work\\debug\\index.html`。
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "tools" / "eval"))

sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json          # noqa: E402
from schematic_parser.metrics import EvalConfig                          # noqa: E402
from stage2_eval import build_index, resolve                             # noqa: E402

import ledger as L                                                       # noqa: E402
import report as R                                                       # noqa: E402


def pred_file(pred_dir: Path, case: str) -> Path | None:
    for name in ("%s_Pred.json" % case, "%s.json" % case):
        p = pred_dir / name
        if p.exists():
            return p
    return None


def stage_counts(p: Path):
    """从一份产物 JSON 里数出「元件 / 有真位号的元件 / 网络 / 引脚」。"""
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    comps = d.get("components") or {}
    n = len(comps)
    n_named = sum(1 for k in comps if not re.match(r"^p\d{3,4}$", str(k)))
    n_fallback = n - n_named
    return {"n_comp": n, "n_named": n_named, "n_fallback": n_fallback,
            "n_net": len(d.get("nets") or {}), "n_pin": len(d.get("pins") or {})}


def build_stages(case, runs: Path, log_entry):
    out = []
    s1 = runs / "stage1" / ("%s.stage1.json" % case)
    info = {}
    if s1.exists():
        try:
            d = json.loads(s1.read_text(encoding="utf-8"))
            info = d.get("stats", {}) or {}
            w = d.get("wires", {}) or {}
        except Exception:
            info, w = {}, {}
    else:
        w = {}
    lg = log_entry or {}
    t1 = (lg.get("stage1") or {}).get("timing", {})
    out.append({"name": "① 阶段一 · 检测 + 提线",
                "n": "元件 %s ｜ 文字框 %s ｜ 网络 %s ｜ 引脚 %s ｜ 割点 %s ｜ %.1fs"
                     % (info.get("n_components", "—"), info.get("n_texts", "—"),
                        info.get("n_nets", "—"), info.get("n_pins", "—"),
                        w.get("n_junctions", "—"), t1.get("total_s", 0.0))})
    for name, sub in (("② 阶段二 · 文字识别 + 装配", "stage2"),
                      ("③ 阶段三 · LLM 仲裁", "stage3"),
                      ("④ VLM 补强（五条通道）", "stage3_vlm")):
        c = stage_counts(runs / sub / ("%s.json" % case))
        if c:
            out.append({"name": name,
                        "n": "元件 %d（有位号 %d ｜ 兜底 pNNN %d）｜ 网络 %d ｜ 引脚组 %d"
                             % (c["n_comp"], c["n_named"], c["n_fallback"],
                                c["n_net"], c["n_pin"])})
        else:
            out.append({"name": name, "n": "（这一版没有这个阶段的产物）"})
    out.append({"name": "⑤ 本地评分", "n": "见上"})
    return {"steps": out, "detail_html": ""}


def render_case(case_id, pred_path, gt_path, gt_doc, split, cfg, runs, log_entry, outdir):
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    led = L.build_ledger(gt_doc, pred, case_id, cfg=cfg, split=split)
    led["headroom"] = L.headroom(led)
    png = Path(gt_path)
    if not png.exists():                       # 退回我们自己的输入图
        alt = _REPO / "work" / "_submit_in220" / ("%s.png" % case_id)
        if alt.exists():
            png = alt
    b64 = base64.b64encode(png.read_bytes()).decode("ascii")
    from PIL import Image
    w, h = Image.open(png).size
    meta = {
        "img_b64": b64, "img_w": w, "img_h": h,
        "gt_path": str(Path(gt_path).relative_to(_REPO)) if str(gt_path).startswith(str(_REPO)) else gt_path,
        "pred_path": str(pred_path.relative_to(_REPO)) if str(pred_path).startswith(str(_REPO)) else str(pred_path),
        "headroom": led["headroom"],
        "stages": build_stages(case_id, runs, log_entry),
        "config_note": led["config"],
    }
    htmltxt = R.render(led, meta)
    out = outdir / ("%s.html" % case_id)
    out.write_text(htmltxt, encoding="utf-8")
    return out, led


def write_index(rows, outdir, pred_dir, runs):
    """总览页：把所有 case 按丢分排序，找最值得点进去看的图。"""
    def srt(k):
        return sorted(rows, key=lambda r: -r["lost"].get(k, 0))

    def table(rows_, key, cap):
        h = ['<table><thead><tr><th>case</th><th>split</th><th>总分</th>'
             '<th>修满%s能到</th><th>%s 丢分</th><th>元件 F1</th><th>元件 tp/fp/fn</th>'
             '<th>引脚 F1</th><th>网络 F1</th><th>导线 F1</th></tr></thead><tbody>' % (cap, cap)]
        for r in rows_[:60]:
            h.append(
                '<tr><td><a href="%s">%s</a></td><td class="d">%s</td>'
                '<td class="b">%.2f</td><td class="g">%.2f</td>'
                '<td class="r">%.2f</td><td>%.3f</td>'
                '<td class="d">%d/%d/%d</td><td>%.3f</td><td>%.3f</td><td>%.3f</td></tr>'
                % (r["file"], r["case"], r["split"], r["score"],
                   r["score"] + r["lost"].get(key, 0), r["lost"].get(key, 0),
                   r["comp"], r["comp_tp"], r["comp_fp"], r["comp_fn"],
                   r["pin"], r["net"], r["line"]))
        h.append("</tbody></table>")
        return "".join(h)

    n = len(rows)
    avg = sum(r["score"] for r in rows) / max(1, n)
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r["score"])
    sp = " ｜ ".join("%s %d 张 均分 <b>%.2f</b>" % (k, len(v), sum(v) / len(v))
                     for k, v in sorted(by_split.items()))
    bad = [r for r in rows if not r["selfcheck_ok"]]
    warn = ("")
    if bad:
        warn = ('<div class="w">⚠ %d 张的自校验与本地 <code>evaluate_case</code> 不一致：%s'
                '</div>' % (len(bad), ", ".join(r["case"] for r in bad[:20])))
    CAPS = {"component": "元件", "pin": "引脚", "net_hypergraph": "网络", "net_line": "导线"}
    body = "".join(
        ('<h2>按<b>%s</b>丢分排序（点 case 进单图调试）</h2>' % CAPS[k])
        + table(srt(k), k, CAPS[k])
        for k in ("net_hypergraph", "pin", "component", "net_line"))
    htmltxt = (
        '<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">'
        '<title>调试总览 · %d 张</title><style>'
        'body{background:#0d1117;color:#e6edf3;font:13px/1.6 -apple-system,Segoe UI,'
        'Microsoft YaHei,sans-serif;margin:0;padding:24px 30px}'
        'h1{font-size:20px;margin:0 0 6px}h2{font-size:15px;margin:26px 0 8px;'
        'border-left:3px solid #1f6feb;padding-left:8px}'
        'a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}'
        'table{border-collapse:collapse;width:100%%;font-size:12px}'
        'th,td{border-bottom:1px solid #21262d;padding:4px 8px;text-align:right}'
        'th{color:#8b949e;font-weight:600;position:sticky;top:0;background:#0d1117}'
        'td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}'
        '.d{color:#8b949e}.b{font-weight:700}.r{color:#f85149}.g{color:#3fb950}'
        '.w{background:#f8514922;border:1px solid #f8514966;border-radius:6px;padding:10px;'
        'margin:12px 0}'
        '</style></head><body>'
        '<h1>单图调试总览 · %d 张</h1>'
        '<div style="color:#8b949e">平均 <b style="color:#3fb950;font-size:16px">%.2f</b>'
        ' / 100 ｜ %s<br>预测 <code>%s</code> ｜ 阶段产物 <code>%s</code></div>'
        '%s%s</body></html>'
        % (n, n, avg, sp, pred_dir, runs, warn, body))
    (outdir / "index.html").write_text(htmltxt, encoding="utf-8")
    return outdir / "index.html"


def main(argv=None):
    ap = argparse.ArgumentParser(description="单样例调试器（逐条对账 + 图纸叠加）")
    ap.add_argument("--case", default=None, help="case_id，如 0001 / 1")
    ap.add_argument("--all", action="store_true", help="全部 220 张 + 总览页")
    ap.add_argument("--pred", default=str(_REPO / "work" / "_submit_out220z" / "final_results"))
    ap.add_argument("--runs", default=str(_REPO / "work" / "_submit_out220z" / "runs"))
    ap.add_argument("--log", default=str(_REPO / "work" / "_submit_out220z" / "decision_log.json"))
    ap.add_argument("--out", default=str(_REPO / "work" / "debug"))
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--profile", choices=("latest", "protocol2", "protocol"), default="latest")
    ap.add_argument("--bbox-tol", type=float, default=None)
    ap.add_argument("--pin-tol", type=float, default=None)
    ap.add_argument("--line-tol", type=float, default=None)
    args = ap.parse_args(argv)

    cfg = EvalConfig.for_profile(args.profile)
    for field in ("bbox_tol", "pin_tol", "line_tol"):
        if getattr(args, field) is not None:
            setattr(cfg, field, getattr(args, field))
    pred_dir, runs, outdir = Path(args.pred), Path(args.runs), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    exact, digits = build_index(args.data_root)

    logmap = {}
    lp = Path(args.log)
    if lp.exists():
        try:
            for c in json.loads(lp.read_text(encoding="utf-8")).get("cases", []):
                logmap[c.get("case")] = c
        except Exception:
            pass

    cases = []
    if args.all:
        for f in sorted(pred_dir.glob("*.json")):
            if f.name.startswith("_"):
                continue
            st = f.stem[:-5] if f.stem.endswith("_Pred") else f.stem
            cases.append(st)
    else:
        if not args.case:
            ap.error("要么 --case，要么 --all")
        cases = [args.case]

    index = []
    for cid in cases:
        case, how = resolve(cid, exact, digits)
        if case is None:
            print("!! %s 配不上 GT（%s）" % (cid, how)); continue
        pf = pred_file(pred_dir, case.case_id) or pred_file(pred_dir, cid)
        if pf is None:
            print("!! %s 找不到预测文件" % cid); continue
        gt_doc = load_json(case.json_path)
        out, led = render_case(case.case_id, pf, case.png_path, gt_doc, case.split,
                               cfg, runs, logmap.get(case.case_id), outdir)
        s = led["score"]
        parts = {p["key"]: p for p in s["parts"]}
        index.append({
            "case": case.case_id, "split": case.split, "score": s["total"],
            "official": s["official"],
            "comp": parts["component"]["f1"], "pin": parts["pin"]["f1"],
            "net": parts["net_hypergraph"]["f1"], "line": parts["net_line"]["f1"],
            "lost": {"component": parts["component"]["lost"], "pin": parts["pin"]["lost"],
                     "net_hypergraph": parts["net_hypergraph"]["lost"],
                     "net_line": parts["net_line"]["lost"]},
            "comp_tp": parts["component"]["tp"], "comp_fp": parts["component"]["fp"],
            "comp_fn": parts["component"]["fn"],
            "file": out.name,
            "selfcheck_ok": all(c["ok"] for c in led["selfcheck"]),
        })
        print("%-24s %7.3f  (评分器 %7.3f)  元件 %5.3f  引脚 %5.3f  网络 %5.3f  导线 %5.3f  → %s"
              % (case.case_id, s["total"], s["official"], parts["component"]["f1"],
                 parts["pin"]["f1"], parts["net_hypergraph"]["f1"],
                 parts["net_line"]["f1"], out.name))

    if index:
        (outdir / "_index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
        print("写到 %s（%d 个 case）" % (outdir / "_index.json", len(index)))
        if args.all:
            print("总览页 → %s" % write_index(index, outdir, pred_dir, runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
