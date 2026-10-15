# -*- coding: utf-8 -*-
"""§5.172 解码侧实验：对同一份 logits 跑 贪心 / 前缀束搜索 / 空白偏置，比较 ruler 上的 exact。
只读 logits，不改 `textrec/infer.py` 的默认行为。判据见 RESULTS §5.172（预注册）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

EDA = Path(__file__).resolve().parents[2]
if str(EDA) not in sys.path:
    sys.path.insert(0, str(EDA))

import numpy as np
import torch
from PIL import Image

from textrec import config as C
from textrec.charset import BLANK, Charset
from textrec.dataset import crop_gray
from textrec.model import CRNN
from textrec.train import levenshtein

WORK = EDA.parent / "work"
NEG = -1e30


def lenbucket(t: str) -> str:
    n = len(t)
    return "1-2" if n <= 2 else "3-4" if n <= 4 else "5-7" if n <= 7 else "8-11" if n <= 11 else "12+"


def greedy(logp: np.ndarray, cs: Charset, beta: float = 0.0) -> str:
    row = logp
    if beta:
        row = logp.copy()
        row[:, BLANK] -= beta
    ids = row.argmax(1)
    out, prev = [], -1
    for i in ids:
        i = int(i)
        if i != prev and i != BLANK:
            ch = cs.itos.get(i)
            if ch is not None:
                out.append(ch)
        prev = i
    return "".join(out)


def beam(logp: np.ndarray, cs: Charset, beam_w: int = 10, beta: float = 0.0) -> str:
    beams = {(): (0.0, NEG)}
    T = logp.shape[0]
    for t in range(T):
        row = logp[t]
        if beta:
            row = row.copy()
            row[BLANK] -= beta
        idx = np.argsort(row)[::-1][: beam_w + 1]
        new: dict = {}
        for pref, (pb, pnb) in beams.items():
            tot = np.logaddexp(pb, pnb)
            for c in idx:
                c = int(c)
                lp = float(row[c])
                if c == BLANK:
                    a, b = new.get(pref, (NEG, NEG))
                    new[pref] = (np.logaddexp(a, tot + lp), b)
                    continue
                last = pref[-1] if pref else None
                if c == last:
                    a, b = new.get(pref, (NEG, NEG))
                    new[pref] = (a, np.logaddexp(b, pnb + lp))
                    k2 = pref + (c,)
                    a2, b2 = new.get(k2, (NEG, NEG))
                    new[k2] = (a2, np.logaddexp(b2, tot + lp))
                else:
                    k2 = pref + (c,)
                    a2, b2 = new.get(k2, (NEG, NEG))
                    new[k2] = (a2, np.logaddexp(b2, tot + lp))
        items = sorted(new.items(), key=lambda kv: -np.logaddexp(kv[1][0], kv[1][1]))[:beam_w]
        beams = dict(items)
    best = max(beams.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]))
    return "".join(cs.itos[c] for c in best[0] if c in cs.itos)


def ops(gt: str, pr: str):
    n, m = len(gt), len(pr)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0], op[i][0] = i, "del"
    for j in range(1, m + 1):
        d[0][j], op[0][j] = j, "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt[i - 1] == pr[j - 1]:
                d[i][j], op[i][j] = d[i - 1][j - 1], "eq"
            else:
                cand = (d[i - 1][j - 1] + 1, d[i - 1][j] + 1, d[i][j - 1] + 1)
                k = cand.index(min(cand))
                d[i][j] = cand[k]
                op[i][j] = ("sub", "del", "ins")[k]
    c = Counter()
    i, j = n, m
    while i > 0 or j > 0:
        o = op[i][j]
        c[o] += 1
        if o in ("eq", "sub"):
            i, j = i - 1, j - 1
        elif o == "del":
            i -= 1
        else:
            j -= 1
    return c, d[n][m]


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0 or min(b, c) == n:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(WORK / "fullpop_r8" / "ruler.jsonl"))
    ap.add_argument("--ckpt", default=str(EDA / "weights" / "textrec_best.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--arms", default="gold,gbeam10,beam10b1,gbeam1,gbeam05,beam10b15")
    ap.add_argument("--dump", default="")
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in Path(a.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_img = defaultdict(list)
    for i, r in enumerate(rows):
        by_img[r["image"]].append((i, r))

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cs = Charset(ck["charset"])
    cfg = ck.get("config", {}) or {}
    img_h, max_w = cfg.get("img_h", C.IMG_H), cfg.get("max_w", C.MAX_W)
    model = CRNN(cs.num_classes)
    model.load_state_dict(ck["model"])
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    print("模型 %s | 字表 %d | device %s" % (Path(a.ckpt).name, cs.num_classes, dev))

    ARMS = a.arms.split(",")
    preds = {arm: {} for arm in ARMS}

    def decode_all(logp_np, meta):
        out = {}
        for arm in ARMS:
            if arm == "gold":
                out[arm] = greedy(logp_np, cs, 0.0)
            elif arm.startswith("gbeam"):
                b = arm[len("gbeam"):]
                out[arm] = greedy(logp_np, cs, float(b[0] + "." + b[1:]) if len(b) > 1 else float(b))
            elif arm.startswith("beam"):
                rest = arm[len("beam"):]
                if "b" in rest:
                    w, bt = rest.split("b", 1)
                    beta = float(bt[0] + "." + bt[1:]) if len(bt) > 1 else float(bt)
                else:
                    w, beta = rest, 0.0
                out[arm] = beam(logp_np, cs, int(w) if w else 10, beta)
            else:
                raise ValueError(arm)
        return out

    with torch.no_grad():
        for p, pairs in by_img.items():
            with Image.open(p) as im:
                im = im.convert("L")
                arrs, widths, roles = [], [], []
                for _, r in pairs:
                    crop = crop_gray(im, r["bbox"], img_h, max_w)
                    x = np.asarray(crop, dtype=np.float32) / 255.0
                    x = (x - 0.5) / 0.5
                    arrs.append(x)
                    widths.append(x.shape[1])
                    roles.append(C.ROLE_ID.get(str(r["role"]), C.N_ROLES - 1))
            for s in range(0, len(pairs), a.batch):
                part = list(range(s, min(len(pairs), s + a.batch)))
                mw = max(widths[i] for i in part)
                h = arrs[0].shape[0]
                batch = torch.full((len(part), 1, h, mw), 1.0, dtype=torch.float32)
                for k, i in enumerate(part):
                    batch[k, 0, :, : widths[i]] = torch.from_numpy(arrs[i])
                batch = batch.to(dev)
                role_t = torch.tensor([roles[i] for i in part], dtype=torch.long, device=dev)
                logp = model(batch, role_t).log_softmax(2).cpu().numpy()   # T,B,C
                for k, i in enumerate(part):
                    tmax = max(1, min(logp.shape[0], widths[i] // 8))
                    d = decode_all(logp[:tmax, k, :], pairs[i][1])
                    gi = pairs[i][0]
                    for arm in ARMS:
                        preds[arm][gi] = d[arm]

    # ---- 统计 ----
    stat = {arm: defaultdict(lambda: [0, 0, 0]) for arm in ARMS}     # (role,bucket) -> [n, exact, lev]
    tot = {arm: [0, 0, 0] for arm in ARMS}
    wrong_keys = {arm: [] for arm in ARMS}
    for i, r in enumerate(rows):
        gt = r["text"]
        for arm in ARMS:
            pr = preds[arm][i]
            k = (r["role"], lenbucket(gt))
            s = stat[arm][k]
            s[0] += 1
            s[1] += int(pr == gt)
            s[2] += levenshtein(pr, gt)
            tot[arm][0] += 1
            tot[arm][1] += int(pr == gt)
            tot[arm][2] += levenshtein(pr, gt)
            if pr != gt:
                wrong_keys[arm].append(i)

    print("\n%-10s %7s %8s %8s" % ("arm", "wrong", "exact", "CER"))
    for arm in ARMS:
        n, ex, lv = tot[arm]
        print("%-10s %7d %8.4f %8.4f" % (arm, n - ex, ex / n, lv / sum(len(r["text"]) for r in rows)))

    base = "gold"
    WSET = {arm: set(wrong_keys[arm]) for arm in ARMS}
    if "gold" in ARMS:
        gv5 = stat[base][("value", "5-7")][0] + stat[base][("value", "8-11")][0] + stat[base][("value", "12+")][0]
        gw5 = sum(stat[base][("value", b)][0] - stat[base][("value", b)][1] for b in ("5-7", "8-11", "12+"))
        print("\n[自检门] A0 全 ruler 错 %d（冻结 908）；value>=5 错 %d / n=%d（冻结 309 / 389）"
              % (len(wrong_keys[base]), gw5, gv5))

    print("\n主判据格 value>=5（n=%d）：" % gv5)
    for arm in ARMS:
        if arm == base:
            continue
        b = c = 0
        for i in range(len(rows)):
            r = rows[i]
            if r["role"] != "value" or len(r["text"]) < 5:
                continue
            wc = i in set(wrong_keys[base])
            wt = i in set(wrong_keys[arm])
            if wc and not wt:
                b += 1
            elif wt and not wc:
                c += 1
        ex_a = (gv5 - gw5) / gv5
        w_arm = sum(stat[arm][("value", bb)][0] - stat[arm][("value", bb)][1] for bb in ("5-7", "8-11", "12+"))
        ex_b = (gv5 - w_arm) / gv5
        delta = ex_b - ex_a
        disc = b + c
        se = math.sqrt(max(0.0, disc - (b - c) ** 2 / gv5)) / gv5 if disc else 0.0
        print("  %-10s exact %.4f (错 %3d)  Δ%+.4f  配对 b=%d c=%d  95%%CI [%+.4f, %+.4f]  p=%.4f"
              % (arm, ex_b, w_arm, delta, b, c, delta - 1.96 * se, delta + 1.96 * se, mcnemar(b, c)))

    print("\n次终点（主终点错误样本的操作构成 / 长度偏置）：")
    for arm in ARMS:
        S = D = I = 0
        dl, short = [], 0
        for i in wrong_keys[arm]:
            r = rows[i]
            if r["role"] != "value" or len(r["text"]) < 5:
                continue
            c, _ = ops(r["text"], preds[arm][i])
            S += c["sub"]; D += c["del"]; I += c["ins"]
            dl.append(len(r["text"]) - len(preds[arm][i]))
            short += int(len(preds[arm][i]) < len(r["text"]))
        t = max(1, S + D + I)
        print("  %-10s 错样本 %3d | sub %.3f del %.3f ins %.3f | 长度偏置 %+.3f | 偏短占比 %.3f"
              % (arm, len(dl), S / t, D / t, I / t, (sum(dl) / len(dl)) if dl else 0.0,
                 short / max(1, len(dl))))

    print("\n护栏（exact，>=5 格）：")
    roles = sorted({r["role"] for r in rows}) + ["ALL"]
    for role in roles:
        line = "  %-11s" % role
        for arm in ARMS:
            n5 = ex5 = 0
            for i, r in enumerate(rows):
                if (role == "ALL" or r["role"] == role) and len(r["text"]) >= 5:
                    n5 += 1
                    ex5 += int(preds[arm][i] == r["text"])
            line += " %s %s" % (arm, ("%.4f(%d)" % (ex5 / n5, n5)) if n5 else "-")
        print(line)
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as fh:
            for arm in ARMS:
                for i in wrong_keys[arm]:
                    r = rows[i]
                    fh.write(json.dumps({"arm": arm, "image": r["image"], "case": r["case"],
                                         "role": r["role"], "gt": r["text"], "pred": preds[arm][i],
                                         "bbox": r["bbox"]}, ensure_ascii=False) + "\n")
        print("\n错误明细 -> %s" % a.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
