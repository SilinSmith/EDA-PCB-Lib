# -*- coding: utf-8 -*-
"""训练 CRNN+CTC 文字识别器。

    python -m textrec.train --epochs 40 --batch 256
    python -m textrec.train --smoke                 # 1 轮 / 2000 条，验管线

产物（默认写到 `textrec/runs/`）：
    last.pt      每轮覆盖
    best.pt      验证集上 CER 最低的一轮，`infer.Recognizer` 读这个
    history.json 每轮的 loss / CER / 准确率
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .charset import Charset
from .dataset import CropDataset, collate, ctc_input_lens_for
from .manifest import load as load_manifest
from .model import build as build_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------- 指标
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Accum:
    """按角色分桶累计 CER 与完全正确率。"""

    def __init__(self) -> None:
        self.buckets: Dict[str, List[int]] = {}

    def add(self, role: str, pred: str, gt: str) -> None:
        b = self.buckets.setdefault(role, [0, 0, 0, 1, 0])   # ed, len, n, exact, n_exact
        b[0] += levenshtein(pred, gt)
        b[1] += max(1, len(gt))
        b[2] += 1
        b[4] += int(pred == gt)

    def merge(self, other: "Accum") -> None:
        for k, v in other.buckets.items():
            b = self.buckets.setdefault(k, [0, 0, 0, 1, 0])
            for i in range(5):
                b[i] += v[i]

    def summary(self) -> Tuple[float, float, Dict[str, dict]]:
        ed = sum(v[0] for v in self.buckets.values())
        ln = sum(v[1] for v in self.buckets.values())
        n = sum(v[2] for v in self.buckets.values())
        ex = sum(v[4] for v in self.buckets.values())
        per = {}
        for k, v in sorted(self.buckets.items()):
            per[k] = {
                "n": v[2],
                "cer": round(v[0] / max(1, v[1]), 4),
                "exact": round(v[4] / max(1, v[2]), 4),
            }
        return ed / max(1, ln), ex / max(1, n), per


# ---------------------------------------------------------------- 评估
@torch.no_grad()
def evaluate(model, loader, charset: Charset, device, max_batches: int = 0):
    """在验证集上算 CER（字符错误率）与完全正确率，并按角色分桶。"""
    was_training = model.training
    model.eval()
    acc = Accum()
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        imgs = batch["img"].to(device, non_blocking=True)
        roles = batch["role"].to(device, non_blocking=True)
        preds = model.greedy(imgs, roles, charset)
        for p, g, rid in zip(preds, batch["text"], roles.tolist()):
            acc.add(C.ROLES[rid] if 0 <= rid < len(C.ROLES) else "note", p, g)
    cer, exact, per = acc.summary()
    if was_training:
        model.train()
    return cer, exact, per


# ---------------------------------------------------------------- 主流程
def _balance_pin_number(recs: List[dict], max_bucket: int = 3) -> List[dict]:
    """给 `pin_number` 按文本长度分层**复制**到各桶样本数相等（只复制、不丢样本）。

    动机：线上失败模式是「3 位引脚号被 CTC 吞掉一位」（`113`→`13`、位号 `C160`→`C10`），
    而合成语料实测 2 位号占 79%（87000 条里 1 位 8909 / 2 位 70694 / 3 位 7660），模型先验
    天然偏向 2 位。超过 `max_bucket` 位的并进最后一桶。
    """
    buckets: Dict[int, List[dict]] = {}
    out_other: List[dict] = []
    for r in recs:
        if r.get("role") == "pin_number":
            k = min(max(1, len(str(r.get("text") or ""))), max_bucket)
            buckets.setdefault(k, []).append(r)
        else:
            out_other.append(r)
    if len(buckets) < 2:
        return recs
    target = max(len(v) for v in buckets.values())
    before = {k: len(v) for k, v in sorted(buckets.items())}
    out = list(out_other)
    for k in sorted(buckets):
        items = buckets[k]
        for i in range(target):
            out.append(items[i % len(items)])
    print("[balance] pin_number 分桶 %s -> 每桶 %d 条；总样本 %d -> %d"
          % (before, target, len(recs), len(out)))
    return out


def main(argv=None) -> int:
    d = C.DEFAULTS
    ap = argparse.ArgumentParser(description="训练 textrec CRNN+CTC")
    ap.add_argument("--manifest", default=str(C.MANIFEST))
    ap.add_argument("--charset", default=str(C.CHARSET_PATH))
    ap.add_argument("--out", default=str(C.RUNS_DIR))
    ap.add_argument("--epochs", type=int, default=d["epochs"])
    ap.add_argument("--batch", type=int, default=d["batch"])
    ap.add_argument("--lr", type=float, default=d["lr"])
    ap.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    ap.add_argument("--workers", type=int, default=d["workers"])
    ap.add_argument("--seed", type=int, default=d["seed"])
    ap.add_argument("--patience", type=int, default=d["patience"])
    ap.add_argument("--grad-clip", type=float, default=d["grad_clip"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条训练样本")
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="1 轮 / 2000 条，验管线")
    ap.add_argument("--resume", default="", help="从某个 last.pt 继续")
    ap.add_argument("--include-real", action="store_true",
                    help="把真实语料也拿进训练（默认不用，它只做验证）")
    ap.add_argument("--pin-num-balance", choices=("off", "equal"), default="off",
                    help="pin_number 按文本长度分层重采样：'equal' 让 1/2/3 位样本数相等。"
                         "线上失败模式是「3 位号读成 2 位」（`113`→`13`），而合成语料里 2 位占 79%%，"
                         "默认 off 保持原分布")
    args = ap.parse_args(argv)

    if args.smoke:
        args.epochs = 1
        args.limit = args.limit or 2000
        args.val_limit = args.val_limit or 500
        args.batch = min(args.batch, 64)
        args.workers = min(args.workers, 2)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    charset = Charset.load(Path(args.charset))
    print("[charset] " + charset.describe().replace("\n", " "))

    recs = load_manifest(Path(args.manifest))
    train_recs = [r for r in recs if r["split"] == "train"]
    val_recs = [r for r in recs if r["split"] == "val"]
    if args.include_real:
        train_recs += [r for r in recs if r["split"] == "real"]
    if args.limit:
        train_recs = train_recs[: args.limit]
    if args.val_limit:
        val_recs = val_recs[: args.val_limit]
    if args.pin_num_balance == "equal":
        train_recs = _balance_pin_number(train_recs)

    train_ds = CropDataset(train_recs, charset, augment=args.augment)
    val_ds = CropDataset(val_recs, charset, augment=False)
    print("[data] train %d (丢弃未知字符 %d)  val %d (丢弃 %d)"
          % (len(train_ds), train_ds.n_dropped_unk,
             len(val_ds), val_ds.n_dropped_unk))
    if not len(train_ds) or not len(val_ds):
        print("!! 训练集或验证集为空，先跑 python -m textrec.manifest")
        return 2

    dl_kw = dict(num_workers=args.workers, collate_fn=collate,
                 pin_memory=(device.type == "cuda"))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          drop_last=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **dl_kw)

    model = build_model(charset.num_classes).to(device)
    print("[model] CRNN 参数 %.2fM  类别 %d" % (model._n_params / 1e6, charset.num_classes))  # type: ignore[attr-defined]

    ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_epoch = int(ck.get("epoch", 0))
        print("[resume] 从 %s 的第 %d 轮继续" % (args.resume, start_epoch))

    # OneCycle 的步数按「本次还要跑几轮」算，不按 --epochs 的绝对值：
    # 续训时 start_epoch 很大（例如 57），若按绝对值 `--epochs 58` 算 total_steps，
    # warmup（pct_start=0.1）会落在本次训练之外，整个这一轮的学习率只有峰值的
    # 1/10 ~ 1/30（实测 e57 起步 lr 6e-6、峰值 1.5e-4），等于没训。
    remain = max(1, args.epochs - start_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, remain * len(train_dl)),
        pct_start=0.1, anneal_strategy="cos")
    print("[sched] 本次 %d 轮 × %d 步 = %d 步，OneCycle 峰值 lr %.2e"
          % (remain, len(train_dl), remain * len(train_dl), args.lr))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: List[dict] = []
    best_cer = 1e9
    bad = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        infeasible = 0
        for step, batch in enumerate(train_dl):
            imgs = batch["img"].to(device, non_blocking=True)
            roles = batch["role"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            tgt_lens = batch["target_len"].to(device, non_blocking=True)

            logits = model(imgs, roles)                      # T,B,C
            T = logits.shape[0]
            in_lens = ctc_input_lens_for(batch["widths"], T).to(device)
            # CTC 要求 target_len <= input_len，否则该样本是 inf 的（会被 zero_infinity 吃掉）。
            infeasible += int((tgt_lens > in_lens).sum().item())

            loss = ctc(logits.log_softmax(2), targets, in_lens, tgt_lens)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()

            tot += float(loss.item())
            n += 1
            if step % 50 == 0:
                print("  e%d %4d/%4d  loss %.4f  lr %.2e"
                      % (epoch, step, len(train_dl), tot / max(1, n),
                         opt.param_groups[0]["lr"]), flush=True)

        tr_loss = tot / max(1, n)
        val_cer, val_exact, per = evaluate(model, val_dl, charset, device)
        dt = time.time() - t0
        history.append({"epoch": epoch, "loss": round(tr_loss, 5),
                        "val_cer": round(val_cer, 5), "val_exact": round(val_exact, 5),
                        "sec": round(dt, 1), "infeasible": infeasible})
        print("== e%d  loss %.4f  valCER %.4f  valExact %.4f  %.0fs  infeasible %d"
              % (epoch, tr_loss, val_cer, val_exact, dt, infeasible), flush=True)
        print("   per-role: " + json.dumps(per, ensure_ascii=False), flush=True)

        ck = {"model": model.state_dict(), "charset": charset.chars,
              "config": {"img_h": C.IMG_H, "max_w": C.MAX_W,
                         "width_reduction": C.WIDTH_REDUCTION},
              "epoch": epoch + 1, "val_cer": val_cer, "val_exact": val_exact}
        torch.save(dict(ck, opt=opt.state_dict()), out_dir / "last.pt")
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(ck, out_dir / "best.pt")
            bad = 0
            print("   * best.pt 更新 (CER %.4f)" % val_cer, flush=True)
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print("   早停：%d 轮没有改善" % bad)
                break

        (out_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")

    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n完成。best CER %.4f -> %s" % (best_cer, out_dir / "best.pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
