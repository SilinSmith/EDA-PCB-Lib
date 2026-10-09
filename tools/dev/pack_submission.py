#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打包一个「可提交」的目录树：`<out>\\eda-pcb-main\\` + 四个平台文件。

背景
----
平台的 Dockerfile 做的是 `COPY eda-pcb-main /submission/eda-pcb-main`，整个代码树
一起进镜像；四个打包文件（`Dockerfile` / `run.sh` / `install_deps.py` /
`verify_imports.py`）在**上传目录的根**上。所以「可提交的目录」长得像：

    <out>/
      Dockerfile
      run.sh
      install_deps.py
      verify_imports.py
      eda-pcb-main/          <- 代码树

这个脚本只干「复制 + 过滤 + 体检」。它**不改**源仓库里的任何东西
（唯一的例外是 `--plan` 模式下只打印、不落地）。

为什么需要过滤
--------------
源仓库里有几坨**绝不该进镜像**的东西（实测）：
  * `eda_yolo/`    3.5 GB —— 本机 Windows venv（`Lib/site-packages/torch/...dll`）
  * `yolo_part/`   1.1 GB —— 数据集 + `runs/` 训练产物（154 MB 的 best.pt ×N）
  * `textrec/data/manifest.jsonl` 85 MB —— 训练清单，推理期只用 ckpt 里的字符表
默认排除已覆盖 `__pycache__` / `*.pyc` / `*.corrupt` / `.git` / `.gitignore` /
`*.log` / 任意层级的 `runs` `output` `work` 目录；上面那三坨用 `--exclude` 点名。
（`yolo_part/runs` 会被「任意层级 runs」规则自动干掉，但数据集不会。）

用法
----
    python -X utf8 eda-pcb\\tools\\dev\\pack_submission.py --out Y:\\workspace\\EDA\\work\\_subpack
    # 真实的提交包（推荐）：把三坨大件也排掉
    python -X utf8 eda-pcb\\tools\\dev\\pack_submission.py ^
        --out Y:\\workspace\\EDA\\work\\submission_next ^
        --exclude eda_yolo --exclude yolo_part --exclude textrec/data
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------- 默认路径 --

DEFAULT_SRC = Path(r"Y:\workspace\EDA\eda-pcb")
DEFAULT_PACK = Path(
    r"Y:\workspace\EDA\20261009引入\theSixthSubmission_44.29\prediction_result (5)")

PACK_FILES = ("Dockerfile", "run.sh", "install_deps.py", "verify_imports.py")

# ★ 四个打包文件的**权威副本**现在也放进了仓库：`<src>/packaging/`。
#   理由：那两个 py 里有真的需要修的东西（install_deps.py 曾在 build 机看不到 GPU 时
#   `sys.exit(1)` 直接把一次可用提交判成红 build；verify_imports.py 把交付路径不用的
#   httpx 当成硬依赖报警）。放在仓库里才有版本历史，也才随代码一起走。
PACK_DIR_NAME = "packaging"

# 任意层级都排除的**目录名**（整棵子树）
EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".git",
    "runs",
    "output",
    "work",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    # 仓库内那份打包文件副本：它已经在 <out>/ 根上放了一份，
    # 不需要再复制一份到 <out>/eda-pcb-main/packaging/ 里。
    "packaging",
}

# 任意层级都排除的**文件**（glob，按 basename 匹配）
EXCLUDE_FILE_GLOBS = (
    "*.pyc",
    "*.pyo",
    "*.corrupt",
    "*.log",
    ".gitignore",
    ".DS_Store",
    "Thumbs.db",
)

# 镜像里**必须有**的关键文件（相对 `--src` 的 POSIX 路径）。
# 缺席就不许打包 —— 上一次 submission 的 Dockerfile:210-213 明确写了
# 「目录结构一错两个权重都找不到，stage1 第一行就 FileNotFoundError，整批全灭」。
REQUIRED = (
    "weights/pcb43_v2_from_best.pt",     # YOLO 元件检测（assets.py:50）
    "weights/textrec_best.pt",           # CRNN 文字识别（assets.py:70）
    "submission/run.py",                 # 全部逻辑（submission/agent_system 只做 sys.path）
    "schematic_parser/metrics.py",       # 评测指标，全链路共用
    "stage1/pins.py",                    # 引脚通道
)

TOP_N = 10


# ------------------------------------------------------------------ 过滤 --


def _match_extra(rel_posix: str, name: str, extra: list[str]) -> bool:
    """`--exclude` 的匹配规则（宽松一点，三条任中其一就排除）。

    1. 相对路径的**任一段**等于该模式（`--exclude eda_yolo` / `yolo_part`）
    2. 相对路径整体匹配（`--exclude "docs/*.html"`）
    3. basename 匹配（`--exclude "*.bak"`）
    """
    for pat in extra:
        p = pat.replace("\\", "/").strip("/")
        if not p:
            continue
        if p in rel_posix.split("/"):
            return True
        if fnmatch.fnmatch(rel_posix, p):
            return True
        if fnmatch.fnmatch(name, p):
            return True
    return False


def should_skip(rel_posix: str, name: str, is_dir: bool, extra: list[str]) -> bool:
    if is_dir and name in EXCLUDE_DIR_NAMES:
        return True
    if not is_dir:
        for g in EXCLUDE_FILE_GLOBS:
            if fnmatch.fnmatch(name, g):
                return True
    return _match_extra(rel_posix, name, extra)


def iter_tree(src: Path, extra: list[str]):
    """自顶向下遍历，产出 (绝对路径, 相对 src 的 POSIX 路径, 是否目录)。

    被排除的目录整棵剪掉（`dirs[:] = ...`），所以不会白走 3.5 GB 的 venv。
    """
    for root, dirs, files in os.walk(src):
        root_p = Path(root)
        rel_root = root_p.relative_to(src).as_posix()
        rel_root = "" if rel_root == "." else rel_root

        kept_dirs = []
        for d in sorted(dirs):
            rel = ("%s/%s" % (rel_root, d)) if rel_root else d
            if should_skip(rel, d, True, extra):
                continue
            kept_dirs.append(d)
            yield root_p / d, rel, True
        dirs[:] = kept_dirs

        for f in sorted(files):
            rel = ("%s/%s" % (rel_root, f)) if rel_root else f
            if should_skip(rel, f, False, extra):
                continue
            yield root_p / f, rel, False


# ------------------------------------------------------------ 体检 / 复制 --


def check_required(src: Path) -> list[str]:
    missing = []
    for rel in REQUIRED:
        p = src / Path(rel)
        if not p.is_file():
            missing.append(rel)
    return missing


def check_pack_files(pack: Path) -> list[str]:
    return [n for n in PACK_FILES if not (pack / n).is_file()]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0
    return "%.1f GB" % n


def safe_reset(dir_: Path, out_root: Path) -> None:
    """删掉上一轮的 `<out>/eda-pcb-main`。**只允许删 out_root 的直接子目录**。"""
    if not dir_.exists():
        return
    if dir_.resolve().parent != out_root.resolve():
        raise SystemExit("[pack] 拒绝删除不在 --out 下的路径: %s" % dir_)
    shutil.rmtree(dir_)


# ------------------------------------------------------------------ main --


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="打包可提交的 EDA 目录树（代码树 + 四个平台文件）。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                    help="输出目录；会生成 <out>/eda-pcb-main/ 与 <out>/<四个文件>")
    ap.add_argument("--src", default=str(DEFAULT_SRC),
                    help="代码树源目录（默认 %s）" % DEFAULT_SRC)
    ap.add_argument("--pack", default=str(DEFAULT_PACK),
                    help="四个打包文件的来源目录（默认上一次的 prediction_result (5)）")
    ap.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                    help="额外排除的 glob/路径段，可重复。"
                         "例：--exclude eda_yolo --exclude yolo_part --exclude textrec/data")
    ap.add_argument("--plan", action="store_true",
                    help="只体检 + 统计，不落盘")
    ap.add_argument("--quiet", action="store_true", help="只打印摘要")
    a = ap.parse_args(argv)

    src = Path(a.src).resolve()
    pack = Path(a.pack).resolve()
    out = Path(a.out).resolve()
    extra = list(a.exclude)

    # ★ 优先用仓库内的权威副本 <src>/packaging/（若四个文件齐全），否则回落到 --pack。
    #   这样打包不再依赖「上一次平台返回的 prediction_result 目录还在不在」。
    pack_src = pack
    repo_pack = src / PACK_DIR_NAME
    if repo_pack.is_dir() and not check_pack_files(repo_pack):
        pack_src = repo_pack
        print("[pack] 打包文件取自仓库内权威副本: %s" % pack_src)
    else:
        print("[pack] 打包文件取自 --pack: %s" % pack_src)

    # ---------- 0) 体检：不通过就别浪费一次提交机会 ----------
    bad = False
    if not src.is_dir():
        print("[pack] 错误：--src 不是目录: %s" % src, file=sys.stderr)
        bad = True
    if not pack_src.is_dir():
        print("[pack] 错误：打包文件来源不是目录: %s" % pack_src, file=sys.stderr)
        bad = True
    if bad:
        return 2

    missing_req = check_required(src)
    if missing_req:
        print("[pack] 错误：--src 缺少**关键文件**，拒绝打包：", file=sys.stderr)
        for rel in missing_req:
            print("         - %s" % rel, file=sys.stderr)
        print("[pack] 这些路径由 Dockerfile:210-213 / assets.py:50,70 硬性依赖；"
              "缺一个就是整批 case 全灭。", file=sys.stderr)
        return 3

    missing_pack = check_pack_files(pack_src)
    if missing_pack:
        print("[pack] 错误：--pack 缺少打包文件：%s" % ", ".join(missing_pack),
              file=sys.stderr)
        print("[pack] 应放 <src>/%s/ 或 --pack 指向上一次返回的 prediction_result 目录。"
              % PACK_DIR_NAME, file=sys.stderr)
        return 4

    # ---------- 1) 走一遍树，决定拷贝集合 ----------
    files: list[tuple[Path, str]] = []
    dirs: list[str] = []
    n_skipped = 0
    for p, rel, is_dir in iter_tree(src, extra):
        if is_dir:
            dirs.append(rel)
        else:
            files.append((p, rel))
    # 统计被排除掉的东西（只走名字，不读内容；用 os.scandir 快速过）
    for root, ds, fs in os.walk(src):
        rp = Path(root).relative_to(src).as_posix()
        rp = "" if rp == "." else rp
        for d in ds:
            if should_skip(("%s/%s" % (rp, d)) if rp else d, d, True, extra):
                n_skipped += 1
        for f in fs:
            if should_skip(("%s/%s" % (rp, f)) if rp else f, f, False, extra):
                n_skipped += 1
        ds[:] = [d for d in ds
                 if not should_skip(("%s/%s" % (rp, d)) if rp else d, d, True, extra)]

    total_bytes = sum(p.stat().st_size for p, _ in files)
    largest = sorted(files, key=lambda t: t[0].stat().st_size, reverse=True)[:TOP_N]

    if not a.quiet:
        print("[pack] src  = %s" % src)
        print("[pack] pack = %s" % pack)
        print("[pack] out  = %s%s" % (out, "   (--plan：不落盘)" if a.plan else ""))
        if extra:
            print("[pack] 额外排除: %s" % ", ".join(extra))

    if a.plan:
        print("[plan] 将拷贝 %d 个文件 / %s，%d 个目录；排除 %d 项"
              % (len(files), human(total_bytes), len(dirs), n_skipped))
        return 0

    # ---------- 2) 落盘 ----------
    dest_root = out / "eda-pcb-main"
    out.mkdir(parents=True, exist_ok=True)
    safe_reset(dest_root, out)
    dest_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    copied_bytes = 0
    for p, rel in files:
        dst = dest_root / Path(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        copied += 1
        copied_bytes += dst.stat().st_size

    for name in PACK_FILES:
        shutil.copy2(pack_src / name, out / name)
        if not a.quiet:
            print("[pack] + %s  (%s)" % (name, human((out / name).stat().st_size)))

    # ---------- 3) 摘要 ----------
    tops = sorted({rel.split("/")[0] for _, rel in files})
    print("")
    print("=" * 72)
    print("打包完成: %s" % out)
    print("=" * 72)
    print("拷贝文件数 : %d" % copied)
    print("拷贝总字节 : %d  (%s)" % (copied_bytes, human(copied_bytes)))
    print("排除条目数 : %d  (被排除的目录按 1 条计，不展开)" % n_skipped)
    print("打包文件   : %s" % ", ".join(PACK_FILES))
    print("")
    print("最大的 %d 个文件：" % TOP_N)
    for p, rel in largest:
        print("  %10s  %s" % (human(p.stat().st_size), rel))
    print("")
    print("顶层条目（%d 个）：" % len(tops))
    for t in tops:
        tp = dest_root / t
        if tp.is_dir():
            n = sum(1 for _ in tp.rglob("*") if _.is_file())
            b = sum(_.stat().st_size for _ in tp.rglob("*") if _.is_file())
            print("  [D] %-22s %5d files  %10s" % (t, n, human(b)))
        else:
            print("  [F] %-22s           %10s" % (t, human(tp.stat().st_size)))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
