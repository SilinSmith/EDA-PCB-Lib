#!/usr/bin/env python3
"""compat_audit —— 版本适配审计器（EDA 赛题六 · 本地 / 评测机双端可用）

**它解决什么问题**
队友在 `torch 2.13.0+cu130 / numpy 2.4.6 / opencv-python 5.0.0.93` 上写代码，
评测机（基础镜像 `edarace-agent-base:ubuntu24`）是
`torch 2.5.1+cu121 / torchvision 0.20.1+cu121 / numpy 2.5.3 / opencv-python-headless 4.10.0.84`。
两边差了一个大版本号（CUDA 13.0 vs 12.1），一定会有 API 在旧机器上不存在。

**为什么不做「自动改写源码」**
torch 2.13 → 2.5.1 不是可判定的重写，猜规则会改错。这个脚本改走**可验证**的三条路：

1. `scan`  ★ 把仓库里用到的每一个 `torch.* / np.* / cv2.* / ...` 属性链静态收集出来，
          然后在**当前解释器**（= 目标环境）里逐条 `getattr` 解析。
          解析不到的，就是评测机上**必定**崩的地方，直接给出 `文件:行`。
          ——秒级、无副作用、不需要真 GT、不需要跑模型。
2. `shim`  把「机械可还原」的差异（numpy 2.0 删掉的别名、`np.cross` 的 2 元向量语义）
          自动生成一个 `compat_shim.py`。包在最前面 import 一次，旧代码就能在旧机器上跑。
3. `cross` 单独盯住已经害过我们一次的那个坑：numpy ≥ 2 对 2 元向量 `np.cross` 直接抛
          `ValueError: Both input arrays must be (arrays of) 3-dimensional vectors`。
          （第五次提交阶段一 0/10 全 FAIL 就是它。）
4. `manifest`  ★★ 把评测机的「属性面」快照成一个 JSON 文件。
          没有 Docker、装不上评测机那套库的人（比如队友本机是 torch 2.13+cu130），
          拿到这个 JSON 后 `scan --manifest` 就能在**本机**跑出评测机口径的结论 ——
          **不需要 Docker，不需要装 numpy/torch，纯标准库**。

**用法**
    # 看目标环境版本
    python compat_audit.py env

    # 【在评测机镜像里跑一次】导出属性面清单，交给没 Docker 的人
    python compat_audit.py manifest --out evalenv_manifest.json --repo /repo

    # 【谁都能跑】拿清单做审计：结论 = 评测机口径
    python compat_audit.py scan --repo . --manifest evalenv_manifest.json

    # 静态属性链审计（不指定 --manifest 时，口径是**当前解释器**）
    python compat_audit.py scan --repo /submission/eda-pcb-main --out /output/compat.md

    # 生成适配垫片
    python compat_audit.py shim --out /submission/compat_shim.py

    # 退出码：发现 HIGH 级问题返回 1，可直接当提交前闸门
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import pkgutil
import shutil
import subprocess
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- 版本

PROBES = [
    ("python", lambda: sys.version.split()[0]),
    ("torch", lambda: _v("torch")),
    ("torchvision", lambda: _v("torchvision")),
    ("numpy", lambda: _v("numpy")),
    ("scipy", lambda: _v("scipy")),
    ("cv2", lambda: _v("cv2")),
    ("PIL", lambda: _v("PIL")),
    ("ultralytics", lambda: _v("ultralytics")),
    ("yaml", lambda: _v("yaml")),
    ("psutil", lambda: _v("psutil")),
    ("matplotlib", lambda: _v("matplotlib")),
]


def _v(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
    except Exception as e:                       # noqa: BLE001
        return f"<缺失: {type(e).__name__}>"
    return str(getattr(m, "__version__", "?"))


def probe_env(verbose: bool = True) -> dict:
    out: dict[str, str] = {}
    for name, fn in PROBES:
        try:
            out[name] = fn()
        except Exception as e:                   # noqa: BLE001
            out[name] = f"<探测失败: {type(e).__name__}>"
    # GPU / CUDA
    try:
        import torch
        out["torch.cuda.is_available"] = str(torch.cuda.is_available())
        out["torch.version.cuda"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:                       # noqa: BLE001
        out["torch.cuda"] = f"<探测失败: {type(e).__name__}>"
    # opencv 是 headless 还是普通版 / 带不带 contrib：查 dist-info 才准
    # （`cv2.__file__` 里并不含 "headless" 字样，早先那版判据是错的）
    try:
        import cv2
        out["cv2 有 ximgproc(contrib)"] = str(hasattr(cv2, "ximgproc"))
    except Exception:                            # noqa: BLE001
        pass
    try:
        import glob
        import site
        cands: list[str] = []
        dirs = list(sys.path)
        try:
            dirs += list(site.getsitepackages())
        except Exception:                        # noqa: BLE001
            pass
        for d in dirs:
            if not d or not os.path.isdir(d):
                continue
            cands += [os.path.basename(p)
                      for p in glob.glob(os.path.join(d, "*opencv*.dist-info"))]
        out["opencv dist-info"] = ", ".join(sorted(set(cands))) or "<找不到>"
    except Exception as e:                       # noqa: BLE001
        out["opencv dist-info"] = f"<探测失败: {type(e).__name__}>"
    if verbose:
        w = max(len(k) for k in out)
        for k, v in out.items():
            print(f"  {k:<{w}}  {v}")
    return out


# ----------------------------------------------------------------- 静态属性链收集

# 根名 -> (导入名, 人类可读)  。只处理无歧义的第三方根，避免把局部变量当模块。
ROOTS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "np": "numpy",
    "numpy": "numpy",
    "cv2": "cv2",
    "scipy": "scipy",
    "PIL": "PIL",
    "ultralytics": "ultralytics",
    "yaml": "yaml",
    "psutil": "psutil",
}

MAX_DEPTH = 5


def _chain(node: ast.AST) -> str | None:
    """把 `a.b.c` 还原成 'a.b.c'；只要根是 Name 且全链都是 Attribute。"""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    parts.reverse()
    return ".".join(parts)


def collect(repo: Path) -> dict[str, list[str]]:
    """返回 {属性链: ['文件:行', ...]}。"""
    found: dict[str, list[str]] = defaultdict(list)
    for p in sorted(repo.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        try:
            # utf-8-sig：仓库里有 79 个文件带 Windows BOM，用 utf-8 会误判成语法错误
            src = p.read_text(encoding="utf-8-sig", errors="replace")
            tree = ast.parse(src)
        except (SyntaxError, OSError):
            continue
        rel = p.relative_to(repo).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            c = _chain(node)
            if c is None:
                continue
            bits = c.split(".")
            if bits[0] not in ROOTS or len(bits) < 2 or len(bits) > MAX_DEPTH:
                continue
            found[c].append(f"{rel}:{node.lineno}")
    return found


def _resolve(chain: str) -> tuple[bool, str]:
    """在当前解释器里解析属性链。返回 (是否解析成功, 第一个失败的前缀)。"""
    import_root = ROOTS[chain.split(".")[0]]
    try:
        obj = importlib.import_module(import_root)
    except Exception as e:                       # noqa: BLE001
        return False, f"{import_root} 本身导入失败({type(e).__name__})"
    acc = import_root
    for part in chain.split(".")[1:]:
        nxt = f"{acc}.{part}"
        try:
            obj = getattr(obj, part)
        except AttributeError:
            # 中间层可能是尚未导入的子模块（如 torch.nn.functional 没被 import 过）
            try:
                obj = importlib.import_module(nxt)
            except Exception:                    # noqa: BLE001
                return False, nxt
        acc = nxt
    return True, ""


MANIFEST_FORMAT = "compat_audit.manifest/1"


def scan(repo: Path, manifest: dict | None = None, allow: dict[str, str] | None = None
         ) -> tuple[list[dict], int, list[dict], list[dict]]:
    """返回 (missing, 链总数, unknown, accepted)。

    manifest=None  → 在当前解释器里逐条 getattr（口径 = 当前环境）。
    manifest=dict  → 用清单判定：清单里记过的用**记录值**，没记过的走属性面推断；
                     推断不出来的进 unknown（不冤枉，也不放过）。
    allow          → 「已接受」的属性链：仍会列出，但不算失败（例如被 hasattr 保护的调用）。
    """
    allow = allow or {}
    chains = collect(repo)
    missing: list[dict] = []
    unknown: list[dict] = []
    accepted: list[dict] = []
    for c, sites in sorted(chains.items()):
        sites_u = sorted(set(sites))
        if manifest is None:
            ok, failed = _resolve(c)
            if not ok:
                item = {"chain": c, "failed_at": failed, "sites": sites_u}
                (accepted if c in allow else missing).append(item)
            continue
        verdict, why = resolve_via_manifest(c, manifest)
        if verdict == "missing":
            item = {"chain": c, "failed_at": why, "sites": sites_u}
            (accepted if c in allow else missing).append(item)
        elif verdict == "unknown":
            unknown.append({"chain": c, "why": why, "sites": sites_u})
    for it in accepted:
        it["reason"] = allow[it["chain"]]
    return missing, len(chains), unknown, accepted


# ------------------------------------------------------------------ manifest 模式

# 在评测机里快照「属性面」时要额外点名 import 的子模块。
# 原因：dir(pkg) 只列出**已经被导入过**的子模块，这些常用的一定要显式 import 一次，
# 否则清单里就没有它们的属性面，本机 scan 会把相应调用判成 UNKNOWN。
SURFACE_MODULES = [
    "torch", "torchvision", "numpy", "cv2", "scipy", "PIL", "ultralytics",
    "yaml", "psutil", "matplotlib",
    "torch.cuda", "torch.nn", "torch.nn.functional", "torch.nn.init",
    "torch.optim", "torch.utils", "torch.utils.data", "torch.backends",
    "torch.distributed", "torch.linalg", "torch.onnx", "torch.amp",
    "torchvision.transforms", "torchvision.ops", "torchvision.io",
    "torchvision.utils", "torchvision.models",
    "numpy.linalg", "numpy.random", "numpy.fft", "numpy.char", "numpy.ma",
    "numpy.polynomial", "numpy.testing", "numpy.typing",
    "cv2.dnn",
    "scipy.ndimage", "scipy.signal", "scipy.spatial", "scipy.optimize",
    "scipy.interpolate", "scipy.stats", "scipy.cluster", "scipy.linalg",
    "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont", "PIL.ImageOps",
    "PIL.ImageFilter",
    "matplotlib.pyplot", "matplotlib.patches", "matplotlib.colors",
]


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:                            # noqa: BLE001
        return None


def _on_disk_submodules(mod) -> list[str]:
    """包在磁盘上有哪些直接子模块。

    为什么需要它：`dir(mod)` **只列出已经被导入过**的子模块。比如 `scipy.special`
    如果快照时没被 import，dir(scipy) 里就没有它，本机 scan 会把
    `scipy.special.xxx` 误判成「不存在」。问文件系统就没这个问题。
    只补一层：更深的名字没被快照到 → 走 UNKNOWN，不会假通过、也不会冤枉。
    """
    path = getattr(mod, "__path__", None)
    if not path:
        return []
    try:
        return [m.name for m in pkgutil.iter_modules(path)]
    except Exception:                            # noqa: BLE001
        return []


def snapshot_surface() -> dict[str, list[str]]:
    """{模块全名: 该模块的公开属性名列表}，只记名字，不记值（JSON 才装得下）。"""
    surf: dict[str, list[str]] = {}

    def _snap(name: str, mod) -> None:
        if name in surf:
            return
        try:
            surf[name] = sorted(x for x in dir(mod) if not x.startswith("__"))
        except Exception:                        # noqa: BLE001
            pass

    for name in SURFACE_MODULES:
        m = _try_import(name)
        if m is not None:
            _snap(name, m)

    for root in sorted(set(ROOTS.values())):
        m = _try_import(root)
        if m is None:
            continue
        # dir() 漏掉的磁盘子模块，补进根模块的属性面
        extra = _on_disk_submodules(m)
        if extra:
            surf[root] = sorted(set(surf.get(root, [])) | set(extra))
        # 再从命名空间里捡「已经是模块」的属性，自动补上一层
        # （这样 torch.onnx、numpy.rec 之类没点名的也能进清单）
        try:
            items = list(vars(m).items())
        except Exception:                        # noqa: BLE001
            continue
        for _attr, val in items:
            if not isinstance(val, types.ModuleType):
                continue
            sub = getattr(val, "__name__", "")
            if sub.startswith(root):
                _snap(sub, val)
    return surf


def build_manifest(repo: Path | None, env: dict) -> dict:
    # chains 的值："" = 解析成功；否则是「第一个失败的前缀」。
    # 记前缀而不是 bool，本机 scan 才能报出「断在哪」。
    chains: dict[str, str] = {}
    if repo is not None:
        for c in collect(repo):
            ok, failed = _resolve(c)
            chains[c] = "" if ok else (failed or c)
    return {
        "format": MANIFEST_FORMAT,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "env": env,
        "roots": ROOTS,
        "np_removed": NP_REMOVED,
        "chains": chains,
        "surface": snapshot_surface(),
    }


def load_manifest(path: Path) -> dict:
    man = json.loads(Path(path).read_text(encoding="utf-8"))
    fmt = man.get("format")
    if fmt != MANIFEST_FORMAT:
        raise SystemExit(f"[manifest] 格式不对：期望 {MANIFEST_FORMAT}，拿到 {fmt!r}")
    return man


def _chain_missing(v) -> bool:
    """清单里 chains 的值 → 是否「在目标环境不存在」。

    新格式："" = 解析成功，非空字符串 = 第一个失败的前缀。
    老格式：bool（兼容用）。
    """
    if isinstance(v, bool):
        return not v
    return bool(v)


def resolve_via_manifest(chain: str, man: dict) -> tuple[str, str]:
    """返回 ('ok'|'missing'|'unknown', 说明)。

    判定顺序：
      1. 清单的 `chains` 里逐条记录过 → 直接用记录值（最权威）。
      2. 否则沿属性面走：某一段不在属性面里 → missing；
         需要的属性面本身没被快照过 → unknown。
    """
    chains = man.get("chains") or {}
    if chain in chains:
        v = chains[chain]
        if _chain_missing(v):
            return "missing", (v if isinstance(v, str) and v else chain)
        return "ok", ""
    surface = man.get("surface") or {}
    roots = man.get("roots") or ROOTS
    bits = chain.split(".")
    import_root = roots.get(bits[0])
    if not import_root:
        return "unknown", f"根名 {bits[0]} 不在清单里"
    if import_root not in surface:
        return "unknown", f"清单里没有 {import_root} 的属性面"
    acc = import_root
    for part in bits[1:]:
        names = surface.get(acc)
        if names is None:
            return "unknown", f"清单里没有 {acc} 的属性面"
        if part not in names:
            return "missing", f"{acc}.{part}"
        acc = f"{acc}.{part}"
    return "ok", ""


def load_allow(path: Path) -> dict[str, str]:
    """读「已接受」清单：每行一条属性链，可用 `# 原因` 附理由。"""
    allow: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        chain, _, reason = line.partition("#")
        chain = chain.strip()
        if chain:
            allow[chain] = reason.strip() or "（未写理由）"
    return allow


# ------------------------------------------------------------- numpy 2.x 移除别名

# numpy 2.0 删掉的名字 -> 建议替代（None = 没有等价物，只能改源码）
NP_REMOVED = {
    "float_": "float64", "unicode_": "str_", "string_": "bytes_",
    "NaN": "nan", "Inf": "inf", "infty": "inf", "Infinity": "inf",
    "NINF": "-inf", "PINF": "inf", "PZERO": "0.0", "NZERO": "-0.0",
    "round_": "round", "product": "prod", "cumproduct": "cumprod",
    "sometrue": "any", "alltrue": "all", "in1d": "isin",
    "row_stack": "vstack", "trapz": "trapezoid", "msort": "sort",
    "mat": "asmatrix", "asfarray": "asarray(dtype=float)",
    "issubsctype": None, "set_string_function": None, "deprecate": None,
    "safe_eval": None, "issctype": None, "maximum_sctype": None,
    "sctype2char": None, "obj2sctype": None, "find_common_type": None,
    "cast": None, "source": None, "lookfor": None, "who": None,
    "sctypes": None, "byte_bounds": None, "get_array_wrap": None,
    "set_numeric_ops": None, "add_newdoc": None, "disp": None,
    "recfromcsv": None, "recfromtxt": None, "typeDict": None,
}

SHIM_HEADER = '''"""compat_shim —— 由 compat_audit.py 自动生成，请勿手改。

作用：让「在 torch 2.13 / numpy 2.4 上写的代码」在评测机
（torch 2.5.1+cu121 / numpy 2.5.3）上少炸一点。

**怎么用**：在被测代码的最前面 import 一次即可 ——
    import compat_shim  # noqa: F401

**注意**：垫片只还原「机械可还原」的差异。真正的 API 缺失（`torch.xxx` 在新版才有）
垫不了，只能改源码 —— 那部分看 `compat_audit.py scan` 的报告。
生成时间环境见文件末尾。
"""

import warnings as _warnings

'''


def emit_shim(out: Path, env: dict) -> None:
    parts = [SHIM_HEADER]
    parts.append("try:\n    import numpy as _np\nexcept Exception:\n    _np = None\n\n")
    parts.append("# ---- numpy 2.0 移除的别名 ----\n")
    parts.append("if _np is not None:\n")
    parts.append("    _ALIAS = {\n")
    for old, new in NP_REMOVED.items():
        if new is None:
            continue
        parts.append(f"        {old!r}: _np.{new.split('(')[0]},\n")
    parts.append("    }\n")
    parts.append("    for _old, _new in _ALIAS.items():\n")
    parts.append("        if not hasattr(_np, _old):\n")
    parts.append("            try:\n")
    parts.append("                setattr(_np, _old, _new)\n")
    parts.append("            except Exception:\n")
    parts.append("                pass\n\n")
    parts.append('''    # ---- np.cross 的 2 元向量语义（numpy >= 2.0 直接抛 ValueError）----
    # 旧语义：二维向量 (x, y) 的叉积是标量 x1*y2 - y1*x2。
    # numpy 2.0 起要求「(arrays of) 3-dimensional vectors」，2 元向量会抛：
    #   ValueError: Both input arrays must be (arrays of) 3-dimensional vectors,
    #               but they are 2 and 2 dimensional instead.
    # 这个坑真实发生过：第五次提交阶段一 10/10 全 FAIL，调用点在 pipeline/pskel.py 的 rdp()。
    if not getattr(_np.cross, "_compat_shim", False):
        _orig_cross = _np.cross

        def _cross_compat(a, b, *args, **kwargs):
            try:
                return _orig_cross(a, b, *args, **kwargs)
            except ValueError:
                _a = _np.asarray(a, dtype=float)
                _b = _np.asarray(b, dtype=float)
                if _a.shape and _b.shape and _a.shape[-1] == 2 and _b.shape[-1] == 2:
                    _warnings.warn(
                        "compat_shim: np.cross 收到 2 元向量，已按 numpy<2 语义"
                        "返回标量叉积；建议改用显式公式（见 pipeline/pskel.py 的 _cross2）。",
                        RuntimeWarning, stacklevel=2)
                    return _a[..., 0] * _b[..., 1] - _a[..., 1] * _b[..., 0]
                raise

        _cross_compat._compat_shim = True
        _np.cross = _cross_compat

    # ---- np.array(..., copy=False) 在 numpy >= 2.0 语义收紧 ----
    # 2.0 起 copy=False 表示「绝不拷贝」，做不到就抛 ValueError；想要旧行为要用 copy=None。
    # 这里只发一次警告，不改写（改写风险大于收益）。
    _np.array = _np.array  # 占位：保持原样，由 compat_audit.py scan 报告调用点

''')
    parts.append("\n# ---- 生成环境快照 ----\n")
    parts.append("GENERATED_ON = {\n")
    for k, v in env.items():
        parts.append(f"    {k!r}: {v!r},\n")
    parts.append("}\n")
    out.write_text("".join(parts), encoding="utf-8")


# ------------------------------------------------------------------------ numpy cross

def find_cross(repo: Path) -> list[str]:
    hits: list[str] = []
    for p in sorted(repo.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        try:
            src = p.read_text(encoding="utf-8-sig", errors="replace")
            tree = ast.parse(src)
        except (SyntaxError, OSError):
            continue
        rel = p.relative_to(repo).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = _chain(f) if isinstance(f, ast.Attribute) else None
            if name in ("np.cross", "numpy.cross"):
                seg = (ast.get_source_segment(src, node) or "").replace("\n", " ")
                hits.append(f"{rel}:{node.lineno}  {seg[:170]}")
    return hits


# --------------------------------------------------------------------- guard

def run_guard(a) -> int:
    """把 compat_guard 经 sitecustomize 注入子进程，跑目标命令并汇总事件。"""
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().with_name("compat_guard.py")
    if not src.is_file():
        print(f"[guard] 找不到 {src} —— compat_guard.py 必须和 compat_audit.py 放同一目录",
              file=sys.stderr)
        return 2
    shutil.copy2(src, outdir / "compat_guard.py")
    (outdir / "sitecustomize.py").write_text(
        "# 由 compat_audit.py guard 生成：让 Python 启动时自动装上守卫\n"
        "import compat_guard  # noqa: F401\n", encoding="utf-8")

    report = Path(a.report) if a.report else outdir / "compat_guard.json"
    rest = [x for x in a.rest if x != "--"]
    if not rest:
        print("[guard] 没给要运行的命令。用法：compat_audit.py guard -- python3 -u ...",
              file=sys.stderr)
        return 2

    env = dict(os.environ)
    env["PYTHONPATH"] = str(outdir) + os.pathsep + env.get("PYTHONPATH", "")
    env["COMPAT_GUARD_REPORT"] = str(report)
    env["COMPAT_GUARD_NONFATAL"] = "0" if a.fatal else "1"
    print(f"[guard] PYTHONPATH = {env['PYTHONPATH']}")
    print(f"[guard] 报告        = {report}")
    print(f"[guard] 运行        = {' '.join(rest)}")
    print(f"[guard] NONFATAL    = {env['COMPAT_GUARD_NONFATAL']}"
          f"{'（改回旧语义继续跑，一次收全）' if not a.fatal else '（记录后立即抛错）'}")
    sys.stdout.flush()

    rc = subprocess.call(rest, env=env)
    print(f"\n[guard] 目标命令退出码 = {rc}")

    events: list[dict] = []
    if report.is_file():
        try:
            events = json.loads(report.read_text(encoding="utf-8")).get("events", [])
        except Exception as e:                       # noqa: BLE001
            print(f"[guard] 报告读取失败：{type(e).__name__}: {e}", file=sys.stderr)
    print(f"[guard] 收集到 {len(events)} 条兼容事件"
          f"{'（0 条 = 语义级也没问题）' if not events else ''}")
    by: dict[str, list] = defaultdict(list)
    for e in events:
        by[e["kind"]].append(e)
    for kind, items in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"  [{len(items):4d}] {kind}")
        for e in items[:5]:
            print(f"           {e['caller']}")
            print(f"           {e['detail'][:170]}")
        if len(items) > 5:
            print(f"           …还有 {len(items) - 5} 条（见报告）")
    hard = [e for e in events if not e["kind"].endswith("Warning")]
    warn = [e for e in events if e["kind"].endswith("Warning")]
    print(f"[guard] 其中硬问题 {len(hard)} 条、弃用/未来警告 {len(warn)} 条"
          f"（警告只在 --strict 下才算失败）")
    return 1 if (rc != 0 or hard or (a.strict and events)) else 0


# ------------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="compat_audit",
        description="版本适配审计器：在目标环境里找出队友代码会崩的 API。")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env", help="打印当前环境的版本快照")

    ps = sub.add_parser("scan", help="静态属性链审计（在目标环境里跑）")
    ps.add_argument("--repo", required=True)
    ps.add_argument("--out", default="")
    ps.add_argument("--json", dest="json_out", default="")
    ps.add_argument("--manifest", default="",
                    help="评测机属性面清单（由 manifest 子命令导出）。"
                         "给了它就按评测机口径判定，本机无需装 numpy/torch")
    ps.add_argument("--strict-unknown", action="store_true",
                    help="把 UNKNOWN（清单没覆盖到）也算作失败")
    ps.add_argument("--allow", default="",
                    help="「已接受」清单文件：每行一条属性链，可写 `# 原因`。"
                         "命中的仍然列出，但不算失败（例如被 hasattr 保护的调用）")

    pm = sub.add_parser("manifest",
                        help="导出评测机属性面清单，交给没 Docker / 装不上那套库的人")
    pm.add_argument("--out", required=True)
    pm.add_argument("--repo", default="",
                    help="顺便把该仓库现有属性链的解析结果也逐条记进去（可选，强烈建议）")

    pk = sub.add_parser("cross", help="只查 np.cross 调用点")
    pk.add_argument("--repo", required=True)

    ph = sub.add_parser("shim", help="生成 compat_shim.py")
    ph.add_argument("--out", required=True)
    ph.add_argument("--manifest", default="",
                    help="可选：垫片文件末尾记的「生成环境」用清单里的评测机快照，"
                         "而不是本机（免得把本机版本记进去误导人）")

    pg = sub.add_parser(
        "guard",
        help="注入运行期守卫跑目标命令，收集语义级差异（静态扫描扫不到的那一半）")
    pg.add_argument("--outdir", default="/tmp/compat_guard",
                    help="sitecustomize 落盘目录（默认 /tmp/compat_guard）")
    pg.add_argument("--report", default="", help="事件 JSON 落盘路径")
    pg.add_argument("--fatal", action="store_true",
                    help="记完立刻按原样抛错；默认是改回旧语义继续跑，以便一次收全")
    pg.add_argument("--strict", action="store_true",
                    help="把 DeprecationWarning/FutureWarning 也算作失败")
    pg.add_argument("rest", nargs=argparse.REMAINDER,
                    help="要运行的命令，写法：guard -- python3 -u agent_system -input /input ...")

    a = ap.parse_args(argv)

    if a.cmd == "guard":
        return run_guard(a)

    if a.cmd == "env":
        print("== 当前环境 ==")
        probe_env()
        return 0

    if a.cmd == "cross":
        repo = Path(a.repo).resolve()
        hits = find_cross(repo)
        print(f"np.cross 调用点 {len(hits)} 处：")
        for h in hits:
            print(f"  {h}")
        return 1 if hits else 0

    if a.cmd == "shim":
        out = Path(a.out)
        if a.manifest:
            env = dict(load_manifest(Path(a.manifest))["env"])
        else:
            env = probe_env(verbose=False)
        emit_shim(out, env)
        print(f"已写出 {out}  ({out.stat().st_size} B)")
        return 0

    if a.cmd == "manifest":
        out = Path(a.out)
        repo = Path(a.repo).resolve() if a.repo else None
        env = probe_env(verbose=False)
        print("== 导出评测机属性面清单 ==")
        for k, v in env.items():
            print(f"  {k:<24} {v}")
        print(f"  正在快照 {len(SURFACE_MODULES)} 个模块的属性面 …")
        man = build_manifest(repo, env)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(man, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        n_ch = len(man["chains"])
        n_bad = sum(1 for v in man["chains"].values() if _chain_missing(v))
        print(f"  属性面：{len(man['surface'])} 个模块")
        if repo is not None:
            print(f"  逐条记录：{n_ch} 条属性链"
                  f"（其中 {n_bad} 条在本环境不存在，{n_ch - n_bad} 条通过）")
        print(f"[写出] {out}  ({out.stat().st_size:,} B)")
        return 0

    # ---- scan ----
    repo = Path(a.repo).resolve()
    manifest = load_manifest(Path(a.manifest)) if a.manifest else None
    allow = load_allow(Path(a.allow)) if a.allow else {}
    env = dict(manifest["env"]) if manifest else probe_env(verbose=False)
    missing, n_chains, unknown, accepted = scan(repo, manifest, allow)

    lines: list[str] = []
    lines.append("# 版本适配审计报告")
    lines.append("")
    lines.append(f"仓库：`{repo}`")
    lines.append("")
    if manifest:
        lines.append(f"判定口径：**评测机属性面清单**（{a.manifest}，"
                     f"导出于 {manifest.get('generated_at', '?')}）")
    else:
        lines.append("判定口径：**当前解释器**"
                     "（⚠ 未给 --manifest，若本机不是评测机环境，结论不可信）")
    lines.append("")
    lines.append("## 目标环境快照（就是跑这次审计的解释器）")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    for k, v in env.items():
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")
    lines.append(f"## 结论：扫到 {n_chains} 条第三方属性链，"
                 f"其中 **{len(missing)} 条在目标环境不存在**"
                 + (f"，另有 **{len(unknown)} 条无法判定**" if unknown else "")
                 + (f"，**{len(accepted)} 条已接受**（不算失败）" if accepted else ""))
    lines.append("")
    if missing:
        lines.append("| # | 属性链 | 断在哪 | 调用点 |")
        lines.append("|---|---|---|---|")
        for i, m in enumerate(missing, 1):
            sites = "<br>".join(f"`{s}`" for s in m["sites"][:6])
            if len(m["sites"]) > 6:
                sites += f"<br>…还有 {len(m['sites']) - 6} 处"
            lines.append(f"| {i} | `{m['chain']}` | `{m['failed_at']}` | {sites} |")
        lines.append("")
        lines.append("**⇒ 这些属性链在评测机上会直接 `AttributeError`，必须在合入前处理。**")
    else:
        lines.append("**⇒ 未发现缺失属性。** 属性级兼容性通过（注意：这不等于签名级、"
                     "语义级也通过）。")
    lines.append("")
    if accepted:
        lines.append(f"## 已接受（{len(accepted)} 条，来自 --allow）")
        lines.append("")
        lines.append("这些属性链在目标环境不存在，但已确认无害（例如调用点有 `hasattr` 兜底）。")
        lines.append("")
        lines.append("| # | 属性链 | 理由 | 调用点 |")
        lines.append("|---|---|---|---|")
        for i, m in enumerate(accepted, 1):
            sites = "<br>".join(f"`{s}`" for s in m["sites"][:4])
            lines.append(f"| {i} | `{m['chain']}` | {m['reason']} | {sites} |")
        lines.append("")
    if unknown:
        lines.append(f"## 无法判定（{len(unknown)} 条）")
        lines.append("")
        lines.append("这些属性链的「属性面」不在清单覆盖范围内，**既不能说通过、"
                     "也不能说失败**。")
        lines.append("处置：把清单重新导出一份（`manifest` 子命令），或把这几条单独"
                     "拿去评测机里跑一次 `scan`。")
        lines.append("")
        lines.append("| # | 属性链 | 原因 | 调用点 |")
        lines.append("|---|---|---|---|")
        for i, m in enumerate(unknown, 1):
            sites = "<br>".join(f"`{s}`" for s in m["sites"][:4])
            lines.append(f"| {i} | `{m['chain']}` | {m['why']} | {sites} |")
        lines.append("")

    # np.cross 单独一节（静态扫不到「参数是不是 2 元向量」，所以全列出来人工过一遍）
    cross_hits = find_cross(repo)
    lines.append(f"## `np.cross` 调用点（{len(cross_hits)} 处，需人工确认参数维度）")
    lines.append("")
    lines.append("numpy ≥ 2.0 对 2 元向量 `np.cross` 会抛 `ValueError`"
                 "（第五次提交阶段一 0/10 全 FAIL 的根因）。")
    lines.append("")
    for h in cross_hits:
        lines.append(f"- `{h}`")
    if not cross_hits:
        lines.append("- （无）")
    lines.append("")

    text = "\n".join(lines)
    if not manifest:
        print(f"[!] 未指定 --manifest：本次判定用的是**当前解释器**"
              f"（python {env.get('python', '?')} / torch {env.get('torch', '?')} / "
              f"numpy {env.get('numpy', '?')}）。\n"
              f"    若本机不是评测机环境，这个结论不可信 —— "
              f"用 `manifest` 子命令导出清单后加 `--manifest` 再跑。",
              file=sys.stderr)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"\n[写出] {a.out}", file=sys.stderr)
    if a.json_out:
        Path(a.json_out).write_text(
            json.dumps({"env": env, "n_chains": n_chains, "missing": missing,
                        "unknown": unknown, "accepted": accepted,
                        "np_cross": cross_hits,
                        "manifest": a.manifest or None},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"[写出] {a.json_out}", file=sys.stderr)
    return 1 if (missing or (a.strict_unknown and unknown)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
