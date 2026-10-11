"""compat_guard —— 运行期兼容守卫（由 `compat_audit.py guard` 经 sitecustomize 注入）

**它补的是静态扫描扫不到的那一半。**

`compat_audit.py scan` 只能回答「这个名字在目标环境里存不存在」。
真正害过我们一次的那个坑是**语义漂移**——名字在，行为变了：

    np.cross([x1,y1], [x2,y2])
      在 numpy < 2.0  →  返回标量 x1*y2 - y1*x2
      在 numpy >= 2.0 →  ValueError: Both input arrays must be (arrays of)
                          3-dimensional vectors, but they are 2 and 2 dimensional instead.

属性扫描对这种情况完全无感（`np.cross` 存在得好好的）。
第五次提交阶段一 10/10 全 FAIL、导出 0 份，就是它。

**守卫的做法**：把它「变回旧语义继续算」，同时把调用点记下来。
这样一个 process 跑完，能一次性收集到**所有**这类问题点，
而不是崩在第一个上——这正是黑盒提交做不到的事。

通过环境变量控制：
    COMPAT_GUARD_REPORT      报告落盘路径（默认 /tmp/compat_guard.json）
    COMPAT_GUARD_NONFATAL    1（默认）继续跑；0 = 记完就照原样抛错
"""
from __future__ import annotations

import atexit
import json
import os
import traceback
import warnings

REPORT = os.environ.get("COMPAT_GUARD_REPORT", "/tmp/compat_guard.json")
NONFATAL = os.environ.get("COMPAT_GUARD_NONFATAL", "1") != "0"

EVENTS: list[dict] = []
_SELF = os.path.abspath(__file__)


def _caller() -> str:
    """找第一个不在本文件里的栈帧 —— 那才是队友的调用点。"""
    for fr in reversed(traceback.extract_stack()[:-1]):
        if os.path.abspath(fr.filename) != _SELF:
            return f"{fr.filename}:{fr.lineno}  {fr.line}"
    return "<unknown>"


def _record(kind: str, detail: str, caller: str | None = None, **extra) -> None:
    EVENTS.append({"kind": kind, "caller": caller or _caller(),
                   "detail": detail, **extra})


def _dump() -> None:
    try:
        with open(REPORT, "w", encoding="utf-8") as fh:
            json.dump({"events": EVENTS, "n": len(EVENTS)}, fh,
                      ensure_ascii=False, indent=2)
        print(f"[compat_guard] {len(EVENTS)} 条兼容事件 -> {REPORT}", flush=True)
        for e in EVENTS[:40]:
            print(f"[compat_guard]   {e['kind']}: {e['detail']}\n"
                  f"                 {e['caller']}", flush=True)
    except Exception:                              # noqa: BLE001
        pass


atexit.register(_dump)

# --------------------------------------------------------------------- numpy

try:
    import numpy as _np

    # ── np.cross：2 元向量 ─────────────────────────────────────────────────
    if not getattr(_np.cross, "_compat_guard", False):
        _orig_cross = _np.cross

        def _cross_guard(a, b, *args, **kwargs):
            try:
                return _orig_cross(a, b, *args, **kwargs)
            except ValueError as exc:
                _a = _np.asarray(a, dtype=float)
                _b = _np.asarray(b, dtype=float)
                _shape = f"a{_a.shape} b{_b.shape}"
                if _a.shape and _b.shape and _a.shape[-1] == 2 == _b.shape[-1]:
                    _record("np.cross-2d",
                            f"numpy>=2.0 对 2 元向量抛错（{_shape}）：{exc}",
                            shapes=_shape)
                    if not NONFATAL:
                        raise
                    return _a[..., 0] * _b[..., 1] - _a[..., 1] * _b[..., 0]
                raise

        _cross_guard._compat_guard = True
        _cross_guard.__wrapped__ = _orig_cross
        _np.cross = _cross_guard

    # ── np.array(..., copy=False) 语义收紧 ────────────────────────────────
    _orig_array = _np.array

    def _array_guard(*args, **kwargs):
        if kwargs.get("copy", "unset") is False:
            try:
                return _orig_array(*args, **kwargs)
            except ValueError as exc:
                _record("np.array-copy=False",
                        f"numpy>=2.0 的 copy=False 表示「绝不拷贝」：{exc}",
                        hint="numpy 2.0 起想表达旧语义要用 copy=None")
                if not NONFATAL:
                    raise
                kwargs = dict(kwargs)
                kwargs["copy"] = None
                return _orig_array(*args, **kwargs)
        return _orig_array(*args, **kwargs)

    _array_guard.__wrapped__ = _orig_array
    _np.array = _array_guard

    # ── 模块级 __getattr__：numpy 2.x 删掉的名字 ──────────────────────────
    _np_ga = getattr(_np, "__getattr__", None)
    if _np_ga is not None and not getattr(_np_ga, "_compat_guard", False):

        def _np_getattr(name: str):
            try:
                return _np_ga(name)
            except AttributeError as exc:
                _record("numpy-removed-name", f"numpy 2.x 已移除 `np.{name}`：{exc}")
                raise

        _np_getattr._compat_guard = True
        _np.__getattr__ = _np_getattr
except Exception:                                  # noqa: BLE001
    _record("guard-init", "numpy 守卫安装失败（多半是 numpy 没装）")

# --------------------------------------------------------------------- torch

try:
    import torch as _torch

    _orig_load = _torch.load

    def _load_guard(*args, **kwargs):
        if "weights_only" not in kwargs:
            _record("torch.load-无 weights_only",
                    f"torch.load({args[0] if args else '?'}) 没显式传 weights_only。"
                    "2.6 起默认值变成 True；旧版默认 False。显式写死以免两端行为不一致。")
        return _orig_load(*args, **kwargs)

    _load_guard.__wrapped__ = _orig_load
    _torch.load = _load_guard

    _t_ga = getattr(_torch, "__getattr__", None)
    if _t_ga is not None and not getattr(_t_ga, "_compat_guard", False):

        def _t_getattr(name: str):
            try:
                return _t_ga(name)
            except AttributeError as exc:
                _record("torch-缺失名字",
                        f"本环境（torch {getattr(_torch, '__version__', '?')}）"
                        f"没有 `torch.{name}` —— 队友那边可能更高版本才有：{exc}")
                raise

        _t_getattr._compat_guard = True
        _torch.__getattr__ = _t_getattr
except Exception:                                  # noqa: BLE001
    _record("guard-init", "torch 守卫安装失败（多半是 torch 没装）")

# -------------------------------------------------------- 弃用/未来警告收集

_orig_showwarning = warnings.showwarning


def _showwarning(message, category, filename, lineno, file=None, line=None):
    if issubclass(category, (DeprecationWarning, FutureWarning, PendingDeprecationWarning)):
        # 这里不能靠栈回溯取 caller：栈顶永远是 warnings.py 自己的 _showwarnmsg_impl。
        # warning 自带的 filename/lineno 才是真实触发点（stacklevel 已由调用方决定）。
        _record(f"{category.__name__}", f"{message}",
                caller=f"{filename}:{lineno}")
    return _orig_showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = _showwarning
warnings.simplefilter("always", DeprecationWarning)
warnings.simplefilter("always", FutureWarning)
