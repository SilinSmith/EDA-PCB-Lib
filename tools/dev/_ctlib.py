# -*- coding: utf-8 -*-
"""容器式**库路径解析**回归测试（用户 m42543「各种灾难性现场」）。

背景：2026-10-12 之前 `stage2/libpins.py` 的默认库路径写死本机
`WORKSPACE\\work\\data\\cpntLibrary.json`。镜像把 `eda-pcb` 拷进 `/submission` 之后，
它解析成 `/work/data/cpntLibrary.json`（不存在）⇒ `LibCtx.load()` 只打印一行
`[libpins] 库不可用` 就跳过，**不炸管线但白丢实测 +1.60**。后来库随仓库提交到
`eda-pcb/data/`，2026-10-14 起默认库又是 `BoxICLibrary_200cases.json`。

本脚本把「/submission」在临时目录里复刻出来（**没有**宿主 `work/`，模拟判题机），
然后逐现场检查解析结果与「缺席时是否发声且不炸」：

  现场 1  包里 BoxIC + cpnt        ⇒ 取 BoxIC，`load()=True`
  现场 2  包里只剩 cpnt            ⇒ 自动退 cpnt，`load()=True`
  现场 3  两个库都没有             ⇒ 打印一行诊断、`load()=False`、**不抛异常**
  现场 4  `STAGE2_BOXIC_LIB` 指向不存在的文件 ⇒ 照用该路径（失败要发声）

跑法（仓库根）：`python -X utf8 eda-pcb\\tools\\dev\\_ctlib.py`
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EDA = Path(__file__).resolve().parents[2]          # …/eda-pcb
BOXIC = EDA / "data" / "BoxICLibrary_200cases.json"
CPNT = EDA / "data" / "cpntLibrary.json"

# 子进程里跑的探针：打印一行 JSON 供断言。
PROBE = r"""
import json, os, sys
root = sys.argv[1]
sys.path.insert(0, root)
os.chdir(root)
from stage2 import libpins as L
ctx = L.LibCtx()
try:
    ok = bool(ctx.load())
    err = ""
except Exception as e:                              # 不该发生：缺席必须不抛
    ok, err = None, "%s: %s" % (type(e).__name__, e)
print(json.dumps({"path": str(L.lib_path()), "ok": ok, "err": err}))
"""


def build(tmp: Path) -> Path:
    """搭一个最小 /submission：只要 assets.py / envflag.py / stage2/libpins.py / data/。"""
    sub = tmp / "submission"
    (sub / "stage2").mkdir(parents=True)
    (sub / "data").mkdir(parents=True)
    (sub / "stage2" / "__init__.py").write_text("# minimal package\n", encoding="utf-8")
    for f in ("assets.py", "envflag.py"):
        shutil.copyfile(EDA / f, sub / f)
    shutil.copyfile(EDA / "stage2" / "libpins.py", sub / "stage2" / "libpins.py")
    return sub


def run(sub: Path, env: dict = None):
    e = dict(os.environ)
    e.pop("STAGE2_BOXIC_LIB", None)
    e.pop("STAGE2_CPNT_LIB", None)
    if env:
        e.update(env)
    p = subprocess.run([sys.executable, "-X", "utf8", "-c", PROBE, str(sub)],
                       capture_output=True, text=True, encoding="utf-8", env=e)
    lines = [ln for ln in (p.stdout or "").splitlines() if ln.strip()]
    import json
    js = {"path": "", "ok": None, "err": "no stdout"}
    for ln in reversed(lines):
        if ln.startswith("{"):
            try:
                js = json.loads(ln)
                break
            except ValueError:
                pass
    # 诊断可能落在 stdout（`print`）也可能落在 stderr，两处都收。
    txt = (p.stdout or "") + "\n" + (p.stderr or "")
    return js, txt


def main() -> int:
    results = []

    def check(name, cond, detail=""):
        results.append((bool(cond), name, detail))
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                               ("  —— " + detail) if detail else ""))

    with tempfile.TemporaryDirectory(prefix="ctlib_") as td:
        tmp = Path(td)
        sub = build(tmp)

        print("现场 1：包里 BoxIC + cpnt（宿主无 work/）")
        shutil.copyfile(BOXIC, sub / "data" / BOXIC.name)
        shutil.copyfile(CPNT, sub / "data" / CPNT.name)
        js, _ = run(sub)
        check("取包内 BoxIC", js["path"].endswith("BoxICLibrary_200cases.json"), js["path"])
        check("load()=True", js["ok"] is True, str(js))

        print("现场 2：包里只剩 cpnt（BoxIC 缺席）")
        (sub / "data" / BOXIC.name).unlink()
        js, _ = run(sub)
        check("自动退包内 cpnt", js["path"].endswith("cpntLibrary.json"), js["path"])
        check("load()=True", js["ok"] is True, str(js))

        print("现场 3：包里两个库都没有")
        (sub / "data" / CPNT.name).unlink()
        js, err = run(sub)
        check("load()=False", js["ok"] is False, "ok=%r err=%r" % (js["ok"], js["err"]))
        check("不抛异常", js["err"] == "", js["err"])
        check("打了『库不可用』诊断", "库不可用" in (err or ""),
              " | ".join((err or "").splitlines())[:90])

        print("现场 4：STAGE2_BOXIC_LIB 指向不存在的文件")
        js, err = run(sub, {"STAGE2_BOXIC_LIB": str(sub / "data" / "nope.json")})
        check("照用显式路径", js["path"].endswith("nope.json"), js["path"])
        check("失败发声且不抛", js["ok"] is False and js["err"] == "", str(js))

    bad = [r for r in results if not r[0]]
    print("\n================ 汇总：%d 项，失败 %d ================" % (len(results), len(bad)))
    for _, nm, dt in bad:
        print("  [FAIL] %s  %s" % (nm, dt))
    print("全部通过 ✓" if not bad else "有失败项 ✗")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
