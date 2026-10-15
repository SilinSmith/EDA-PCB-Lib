#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build 期装依赖（三档自动降级）。

为什么要写成一个脚本而不是 Dockerfile 里一行 pip：

  第二次提交（2026-10-09）build 到 Step 11 炸了 —— 平台直连
  files.pythonhosted.org 只有 14~60 kB/s，scipy 的 35.3 MB 下到
  7.3 MB 就 ReadTimeoutError，整次提交白扔。

  换国内镜像能解决带宽（阿里云实测 287 kB/s、清华 791 kB/s），但
  把「下载问题」换成「约束问题」是净亏的，所以这里任何一档失败都
  会继续往下试，只有三档全失败才让 build 红：

    第 1 档 严格：torch / torchvision / numpy / opencv-python 四个
            `===` 钉死（最不打扰基础镜像，但 `===` 是最严的算子，
            万一上游版本串跟我读到的不一致就会 ResolutionImpossible）
    第 2 档 宽松：同样的钉子改成区间（torch>=2.5.1 且 <2.6），
            挡住升级、但允许 pip 在区间内挪
    第 3 档 无约束：只保证装上，版本由 pip 定

  每一档成功都会把关键版本打出来，万一 torch 被换掉能立刻看见
  （torch 被换成 CPU 版 = 阶段二掉到 CPU 跑，2 小时预算直接爆）。

  第三次提交（2026-10-09）的结果：**第 1 档就全部装成功**，
  镜像源换对了，torch.cuda.is_available() 仍是 True，
  `Successfully installed ... scipy-1.18.1 ultralytics-8.4.174 ...`。
  那一次 build 最后仍然红了，但**红在末尾的自检上**，原因和 pip 无关：
  opencv-python 的 cv2 需要 libGL.so.1，而基础镜像只有 headless 版没有
  这个 .so —— 于是 `import cv2` / `import ultralytics` 都报
  `ImportError: libGL.so.1: cannot open shared object file`。
  当时的修法是在 Dockerfile 里 apt-get install libgl1 libglib2.0-0t64
  （**这个修法后来被第五次改动整段删掉了**，见下）。

  第四次提交（2026-10-09 19:51）换了个死法：给 build 加了 apt 装 libGL 之后，
  `[Ingestion] ERROR: Image build timeout (30 minutes)`，
  ingestionElapsedTime = 1801.4 s（对比第三次 986.8 s）——
  apt 段自己花掉 800+ 秒，而 pip 段单独就吃掉 16.4 分钟（30 分钟预算的 55%）。

  ★ 第五次的结论是：**前两次都治错了地方**。
    · 不该装 libGL，而该**不让 pip 把 headless cv2 换成普通版** ——
      做法是 ultralytics 本体用 `--no-deps` 装，依赖显式列
      （见下面 ULTRALYTICS_ONLY / EXPLICIT_DEPS）。
    · 于是 apt 整段删掉，下载量从 ~170 MB 降到 ~52 MB，两个死因一起消失。
    证据：基础镜像里那份 headless 的 `cv2/cv2.abi3.so`（65.7 MB）
    引用的 lib*.so 清单里**没有 libGL.so.1、没有 libgthread-2.0.so.0**
    （ffmpeg / openblas / png / z 全是它自带的带 hash 私有 .so）。

只用标准库。上界和下界都从**基础镜像现装版本**现推，不手写死版本号，
这样换基础镜像也能用。
"""
import os
import re
import subprocess
import sys

CONS = "/submission/constraints.txt"
PIN = ("torch", "torchvision", "numpy", "opencv-python")
# 基础镜像装的是 opencv_python_headless-4.10.0.84，但 ultralytics 的
# METADATA 写的是 `Requires-Dist: opencv-python!=4.13.0.90,>=4.7.0`。
# ★ 第五次改动后这个值**只在生成约束时用**（给"万一"留个合理版本号），
#   我们根本不会再让 pip 去装 opencv-python —— 见下面的 ULTRALYTICS_ONLY。
#   （历史教训：钉成同版本号**并不能**让它的 cv2 变成 headless 那份，
#     非 headless 的 cv2 动态依赖 libGL.so.1，第三次提交就死在这一点上。）
OPENCV_FALLBACK = "4.10.0.84"

MIRRORS = [
    ("阿里云 HTTP", "http://mirrors.aliyun.com/pypi/simple/", "mirrors.aliyun.com"),
    ("清华 HTTPS", "https://pypi.tuna.tsinghua.edu.cn/simple/", "pypi.tuna.tsinghua.edu.cn"),
    ("中科大 HTTP", "http://mirrors.ustc.edu.cn/pypi/simple/", "mirrors.ustc.edu.cn"),
]

# ============================================================================
# ★★★★ 第五次改动（2026-10-09）：不再让 pip 解析 ultralytics 的依赖树。
#
# 第四次提交 build 超时（`[Ingestion] ERROR: Image build timeout (30 minutes)`，
# ingestionElapsedTime = 1801.4 s）的根因是下载量太大：约 170 MB。
# 其中 opencv-python 62.5 MB 和 polars-runtime-32 54.5 MB 是**完全不该下**的：
#   · opencv-python —— 基础镜像已经有 opencv-python-headless 提供同一个 cv2
#     模块，但 pip 不认它顶替 METADATA 里的 `opencv-python`，会真去下一个。
#     而那份非 headless 的 cv2 动态依赖 libGL.so.1（镜像里没有），装上反而
#     会把原本能用的 headless cv2 覆盖成 import 就报错的版本 —— 第三次提交
#     的失败原文就是 `libGL.so.1: cannot open shared object file`。
#   · polars —— 用 AST 扫遍 ultralytics 全部 231 个 py 文件，
#     `import polars` **没有一处**在模块级（5 处全在函数内），导入期根本不需要。
#
# 所以：ultralytics 本体用 `--no-deps` 单独装，再把它真正需要、而基础镜像
# 没有的那些**一个一个显式列出来**（EXPLICIT_DEPS）。
# 下载量从 ~170 MB 降到 ~52 MB，apt 段也可以整段删掉（不需要 libGL 了）。
# ============================================================================
ULTRALYTICS_ONLY = ["--no-deps", "ultralytics"]
EXPLICIT_DEPS = [
    "scipy",                 # 我们自己的代码要：linear_sum_assignment / cKDTree
    "matplotlib",            # ultralytics 声明了；utils/plotting.py 函数内会 import
    "nvidia-ml-py",          # 提供 pynvml（gpu_mem / ultralytics 都会用）
    "ultralytics-thop",      # ultralytics 声明了（模型 FLOPs 统计）
    "ultralytics-platform",  # ultralytics 声明了（py>=3.11），__init__ 里懒加载
]

BASE_OPTS = ["--break-system-packages", "--no-cache-dir",
             "--prefer-binary", "--timeout", "60", "--retries", "5"]


def _pip(url, host, extra):
    """在指定镜像源上跑一次 pip install。返回退出码。"""
    cmd = ([sys.executable, "-m", "pip", "install"] + BASE_OPTS
           + ["--index-url", url, "--trusted-host", host] + extra)
    print("  " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def installed():
    """基础镜像现装的版本。用 importlib.metadata，不用 pip（快、不联网）。"""
    import importlib.metadata as md

    out = {}
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").strip()
        if name:
            out[name.lower().replace("_", "-")] = dist.version
    return out


def norm(s):
    return s.strip().lower().replace("_", "-")


def to_range(ver):
    """2.5.1+cu121 -> >=2.5.1,<2.6 ；只认前三段数字。"""
    base = re.sub(r"\+.*$", "", ver)
    parts = base.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None
    # 0.x 的包（比如 opencv 的 4.10.0.84）按 major 收紧
    if major == 0:
        return ">={},<{}".format(base, major + 1)
    return ">={},<{}".format("{}.{}".format(major, minor), "{}.{}".format(major, minor + 1))


def write_constraints(strict):
    have = installed()
    lines = ["# generated at build time from the base image; do not edit"]
    for p in PIN:
        v = have.get(p)
        if p == "opencv-python" and not v:
            # 基础镜像只有 opencv-python-headless
            v = have.get("opencv-python-headless") or OPENCV_FALLBACK
        if not v:
            continue
        if strict:
            lines.append("{}==={}".format(p, v))
        else:
            r = to_range(v)
            lines.append("{}  # base image has {}".format(r if r else p, v))
    text = "\n".join(lines) + "\n"
    open(CONS, "w", encoding="utf-8").write(text)
    return text


def try_install():
    last = ""
    for idx, (label, url, host) in enumerate(MIRRORS):
        print("=" * 70, flush=True)
        print("[mirror {}] {}".format(idx + 1, label), flush=True)
        print("=" * 70, flush=True)
        # 第一段：ultralytics 本体，**不带依赖**。
        # 不带依赖是这一版的核心改动，见 ULTRALYTICS_ONLY 上方的注释。
        rc1 = _pip(url, host, ULTRALYTICS_ONLY)
        # 第二段：补齐真正缺的那几个（这几个正常解析它们自己的小依赖）
        rc2 = _pip(url, host, EXPLICIT_DEPS) if rc1 == 0 else 1
        if rc1 == 0 and rc2 == 0:
            print("[mirror {}] OK".format(label), flush=True)
            return True
        last = label
        print("[mirror {}] 失败 rc1={} rc2={}".format(label, rc1, rc2), flush=True)
    print("所有镜像源都失败，最后一个：{}".format(last), flush=True)
    return False


def main():
    print("### 基础镜像已装（决定约束的输入） ###", flush=True)
    have = installed()
    for p in PIN:
        print("  {:<24} {}".format(p, have.get(p, "(没有)")), flush=True)
    print("  {:<24} {}".format("opencv-python-headless",
                               have.get("opencv-python-headless", "(没有)")), flush=True)

    ok = False
    for attempt, strict in enumerate((True, False), 1):
        print(flush=True)
        print("### 第 {} 档 {} ###".format(attempt, "严格约束(===)" if strict else "宽松约束(区间)"), flush=True)
        text = write_constraints(strict)
        for line in text.splitlines():
            print("      " + line, flush=True)
        if try_install():
            ok = True
            break
        print("### 第 {} 档失败，降级重试 ###".format(attempt), flush=True)

    if not ok:
        # 第 3 档：完全不用约束。
        # 不写一个「只有注释」的 constraints 文件 —— 那样 PIP_CONSTRAINT
        # 仍然指向它，pip 还得去解析（注释理论上合法，但没必要赌）。
        # 直接把变量指向一个**不存在**的路径：pip 对不存在的 PIP_CONSTRAINT
        # 是直接忽略的，等于关掉这个机制。
        print("### 第 3 档：不带任何约束，只求装上 ###", flush=True)
        os.environ["PIP_CONSTRAINT"] = "/submission/constraints.disabled"
        try:
            os.unlink(CONS)
        except OSError:
            pass
        if not try_install():
            print("!!! 装不上，build 必须失败 !!!", flush=True)
            return 1

    # 收尾：证明关键包能 import，并把版本打出来给平台日志留证。
    #
    # ★ 只有「真的 import 不了」才硬失败（cv2 / ultralytics / scipy 缺了，
    #   阶段一会在每张图上抛 ModuleNotFoundError ⇒ 白跑一次提交）。
    # ★ GPU 可见性**不再**算致命（见下面那段注释）：build 机没有显卡不代表
    #   评测机没有，基础镜像自带的就是 CUDA 轮子。
    print(flush=True)
    print("### 装完自检（不通过就让 build 失败）###", flush=True)
    code = (
        "import sys\n"
        "import importlib.metadata as m\n"
        "for p in ('ultralytics','scipy','torch','torchvision','numpy',"
        "'opencv-python','opencv-python-headless','matplotlib','polars'):\n"
        "    try: print('  {:26s} {}'.format(p, m.version(p)))\n"
        "    except Exception: print('  {:26s} (缺失)'.format(p))\n"
        "try:\n"
        "    m.version('opencv-python')\n"
        "    print('  !! 警告：装上了 opencv-python（非 headless）。它会覆盖 headless 的')\n"
        "    print('  !!   cv2，无 GUI 容器里 import cv2 会报 libGL.so.1。')\n"
        "except Exception:\n"
        "    print('  ok: 没有 opencv-python（非 headless），cv2 仍来自 headless 那份。')\n"
        "bad = []\n"
        "import torch\n"
        "print('  torch.cuda.is_available()  ', torch.cuda.is_available())\n"
        "print('  torch.version.cuda        ', torch.version.cuda)\n"
        "if not torch.cuda.is_available():\n"
        "    # ★ 只警告、不阻断 build。\n"
        "    #   build 机看不到 GPU 不等于评测机没有 GPU：平台跑分时注入\n"
        "    #   NVIDIA_VISIBLE_DEVICES，而基础镜像自带的 torch 2.5.1+cu121 本来就是\n"
        "    #   CUDA 轮子 —— build 机的显卡可见性根本不改变镜像内容。\n"
        "    #   旧版这里 bad.append(...) 直接让 build 红掉，是「把一次可用提交判死」的\n"
        "    #   最大单点失败面（20261009 那次提交就是被这一类自检卡住的）。\n"
        "    #   真掉到 CPU 会在跑分时自己暴露（极慢/超时），不需要 build 提前判死。\n"
        "    print('  !! 警告：build 机上 torch.cuda.is_available() = False（不阻断 build）')\n"
        "    print('  !!   评测机注入 GPU 后仍是 CUDA 版；若真掉 CPU，跑分会暴露。')\n"
        "try:\n"
        "    import cv2\n"
        "    print('  cv2                       ', cv2.__version__)\n"
        "except Exception as e:\n"
        "    bad.append('cv2 缺失: %s' % e)\n"
        "    # ★ 第五次改动后我们**不再用 apt 补 .so**。缺 .so 只说明一件事：\n"
        "    #   pip 把基础镜像那份 headless cv2 换成了普通版（覆盖 dist-packages/cv2/）。\n"
        "    #   先看上面 opencv-python / opencv-python-headless 两行的版本号。\n"
        "    import re as _re\n"
        "    for _lib in set(_re.findall(r'([A-Za-z0-9_+\\-]+\\.so(?:\\.[0-9]+)*)', str(e))):\n"
        "        print('    !!! 缺共享库: %s' % _lib)\n"
        "    print('    !!! 修法不是装 apt 包，而是让 --no-deps 生效：')\n"
        "    print('    !!!   ultralytics 必须 --no-deps 装，opencv-python 绝不能进容器。')\n"
        "    print('    !!!   （历史：第三次提交死在这里，第四次用 apt 补 GL 又撞 build 超时。）')\n"
        "try:\n"
        "    import ultralytics, scipy\n"
        "    print('  ultralytics/scipy import  OK')\n"
        "except Exception as e:\n"
        "    bad.append('ultralytics/scipy 导入失败: %s' % e)\n"
        "if bad:\n"
        "    print()\n"
        "    for b in bad: print('!!! 致命: ' + b)\n"
        "    print('!!! build 主动失败，避免用这个环境白交一次提交。')\n"
        "    sys.exit(1)\n"
        "print('  环境自检全部通过，可以提交。')\n"
    )
    rc = subprocess.call([sys.executable, "-c", code])
    if rc != 0:
        print(flush=True)
        print("!!! 自检失败 rc=%d —— 故意让 build 红掉。" % rc, flush=True)
        print("!!! 请把上面这几行发给开发者；不要提交这个包。", flush=True)
        return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
