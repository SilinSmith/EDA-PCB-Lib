# -*- coding: utf-8 -*-
"""GPU 串行闸门：**同一时刻只允许一个自己起的吃卡任务**，且起之前先确认显存够。

## 为什么需要它（用户 m34654 的原话）

> 能不能不要爆显存，显卡同时跑多任务会导致缓存命中降低，这对 LLM 和各种训练是灾难性的，
> 能不能分开来跑这些内容，不要一起抢显存

显存被多进程分掉之后，受害的不是「谁跑得慢一点」：
**LLM 侧的前缀缓存 / KV 会被挤掉重算**，训练侧的 cudnn workspace 与 pinned buffer 也要重新分配，
两边的墙钟都可能翻倍，而且失败模式是 OOM 崩在别人身上（谁后起谁背锅）。
所以纪律是**排队**，不是「小心一点」。

## 三个子命令

    python tools\\dev\\gpu_gate.py status
        看一眼：整卡占用 / 每进程归属（自己人 vs 外人）/ 闸门锁归谁 / 最近 10 条台账。

    python tools\\dev\\gpu_gate.py run --tag crnn_mix66 --need-mb 2000 -- <命令行>
        加锁 → 等显存 → **采样峰值** → 跑 → 释放 → 记台账。
        默认只看「自己人」是否在跑 + 显存够不够；`--exclusive` 则**外人（如 LM Studio 的
        llama-server）在场就等**，适合「这轮训练要独占卡」的场合。

    python tools\\dev\\gpu_gate.py ledger --tail 20
        台账：谁在什么时候占了多少显存、跑了多久、退出码多少。

## 锁与台账

- 锁：`work\\_gpu.lock`（JSON：tag / pid / host / start）。持锁进程死了会被自动接管（stale steal）。
- 台账：`work\\_gpu_ledger.jsonl`（一行一次运行，含 peak_used_mb / free_mb_at_start / exit）。

## 纪律（写死在文档里，别绕过）

1. 任何吃卡的东西（`fast.py`、`submission\\run.py`、`textrec.train`、`fusion.run` 的 LLM 轮、
   跑 YOLO 训练）**都必须从这个闸门里起**。
2. 起之前先 `status` 看外人在不在；外人（LM Studio / llama-server）在场时**先问用户**，
   不要用 `--exclusive` 之外的姿势硬上。
3. 一次只跑一个；要跑多臂就串行，或者把臂之间的差别做成离线（纯 CPU）复算。
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]          # eda-pcb/
WORK = ROOT.parent / "work"                          # Y:\workspace\EDA\work
LOCK = WORK / "_gpu.lock"
LEDGER = WORK / "_gpu_ledger.jsonl"
# 一家的进程名（自己起的）：只用来说明「谁是外人」
OWN_MARKS = ("textrec.train", "fast.py", "submission\\run.py", "submission/run.py",
             "fusion.run", "yolo_part", "train_pcb43", "stage2.run", "ultralytics")


# ---------------------------------------------------------------- nvidia-smi
def _smi(*fields: str) -> list[list[str]]:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-{fields[0]}={fields[1]}",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    rows = []
    for line in (out.stdout or "").strip().splitlines():
        if line.strip():
            rows.append([c.strip() for c in line.split(",")])
    return rows


def gpu_mem() -> dict:
    r = _smi("gpu", "memory.used,memory.total,utilization.gpu")
    if not r:
        return {"used": 0, "total": 0, "util": 0, "ok": False}
    u, t, g = (int(float(x)) for x in r[0][:3])
    return {"used": u, "total": t, "util": g, "ok": True}


# Windows 的 WDDM 会把所有桌面/浏览器图形进程都列进 compute-apps（且 used_memory 是 N/A）。
# 真正的「吃卡租户」只有两种：① 报得出显存数的；② 名字就是常见 LLM/训练后端的。
HEAVY_NAMES = ("llama-server", "python", "python3", "pt_main_thread", "ollama",
               "vllm", "LM Studio", "koboldcpp", "text-generation")


def _cmdlines(pids: list[int]) -> dict[int, str]:
    """一次 PowerShell 取全部命令行（逐个取会起 18 个进程、慢到不能用）。"""
    if not pids:
        return {}
    if os.name != "nt":
        out = {}
        for p in pids:
            try:
                out[p] = Path(f"/proc/{p}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
            except Exception:
                out[p] = ""
        return out
    flt = ",".join(str(p) for p in pids)
    ps = ("Get-CimInstance Win32_Process | Where-Object { " + flt +
          " -contains $_.ProcessId } | ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=40)
    except Exception:
        return {}
    res = {}
    for line in (out.stdout or "").splitlines():
        if "|" in line:
            pid, cmd = line.split("|", 1)
            if pid.strip().isdigit():
                res[int(pid.strip())] = cmd.strip()
    return res


def compute_apps() -> list[dict]:
    """每个 CUDA 计算进程：pid / 可执行名 / 显存（消费卡常给 N/A → None）/ 命令行。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    raw = []
    for line in (out.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        raw.append({"pid": int(parts[0]), "exe": Path(parts[1]).name,
                    "mb": int(parts[2]) if parts[2].isdigit() else None})
    cmds = _cmdlines([a["pid"] for a in raw])
    for a in raw:
        a["cmd"] = cmds.get(a["pid"], "")
        a["heavy"] = bool((a["mb"] or 0) > 200) or any(h.lower() in a["exe"].lower() for h in HEAVY_NAMES)
    return raw


def who(app: dict) -> str:
    """自己人 / 外人 —— 只看命令行标记，不猜。"""
    cmd = app.get("cmd") or ""
    return "own" if any(m in cmd for m in OWN_MARKS) else "foreign"


def gpu_free_after_others() -> int:
    m = gpu_mem()
    return max(0, m["total"] - m["used"])


# ---------------------------------------------------------------- 锁
def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{'1'}}"],
                                 capture_output=True, text=True, timeout=20)
            return (out.stdout or "").strip() == "1"
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def lock_holder() -> dict | None:
    if not LOCK.exists():
        return None
    try:
        d = json.loads(LOCK.read_text(encoding="utf-8-sig"))
    except Exception:
        print(f"[gate] 锁文件读不出来（可能在写一半就崩了），当过期清掉：{LOCK}")
        try:
            LOCK.unlink()
        except Exception:
            pass
        return None
    if _pid_alive(int(d.get("pid", 0))):
        return d
    print(f"[gate] 清掉过期锁（pid {d.get('pid')} 已不在）")
    try:
        LOCK.unlink()
    except Exception:
        pass
    return None


def acquire(tag: str, wait_sec: int) -> dict:
    t0 = time.time()
    while True:
        h = lock_holder()
        if h is None:
            break
        if time.time() - t0 > wait_sec:
            raise SystemExit(f"[gate] 等锁超时（{wait_sec}s）：现在持有者 tag={h.get('tag')} pid={h.get('pid')}")
        print(f"[gate] 等锁：{h.get('tag')} (pid {h.get('pid')}, 起于 {h.get('start')}) —— 10s 后重看")
        time.sleep(10)
    d = {"tag": tag, "pid": os.getpid(), "host": os.environ.get("COMPUTERNAME", ""),
         "start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "need_mb": None}
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return d


def release():
    try:
        LOCK.unlink()
    except Exception:
        pass


def log_ledger(row: dict):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 峰值采样
class Sampler(threading.Thread):
    def __init__(self, every: float = 5.0):
        super().__init__(daemon=True)
        self.every, self.stop = every, threading.Event()
        self.peak_used = 0
        self.peak_own = 0

    def run(self):
        while not self.stop.is_set():
            m = gpu_mem()
            self.peak_used = max(self.peak_used, m["used"])
            own = 0
            for a in compute_apps():
                if a["heavy"] and who(a) == "own" and a.get("mb"):
                    own += int(a["mb"])
            self.peak_own = max(self.peak_own, own)
            self.stop.wait(self.every)


# ---------------------------------------------------------------- 子命令
def cmd_status(args) -> int:
    m = gpu_mem()
    apps = compute_apps()
    heavy = [a for a in apps if a["heavy"]]
    light = len(apps) - len(heavy)
    own = [a for a in heavy if who(a) == "own"]
    foreign = [a for a in heavy if who(a) == "foreign"]
    print(f"整卡：{m['used']} / {m['total']} MiB 已用，空闲 {m['total'] - m['used']} MiB，利用率 {m['util']}%")
    print(f"闸门锁：{lock_holder() or '（没人持有）'}")
    print(f"自己人的吃卡进程：{len(own)} 个" + ("  ← 超过 1 个就违反纪律" if len(own) > 1 else ""))
    for a in own:
        print(f"  own  pid={a['pid']:<7} mb={a['mb']}  {(a['cmd'] or a['exe'])[:120]}")
    print(f"外人的吃卡进程：{len(foreign)} 个" + ("  ← 起训练前先问用户" if foreign else ""))
    for a in foreign:
        print(f"  ext  pid={a['pid']:<7} mb={a['mb']}  {a['exe']}  {(a['cmd'] or '')[:90]}")
    print(f"（另有 {light} 个桌面/浏览器图形进程，不占算力，忽略）")
    if LEDGER.exists():
        tail = LEDGER.read_text(encoding="utf-8").strip().splitlines()[-args.tail:]
        print(f"台账尾 {len(tail)} 条：")
        for line in tail:
            try:
                r = json.loads(line)
                print(f"  {r.get('start')}  {r.get('tag'):<16} 峰值{r.get('peak_used_mb')}MiB "
                      f"用{r.get('sec')}s 退出{r.get('exit')}  {r.get('cmd','')[:70]}")
            except Exception:
                pass
    return 0


def cmd_run(args) -> int:
    if not args.cmd:
        raise SystemExit("用法：gpu_gate.py run --tag X --need-mb N -- <命令行>")
    mode = "exclusive" if args.exclusive else "shared"
    print(f"[gate] tag={args.tag} mode={mode} need={args.need_mb}MiB headroom={args.headroom}MiB")
    lock = acquire(args.tag, args.wait_sec)
    lock["need_mb"] = args.need_mb
    LOCK.write_text(json.dumps(lock, ensure_ascii=False), encoding="utf-8")
    t0 = time.time()
    rc, peak = 1, 0
    try:
        while True:
            m = gpu_mem()
            apps = [a for a in compute_apps() if a["heavy"]]
            own = [a for a in apps if who(a) == "own"]
            foreign = [a for a in apps if who(a) == "foreign"]
            free = m["total"] - m["used"]
            bad = []
            if own:
                bad.append(f"自己人已在跑 {[a['pid'] for a in own]}")
            if args.exclusive and foreign:
                bad.append(f"--exclusive：外人在场 pid={[a['pid'] for a in foreign]}")
            if free < args.need_mb + args.headroom:
                bad.append(f"空闲 {free} < 需要 {args.need_mb}+{args.headroom}")
            if not bad:
                break
            if time.time() - t0 > args.wait_sec:
                raise SystemExit(f"[gate] 等资源超时（{args.wait_sec}s）：{'；'.join(bad)}")
            print(f"[gate] 等资源：{'；'.join(bad)} —— 15s 后重看")
            time.sleep(15)
        m0 = gpu_mem()
        print(f"[gate] 开始：空闲 {m0['total'] - m0['used']} MiB，利用率 {m0['util']}%，命令 {' '.join(shlex.quote(c) for c in args.cmd)}")
        s = Sampler()
        s.start()
        try:
            rc = subprocess.call(args.cmd)
        finally:
            s.stop.set()
            s.join(timeout=10)
        peak = s.peak_used
    finally:
        sec = round(time.time() - t0, 1)
        release()
        log_ledger({"tag": args.tag, "start": lock["start"], "sec": sec, "exit": rc,
                    "peak_used_mb": peak, "need_mb": args.need_mb, "mode": mode,
                    "cmd": " ".join(args.cmd)[:400]})
        print(f"[gate] 结束：退出码 {rc}，用时 {sec}s，峰值整卡占用 {peak} MiB（锁已释放）")
    return rc


def cmd_ledger(args) -> int:
    if not LEDGER.exists():
        print("台账还没有")
        return 0
    for line in LEDGER.read_text(encoding="utf-8").strip().splitlines()[-args.tail:]:
        r = json.loads(line)
        print(f"{r.get('start')}  {r.get('tag'):<18} {r.get('mode'):<9} 峰值{r.get('peak_used_mb')}MiB "
              f"用时{r.get('sec')}s 退出{r.get('exit')}  {r.get('cmd','')[:80]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(add_help=True, description="GPU 串行闸门（见文件头注释）")
    sub = ap.add_subparsers(dest="what", required=True)

    p = sub.add_parser("status", help="看整卡占用 / 归属 / 锁 / 台账")
    p.add_argument("--tail", type=int, default=10)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("run", help="排队后运行一个吃卡命令")
    p.add_argument("--tag", required=True)
    p.add_argument("--need-mb", type=int, default=0, help="这次至少要多少空闲显存")
    p.add_argument("--headroom", type=int, default=1024, help="额外留白（默认 1 GiB）")
    p.add_argument("--wait-sec", type=int, default=3600, help="最多等多久（秒）")
    p.add_argument("--exclusive", action="store_true", help="外人在场就等（默认只警告不拦）")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("ledger", help="看台账")
    p.add_argument("--tail", type=int, default=20)
    p.set_defaults(fn=cmd_ledger)

    args = ap.parse_args(argv)
    if args.what == "run":
        args.cmd = [c for c in (args.cmd or []) if c != "--"]
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
