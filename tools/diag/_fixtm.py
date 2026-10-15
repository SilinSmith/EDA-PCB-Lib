"""把队友分支 worktree 里的「硬编码主仓绝对路径」和 traceback 影子变量就地修好。

用法：python work/_fixtm.py <worktree 根>

为什么需要：`stage1/pintrace.py` 是**库模块**却写了
`sys.path.insert(0, r"Y:\\workspace\\EDA\\eda-pcb")`，于是任何在
worktree / 别的 checkout / 容器里跑的进程，一旦 import 到它，
后续 `import stage2.*` 就静默地从**主仓**加载模块 —— 实测表现为
`stage2.libpins` 之类的 ImportError，且 `submission/run.py` 的
`import traceback` 影子变量会把该异常吞成 UnboundLocalError。
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()

SELF_ROOT = ('__import__("os").path.dirname(__import__("os").path.dirname('
             '__import__("os").path.abspath(__file__)))')

EDITS = [
    ("stage1/pintrace.py",
     'sys.path.insert(0, r"Y:\\workspace\\EDA\\eda-pcb")',
     'sys.path.insert(0, %s)  # 本地修复：原为硬编码主仓绝对路径' % SELF_ROOT),
    ("pipeline/config.py",
     "REPO = r'Y:\\workspace\\EDA\\eda-pcb'",
     'REPO = %s  # 本地修复：原为硬编码主仓绝对路径' % SELF_ROOT),
    ("submission/run.py",
     '        except Exception as exc:                    # noqa: BLE001\n'
     '            import traceback\n'
     '            traceback.print_exc()',
     '        except Exception as exc:                    # noqa: BLE001\n'
     '            traceback.print_exc()'),
]


def main():
    for rel, old, new in EDITS:
        p = ROOT / rel
        if not p.exists():
            print("skip (missing)", rel)
            continue
        s = p.read_text(encoding="utf-8")
        if old not in s:
            print("skip (already patched?)", rel)
            continue
        p.write_text(s.replace(old, new, 1), encoding="utf-8")
        print("patched", rel)


if __name__ == "__main__":
    main()
