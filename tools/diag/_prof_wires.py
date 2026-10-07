"""对最慢的 case 做 cProfile，定位 pskel 到底慢在哪。用法：python _prof_wires.py 0107"""
import cProfile, pstats, io, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
import cv2
from schematic_parser import data as D
from stage1 import wires as W

cid = sys.argv[1] if len(sys.argv) > 1 else "0107"
case = [c for c in D.discover(D.DEFAULT_DATA_ROOT) if c.case_id == cid][0]
bgr = cv2.imread(str(case.png_path))
print(f"{cid}  {bgr.shape[1]}x{bgr.shape[0]}  {bgr.shape[1]*bgr.shape[0]/1e6:.2f} Mpx")

pr = cProfile.Profile()
t0 = time.time()
pr.enable()
res, secs = W.extract_wires(bgr, max_pixels=99_000_000)   # 强制不降采样，看原尺寸代价
pr.disable()
print(f"extract_wires {secs:.1f} s   (wall {time.time()-t0:.1f} s)  scale={res['scale']}")

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(18)
out = s.getvalue()
print("\n".join(out.splitlines()[:46]))

print("\n--- 按 tottime 排（自身耗时）---")
s2 = io.StringIO()
pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(14)
print("\n".join(s2.getvalue().splitlines()[4:24]))
