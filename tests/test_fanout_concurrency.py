# -*- coding: utf-8 -*-
"""扇出的**规模口径**必须锁住（8 卡那天唯一会自动生效的并发决策）。

`submission/run.py` 在阶段二开启 case 级扇出，而扇出是比赛日唯一「不用改代码、
拿到的卡越多越赚」的机制 —— 所以它的三件事必须有回归锁：

1. `_resolve_case_workers()`：`-case-workers` > `EDA_CASE_WORKERS` > 自动（可见 GPU 数）；
   **单卡/无卡必须回落串行**（本机与官方练习环境都是 1 张卡，这条决定「本地读数」和
   「平台读数」是不是同一条链）。
2. `_lane_device()`：lane i 独占一张卡（`-device 0` → 0/1/…/N-1 轮转；`cuda:i` 同理；
   非数字原样透传）。
3. `_child_concurrency()`：整体并发 = lanes × 每 lane 并发 —— lane 数 ≥ 4 时每 lane 的
   VLM 并发**减半**、stage1 的进程内并行让给 lane 之间（否则 8 lane × 4 = 32 路并发
   打爆外部端点，靠 `VLM_RETRY` 兜回来的是**时间**，而超时按双倍罚）。
"""
import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "submission_run", Path(__file__).resolve().parent.parent / "submission" / "run.py")
R = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(R)


class _Args:
    def __init__(self, **kw):
        self.case_workers = -1
        self.vlm_workers = 4
        self.workers = 1
        self.device = "0"
        self.input_dir = "in"
        self.__dict__.update(kw)


class ResolveCaseWorkersTests(unittest.TestCase):
    def setUp(self):
        self._orig = R._visible_cuda_devices
        self._env = R.os.environ.get("EDA_CASE_WORKERS")
        R.os.environ.pop("EDA_CASE_WORKERS", None)

    def tearDown(self):
        R._visible_cuda_devices = self._orig
        if self._env is None:
            R.os.environ.pop("EDA_CASE_WORKERS", None)
        else:
            R.os.environ["EDA_CASE_WORKERS"] = self._env

    def test_explicit_wins_and_is_clamped_by_case_count(self):
        self.assertEqual(R._resolve_case_workers(_Args(case_workers=8), 3), 3)
        self.assertEqual(R._resolve_case_workers(_Args(case_workers=8), 50), 8)
        self.assertEqual(R._resolve_case_workers(_Args(case_workers=0), 50), 1)

    def test_env_is_only_a_fallback_for_the_negative_default(self):
        R.os.environ["EDA_CASE_WORKERS"] = "3"
        self.assertEqual(R._resolve_case_workers(_Args(case_workers=-1), 50), 3)
        self.assertEqual(R._resolve_case_workers(_Args(case_workers=5), 50), 5)

    def test_auto_uses_visible_gpu_count(self):
        R._visible_cuda_devices = lambda: 8
        self.assertEqual(R._resolve_case_workers(_Args(case_workers=-1), 50), 8)
        self.assertEqual(R._resolve_case_workers(_Args(case_workers=-1), 5), 5)

    def test_single_or_no_gpu_stays_serial(self):
        for ndev in (0, 1):
            R._visible_cuda_devices = lambda n=ndev: n
            self.assertEqual(R._resolve_case_workers(_Args(case_workers=-1), 50), 1)


class LaneDeviceTests(unittest.TestCase):
    def setUp(self):
        self._orig = R._visible_cuda_devices

    def tearDown(self):
        R._visible_cuda_devices = self._orig

    def test_digit_base_rotates_over_visible_devices(self):
        R._visible_cuda_devices = lambda: 8
        got = [R._lane_device("0", i) for i in range(9)]
        self.assertEqual(got, ["0", "1", "2", "3", "4", "5", "6", "7", "0"])

    def test_cuda_base_rotates_too(self):
        R._visible_cuda_devices = lambda: 4
        self.assertEqual([R._lane_device("cuda:2", i) for i in range(4)],
                         ["cuda:2", "cuda:3", "cuda:0", "cuda:1"])

    def test_non_numeric_base_passes_through(self):
        R._visible_cuda_devices = lambda: 8
        self.assertEqual(R._lane_device("auto", 3), "auto")
        self.assertEqual(R._lane_device("cpu", 3), "cpu")

    def test_without_cuda_the_base_is_kept(self):
        R._visible_cuda_devices = lambda: 0
        self.assertEqual(R._lane_device("0", 5), "0")


class ChildConcurrencyTests(unittest.TestCase):
    """★ 期望值来自 `docs/RESULTS.md` §5.61 实测（官方 10 张、`-lean`、1 卡）：

        lane 1 × vw 4 = 4 路在飞 → 基线
        lane 2 × vw 4 = 8 路在飞 → 10/10 逐字节相同，80.04
        lane 4 × vw 2 = 8 路在飞 → 10/10 逐字节相同，80.04
        lane 8 × vw 2 = 16 路在飞 → 只有 4/10 相同，77.25（-2.79）
    拐点在 8 与 16 之间，所以规则是「在飞 ≤ `VLM_INFLIGHT_CAP`（8）」，不是「减半」。
    """

    def test_single_lane_is_untouched(self):
        self.assertEqual(R._child_concurrency(_Args(vlm_workers=4, workers=2), 1), (4, 2))

    def test_two_lanes_still_fit_the_cap(self):
        self.assertEqual(R._child_concurrency(_Args(vlm_workers=4, workers=2), 2), (4, 1))

    def test_four_lanes_get_exactly_the_cap(self):
        self.assertEqual(R._child_concurrency(_Args(vlm_workers=4, workers=2), 4), (2, 1))

    def test_eight_lanes_fall_back_to_one_worker(self):
        self.assertEqual(R._child_concurrency(_Args(vlm_workers=4, workers=2), 8), (1, 1))

    def test_never_reaches_zero_even_above_the_cap(self):
        for n in (8, 12, 50):
            vw, s1 = R._child_concurrency(_Args(vlm_workers=4, workers=1), n)
            self.assertEqual(vw, 1)
            self.assertEqual(s1, 1)

    def test_low_explicit_worker_count_is_not_raised(self):
        self.assertEqual(R._child_concurrency(_Args(vlm_workers=1, workers=1), 8), (1, 1))

    def test_inflight_never_exceeds_the_cap_below_it(self):
        for n in range(1, R.VLM_INFLIGHT_CAP + 1):
            vw, _s1 = R._child_concurrency(_Args(vlm_workers=4), n)
            self.assertLessEqual(n * vw, R.VLM_INFLIGHT_CAP)

    def test_the_cap_is_the_measured_eight(self):
        self.assertEqual(R.VLM_INFLIGHT_CAP, 8)


class ChildArgvWiringTests(unittest.TestCase):
    """`_child_concurrency` 的结论必须**真的进了子进程命令行**，否则白算。"""

    def _argv(self, n_lanes, **kw):
        a = _Args(**kw)
        vw, s1 = R._child_concurrency(a, n_lanes)
        return R._child_argv(a, Path("."), ["1", "2"], "cuda:3", vw, s1)

    def test_lane_child_is_always_serial_in_case_fanout(self):
        argv = self._argv(2)
        self.assertEqual(argv[argv.index("-case-workers") + 1], "1")

    def test_small_fanout_hands_four_vlm_workers_to_the_lane(self):
        argv = self._argv(2)
        self.assertEqual(argv[argv.index("-vlm-workers") + 1], "4")

    def test_big_fanout_hands_one_vlm_worker_and_serial_stage1(self):
        argv = self._argv(8, workers=2)
        self.assertEqual(argv[argv.index("-vlm-workers") + 1], "1")
        self.assertEqual(argv[argv.index("-workers") + 1], "1")

    def test_lane_device_is_passed_through(self):
        argv = self._argv(8)
        self.assertEqual(argv[argv.index("-device") + 1], "cuda:3")


if __name__ == "__main__":
    unittest.main()
