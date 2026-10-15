"""Fan-out publication must upgrade a lane product, never freeze a stale one (§5.54).

背景：`submission/run.py` 的阶段二**渐进导出**（用户 m42543 的灾难现场加固）会在
阶段二刚跑完就把**还没融合**的文档写进 lane 的 `final_results/`，随后阶段三 /
融合层再覆盖它。父进程从前「只补缺不覆盖」⇒ 抢到的第一份（stage2 档）**永久胜出**：
8 路扇出单卡实测 9/10 张交付的是缺 `__pin_index`、没过融合层的 stage2 文档。
现在改成 `_publish_one`：**按源文件新旧升级**，拷贝前验 JSON，先落同卷暂存再
`os.replace`。
"""
import importlib.util
import json
import os
import time
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "submission_run", Path(__file__).resolve().parent.parent / "submission" / "run.py")
R = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(R)


def _doc(tag, pin_index=False):
    d = {"case": "1",
         "components": {"U1": {"Name": tag, "Type": "box"}},
         "pins": {"U1": {"1": "IN"}},
         "nets": [{"net_id": "N1", "hyperGraph": "U1.1"}]}
    if pin_index:
        d["__pin_index"] = {"U1": {"1": {"x": 1.0, "y": 2.0}}}
    return d


class PublishUpgradeTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.lane = base / "lane0" / "final_results"
        self.main = base / "out" / "final_results"
        self.stage = base / "out" / "runs" / ".pub"
        self.lane.mkdir(parents=True, exist_ok=True)
        self.main.mkdir(parents=True, exist_ok=True)
        self.src = self.lane / "1_Pred.json"
        self.dst = self.main / "1_Pred.json"
        self.state = {}

    def tearDown(self):
        self._tmp.cleanup()

    def publish(self, state=None):
        return R._publish_one(self.src, self.dst, self.state if state is None else state,
                              self.stage)

    def write(self, tag, pin_index=False, mtime=None):
        self.src.write_text(json.dumps(_doc(tag, pin_index), ensure_ascii=False),
                            encoding="utf-8")
        if mtime is not None:
            os.utime(str(self.src), (mtime, mtime))

    def name(self):
        return json.loads(self.dst.read_text(encoding="utf-8"))["components"]["U1"]["Name"]

    def test_first_publish_lands(self):
        self.write("s2")
        self.assertTrue(self.publish())
        self.assertEqual(self.name(), "s2")

    def test_newer_lane_product_upgrades_stale_published_one(self):
        """阶段二档先发布、融合档后到 ⇒ 主目录必须换成融合档（§5.54 的回归点）。"""
        self.write("s2")
        self.assertTrue(self.publish())
        time.sleep(0.02)
        self.write("fusion", pin_index=True)
        self.assertTrue(self.publish())
        got = json.loads(self.dst.read_text(encoding="utf-8"))
        self.assertEqual(got["components"]["U1"]["Name"], "fusion")
        self.assertIn("__pin_index", got)

    def test_unchanged_source_is_not_recopied(self):
        self.write("fusion", pin_index=True)
        self.assertTrue(self.publish())
        self.assertFalse(self.publish())

    def test_older_source_never_overwrites_newer_output(self):
        self.write("fusion", pin_index=True)
        self.assertTrue(self.publish())
        self.write("OLDER", mtime=self.src.stat().st_mtime - 600)
        self.assertFalse(self.publish())
        self.assertEqual(self.name(), "fusion")

    def test_torn_json_is_refused_and_keeps_previous(self):
        self.write("fusion", pin_index=True)
        self.assertTrue(self.publish())
        self.src.write_text('{"case": "1", "components": {"U1"', encoding="utf-8")
        future = time.time() + 5
        os.utime(str(self.src), (future, future))
        self.assertFalse(self.publish())
        self.assertEqual(self.name(), "fusion")

    def test_without_memory_falls_back_to_mtime_comparison(self):
        fresh = {}
        self.dst.write_text(json.dumps(_doc("NEWER_MAIN"), ensure_ascii=False),
                            encoding="utf-8")
        self.write("lane_old", mtime=self.dst.stat().st_mtime - 600)
        self.assertFalse(self.publish(state=fresh))
        self.assertEqual(self.name(), "NEWER_MAIN")
        self.write("lane_new", mtime=time.time() + 5)
        self.assertTrue(self.publish(state=fresh))
        self.assertEqual(self.name(), "lane_new")

    def test_staging_dir_is_left_empty(self):
        self.write("fusion", pin_index=True)
        self.assertTrue(self.publish())
        self.assertEqual(sorted(p.name for p in self.stage.glob("*")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
