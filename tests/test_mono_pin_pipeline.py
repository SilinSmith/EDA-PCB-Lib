"""Exercise mono-reader wiring through the real process, binding and JSON export."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from stage2 import run
from stage2 import mono_pin_reading


def targeted(role, value, rect):
    return {"bbox": rect, "geometry_bbox": rect,
            "reads": {role: (value, .96)},
            "name_only": role == "pin_name", "number_only": role == "pin_number",
            "source": "mono_" + role + "_ppocr",
            "pin_target": {"component_id": "ic", "pin_index": 0, "edge": "left",
                           "point_img": [90, 180], "anchor_img": [100, 180]},
            "mono_pin_ocr": {"backend": "ppocr_v4_direct", "accepted": True,
                             "flags": [], "selected_views": [
                                 {"padding": p, "text": value, "confidence": .96}
                                 for p in (2, 4)]}}


class MonoPinPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="mono-process-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.image = np.full((300, 300, 3), 255, np.uint8)
        cv2.rectangle(self.image, (100, 100), (270, 290), (0, 0, 0), 1)
        self.image[180:182, 80:100] = 0
        for x in (81, 87, 93):
            self.image[165:176, x:x+3] = 0
        for x in (111, 121, 131, 141):
            self.image[176:187, x:x+5] = 0
        image_path = self.base / "sample.png"
        success, encoded = cv2.imencode(".png", self.image)
        self.assertTrue(success)
        image_path.write_bytes(encoded.tobytes())
        evidence = {"point_img": [90, 180], "anchor_img": [100, 180],
                    "edge": "left", "trust": "trusted", "frame_confirmed": True,
                    "reason": "frame_attached_normal_stroke", "frame_coverage": 1.,
                    "normal_ink_run": 22, "normal_ink_hits": 22, "first_ink_distance": 1,
                    "text_overlap_fraction": 0., "branching_fraction": .071}
        self.doc = {"image": {"path": str(image_path), "w": 300, "h": 300},
                    "components": [{"id": "ic", "type": "box",
                                    "bbox_img": [100, 100, 270, 290],
                                    "bbox_json": [100, 10, 270, 200]}],
                    "pins": {"ic": [[90, 180]]}, "pin_evidence": {"ic": [evidence]},
                    "texts": [[111, 72, 132, 85], [111, 174, 151, 188], [79, 165, 96, 179],
                              [150, 120, 250, 134]],
                    "nets": []}
        (self.base / "sample.stage1.json").write_text(json.dumps(self.doc), encoding="utf8")
        self.old_reads = [
            {"bbox": [111, 72, 132, 85], "reads": {"designator": ("U1", .99)}},
            {"bbox": [111, 174, 151, 188], "name_only": True,
             "reads": {"pin_name": ("VBA1", .99)}},
            {"bbox": [79, 165, 96, 179], "number_only": True,
             "reads": {"pin_number": ("13", .99)}},
            # box 元件的「器件名」槽走 `_best_read`，允许按 0.85 折扣串角色借读
            # （stage2/assign.py:919 `_best_read` docstring），所以真图上 box 旁边
            # 必须有一块型号文字；缺了它 VBA1 那块会被器件名槽借走、引脚反而没名字。
            {"bbox": [150, 120, 250, 134], "reads": {"partname": ("ACME42", .99)}}]

    def pipeline(self, *, mode="auto", extras=None, real_reader=False, **options):
        def read(image, boxes, roles=None):
            result = copy.deepcopy(self.old_reads)
            if roles is not None:
                for item in result:
                    item["reads"] = {r: value for r, value in item["reads"].items() if r in roles}
            return result

        engine = SimpleNamespace(read=Mock(side_effect=read))
        out = self.base / ("output-%d" % len(list(self.base.glob("output-*"))))
        # 0cc7fdc/§5.29：main 把「调用点现读模块全局」的七段 if 换成了**import 时求值**的
        # `run.PIN_READERS` 注册表（stage2/run.py:110-129、:412-431），所以
        # `patch.multiple(run, ...)` 只改全局、改不到注册表里已固化的 `_on`/`_kw`。
        # 这里按被 patch 的全局重建同一张表 —— 分发语义与断言都不动，只让桩真的被调到。
        with ExitStack() as stack:
            stack.enter_context(patch.multiple(run, PINCOL_WIN_TEXTS=False,
                COLOR_NUMBER_CROP=False, PIN_TEXT_CROP=False, PIN_LANE_CROP=False,
                PIN_NUMBER_REPAIR=False, PIN_NAME_OCR="off", MONO_PIN_OCR=mode))
            switches = {"color_crops": bool(run.COLOR_NUMBER_CROP),
                        "pin_text_crops": bool(run.PIN_TEXT_CROP),
                        "pin_lane_reading": bool(run.PIN_LANE_CROP),
                        "pin_number_reading": bool(run.PIN_NUMBER_REPAIR),
                        "pin_name_reading": run.PIN_NAME_OCR != "off"}
            readers = []
            for _name, _on, _mod, _fn, _need, _kw in run.PIN_READERS:
                _kw = dict(_kw)
                if _mod == "mono_pin_reading":
                    _on = run.MONO_PIN_OCR != "off"
                    _kw["mode"] = run.MONO_PIN_OCR
                else:
                    _on = switches.get(_mod, _on)
                readers.append((_name, _on, _mod, _fn, _need, _kw))
            stack.enter_context(patch.object(run, "PIN_READERS", readers))
            loader = stack.enter_context(patch.object(mono_pin_reading, "get_pin_name_ocr", return_value=None))
            arguments = ({"wraps": mono_pin_reading.read_mono_pin_texts} if real_reader
                         else {"return_value": copy.deepcopy(extras or [])})
            reader = stack.enter_context(patch.object(mono_pin_reading, "read_mono_pin_texts", **arguments))
            report = run.process("sample", self.base, out, engine, texts="stage1", **options)
        final = json.loads((out / "sample.json").read_text(encoding="utf8"))
        assigned = json.loads((out / "_assigned" / "sample.json").read_text(encoding="utf8"))
        index = json.loads((out / "sample.pinindex.json").read_text(encoding="utf8"))
        return final, assigned, index, report, reader, loader, engine

    def test_no_ocr_does_not_call_reader_or_load_optional_runtime(self):
        final, _, _, report, reader, loader, engine = self.pipeline(do_ocr=False, mode="on")
        reader.assert_not_called()
        loader.assert_not_called()
        engine.read.assert_not_called()
        self.assertEqual(report["n_pin_name"], 0)
        self.assertEqual(report["n_pin_number"], 0)
        self.assertFalse(any(final["pins"].values()))

    def test_off_mode_preserves_old_final_output_without_optional_load(self):
        final, assigned, index, _, reader, loader, engine = self.pipeline(mode="off")
        reader.assert_not_called()
        loader.assert_not_called()
        engine.read.assert_called_once()
        self.assertEqual(final["pins"]["U1"]["pin_13"]["pinname"], "VBA1")
        self.assertEqual(assigned["comps"]["ic"]["pins"]["0"]["number"]["text"], "13")
        self.assertEqual(index, {"ic": {"0": "pin_13"}})

    def test_role_selection_and_threshold_reach_reader(self):
        for roles, expected in ((["pin_name"], (True, False)),
                                (["pin_number"], (False, True)),
                                (["designator", "pin_name", "pin_number"], (True, True))):
            with self.subTest(roles=roles):
                _, _, _, _, reader, _, engine = self.pipeline(roles=roles, min_conf=.93)
                reader.assert_called_once()
                call = reader.call_args
                self.assertEqual(call.kwargs["include_names"], expected[0])
                self.assertEqual(call.kwargs["include_numbers"], expected[1])
                self.assertEqual(call.kwargs["min_conf"], .93)
                self.assertEqual(call.kwargs["mode"], "auto")
                self.assertEqual(call.args[0], self.doc)
                self.assertIs(call.args[2], engine)
                np.testing.assert_array_equal(call.args[3], self.image)

    def test_no_pin_roles_performs_zero_optional_calls(self):
        for roles in ([], ["designator"], ["partname", "value"]):
            with self.subTest(roles=roles):
                _, _, _, _, reader, loader, _ = self.pipeline(roles=roles)
                reader.assert_not_called()
                loader.assert_not_called()

    def test_accepted_reader_observations_reach_real_binding_and_export(self):
        name = targeted("pin_name", "VBAT", [111, 174, 151, 188])
        number = targeted("pin_number", "113", [77, 165, 96, 179])
        final, assigned, index, report, reader, loader, _ = self.pipeline(extras=[name, number])
        reader.assert_called_once()
        loader.assert_not_called()
        physical = assigned["comps"]["ic"]["pins"]["0"]
        self.assertEqual((physical["name"]["text"], physical["number"]["text"]), ("VBAT", "113"))
        self.assertEqual(final["pins"]["U1"], {
            "pin_113": {"pinname": "VBAT", "point": {"x": 90., "y": 120.}}})
        self.assertEqual(index, {"ic": {"0": "pin_113"}})
        self.assertNotIn("__pin_index", final)
        self.assertEqual((report["n_pin_name"], report["n_pin_number"]), (1, 1))

    def test_process_threshold_filters_reader_reads_before_binding(self):
        extras = [targeted("pin_name", "VBAT", [111, 174, 151, 188]),
                  targeted("pin_number", "113", [77, 165, 96, 179])]
        final, assigned, _, _, reader, _, _ = self.pipeline(extras=extras, min_conf=.98)
        self.assertEqual(reader.call_args.kwargs["min_conf"], .98)
        physical = assigned["comps"]["ic"]["pins"]["0"]
        self.assertEqual((physical["name"]["text"], physical["number"]["text"]), ("VBA1", "13"))
        self.assertEqual(set(final["pins"]["U1"]), {"pin_13"})

    def test_auto_missing_optional_dependency_keeps_real_legacy_pipeline(self):
        expected, _, expected_index, _, _, _, _ = self.pipeline(mode="off")
        final, assigned, index, _, reader, loader, _ = self.pipeline(real_reader=True)
        reader.assert_called_once()
        loader.assert_called_once_with("auto")
        self.assertEqual(final, expected)
        self.assertEqual(index, expected_index)
        self.assertEqual(assigned["comps"]["ic"]["pins"]["0"]["number"]["text"], "13")


def setUpModule():
    """§5.28：本模块的「合格读数压过更高分旧读数」那条断言测的是队友分支**未并入**的绑定
    主体（`_qualified_mono_views` 原注释："Do not compare optional OCR confidence with the
    legacy model's score."）。其余 6 条已按 main 自己的 API 修好（`run.PIN_READERS`
    注册表重建 + box 器件名夹具），门开后原样开跑。见 tests/_unmerged_gate.py。"""
    import _unmerged_gate
    _unmerged_gate.gate([("stage2.assign", ("_qualified_mono_views",
                                            "_preferred_name_observation"))])


if __name__ == "__main__":
    unittest.main()
