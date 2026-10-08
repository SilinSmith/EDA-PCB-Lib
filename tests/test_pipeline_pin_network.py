"""Pipeline integration checks without detector weights, OCR, or image files."""
import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from stage1 import pipeline


class PipelinePinNetworkTests(unittest.TestCase):
    def run_case(self, components, segments, direction_guard=False,
                 attachment_guard=False, supplied_pins=None, pin_synth=None):
        image = np.full((480, 480, 3), 255, dtype=np.uint8)
        hypotheses = SimpleNamespace(case="", to_dict=lambda: {})
        detector = Mock()
        detector.detect.return_value = (components, hypotheses, {}, {})
        wires = {"segments": segments, "endpoints": [], "junctions": [],
                 "n_segments": len(segments), "scale": 1.0}
        case = SimpleNamespace(case_id="integration", png_path="unused.png")
        pins_module = pipeline.nets_mod.pins_mod

        with ExitStack() as stack:
            stack.enter_context(patch.object(pipeline.cv2, "imread", return_value=image))
            stack.enter_context(patch.object(pipeline.wires_mod, "extract_wires",
                                             return_value=(wires, 0.0)))
            stack.enter_context(patch.object(pipeline, "DET", None))
            stack.enter_context(patch.object(pipeline, "BIGIC_ON", False))
            stack.enter_context(patch.object(pipeline, "PINC_ON", pin_synth is not None))
            stack.enter_context(patch.object(pipeline, "PIN_DIRECTION_GUARD", direction_guard))
            stack.enter_context(patch.object(pipeline, "PIN_ATTACHMENT_GUARD", attachment_guard))
            # Keep graph parameters independent of a caller's experimental env.
            stack.enter_context(patch.object(pipeline.nets_mod, "PIN_ATTR", "nearest"))
            for name, value in (("WIRE_TOL", 1.0), ("WIRE_EXT", 0.0),
                                ("WIRE_WELD", 0.0), ("WIRE_OVER", 0.0)):
                stack.enter_context(patch.object(pipeline.nets_mod.labels_mod, name, value))
            if pin_synth is not None:
                stack.enter_context(patch.object(pipeline.pincol_mod, "enumerate_pins",
                                                 return_value={}))
                stack.enter_context(patch.object(pipeline.pincol_mod, "merge_into",
                                                 side_effect=lambda pins, cols:
                                                 (pins, pin_synth, {})))
            if supplied_pins is None:
                pin_spy = stack.enter_context(patch.object(
                    pins_module, "find_pins", wraps=pins_module.find_pins))
            else:
                pin_spy = stack.enter_context(patch.object(
                    pins_module, "find_pins", return_value=supplied_pins))
            net_spy = stack.enter_context(patch.object(
                pipeline.nets_mod, "build_nets", wraps=pipeline.nets_mod.build_nets))
            # Existing run_case kwargs stay sufficient: no guard argument is
            # required at this public boundary, and an injected detector wins.
            doc = pipeline.run_case(case, "unused.pt", det=detector, do_text=False,
                                    do_rays=False, use_cv=False)
            return doc, pin_spy.call_args, net_spy.call_args

    @staticmethod
    def component(cid, box):
        return {"id": cid, "type": "box", "bbox_img": box, "pending": False,
                "center_img": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]}

    def test_direction_flag_reaches_real_pin_finder_and_changes_contacts(self):
        components = [self.component("ic", [100, 100, 200, 300])]
        # A parallel border stroke is a legacy contact but is rejected by the
        # direction-aware finder. The detector and skeleton stages are mocked;
        # the pin finder and graph assembly remain real.
        segments = [[[100, 140], [100, 260]]]
        for enabled, expected_count in ((False, 1), (True, 0)):
            with self.subTest(enabled=enabled):
                doc, pin_call, net_call = self.run_case(
                    components, segments, direction_guard=enabled)
                self.assertEqual(pin_call.kwargs["direction_guard"], enabled)
                self.assertFalse(net_call.kwargs["guard_attachments"])
                self.assertEqual(len(doc["pins"]["ic"]), expected_count)
                self.assertEqual(doc["errors"], [])

    def test_attachment_guard_and_synth_metadata_survive_pipeline_serialization(self):
        components = [self.component("ic", [400, 400, 420, 420]),
                      self.component("obstacle", [-1, 2, 1, 6])]
        pins = {"ic": [(0, 0), (100, 0), (200, 0), (300, 0)]}
        segments = [[[-2, 8], [2, 8]], [[98, -4], [102, -4]],
                    [[98, 4], [102, 4]], [[198, 4], [202, 4]],
                    [[298, 0], [302, 0]]]
        synth = {"ic": [0, 0, 1, 0]}
        doc, pin_call, net_call = self.run_case(
            components, segments, direction_guard=True, attachment_guard=True,
            supplied_pins=pins, pin_synth=synth)
        self.assertTrue(pin_call.kwargs["direction_guard"])
        self.assertTrue(net_call.kwargs["guard_attachments"])
        self.assertEqual(net_call.kwargs["components"], components)
        self.assertEqual(net_call.kwargs["pin_synth"], synth)
        self.assertEqual(doc["errors"], [])
        self.assertEqual([net["pins"] for net in doc["nets"]], [[["ic", 3]]])
        # Exercise the stage1 JSON boundary, rather than checking only a
        # transient report returned by build_nets.
        saved = json.loads(json.dumps(doc))["pin_join"]
        self.assertTrue(saved["direction_guard"])
        self.assertTrue(saved["attachment_guard"])
        self.assertEqual(saved["n_blocked"], 1)
        self.assertEqual(saved["n_ambiguous"], 1)
        self.assertEqual(saved["n_orphans"], 3)
        self.assertEqual(saved["n_weak"], 0)
        self.assertEqual(saved["blocked"][0]["pin"], 0)
        self.assertEqual(saved["ambiguous"][0]["pin"], 1)
        self.assertEqual({entry["pin"] for entry in saved["orphans"]}, {0, 1, 2})

    def test_disabled_guards_preserve_legacy_attachment_and_empty_diagnostics(self):
        components = [self.component("ic", [400, 400, 420, 420]),
                      self.component("obstacle", [-1, 2, 1, 6])]
        pins = {"ic": [(0, 0), (100, 0)]}
        segments = [[[-2, 8], [2, 8]], [[98, -4], [102, -4]],
                    [[98, 4], [102, 4]]]
        doc, pin_call, net_call = self.run_case(
            components, segments, supplied_pins=pins)
        self.assertFalse(pin_call.kwargs["direction_guard"])
        self.assertFalse(net_call.kwargs["guard_attachments"])
        self.assertEqual(doc["errors"], [])
        self.assertEqual(doc["stats"]["n_pins_in_nets"], 2)
        diagnostics = json.loads(json.dumps(doc))["pin_join"]
        for name in ("orphans", "weak", "blocked", "ambiguous"):
            self.assertEqual(diagnostics[name], [])
            self.assertEqual(diagnostics["n_" + name], 0)


if __name__ == "__main__":
    unittest.main()
