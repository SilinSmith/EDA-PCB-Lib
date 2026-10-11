"""Measured black rows retain final pin/network/evidence index alignment."""
import unittest
from unittest.mock import patch

import numpy as np

from stage1 import pipeline
import test_pipeline_pin_network as fixture


class MonoFramePipelineTests(unittest.TestCase):
    def test_append_and_calibration_align_flags_and_protect_color_sides(self):
        pins = {"ic": [[88, 150], [212, 160]]}
        color = {"edge": "right", "frame_confirmed": True,
                 "confirmed_edges": ["right"], "point_img": [212, 160]}
        recovered = {"ic": [[88, 151], [212, 160], [88, 170]]}
        audit = {"ic": {"added": [{"index": 2}], "calibrated": [{"index": 0}]}}
        with patch.object(pipeline.evidence_mod, "recover_frame_pins",
                          return_value=(recovered, {"ic": {0: 0, 1: 1}}, audit)) as read:
            result = pipeline.apply_mono_frame_pins([], pins, np.zeros((1, 1, 3)),
                np.zeros((1, 1)), [], {"ic": [0, 0]}, {"ic": [None, color]},
                {"ic": [None, True]})
        new, flags, overlays, connected, log = result
        self.assertEqual(read.call_args.kwargs["skip_sides"], {"ic": {"right"}})
        self.assertEqual(new, recovered)
        self.assertEqual(flags, {"ic": [1, 0, 1]})
        self.assertEqual(overlays, {"ic": [None, color, None]})
        self.assertEqual(connected, {"ic": [None, True, None]})
        self.assertEqual(log["ic"]["old_index_map"], {0: 0, 1: 1})
        self.assertEqual(pins, {"ic": [[88, 150], [212, 160]]})

    def test_invalid_index_map_and_color_motion_are_rejected(self):
        for new, mapping, color in (({"ic": [[88, 150]]}, {"ic": {0: 1}}, {}),
                ({"ic": [[89, 150]]}, {"ic": {0: 0}}, {"ic": [{
                    "edge": "left", "frame_confirmed": True}]})):
            with self.subTest(new=new, mapping=mapping), patch.object(
                    pipeline.evidence_mod, "recover_frame_pins", return_value=(new, mapping, {})):
                with self.assertRaises(ValueError):
                    pipeline.apply_mono_frame_pins([], {"ic": [[88, 150]]},
                        None, None, [], {}, color, {})

    def test_new_points_reach_networks_and_final_evidence_in_same_order(self):
        helper = fixture.PipelinePinNetworkTests()
        comps = [helper.component("ic", [100, 100, 200, 200])]
        recovered = {"ic": [[88, 150], [88, 170]]}
        measured = {"point_img": [88, 170], "anchor_img": [100, 170],
                    "edge": "left", "trust": "trusted", "frame_confirmed": True}
        audit = {"ic": {"added": [{"index": 1, "estimated_endpoint": True,
                                   "evidence": measured}], "calibrated": []}}
        with patch.object(pipeline, "MONO_FRAME_PINS_ON", True), patch.object(
                pipeline.evidence_mod, "recover_frame_pins",
                return_value=(recovered, {"ic": {0: 0}}, audit)):
            doc, _, net = helper.run_case(comps, [[[0, 150], [90, 150]], [[0, 170], [90, 170]]],
                                           supplied_pins={"ic": [(88, 150)]})
        self.assertEqual(doc["pins"]["ic"], recovered["ic"])
        self.assertEqual(net.args[0], recovered)
        self.assertEqual(net.kwargs["pin_synth"], {"ic": [0, 1]})
        self.assertEqual([p["point_img"] for p in doc["pin_evidence"]["ic"]], recovered["ic"])
        self.assertEqual([n["pins"] for n in doc["nets"]], [[["ic", 0]], [["ic", 1]]])
        self.assertTrue(doc["pin_mono"]["ic"]["added"][0]["estimated_endpoint"])
        self.assertEqual(doc["pin_evidence"]["ic"][1]["anchor_img"], [100, 170])
        self.assertTrue(doc["pin_evidence"]["ic"][1]["estimated_endpoint"])

    def test_exact_measured_anchor_wins_over_neighbour_search_and_stale_is_rejected(self):
        old = {"ic": [{"point_img": [330, 311.5], "anchor_img": [318, 309.5]}]}
        measured = {"point_img": [330, 311.5], "anchor_img": [318, 311.5], "trust": "trusted"}
        audit = {"ic": {"calibrated": [{"index": 0, "evidence": measured}]}}
        actual = pipeline.overlay_mono_evidence(old, audit, {"ic": [[330, 311.5]]})
        self.assertEqual(actual["ic"][0]["anchor_img"], [318, 311.5])
        self.assertEqual(old["ic"][0]["anchor_img"], [318, 309.5])
        with self.assertRaises(ValueError):
            pipeline.overlay_mono_evidence(old, audit, {"ic": [[330, 320]]})

    def test_off_mode_never_calls_recovery(self):
        helper = fixture.PipelinePinNetworkTests()
        comps = [helper.component("ic", [100, 100, 200, 200])]
        with patch.object(pipeline, "MONO_FRAME_PINS_ON", False), patch.object(
                pipeline.evidence_mod, "recover_frame_pins") as read:
            doc, _, _ = helper.run_case(comps, [[[0, 150], [90, 150]]],
                                        supplied_pins={"ic": [(88, 150)]})
        read.assert_not_called()
        self.assertEqual(doc["pins"]["ic"], [[88., 150.]])
        self.assertEqual(doc["pin_mono"], {})

    def test_failure_keeps_old_geometry_and_network_and_reports_error(self):
        helper = fixture.PipelinePinNetworkTests()
        comps = [helper.component("ic", [100, 100, 200, 200])]
        def broken(components, copied, *args, **kwargs):
            copied["ic"][0][1] = 160
            raise ValueError("invalid recovery")
        with patch.object(pipeline, "MONO_FRAME_PINS_ON", True), patch.object(
                pipeline.evidence_mod, "recover_frame_pins", side_effect=broken):
            doc, _, _ = helper.run_case(comps, [[[0, 150], [90, 150]]],
                                        supplied_pins={"ic": [[88, 150]]})
        self.assertEqual(doc["pins"]["ic"], [[88., 150.]])
        self.assertEqual(doc["nets"][0]["pins"], [["ic", 0]])
        self.assertTrue(any(e.startswith("pin_mono:") for e in doc["errors"]))


if __name__ == "__main__":
    unittest.main()
