"""Terminal geometry and real pipeline regressions, including unseen rotations."""
import copy
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from stage1 import pipeline
from stage1.passive_pins import refine_pins
from stage2 import assemble


class PassivePinTests(unittest.TestCase):
    def resistor(self):
        image = np.full((220, 220, 3), 255, np.uint8)
        cv2.rectangle(image, (96, 90), (104, 110), (0, 0, 200), 1)
        cv2.line(image, (100, 75), (100, 90), (0, 0, 200), 1)
        cv2.line(image, (100, 110), (100, 125), (0, 0, 200), 1)
        cv2.line(image, (100, 20), (100, 74), (0, 140, 0), 1)
        cv2.line(image, (100, 126), (100, 180), (0, 140, 0), 1)
        comp = dict(id="r", type="r", bbox_img=[96, 90, 104, 110],
                    bbox_raw_img=[96, 90, 104, 110], center_img=[100, 100], pending=False)
        old = {"r": [(100, 80), (114, 100), (100, 120), (86, 100)]}
        return image, comp, old

    def test_colour_transition_uses_observed_wire_anchor_and_removes_side_contacts(self):
        image, comp, old = self.resistor()
        result, report = refine_pins([comp], old, image)
        self.assertEqual(len(result["r"]), 2)
        np.testing.assert_allclose(result["r"], [(100, 74.5), (100, 125.5)], atol=0.1)
        self.assertEqual(set(report["r"]["sources"]), {"body_attached_red_green_transition"})

    def test_quarter_turn_and_contact_reordering_preserve_terminals(self):
        image, comp, old = self.resistor()
        baseline, _ = refine_pins([comp], old, image)
        rotate = lambda p: (p[1], 219-p[0])
        box = np.asarray([rotate(comp["bbox_img"][:2]), rotate(comp["bbox_img"][2:])])
        turned = dict(comp, bbox_img=[*box.min(axis=0), *box.max(axis=0)])
        turned["bbox_raw_img"] = turned["bbox_img"]
        result, _ = refine_pins([turned], {"r": [rotate(p) for p in old["r"][::-1]]}, np.rot90(image))
        np.testing.assert_allclose(sorted(result["r"]), sorted(rotate(p) for p in baseline["r"]), atol=1)

    def test_monochrome_candidates_are_unchanged_even_with_visible_strokes(self):
        image = np.full((220, 220, 3), 255, np.uint8)
        _, comp, _ = self.resistor()
        old = {"r": [(100, 80), (86, 100)]}
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})
        cv2.line(image, (100, 110), (100, 125), (0, 0, 0), 1)
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_nc_cross_is_not_accepted_as_a_continuous_green_wire(self):
        image, comp, old = self.resistor()
        image[20:75, 99:102] = 255
        cv2.line(image, (96, 69), (104, 77), (0, 140, 0), 1)
        cv2.line(image, (96, 77), (104, 69), (0, 140, 0), 1)
        result, _ = refine_pins([comp], old, image)
        self.assertEqual(result["r"], [(100, 125.5)])

    def test_two_neutral_transition_pixels_do_not_erase_a_visible_terminal(self):
        image, comp, old = self.resistor()
        image[73:75, 99:102] = 255
        result, report = refine_pins([comp], old, image)
        self.assertEqual(len(result["r"]), 2)
        self.assertEqual(report["r"]["sources"][0], "body_attached_red_green_transition")

    def test_single_port_does_not_choose_between_two_confirmed_contacts(self):
        image, comp, old = self.resistor()
        comp["type"] = "gnd"
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_single_port_has_one_terminal_when_one_transition_is_confirmed(self):
        image, comp, old = self.resistor()
        comp["type"] = "v"
        image[110:181, 99:102] = 255
        result, report = refine_pins([comp], old, image)
        self.assertEqual(len(result["r"]), 1)
        np.testing.assert_allclose(result["r"][0], (100, 74.5), atol=0.1)
        self.assertNotIn("published_points_img", report["r"])

    def test_unconnected_nearby_red_symbol_cannot_supply_a_contact(self):
        image, comp, old = self.resistor()
        # Detached coloured strokes outside the body do not qualify.
        image[75:90, 99:102] = 255
        image[111:126, 99:102] = 255
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertEqual(report, {})

    def test_single_port_keeps_measured_boundary_despite_box_centre_offset(self):
        image, comp, old = self.resistor()
        comp.update(type="gnd", bbox_img=[95.2, 90, 103.2, 110])
        image[110:181, 99:102] = 255
        old["r"] = [(104.8, 80), (86, 100)]
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result["r"], [(100, 74.5)])
        self.assertEqual(report["r"]["coordinate_rule"], "red_green_pixel_boundary")

    def test_already_measured_points_still_carry_evidence_to_override_coordinate_stamp(self):
        image, comp, old = self.resistor()
        old["r"] = [(100, 74.5), (100, 125.5)]
        result, report = refine_pins([comp], old, image)
        self.assertEqual(result, old)
        self.assertTrue(report)

    def test_ic_candidates_and_input_objects_are_preserved(self):
        image, comp, old = self.resistor()
        ic = dict(comp, id="ic", type="box")
        old["ic"] = [(80, 100), (120, 100)]
        original = copy.deepcopy(([comp, ic], old, image))
        result, report = refine_pins([comp, ic], old, image)
        self.assertEqual(result["ic"], old["ic"])
        self.assertNotIn("ic", report)
        self.assertEqual([comp, ic], original[0])
        self.assertEqual(old, original[1])
        np.testing.assert_array_equal(image, original[2])

    def test_unknown_types_and_invalid_box_fall_back(self):
        image, comp, old = self.resistor()
        for replacement in (dict(type="switch"), dict(bbox_img=[0, 0, 0, 0])):
            with self.subTest(replacement=replacement):
                result, report = refine_pins([dict(comp, **replacement)], old, image)
                self.assertEqual(result, old)
                self.assertEqual(report, {})

    def test_boxes_outside_image_do_not_wrap_negative_slices(self):
        image, comp, old = self.resistor()
        for box in ([-200, 90, -50, 110], [96, -200, 104, -50],
                    [250, 90, 300, 110], [96, 250, 104, 300]):
            with self.subTest(box=box):
                outside = dict(comp, bbox_img=box, bbox_raw_img=box)
                result, report = refine_pins([outside], old, image)
                self.assertEqual(result, old)
                self.assertEqual(report, {})


class PassivePinPipelineTests(unittest.TestCase):
    def run_pipeline(self, enabled, stale_flags=False):
        image, comp, old = PassivePinTests().resistor()
        hypotheses = SimpleNamespace(case="", to_dict=lambda: {})
        detector = Mock()
        detector.detect.return_value = ([comp], hypotheses, {}, {})
        segments = [[[100, 20], [100, 74]], [[100, 126], [100, 180]]]
        wires = dict(segments=segments, endpoints=[], junctions=[], n_segments=2, scale=1)
        with ExitStack() as stack:
            stack.enter_context(patch.object(pipeline.cv2, "imread", return_value=image))
            stack.enter_context(patch.object(pipeline.wires_mod, "extract_wires", return_value=(wires, 0)))
            stack.enter_context(patch.object(pipeline.nets_mod, "build_pins", return_value=old))
            if stale_flags:
                stack.enter_context(patch.object(pipeline.pincol_mod, "enumerate_pins", return_value={}))
                stack.enter_context(patch.object(pipeline.pincol_mod, "merge_into",
                    return_value=(old, {"r": [True] * 4}, {})))
                stack.enter_context(patch.object(pipeline, "apply_color_pins",
                    return_value=(old, {"r": [True] * 4}, {}, {"r": [False] * 4}, {})))
            for key, value in (("BIGIC_ON", False), ("PINC_ON", stale_flags),
                               ("PIN_COLOR_ON", stale_flags), ("NON_IC_PINS_ON", enabled)):
                stack.enter_context(patch.object(pipeline, key, value))
            for key, value in (("WIRE_TOL", 1), ("WIRE_EXT", 0), ("WIRE_WELD", 0), ("WIRE_OVER", 0)):
                stack.enter_context(patch.object(pipeline.nets_mod.labels_mod, key, value))
            doc = pipeline.run_case(SimpleNamespace(case_id="integration", png_path="unused"),
                                    "unused", det=detector, do_text=False, do_rays=False, use_cv=False)
        self.assertEqual(doc["errors"], [])
        return doc

    def test_normal_entry_refines_before_network_indexing_and_can_be_disabled(self):
        baseline, candidate = self.run_pipeline(False), self.run_pipeline(True)
        self.assertEqual(len(baseline["pins"]["r"]), 4)
        self.assertEqual(len(candidate["pins"]["r"]), 2)
        self.assertTrue(candidate["passive_pin_report"])
        self.assertEqual(baseline["passive_pin_report"], {})
        self.assertEqual([n["pins"] for n in candidate["nets"]], [[["r", 0]], [["r", 1]]])

    def test_replacing_terminals_clears_stale_index_based_connection_flags(self):
        doc = self.run_pipeline(True, stale_flags=True)
        self.assertEqual([n["pins"] for n in doc["nets"]], [[["r", 0]], [["r", 1]]])


class PassivePinPublicationTests(unittest.TestCase):
    def fixture(self):
        comp = dict(id="p", type="r", bbox_json=[46, 20, 54, 40])
        stage = dict(image=dict(h=100), components=[comp], pins={"p": [[50, 45], [50, 95]]},
            passive_pin_report={"p": dict(type="r", after=[[50, 45], [50, 95]],
                coordinate_rule="red_green_pixel_boundary",
                sources=["body_attached_red_green_transition"] * 2)},
            wire_detail=dict(segments=[[[50, 10], [50, 45]]]),
            nets=[dict(pins=[["p", 0]], seg_index=[0])])
        assigned = dict(comps={"p": dict(designator=dict(text="R1"))})
        return stage, assigned

    def test_assembly_publishes_actual_colour_boundary_instead_of_coordinate_stamp(self):
        stage, assigned = self.fixture()
        with patch.object(assemble, "DROP_UNKEYED", False), \
             patch.object(assemble, "NET_DROP_UNKEYED", False):
            doc = assemble.build_doc(stage, assigned, "publication")
        key = next(iter(doc["components"]))
        self.assertEqual(sorted((p["point"]["x"], p["point"]["y"]) for p in doc["pins"][key].values()),
                         [(50, 5), (50, 55)])
        number = doc["__pin_index"]["p"]["0"].removeprefix("pin_")
        self.assertEqual(next(iter(doc["nets"].values()))["hyperGraph"], f"({key}.{number})")
        self.assertEqual(stage["pins"]["p"], [[50, 45], [50, 95]])

    def test_ic_stale_changed_type_and_invalid_boundary_records_are_ignored(self):
        stage, _ = self.fixture()
        for ctype, points, value in (("box", [[50, 45], [50, 95]], [[50, 45], [50, 95]]),
                                      ("l", [[50, 45], [50, 95]], [[50, 45], [50, 95]]),
                                      ("r", [[50, 47], [50, 95]], [[50, 45], [50, 95]]),
                                      ("r", [[50, 45], [50, 95]], [[float("nan"), 45], [50, 95]])):
            with self.subTest(ctype=ctype, points=points, value=value):
                stage["passive_pin_report"]["p"]["after"] = value
                self.assertEqual(assemble._measured_passive_points(stage, "p", ctype, points), {})


if __name__ == "__main__":
    unittest.main()
