"""Main's recovery passes must respect existing measured contacts."""
import importlib
import unittest
from unittest.mock import patch

from stage2 import assemble

assign_module = importlib.import_module("stage2.assign")


class MainPinCompatibilityTests(unittest.TestCase):
    def bind(self, merge, measured):
        points = [[90, 130], [90, 140]]
        components = [dict(id="ic", type="box", bbox_img=[100, 100, 200, 200])]
        records = [dict(bbox=[76, 119, 84, 126], reads={"pin_number": ("3", .99)}),
                   dict(bbox=[110, 136, 123, 145], reads={"pin_name": ("EN", .99)})]
        evidence = {"ic": [dict(point_img=p, anchor_img=[100, p[1]], edge="left",
                                trust="trusted", frame_confirmed=True,
                                source="red_green_transition") for p in points]} if measured else None
        with patch.object(assign_module, "STUB_MERGE", merge):
            return assign_module.assign(components, {"ic": points}, records,
                                        pin_evidence=evidence)["comps"]["ic"]["pins"]

    def test_measured_adjacent_rows_keep_their_own_number_and_name(self):
        off = self.bind(0, True)
        on = self.bind(12, True)
        self.assertEqual(on, off)
        self.assertEqual(on[0]["number"]["text"], "3")
        self.assertIsNone(on[0]["name"])
        self.assertEqual(on[1]["name"]["text"], "EN")
        self.assertIsNone(on[1]["number"])

    def test_unmeasured_legacy_phantom_slot_recovery_remains_available(self):
        on = self.bind(12, False)
        self.assertEqual(on[1]["number"]["text"], "3")
        self.assertEqual(on[1]["merged_from"], 0)

    def test_diode_to_led_type_refinement_preserves_observed_terminal_positions(self):
        points = [[70, 150], [230, 150]]
        component = dict(id="d", type="d", bbox_img=[100, 140, 200, 160],
                         bbox_json=[100, 140, 200, 160], bbox_raw_img=[100, 140, 200, 160])
        stage = dict(image=dict(w=300, h=300), components=[component], pins={"d": points},
                     passive_pin_report={"d": dict(type="d", coordinate_rule="image_color_terminal_rules",
                                                   after=points, sources=["body_attached_red_green_transition"] * 2)}, nets=[])
        assigned = dict(comps={"d": dict(designator=dict(text="LED1"), ctype_fix="led")})
        doc = assemble.build_doc(stage, assigned)
        self.assertEqual(doc["components"]["LED1"]["type"], "led")
        self.assertEqual({(v["point"]["x"], v["point"]["y"])
                          for v in doc["pins"]["LED1"].values()}, {(70., 150.), (230., 150.)})


if __name__ == "__main__":
    unittest.main()
