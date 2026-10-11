"""A text-shaped contact must not steal a real contact's internal label."""
import importlib
import json
import unittest
from unittest.mock import patch

from stage2 import structmatch

assign = importlib.import_module("stage2.assign")


def evidence(point, edge, trust, anchor=None):
    return {"point_img": list(point), "edge": edge, "trust": trust,
            "confidence": 0.9 if trust == "trusted" else 0.3,
            "anchor_img": list(anchor or point), "reason": "test frame contact"}


def cand(text, box):
    return {"text": text, "center": [(box[0] + box[2]) / 2,
                                     (box[1] + box[3]) / 2],
            "bbox": box, "score": 1.0}


def binding(box, pins, pool, records=None, prioritize=True):
    def score(cc, pi):
        edge = structmatch.pin_edge(box, pins, pi, records)
        anchor = structmatch.pin_anchor(pins, pi, records)
        return 1.0 / (1.0 + abs(structmatch.along(box, edge, *anchor)
                               - structmatch.along(box, edge, *cc["center"])) / 20)
    return structmatch.rebind(box, pins, pool, lambda cc: cc["text"], score,
                              pin_evidence=records, prioritize_trusted=prioritize)[0]


class TrustedPinBindingTests(unittest.TestCase):
    def test_en_true_left_contact_wins_over_closer_uncertain_bottom(self):
        # Coordinates from the official case 3 intermediate contact audit.
        box = [816.11, 405.0, 929.0, 501.34]
        pins = [[804.1, 483.5], [838.0, 513.3]]
        pool = [cand("EN", [824, 482, 842, 492])]
        self.assertEqual(binding(box, pins, pool), {1: 0})
        records = [evidence(pins[0], "left", "trusted", [816.11, 483.5]),
                   evidence(pins[1], "bottom", "uncertain", [838.0, 501.34])]
        self.assertEqual(binding(box, pins, pool, records), {0: 0})

    def test_uncertain_real_side_keeps_label_unreachable_from_trusted_side(self):
        box = [100, 100, 300, 300]
        pins = [[90, 250], [200, 90]]
        records = [evidence(pins[0], "left", "trusted", [100, 250]),
                   evidence(pins[1], "top", "uncertain", [200, 100])]
        pool = [cand("LEFT", [105, 243, 146, 257]),
                cand("TOP", [190, 106, 210, 118])]
        self.assertEqual(binding(box, pins, pool, records), {0: 0, 1: 1})

    def test_projection_handles_long_and_rotated_labels_at_corners(self):
        box = [100, 100, 300, 300]
        pins = [[90, 115], [110, 90]]
        records = [evidence(pins[0], "left", "trusted", [100, 115]),
                   evidence(pins[1], "top", "uncertain", [110, 100])]
        # A long name's center is far to the right; its y projection is local.
        pool = [cand("VDDIO_REGULATOR", [105, 109, 270, 121])]
        self.assertEqual(binding(box, pins, pool, records), {0: 0})
        # Quarter-turn the complete case, including the vertical OCR box.
        rot = lambda pt: [-pt[1], pt[0]]
        rpins = [rot(pt) for pt in pins]
        rrecords = [evidence(rpins[0], "top", "trusted", rot([100, 115])),
                    evidence(rpins[1], "right", "uncertain", rot([110, 100]))]
        rpool = [cand("VDDIO_REGULATOR", [-121, 105, -109, 270])]
        self.assertEqual(binding([-300, 100, -100, 300], rpins, rpool, rrecords), {0: 0})

    def test_pair_gate_does_not_spread_a_local_label_to_another_trusted_row(self):
        box = [100, 100, 300, 300]
        pins = [[90, 140], [90, 200], [90, 260]]
        records = [evidence(p, "left", "trusted", [100, p[1]]) for p in pins]
        pool = [cand("A", [105, 136, 115, 144]), cand("B", [105, 196, 115, 204])]
        got = binding(box, pins, pool, records)
        self.assertEqual(got, {0: 0, 1: 1})
        self.assertEqual(len(set(got.values())), len(got))

    def test_rejected_points_cannot_take_names_or_numbers(self):
        box = [100, 100, 300, 300]
        pins = [[90, 150], [200, 310]]
        records = [evidence(pins[0], "left", "uncertain"),
                   evidence(pins[1], "bottom", "rejected")]
        pool = [cand("EN", [105, 143, 120, 157]), cand("4", [190, 305, 200, 315])]
        for prioritize in (False, True):
            got = binding(box, pins, pool, records, prioritize)
            self.assertNotIn(1, got)
            self.assertEqual(len(set(got.values())), len(got))

    def test_number_ladder_keeps_full_mixed_trust_row(self):
        box = [100, 100, 300, 300]
        pins = [[90, 130], [90, 170], [90, 210], [90, 250]]
        records = [evidence(p, "left", "trusted" if i % 2 else "uncertain")
                   for i, p in enumerate(pins)]
        pool = [cand(str(i + 1), [75, p[1] - 5, 85, p[1] + 5])
                for i, p in enumerate(pins)]
        self.assertEqual(binding(box, pins, pool, records, False),
                         {0: 0, 1: 1, 2: 2, 3: 3})

    def test_stale_evidence_does_not_attach_trust_to_another_point(self):
        pins = [[90, 150], [200, 310]]
        records = [evidence(pins[1], "bottom", "trusted"),
                   evidence(pins[0], "left", "rejected")]
        checked = structmatch.normalize_pin_evidence(pins, records)
        self.assertEqual(checked, [])
        self.assertFalse(structmatch.normalize_pin_evidence(pins, records[:1]))

    def test_valid_uncertain_evidence_is_not_confused_with_invalid_correspondence(self):
        pins = [[90, 150], [200, 310]]
        records = [evidence(pins[0], "left", "uncertain"),
                   evidence(pins[1], "bottom", "uncertain")]
        checked = structmatch.normalize_pin_evidence(pins, records)
        self.assertEqual([r["trust"] for r in checked], ["uncertain", "uncertain"])
        self.assertEqual([r["edge"] for r in checked], ["left", "bottom"])

    def test_mixed_valid_and_invalid_evidence_keeps_valid_trust_and_falls_back_locally(self):
        pins = [[90, 150], [200, 310]]
        records = [evidence(pins[0], "left", "trusted", [100, 150]),
                   evidence([0, 0], "left", "rejected", [100, 200])]
        checked = structmatch.normalize_pin_evidence(pins, records)
        self.assertEqual([r["trust"] for r in checked], ["trusted", "uncertain"])
        self.assertEqual(checked[0]["anchor_img"], [100, 150])
        self.assertNotIn("edge", checked[1])
        self.assertNotIn("anchor_img", checked[1])

    @staticmethod
    def in_lx_case():
        box = [816.11, 404.56, 934.1, 501.34]
        pins = [[804.1, 422.5], [946.1, 454.0]]
        records = [evidence(pins[0], "left", "trusted", [815.5, 425.5]),
                   evidence(pins[1], "right", "trusted", [933.5, 455.0])]
        pool = [dict(cand("IN", [825, 424, 837, 434]), score=.85),
                dict(cand("LX", [909, 424, 926, 434]), score=.99)]
        return box, pins, pool, records

    def test_missing_right_contact_does_not_move_lx_to_true_left_in_contact(self):
        # The right top pin is missing from stage1 in actual official case 3.
        # Along-row evidence alone gave the higher-scoring LX to the left pin.
        box, pins, pool, records = self.in_lx_case()
        self.assertEqual(binding(box, pins, pool, records), {0: 0})

    def test_present_right_contact_can_take_its_own_lx(self):
        box, pins, pool, records = self.in_lx_case()
        pins.append([946.1, 422.5])
        records.append(evidence(pins[2], "right", "trusted", [933.5, 425.5]))
        self.assertEqual(binding(box, pins, pool, records), {0: 0, 2: 1})

    def test_normal_side_constraint_survives_horizontal_mirror(self):
        box, pins, pool, records = self.in_lx_case()
        total = box[0] + box[2]
        pins = [[total - x, y] for x, y in pins]
        records = [evidence(p, "right" if r["edge"] == "left" else "left", r["trust"],
                            [total - r["anchor_img"][0], r["anchor_img"][1]])
                   for p, r in zip(pins, records)]
        pool = [dict(cand(c["text"], [total - c["bbox"][2], c["bbox"][1],
                                     total - c["bbox"][0], c["bbox"][3]]), score=c["score"])
                for c in pool]
        self.assertEqual(binding(box, pins, pool, records), {0: 0})

    def test_normal_side_constraint_survives_top_bottom_quarter_turn(self):
        box, pins, pool, records = self.in_lx_case()
        rotate = lambda p: [-p[1], p[0]]
        pins = [rotate(p) for p in pins]
        records = [evidence(p, "top" if r["edge"] == "left" else "bottom", r["trust"],
                            rotate(r["anchor_img"])) for p, r in zip(pins, records)]
        pool = [dict(cand(c["text"], [-c["bbox"][3], c["bbox"][0],
                                     -c["bbox"][1], c["bbox"][2]]), score=c["score"])
                for c in pool]
        box = [-box[3], box[0], -box[1], box[2]]
        self.assertEqual(binding(box, pins, pool, records), {0: 0})

    def test_uncertain_phase_cannot_restore_cross_edge_name_assignment(self):
        box, pins, pool, records = self.in_lx_case()
        records[0]["trust"] = "uncertain"
        self.assertEqual(binding(box, pins, pool, records), {0: 0})

    def test_full_width_name_remains_ambiguous_instead_of_being_dropped(self):
        box = [100, 100, 300, 300]
        candidate = cand("VDDIO_REGULATOR", [105, 144, 297, 156])
        self.assertTrue(structmatch._candidate_faces_edge(candidate, box, "left"))
        self.assertTrue(structmatch._candidate_faces_edge(candidate, box, "right"))

    def test_missing_name_bbox_keeps_legacy_projection_eligibility(self):
        candidate = dict(text="EN", center=[110, 150], score=1.0)
        self.assertTrue(structmatch._candidate_faces_edge(candidate, [100, 100, 300, 300], "right"))

    def test_all_invalid_evidence_preserves_legacy_pincol_window_names(self):
        box = [816.11, 405, 934.1, 501.34]
        components = [dict(id="ic", type="box", bbox_img=box)]
        pins = {"ic": [[804.1, 422.5], [804.1, 453], [804.1, 483.5], [838, 513.3]]}
        names = [("IN", [824, 422, 842, 432]), ("GND", [824, 452, 851, 462]),
                 ("EN", [824, 482, 842, 492])]
        texts = [dict(bbox=bb, reads={"pin_name": (name, .95)}) for name, bb in names]
        cols = {"ic": [dict(in_win=bb) for _name, bb in names]}
        stale = {"ic": [evidence([0, 0], "left", "trusted") for _p in pins["ic"]]}
        with patch.object(assign, "PINCOL_GUIDE", True), patch.object(assign, "PIN_TRUST_BIND", True):
            old = assign.assign(components, pins, texts, pincol=cols)
            new = assign.assign(components, pins, texts, pincol=cols, pin_evidence=stale)
        self.assertEqual(json.dumps(old, sort_keys=True), json.dumps(new, sort_keys=True))
        self.assertEqual(new["comps"]["ic"]["pins"][2]["name"]["text"], "EN")

    def test_number_assignment_uses_original_points_and_no_evidence_on_7_8_slices(self):
        # Candidate coordinates and number-box geometry from the official 7/8
        # regression audit. These are local path checks, not full-image scores.
        cases = [
            ("7", [822.36, 323.89, 959.99, 441.53],
             [[810.4, 423], [810.4, 381]], ["left", "left"],
             [[821.5, 424], [821.5, 384]], ["trusted", "trusted"],
             [("7", .9614, [815, 413, 822, 423]),
              ("8", .9771, [814, 374, 822, 384])]),
            ("8", [994.93, 453.68, 1172.31, 770.37],
             [[982.9, 501.4], [1184.3, 493.5]], ["left", "right"],
             [[994, 501.4], [1171.5, 493.5]], ["uncertain", "trusted"],
             [("8", .8735, [987, 483, 994, 493]),
              ("7", .4640, [967, 490, 996, 505]),
              ("7", .9752, [1174, 482, 1181, 492])]),
        ]
        for case, box, points, edges, anchors, trusts, values in cases:
            with self.subTest(case=case):
                components = [dict(id="ic", type="box", bbox_img=box)]
                pins = {"ic": points}
                records = {"ic": [evidence(p, e, t, a)
                                   for p, e, t, a in zip(points, edges, trusts, anchors)]}
                texts = [dict(bbox=bb, reads={"pin_number": (text, conf)})
                         for text, conf, bb in values]
                old = assign.assign(components, pins, texts)
                with patch.object(assign.structmatch, "rebind", wraps=structmatch.rebind) as spy:
                    new = assign.assign(components, pins, texts, pin_evidence=records)
                self.assertEqual(old["comps"]["ic"]["pins"], new["comps"]["ic"]["pins"])
                calls = [call for call in spy.call_args_list
                         if call.kwargs.get("prioritize_trusted") is False]
                self.assertTrue(calls)
                for call in calls:
                    self.assertFalse(call.kwargs.get("pin_evidence"))
                    # Check the scoring closure too: not only the rebind flag.
                    pool, score_fn = call.args[2], call.args[4]
                    for candidate in pool:
                        for pi, point in enumerate(points):
                            edge = structmatch.edge_of_point(box, *point)
                            delta = abs(structmatch.along(box, edge, *point)
                                        - structmatch.along(box, edge, *candidate["center"]))
                            expected = candidate["score"] / (1 + delta / assign.PINNUM_DECAY)
                            self.assertAlmostEqual(score_fn(candidate, pi), expected, places=10)

    def test_assign_threads_evidence_but_absent_or_switch_off_is_identical(self):
        box = [816.11, 405.0, 929.0, 501.34]
        components = [{"id": "ic", "type": "box", "bbox_img": box}]
        pins = {"ic": [[804.1, 483.5], [838.0, 513.3]]}
        records = {"ic": [evidence(pins["ic"][0], "left", "trusted", [816.11, 483.5]),
                          evidence(pins["ic"][1], "bottom", "uncertain", [838, 501.34])]}
        texts = [{"bbox": [824, 482, 842, 492], "reads": {"pin_name": ("EN", 0.95)}},
                 {"bbox": [816, 505, 921, 517],
                  "reads": {"partname": ("SY8113BADC", 0.99)}},
                 {"bbox": [804, 478, 812, 487], "reads": {"pin_number": ("4", 0.95)}}]
        with patch.object(assign, "PIN_TRUST_BIND", False):
            old = assign.assign(components, pins, texts)
            disabled = assign.assign(components, pins, texts, pin_evidence=records)
        with patch.object(assign, "PIN_TRUST_BIND", True):
            no_evidence = assign.assign(components, pins, texts)
            new = assign.assign(components, pins, texts, pin_evidence=records)
        dumps = lambda doc: json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(dumps(old), dumps(no_evidence))
        self.assertEqual(dumps(old), dumps(disabled))
        self.assertEqual(old["comps"]["ic"]["pins"][1]["name"]["text"], "EN")
        self.assertIsNone(old["comps"]["ic"]["pins"][0]["name"])
        true_pin = new["comps"]["ic"]["pins"][0]
        self.assertEqual(true_pin["name"]["text"], "EN")
        self.assertEqual(true_pin["number"]["text"], "4")
        self.assertIsNone(new["comps"]["ic"]["pins"][1]["name"])


if __name__ == "__main__":
    unittest.main()
