"""Merged OCR strokes across a calibrated IC body must not become pin names.

These fixtures exercise observable assignment behavior, including legitimate
numeric names and imprecise detector boxes, rather than duplicating the guard.
The official case-3 slice uses recorded image coordinates, not GT names to
drive the matching algorithm. No dataset, OCR model or image file is required.
"""
import copy
import importlib
import json
import unittest
from unittest.mock import patch

from stage2 import structmatch
from stage2 import assemble

assign = importlib.import_module("stage2.assign")


def text(value, rect, score=1.0):
    return {"text": value, "bbox": list(rect), "score": score,
            "center": [(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2]}


def contact(point, side="left", trust="trusted", anchor=None, confirmed=True):
    return {"point_img": list(point), "edge": side, "trust": trust,
            "anchor_img": list(point if anchor is None else anchor),
            "frame_confirmed": confirmed}


def bind(box, pins, pool, records=None, prioritize=True):
    # A deliberately strong false candidate exposes eligibility errors instead
    # of letting an unrelated distance weight hide them.
    return structmatch.rebind(
        box, pins, pool, lambda c: c["text"], lambda c, pi: c["score"],
        ladder=False, pin_evidence=records, prioritize_trusted=prioritize)[0]


class PinNameFrameGuardTests(unittest.TestCase):
    box = [100, 100, 300, 300]

    def test_high_score_merged_rectangle_is_rejected_on_each_body_side(self):
        cases = [
            ("left", [90, 150], [100, 150], [95, 145, 115, 155], [110, 145, 130, 155]),
            ("right", [310, 150], [300, 150], [285, 145, 305, 155], [270, 145, 290, 155]),
            ("top", [150, 90], [150, 100], [145, 95, 155, 115], [145, 110, 155, 130]),
            ("bottom", [150, 310], [150, 300], [145, 285, 155, 305], [145, 270, 155, 290]),
        ]
        for side, point, anchor, merged, good in cases:
            with self.subTest(side=side):
                false = text("1K", merged, 10.0)
                self.assertTrue(100 < false["center"][0] < 300)
                self.assertTrue(100 < false["center"][1] < 300)
                records = [contact(point, side, anchor=anchor)]
                self.assertEqual(bind(self.box, [point], [false, text("GND", good)], records),
                                 {0: 1})

    def test_valid_alphanumeric_and_numeric_names_are_not_blacklisted(self):
        for value in ("1", "12", "A0", "D7", "VDD3V3", "GPIO_01"):
            with self.subTest(name=value):
                point = [90, 150]
                self.assertEqual(bind(self.box, [point], [text(value, [105, 145, 155, 155])],
                                      [contact(point, anchor=[100, 150])]), {0: 0})

    def test_long_name_crossing_symbol_center_stays_eligible(self):
        point = [90, 150]
        pool = [text("VDDIO_REGULATOR", [105, 145, 295, 155])]
        self.assertEqual(bind(self.box, [point], pool,
                              [contact(point, anchor=[100, 150])]), {0: 0})

    def test_quarter_turn_keeps_long_name_and_rejects_external_stroke(self):
        # A rotation changes the name's long axis, not its body containment.
        box = [-300, 100, -100, 300]
        point = [-150, 90]
        records = [contact(point, "top", anchor=[-150, 100])]
        pool = [text("1K", [-155, 95, -145, 115], 10.0),
                text("VDDIO_REGULATOR", [-155, 105, -145, 295])]
        self.assertEqual(bind(box, [point], pool, records), {0: 1})

    def test_small_ocr_halo_does_not_discard_a_real_name(self):
        cases = [("left", [90, 150], [100, 150], [99, 145, 119, 155]),
                 ("right", [310, 150], [300, 150], [281, 145, 301, 155]),
                 ("top", [150, 90], [150, 100], [145, 99, 155, 119]),
                 ("bottom", [150, 310], [150, 300], [145, 281, 155, 301])]
        for side, point, anchor, rect in cases:
            with self.subTest(side=side):
                self.assertEqual(bind(self.box, [point], [text("EN", rect)],
                                      [contact(point, side, anchor=anchor)]), {0: 0})

    def test_detector_box_inset_does_not_replace_calibrated_body(self):
        # Real frame is at x=100 while detector left side is x=106. A real
        # name at x=102 lies outside the detector but inside the actual body.
        box = [106, 100, 300, 300]
        point = [90, 150]
        pool = [text("EN", [102, 145, 122, 155])]
        self.assertEqual(bind(box, [point], pool,
                              [contact(point, anchor=[100, 150])]), {0: 0})

    def test_detector_box_outset_does_not_accept_cross_body_rectangle(self):
        # Center and the complete OCR box are inside the loose detector box,
        # but the merged stroke crosses the confirmed frame at x=106.
        point = [90, 150]
        pool = [text("1", [101, 145, 121, 155], 10),
                text("GND", [110, 145, 135, 155])]
        self.assertEqual(bind(self.box, [point], pool,
                              [contact(point, anchor=[106, 150])]), {0: 1})

    def test_consistent_subpixel_body_measurements_keep_frame_filter(self):
        pins = [[90, 150], [90, 220]]
        records = [contact(pins[0], anchor=[100, 150]),
                   contact(pins[1], anchor=[100.8, 220])]
        self.assertEqual(bind(self.box, pins, [text("merged", [95, 145, 115, 155])],
                              records), {})

    def test_contradictory_side_measurements_do_not_create_hard_boundary(self):
        pins = [[90, 150], [90, 220]]
        records = [contact(pins[0], anchor=[100, 150]),
                   contact(pins[1], anchor=[104, 220])]
        pool = [text("EN", [95, 145, 115, 155])]
        self.assertEqual(bind(self.box, pins, pool, records), {0: 0})

    def test_disagreement_on_one_side_does_not_disable_another_side(self):
        pins = [[90, 150], [90, 220], [310, 150]]
        records = [contact(pins[0], anchor=[100, 150]),
                   contact(pins[1], anchor=[104, 220]),
                   contact(pins[2], "right", anchor=[300, 150])]
        pool = [text("LEFT", [95, 145, 115, 155]),
                text("RIGHT_MERGED", [285, 145, 305, 155])]
        self.assertEqual(bind(self.box, pins, pool, records), {0: 0})

    def test_anchor_far_from_detector_side_is_neutral(self):
        point = [90, 150]
        records = [contact(point, anchor=[120, 150])]
        self.assertEqual(bind(self.box, [point], [text("EN", [95, 145, 115, 155])],
                              records), {0: 0})

    def test_missing_unconfirmed_or_invalid_anchor_is_neutral(self):
        point = [90, 150]
        pool = [text("EN", [95, 145, 115, 155])]
        records = [contact(point, anchor=[100, 150], confirmed=False),
                   contact(point, anchor=[100, 150], confirmed=1),
                   contact(point, anchor=[float("nan"), 150]),
                   contact(point, anchor=[100]),
                   contact(point, anchor=[100, 150])]
        records[-1].pop("anchor_img")
        for rec in records:
            with self.subTest(record=rec):
                self.assertEqual(bind(self.box, [point], pool, [rec]), {0: 0})

    def test_stale_point_correspondence_cannot_install_a_false_boundary(self):
        point = [90, 150]
        stale = [contact([90, 151], anchor=[100, 150])]
        pool = [text("EN", [95, 145, 115, 155])]
        self.assertEqual(bind(self.box, [point], pool, stale),
                         bind(self.box, [point], pool))
        self.assertEqual(bind(self.box, [point], pool, stale), {0: 0})

    def test_invalid_local_record_cannot_contribute_its_old_frame(self):
        pins = [[90, 150], [310, 200]]
        records = [contact([90, 151], anchor=[100, 150]),
                   contact(pins[1], "right", anchor=[300, 200])]
        # Fresh right evidence must not lend validity to stale left geometry.
        pool = [text("LEFT", [95, 145, 115, 155])]
        self.assertEqual(bind(self.box, pins, pool, records), {0: 0})

    def test_unmeasured_side_is_not_assumed_to_be_detector_edge(self):
        point = [90, 150]
        pool = [text("TOP_TOUCH", [105, 95, 125, 155])]
        # Only the left frame has evidence. Crossing the unmeasured top side
        # is insufficient grounds to discard this vertically extended box.
        self.assertEqual(bind(self.box, [point], pool,
                              [contact(point, anchor=[100, 150])]), {0: 0})

    def test_unrecognized_evidence_side_falls_back_to_legacy_matching(self):
        point = [90, 150]
        pool = [text("EN", [95, 145, 115, 155])]
        records = [contact(point, "unknown", anchor=[100, 150])]
        self.assertEqual(bind(self.box, [point], pool, records),
                         bind(self.box, [point], pool))

    def test_trusted_and_uncertain_passes_share_crossing_exclusion(self):
        pins = [[90, 150], [90, 220]]
        records = [contact(pins[0], anchor=[100, 150]),
                   contact(pins[1], trust="uncertain", anchor=[100, 220])]
        pool = [text("MERGED_A", [95, 145, 115, 155], 10),
                text("A", [110, 145, 130, 155]),
                text("MERGED_B", [95, 215, 115, 225], 10),
                text("B", [110, 215, 130, 225])]
        self.assertEqual(bind(self.box, pins, pool, records), {0: 1, 1: 3})

    def test_only_uncertain_contact_still_filters_a_confirmed_body_side(self):
        point = [90, 150]
        records = [contact(point, trust="uncertain", anchor=[100, 150])]
        self.assertEqual(bind(self.box, [point], [text("MERGED", [95, 145, 115, 155])],
                              records), {})

    def test_filter_keeps_original_pool_indices_and_does_not_mutate_input(self):
        pins = [[90, 150], [90, 220]]
        records = [contact(p, anchor=[100, p[1]]) for p in pins]
        pool = [text("MERGED", [95, 145, 115, 155]),
                text("A", [110, 145, 130, 155]),
                text("MERGED", [95, 215, 115, 225]),
                text("B", [110, 215, 130, 225])]
        identities = [id(c) for c in pool]
        self.assertEqual(bind(self.box, pins, pool, records), {0: 1, 1: 3})
        self.assertEqual([id(c) for c in pool], identities)
        self.assertEqual([c["text"] for c in pool], ["MERGED", "A", "MERGED", "B"])

    def test_number_matching_does_not_use_name_containment_guard(self):
        point = [90, 150]
        records = [contact(point, anchor=[100, 150])]
        pool = [text("4", [95, 145, 115, 155])]
        self.assertEqual(bind(self.box, [point], pool, records, prioritize=False),
                         bind(self.box, [point], pool, prioritize=False))
        self.assertEqual(bind(self.box, [point], pool, records, prioritize=False), {0: 0})

    def test_missing_text_rectangle_is_neutral(self):
        point = [90, 150]
        pool = [{"text": "EN", "center": [110, 150], "score": 1.0}]
        self.assertEqual(bind(self.box, [point], pool,
                              [contact(point, anchor=[100, 150])]), {0: 0})

    def test_nested_ocr_rectangle_gets_same_crossing_filter(self):
        point = [90, 150]
        false = text("merged", [95, 145, 115, 155])
        false["rec"] = {"bbox": false.pop("bbox")}
        self.assertEqual(bind(self.box, [point], [false],
                              [contact(point, anchor=[100, 150])]), {})

    def test_official_case_3_en_and_external_partname_keep_separate_roles(self):
        # Recorded official case-3 text/frame geometry (image coordinates).
        # Role reads isolate guard behavior from OCR spelling quality; the
        # frozen OCR misspelled this visible partname as S1313ADC.
        box = [816.11, 404.56, 934.1, 501.34]
        points = [[804.1, 483.5], [838.0, 513.3]]
        records = [contact(points[0], anchor=[815.5, 483.5]),
                   contact(points[1], "bottom", "rejected", [839.0, 500.5])]
        texts = [dict(bbox=[824, 482, 842, 492], reads={"pin_name": ("EN", .95)}),
                 dict(bbox=[816, 504, 903, 514], reads={"partname": ("SY8113BADC", .99)}),
                 dict(bbox=[808, 473, 816, 483], reads={"pin_number": ("4", .95)})]
        with patch.object(assign, "PIN_TRUST_BIND", True):
            got = assign.assign([dict(id="ic", type="box", bbox_img=box)],
                                {"ic": points}, texts, pin_evidence={"ic": records})
        ic = got["comps"]["ic"]
        self.assertEqual(ic["pins"][0]["name"]["text"], "EN")
        self.assertEqual(ic["pins"][0]["number"]["text"], "4")
        self.assertIsNone(ic["pins"][1]["name"])
        self.assertEqual(ic["partname"]["text"], "SY8113BADC")

    @staticmethod
    def bind_with_reason(box, pins, pool, records, guard=True, score=None):
        rejected = []
        chosen, _ = structmatch.rebind(
            box, pins, pool, lambda c: c["text"],
            score or (lambda c, pi: c["score"]), ladder=False,
            pin_evidence=records, prioritize_trusted=True,
            name_frame_guard=guard, frame_rejected_name_pins=rejected)
        return chosen, rejected

    def test_empty_slot_is_flagged_only_when_its_former_name_was_rejected(self):
        pins = [[90, 150], [90, 220]]
        records = [contact(p, anchor=[100, p[1]]) for p in pins]
        pool = [text("EN", [95, 145, 115, 155])]
        unguarded, unguarded_reason = self.bind_with_reason(self.box, pins, pool, records, False)
        guarded, guarded_reason = self.bind_with_reason(self.box, pins, pool, records)
        self.assertEqual(unguarded, {0: 0})
        self.assertEqual(guarded, {})
        self.assertEqual(guarded_reason, [0])
        # Pin 1 was already empty; the new filter did not cause its emptiness.
        self.assertEqual(unguarded_reason, [])

    def test_removed_but_previously_unmatched_text_does_not_flag_empty_slot(self):
        point = [90, 150]
        records = [contact(point, anchor=[100, 150])]
        # Along the wrong row, this crossing word was never eligible for the
        # contact in either run. A rejection alone must not imply causality.
        pool = [text("EN", [95, 195, 115, 205])]
        self.assertEqual(self.bind_with_reason(self.box, [point], pool, records), ({}, []))
        self.assertEqual(self.bind_with_reason(self.box, [point], pool, records, False), ({}, []))

    def test_loss_of_eligible_name_after_global_rematching_is_not_flagged(self):
        # Removing a second, crossed word changes the optimum: the legitimate
        # word is now taken by the second contact. The first slot is empty,
        # but its former word was not rejected, so it must not get the flag.
        pins = [[90, 150], [90, 153]]
        records = [contact(p, anchor=[100, p[1]]) for p in pins]
        pool = [text("A", [101, 145, 121, 155]),
                text("EN", [95, 149, 115, 159])]
        def score(c, pi):
            return (2 if pi == 0 else 3) if c["text"] == "A" else (0 if pi == 0 else 10)
        self.assertEqual(self.bind_with_reason(self.box, pins, pool, records, False, score),
                         ({0: 0, 1: 1}, []))
        self.assertEqual(self.bind_with_reason(self.box, pins, pool, records, True, score),
                         ({1: 0}, []))

    def test_no_actual_text_rejection_does_not_run_counterfactual_matching(self):
        point = [90, 150]
        records = [contact(point, anchor=[100, 150])]
        pool = [text("EN", [110, 145, 130, 155])]
        with patch.object(structmatch, "rebind", wraps=structmatch.rebind) as spy:
            self.assertEqual(self.bind_with_reason(self.box, [point], pool, records), ({0: 0}, []))
        self.assertEqual(spy.call_count, 1)

    def test_guard_disabled_preserves_name_and_does_not_report_rejection(self):
        point = [90, 150]
        records = [contact(point, anchor=[100, 150])]
        pool = [text("EN", [95, 145, 115, 155])]
        with patch.object(structmatch, "rebind", wraps=structmatch.rebind) as spy:
            self.assertEqual(self.bind_with_reason(self.box, [point], pool, records, False),
                             ({0: 0}, []))
        self.assertEqual(spy.call_count, 1)

    def test_rejection_reason_keeps_original_contact_and_candidate_indices(self):
        pins = [[310, 220], [90, 150]]
        records = [contact(pins[0], "right", anchor=[300, 220]),
                   contact(pins[1], anchor=[100, 150])]
        pool = [text("EN", [95, 145, 115, 155]),
                text("GND", [270, 215, 290, 225])]
        self.assertEqual(self.bind_with_reason(self.box, pins, pool, records, False),
                         ({0: 1, 1: 0}, []))
        self.assertEqual(self.bind_with_reason(self.box, pins, pool, records), ({0: 1}, [1]))

    def test_assign_flag_and_assembly_restore_only_complete_duplicate_contact(self):
        points = [[90, 150], [90, 220]]
        components = [dict(id="ic", type="box", bbox_img=self.box, bbox_json=self.box)]
        records = [contact(p, anchor=[100, p[1]]) for p in points]
        texts = [dict(bbox=[105, 75, 120, 85], reads={"designator": ("U1", .99)}),
                 dict(bbox=[95, 145, 115, 155], reads={"pin_name": ("EN", .99)}),
                 dict(bbox=[110, 215, 140, 225], reads={"pin_name": ("GND", .99)}),
                 dict(bbox=[80, 145, 88, 155], reads={"pin_number": ("8", .99)}),
                 dict(bbox=[80, 215, 88, 225], reads={"pin_number": ("8", .99)})]
        with patch.object(assign, "PIN_TRUST_BIND", True):
            assigned = assign.assign(components, {"ic": points}, texts,
                                     pin_evidence={"ic": records})
        pinfo = assigned["comps"]["ic"]["pins"]
        self.assertTrue(pinfo[0]["name_rejected_by_frame"])
        self.assertIsNone(pinfo[0]["name"])
        self.assertEqual(pinfo[1]["name"]["text"], "GND")
        self.assertNotIn("name_rejected_by_frame", pinfo[1])
        self.assertEqual([pinfo[k]["number"]["text"] for k in range(2)], ["8", "8"])
        # The flag must survive the JSON boundary and expose the surviving
        # contact's own coordinate/net, never copy GND onto the rejected point.
        snapshot = copy.deepcopy(assigned)
        stage1 = {"image": {"w": 400, "h": 400}, "components": components,
                  "pins": {"ic": points},
                  "wire_detail": {"segments": [[p, [0, p[1]]] for p in points]},
                  "nets": [{"pins": [["ic", k]], "seg_index": [k]} for k in range(2)]}
        for string_keys in (False, True):
            with self.subTest(string_keys=string_keys):
                reads = json.loads(json.dumps(assigned)) if string_keys else assigned
                with patch.object(assemble, "BOX_PIN_DUPLICATE_BEST", True), \
                        patch.object(assemble, "BOX_PIN_PARTIAL", True), \
                        patch.object(assemble, "BOX_PIN_DROP_UNREAD", True), \
                        patch.object(assemble, "BOX_PIN_DROP_MIXED_ONLY", True), \
                        patch.object(assemble, "BOX_PIN_DROP_CAP", False), \
                        patch.object(assemble, "BOX_PIN_NEED_NAME", True):
                    doc = assemble.build_doc(stage1, reads, "synthetic")
                self.assertEqual(doc["pins"]["U1"], {
                    "pin_8": {"pinname": "GND", "point": {"x": 90., "y": 180.}}})
                self.assertEqual(doc["__pin_index"]["ic"], {"1": "pin_8"})
                self.assertEqual(len(doc["nets"]), 1)
                net = next(iter(doc["nets"].values()))
                self.assertEqual(net["hyperGraph"], "(U1.8)")
                self.assertEqual(next(iter(net["edges"].values()))[0]["y"], 180.)
        self.assertEqual(assigned, snapshot)

    def test_assign_invalid_claimed_frame_anchor_keeps_neutral_name_ranking(self):
        point = [90, 150]
        components = [dict(id="ic", type="box", bbox_img=self.box)]
        texts = [dict(bbox=[110, 145, 140, 155], reads={"pin_name": ("GND", .99)}),
                 dict(bbox=[110, 148, 140, 158], reads={"pin_name": ("VDD", .99)})]
        neutral = contact(point, confirmed=False)
        neutral.pop("anchor_img")
        variants = [neutral,
                    contact(point, anchor=[100, 250]),  # Correct side, stale row.
                    contact(point, anchor=[120, 150])]  # Correct row, wrong side.
        outputs = []
        for record in variants:
            with self.subTest(record=record):
                evidence = {"ic": [record]}
                original = copy.deepcopy(evidence)
                with patch.object(assign, "PIN_TRUST_BIND", True), \
                        patch.object(assign, "PIN_ALONG", True):
                    got = assign.assign(components, {"ic": [point]}, texts,
                                        pin_evidence=evidence)
                name = got["comps"]["ic"]["pins"][0]["name"]
                self.assertEqual(name["text"], "GND")
                outputs.append(name)
                self.assertEqual(evidence, original)
        self.assertEqual(outputs, [outputs[0]] * len(variants))

    def test_assign_uncertain_name_with_five_pixel_gap_keeps_unconfirmed_frame_behavior(self):
        point = [90, 150]
        components = [dict(id="ic", type="box", bbox_img=self.box)]
        # The text projection starts 5 pixels from this uncertain contact's
        # row. A cache with no confirmed body must retain its former name,
        # rather than newly applying a precise directional corridor gate.
        texts = [dict(bbox=[110, 155, 140, 165], reads={"pin_name": ("GND", .99)})]
        unconfirmed = contact(point, trust="uncertain", anchor=[100, 150], confirmed=False)
        missing_confirmation = dict(unconfirmed)
        missing_confirmation.pop("frame_confirmed")
        missing_anchor = dict(missing_confirmation)
        missing_anchor.pop("anchor_img")
        for record in (unconfirmed, missing_confirmation, missing_anchor):
            with self.subTest(record=record):
                evidence = {"ic": [record]}
                original = copy.deepcopy(evidence)
                with patch.object(assign, "PIN_TRUST_BIND", True):
                    got = assign.assign(components, {"ic": [point]}, texts,
                                        pin_evidence=evidence)
                pin = got["comps"]["ic"]["pins"][0]
                self.assertEqual(pin["name"]["text"], "GND")
                self.assertNotIn("name_rejected_by_frame", pin)
                self.assertEqual(evidence, original)


if __name__ == "__main__":
    unittest.main()
