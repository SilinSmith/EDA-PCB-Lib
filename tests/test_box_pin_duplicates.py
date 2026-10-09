"""Conflicting OCR numbers must retain the right contact and network mapping."""
import unittest
from unittest.mock import patch

from stage2 import assemble


class BoxPinDuplicateTests(unittest.TestCase):
    @staticmethod
    def read(number, name, nc=.99, nn=.99):
        return {"number": {"text": number, "conf": nc},
                "name": {"text": name, "conf": nn}}

    def build(self, reads, enabled=True, string_keys=False, partial=True,
              need_name=True, evidence=None):
        points = [[212, 120], [212, 180]]
        stage1 = {
            "image": {"w": 300, "h": 300},
            "components": [{"id": "ic", "type": "box",
                            "bbox_img": [100, 100, 200, 200],
                            "bbox_json": [100, 100, 200, 200]}],
            "pins": {"ic": points},
            "wire_detail": {"segments": [[p, [280, p[1]]] for p in points]},
            "nets": [{"pins": [["ic", k]], "seg_index": [k]} for k in range(2)],
        }
        assigned = {"comps": {"ic": {"designator": {"text": "U1"},
                         "pins": {str(k) if string_keys else k: r
                                  for k, r in enumerate(reads)}}}}
        if evidence is not None:
            stage1["pin_evidence"] = {"ic": evidence}
        with patch.object(assemble, "BOX_PIN_DUPLICATE_BEST", enabled), \
                patch.object(assemble, "BOX_PIN_PARTIAL", partial), \
                patch.object(assemble, "BOX_PIN_DROP_UNREAD", True), \
                patch.object(assemble, "BOX_PIN_DROP_MIXED_ONLY", True), \
                patch.object(assemble, "BOX_PIN_DROP_CAP", False), \
                patch.object(assemble, "BOX_PIN_NEED_NAME", need_name):
            return assemble.build_doc(stage1, assigned, "synthetic")

    def test_stronger_complete_duplicate_keeps_own_point_and_only_own_net(self):
        for string_keys in (False, True):
            with self.subTest(string_keys=string_keys):
                doc = self.build([self.read("8", "VOUT", .47, .95),
                                  self.read("8", "GND", .96, .99)],
                                 string_keys=string_keys)
                self.assertEqual(doc["pins"]["U1"], {
                    "pin_8": {"pinname": "GND", "point": {"x": 212., "y": 120.}}})
                self.assertEqual(doc["__pin_index"]["ic"], {"1": "pin_8"})
                self.assertEqual(len(doc["nets"]), 1)
                net = next(iter(doc["nets"].values()))
                self.assertEqual(net["hyperGraph"], "(U1.8)")
                self.assertEqual(next(iter(net["edges"].values()))[0]["y"], 120.)

    def test_name_quality_also_matters_when_numbers_are_confident(self):
        doc = self.build([self.read("2", "R1", .999, .51),
                          self.read("2", "2", .998, .999)])
        self.assertEqual(doc["pins"]["U1"]["pin_2"]["pinname"], "2")

    def test_frame_rejected_incumbent_cannot_block_complete_same_number_contact(self):
        # A frame guard can reject the first contact's mixed number/marker
        # name. Another physical contact still has a valid complete read.
        for first in (self.read("8", "", .4, .4),
                      {"number": {"text": "8", "conf": .99}, "name": None}):
            first["name_rejected_by_frame"] = True
            for string_keys in (False, True):
                with self.subTest(first=first, string_keys=string_keys):
                    doc = self.build([first, self.read("8", "GND")], string_keys=string_keys)
                    self.assertEqual(doc["pins"]["U1"], {
                        "pin_8": {"pinname": "GND", "point": {"x": 212., "y": 120.}}})
                    self.assertEqual(doc["__pin_index"]["ic"], {"1": "pin_8"})
                    self.assertEqual(len(doc["nets"]), 1)
                    net = next(iter(doc["nets"].values()))
                    self.assertEqual(net["hyperGraph"], "(U1.8)")
                    self.assertEqual(next(iter(net["edges"].values()))[0]["y"], 120.)

    def test_color_complete_read_replaces_any_unnamed_incumbent(self):
        for source in ("red_green_transition", "red_stub_endpoint", "red_green_transition_uncertain"):
            with self.subTest(source=source):
                evidence = [self.evidence("uncertain", [212, 120]),
                            self.evidence("trusted", [212, 180], source=source)]
                doc = self.build([self.read("8", "", .4, .4), self.read("8", "GND")], evidence=evidence)
                self.assertEqual(doc["__pin_index"]["ic"], {"1": "pin_8"})
        self.assertEqual(doc["pins"]["U1"], {
            "pin_8": {"pinname": "GND", "point": {"x": 212., "y": 120.}}})
        self.assertEqual(doc["__pin_index"]["ic"], {"1": "pin_8"})
        self.assertEqual(len(doc["nets"]), 1)
        net = next(iter(doc["nets"].values()))
        self.assertEqual(net["hyperGraph"], "(U1.8)")
        self.assertEqual(next(iter(net["edges"].values()))[0]["y"], 120.)

    @staticmethod
    def evidence(trust, point, confirmed=True, source=None):
        return {"trust": trust, "point_img": point, "frame_confirmed": confirmed, "source": source}

    def test_unnamed_non_color_incumbent_keeps_previous_policy(self):
        for evidence in (None, [self.evidence("uncertain", [212, 120]),
                               self.evidence("trusted", [212, 180])]):
            with self.subTest(evidence=evidence):
                doc = self.build([self.read("8", "", .4, .4), self.read("8", "GND")], evidence=evidence)
                self.assertEqual(doc["pins"]["U1"], {})
                self.assertEqual(doc["__pin_index"], {})

    def test_invalid_or_unconfirmed_color_claim_cannot_replace_unnamed_incumbent(self):
        claims = (self.evidence("trusted", [212, 180], source="other"),
                  self.evidence("trusted", [212, 180], source=["red_green_transition"]),
                  self.evidence("trusted", [212, 180], False, source="red_green_transition"),
                  self.evidence("trusted", [212, 120], source="red_green_transition"),
                  self.evidence("uncertain", [212, 180], source="red_green_transition"))
        for claim in claims:
            with self.subTest(claim=claim):
                doc = self.build([self.read("8", ""), self.read("8", "GND")],
                                 evidence=[self.evidence("uncertain", [212, 120]), claim])
                self.assertEqual(doc["pins"]["U1"], {})

    def test_non_color_contact_ranks_keep_old_confidence_order(self):
        evidence = [self.evidence("trusted", [212, 120]),
                    self.evidence("rejected", [212, 180])]
        doc = self.build([self.read("8", "FIRST", .65, .65),
                          self.read("8", "SECOND", .99, .99)], evidence=evidence)
        self.assertEqual(doc["__pin_index"]["ic"], {"1": "pin_8"})

    def test_trusted_contact_beats_more_confident_text_contact_and_keeps_own_net(self):
        evidence = [self.evidence("rejected", [212, 120]),
                    self.evidence("trusted", [212, 180], source="red_green_transition")]
        doc = self.build([self.read("8", "FALSE", .999, .999),
                          self.read("8", "VDDREG", .65, .65)], evidence=evidence)
        self.assertEqual(doc["pins"]["U1"], {
            "pin_8": {"pinname": "VDDREG", "point": {"x": 212., "y": 120.}}})
        self.assertEqual(doc["__pin_index"]["ic"], {"1": "pin_8"})
        self.assertEqual(len(doc["nets"]), 1)
        net = next(iter(doc["nets"].values()))
        self.assertEqual(net["hyperGraph"], "(U1.8)")
        self.assertEqual(next(iter(net["edges"].values()))[0]["y"], 120.)

    def test_later_text_candidate_cannot_replace_trusted_contact_by_confidence(self):
        evidence = [self.evidence("trusted", [212, 120], source="red_stub_endpoint"),
                    self.evidence("uncertain", [212, 180])]
        doc = self.build([self.read("8", "VDDREG", .65, .65),
                          self.read("8", "FALSE", .999, .999)], evidence=evidence)
        self.assertEqual(doc["__pin_index"]["ic"], {"0": "pin_8"})

    def test_unconfirmed_missing_and_stale_contact_evidence_keep_quality_order(self):
        invalid_evidence = (
            [self.evidence("rejected", [212, 120]),
             self.evidence("trusted", [212, 180], False, source="red_green_transition")],
            [self.evidence("rejected", [212, 120]), None],
            [self.evidence("rejected", [212, 120]),
             self.evidence("trusted", [212, 120], source="red_green_transition")],
        )
        for evidence in invalid_evidence:
            with self.subTest(evidence=evidence):
                doc = self.build([self.read("8", "FIRST", .99, .99),
                                  self.read("8", "SECOND", .65, .65)], evidence=evidence)
                self.assertEqual(doc["__pin_index"]["ic"], {"0": "pin_8"})

    def test_contact_evidence_does_not_change_disabled_switches(self):
        evidence = [self.evidence("rejected", [212, 120]),
                    self.evidence("trusted", [212, 180], source="red_green_transition")]
        reads = [self.read("8", "FIRST", .99, .99), self.read("8", "SECOND", .65, .65)]
        for switches in ({"enabled": False}, {"need_name": False}, {"partial": False}):
            with self.subTest(switches=switches):
                self.assertEqual(self.build(reads, evidence=evidence, **switches),
                                 self.build(reads, **switches))

    def test_frozen_case8_vddreg_uses_own_contact_index_and_network(self):
        # Contact 14's old grayscale guess competes with the actual color-rule
        # VDDREG/5 contact measured at (1191,513). Keep the color contact's own
        # point and network; its OCR read comes from the frozen _assigned pool.
        points = [[982.9, 680.4], [1191.0, 513.0]]
        stage1 = {
            "image": {"w": 2229, "h": 1192},
            "components": [{"id": "c0000", "type": "box",
                            "bbox_img": [994.93, 453.68, 1172.31, 770.37],
                            "bbox_json": [994.93, 421.63, 1172.31, 738.32]}],
            "pins": {"c0000": points},
            "pin_evidence": {"c0000": [self.evidence("uncertain", points[0]),
                                        self.evidence("trusted", points[1], source="red_green_transition")]},
            # The actual unnamed contact has no network. The real right-side
            # contact is in net_7. A short marker here identifies that same net.
            "wire_detail": {"segments": [[[1191.0, 513.0], [1231.0, 513.0]]]},
            "nets": [{"id": "net_7", "pins": [["c0000", 1]], "seg_index": [0]}],
        }
        assigned = {"comps": {"c0000": {"designator": {"text": "U1"}, "pins": {
            "0": {"number": {"text": "5", "conf": .6476764834590687}, "name": None},
            "1": {"number": {"text": "5", "conf": .9662621787068312},
                  "name": {"text": "VDDREG", "conf": .6651882566996943}},
        }}}}
        with patch.object(assemble, "BOX_PIN_DUPLICATE_BEST", True), \
                patch.object(assemble, "BOX_PIN_PARTIAL", True), \
                patch.object(assemble, "BOX_PIN_DROP_UNREAD", True), \
                patch.object(assemble, "BOX_PIN_DROP_MIXED_ONLY", True), \
                patch.object(assemble, "BOX_PIN_DROP_CAP", False), \
                patch.object(assemble, "BOX_PIN_NEED_NAME", True):
            doc = assemble.build_doc(stage1, assigned, "8")
        self.assertEqual(doc["pins"]["U1"], {
            "pin_5": {"pinname": "VDDREG", "point": {"x": 1191., "y": 679.}}})
        self.assertEqual(doc["__pin_index"], {"c0000": {"1": "pin_5"}})
        self.assertEqual(len(doc["nets"]), 1)
        net = next(iter(doc["nets"].values()))
        self.assertEqual(net["hyperGraph"], "(U1.5)")
        self.assertEqual(next(iter(net["edges"].values()))[0], {"x": 1191., "y": 679.})

    def test_unknown_complete_read_cannot_promote_unnamed_incumbent(self):
        for invalid in (None, "bad", float("nan"), float("inf"), -1, 2):
            with self.subTest(conf=invalid):
                first = dict(self.read("8", ""), name_rejected_by_frame=True)
                doc = self.build([first, self.read("8", "GND", invalid)])
                self.assertEqual(doc["pins"]["U1"], {})
                self.assertEqual(doc["__pin_index"], {})

    def test_disabled_switch_keeps_unnamed_incumbent_policy(self):
        first = dict(self.read("8", ""), name_rejected_by_frame=True)
        doc = self.build([first, self.read("8", "GND")], enabled=False)
        self.assertEqual(doc["pins"]["U1"], {})
        self.assertEqual(doc["__pin_index"], {})

    def test_names_optional_preserves_frame_rejected_first_contact(self):
        raw = [{"num": "8", "nm": "", "quality": .99, "name_rejected_by_frame": True},
               {"num": "8", "nm": "GND", "quality": .99}]
        with patch.object(assemble, "BOX_PIN_NEED_NAME", False), \
                patch.object(assemble, "BOX_PIN_DUPLICATE_BEST", True):
            self.assertEqual(assemble._box_number_winners(raw), {8: 0})

    def test_valid_complete_successors_still_use_existing_quality_order(self):
        raw = [{"num": "8", "nm": "", "quality": None, "name_rejected_by_frame": True},
               {"num": "8", "nm": "GND", "quality": .5},
               {"num": "8", "nm": "VOUT", "quality": .9}]
        with patch.object(assemble, "BOX_PIN_NEED_NAME", True), \
                patch.object(assemble, "BOX_PIN_DUPLICATE_BEST", True):
            self.assertEqual(assemble._box_number_winners(raw), {8: 2})

    def test_unnamed_later_duplicate_cannot_replace_complete_read(self):
        doc = self.build([self.read("8", "GND", .4, .4), self.read("8", "")])
        self.assertEqual(doc["__pin_index"]["ic"], {"0": "pin_8"})

    def test_equal_unknown_and_invalid_confidence_preserve_first(self):
        for invalid in (None, "bad", float("nan"), float("inf"), -1, 2):
            with self.subTest(conf=invalid):
                doc = self.build([self.read("8", "GND", invalid),
                                  self.read("8", "VOUT")])
                self.assertEqual(doc["__pin_index"]["ic"], {"0": "pin_8"})
        doc = self.build([self.read("8", "GND"), self.read("8", "VOUT")])
        self.assertEqual(doc["__pin_index"]["ic"], {"0": "pin_8"})

    def test_disabled_switch_preserves_first_read(self):
        doc = self.build([self.read("8", "VOUT", .4, .4), self.read("8", "GND")],
                         enabled=False)
        self.assertEqual(doc["__pin_index"]["ic"], {"0": "pin_8"})

    def test_unique_numbers_are_unchanged(self):
        reads = [self.read("6", "VOUT", .4, .4), self.read("8", "GND")]
        self.assertEqual(self.build(reads), self.build(reads, enabled=False))

    def test_disabled_partial_number_policy_is_unchanged(self):
        reads = [self.read("8", "VOUT", .4, .4), self.read("8", "GND")]
        self.assertEqual(self.build(reads, partial=False),
                         self.build(reads, enabled=False, partial=False))


if __name__ == "__main__":
    unittest.main()
