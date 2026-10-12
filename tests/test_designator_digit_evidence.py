"""Observed local designation digits must outrank internal letter-only labels."""
import importlib
import unittest

assignment = importlib.import_module("stage2.assign")
arbitration = importlib.import_module("stage3.arbitrate")


def text(box, value, confidence):
    return {"bbox": box, "reads": {"designator": (value, confidence)}}


class DesignatorDigitEvidenceTests(unittest.TestCase):
    body = {"id": "ic", "type": "box", "bbox_img": [100, 100, 200, 200]}

    def assign(self, texts, components=None):
        return assignment.assign(components or [self.body], {}, texts, global_match=False)

    def test_local_u5_beats_internal_ud_without_inventing_a_digit(self):
        result = self.assign([text([145, 145, 155, 155], "UD", .923),
                              text([100, 90, 110, 97], "U5", .999)])
        chosen = result["comps"]["ic"]["designator"]
        self.assertEqual((chosen["text"], chosen["canon"]), ("U5", "U5"))
        self.assertEqual({a["canon"] for a in result["desig_slots"]["ic"]}, {"U5"})

    def test_distant_or_negative_score_digit_does_not_remove_fallback(self):
        for digit_box, confidence in (([100, 76, 110, 84], .999),
                                      ([100, 81, 110, 89], .0)):
            with self.subTest(digit_box=digit_box):
                result = self.assign([text([145, 145, 155, 155], "UD", .923),
                                      text(digit_box, "U5", confidence)])
                self.assertEqual(result["comps"]["ic"]["designator"]["canon"], "U0")
                self.assertIn("U0", {a["canon"] for a in result["desig_slots"]["ic"]})

    def test_without_numeric_evidence_letter_fallback_is_preserved(self):
        result = self.assign([text([145, 145, 155, 155], "UD", .923)])
        self.assertEqual(result["comps"]["ic"]["designator"]["canon"], "U0")

    def test_numeric_candidates_keep_existing_geometric_order(self):
        result = self.assign([text([145, 145, 155, 155], "UD", .999),
                              text([100, 90, 110, 97], "U5", .999),
                              text([130, 130, 140, 140], "U6", .8)])
        self.assertEqual(result["comps"]["ic"]["designator"]["canon"], "U6")
        self.assertEqual({a["canon"] for a in result["desig_slots"]["ic"]}, {"U5", "U6"})

    def test_stage3_off_keeps_true_owner_and_drops_lower_confidence_collision(self):
        other = {"id": "reg", "type": "box", "bbox_img": [300, 300, 400, 400]}
        components = [self.body, other]
        result = self.assign([text([145, 145, 155, 155], "UD", .923),
                              text([100, 90, 110, 97], "U5", .999),
                              text([300, 290, 310, 297], "U5", .84)], components)
        hy = arbitration.build("observed-digit", result, components)
        arbitration.arbitrate(hy, mode="off", log=lambda *a: None)
        arbitration.apply(result, hy)
        arbitration.dedup(result, log=lambda *a: None)
        self.assertEqual(result["comps"]["ic"]["designator"]["canon"], "U5")
        self.assertIsNone(result["comps"]["reg"]["designator"])


def setUpModule():
    """§5.28：本模块测的是队友分支**未并入**的位号数字证据（见 tests/_unmerged_gate.py）。"""
    import _unmerged_gate
    _unmerged_gate.gate([("stage2.assign", ("PIN_DIGIT_ENTITY", "_digit_entities"))])


if __name__ == "__main__":
    unittest.main()
