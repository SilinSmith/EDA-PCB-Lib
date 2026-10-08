"""Same-box OCR disagreement must not become a global majority prior."""
import importlib
import unittest
from unittest.mock import patch

assign = importlib.import_module("stage2.assign")


class PinNumberConsensusTests(unittest.TestCase):
    def test_uncertain_glyph_uses_stronger_same_box_majority(self):
        reads = {"designator": ("87", 0.566), "value": ("87", 0.486),
                 "pin_name": ("97", 0.408), "pin_number": ("67", 0.524),
                 "net_label": ("87", 0.484)}
        with patch.object(assign, "PINNUM_CONSENSUS", True):
            self.assertEqual(assign._best_number_read(reads)[1:3], ("87", 0.566))
        with patch.object(assign, "PINNUM_CONSENSUS", False):
            self.assertEqual(assign._best_number_read(reads),
                             assign._best_read(reads, "pin_number", assign.fit_pinnum))

    def test_confident_target_and_lower_confidence_majority_remain_exact(self):
        for target_conf, other_conf in ((0.8, 0.9), (0.59, 0.58)):
            with self.subTest(target_conf=target_conf):
                reads = {role: ("87", other_conf) for role in
                         ("designator", "value", "pin_name", "net_label")}
                reads["pin_number"] = ("67", target_conf)
                with patch.object(assign, "PINNUM_CONSENSUS", True):
                    self.assertEqual(assign._best_number_read(reads),
                                     assign._best_read(reads, "pin_number", assign.fit_pinnum))

    def test_minority_and_digit_clipping_cannot_override(self):
        for others in (("87", "87", "97", "97"), ("7", "7", "7", "7")):
            with self.subTest(others=others):
                reads = dict(zip(("designator", "value", "pin_name", "net_label"),
                                 [(text, 0.6) for text in others]))
                reads["pin_number"] = ("67", 0.55)
                with patch.object(assign, "PINNUM_CONSENSUS", True):
                    self.assertEqual(assign._best_number_read(reads),
                                     assign._best_read(reads, "pin_number", assign.fit_pinnum))

    def test_unchanged_read_keeps_original_confidence_and_score(self):
        reads = {role: ("67", conf) for role, conf in
                 (("pin_number", 0.55), ("designator", 0.60),
                  ("value", 0.59), ("pin_name", 0.58), ("net_label", 0.57))}
        with patch.object(assign, "PINNUM_CONSENSUS", True):
            self.assertEqual(assign._best_number_read(reads),
                             assign._best_read(reads, "pin_number", assign.fit_pinnum))

    def test_assignment_path_uses_consensus_only_for_pin_number(self):
        components = [{"id": "ic", "type": "box", "bbox_img": [100, 100, 200, 200]}]
        reads = {"designator": ("87", 0.566), "value": ("87", 0.486),
                 "pin_name": ("97", 0.408), "pin_number": ("67", 0.524),
                 "net_label": ("87", 0.484)}
        texts = [{"bbox": [80, 146, 96, 154], "reads": reads}]
        for enabled, expected in ((False, "67"), (True, "87")):
            with self.subTest(enabled=enabled), patch.object(assign, "PINNUM_CONSENSUS", enabled):
                result = assign.assign(components, {"ic": [[100, 150]]}, texts)
                self.assertEqual(result["comps"]["ic"]["pins"][0]["number"]["text"], expected)


if __name__ == "__main__":
    unittest.main()
