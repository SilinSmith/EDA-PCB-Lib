"""Guard the evaluation boundary and allow legitimate duplicate-winner repair."""
import copy
import unittest

from tools.eval.pin_number_ocr_ab import audit_number_records, duplicate_number_audit


def numeral_record():
    return {
        "reads": {"pin_number": ["13", .97]},
        "pin_target": {"component_id": "b0", "pin_index": 0},
        "pin_number_ocr": {"selected_views": [
            {"sampling": {"model_input_sha256": "a" * 64}},
            {"sampling": {"model_input_sha256": "b" * 64}},
        ]},
    }


class AuditNumberRecordsTests(unittest.TestCase):
    def test_valid_number_only_observation(self):
        audit = audit_number_records([numeral_record()], .95)
        self.assertEqual(audit["records"], 1)
        self.assertEqual(len(audit["input_sha_traces"]), 2)

    def test_empty_disabled_reader_is_valid(self):
        self.assertEqual(audit_number_records([])["records"], 0)

    def test_role_leak_is_rejected(self):
        record = numeral_record()
        record["reads"]["pin_name"] = ["EN", .99]
        with self.assertRaisesRegex(AssertionError, "leaked"):
            audit_number_records([record])

    def test_minimum_confidence_is_not_bypassed(self):
        with self.assertRaisesRegex(AssertionError, "min_conf"):
            audit_number_records([numeral_record()], .99)

    def test_every_observation_needs_its_own_actual_input_hash(self):
        second = copy.deepcopy(numeral_record())
        second.pop("pin_number_ocr")
        with self.assertRaisesRegex(AssertionError, "actual model input SHA"):
            audit_number_records([numeral_record(), second])

    def test_non_numeric_read_is_rejected(self):
        record = numeral_record()
        record["reads"]["pin_number"] = ["I3", .99]
        with self.assertRaisesRegex(AssertionError, "digit sequence"):
            audit_number_records([record])

    def test_target_is_required(self):
        record = numeral_record()
        record.pop("pin_target")
        with self.assertRaisesRegex(AssertionError, "physical target"):
            audit_number_records([record])


class DuplicateNumberAuditTests(unittest.TestCase):
    def test_observed_digit_repair_may_legitimately_change_winner(self):
        stage = {"components": [{"id": "b0", "type": "box"}]}
        assigned = {
            "off": {"comps": {"b0": {"pins": {
                "0": {"number": {"text": "1"}}, "1": {"number": {"text": "1"}},
            }}}},
            "on": {"comps": {"b0": {"pins": {
                "0": {"number": {"text": "13"}}, "1": {"number": {"text": "1"}},
            }}}},
        }
        indexes = {"off": {"b0": {"0": "pin_1", "1": "pin_2"}},
                   "on": {"b0": {"0": "pin_13", "1": "pin_1"}}}
        checks = duplicate_number_audit(stage, assigned, indexes)
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["winners"], {"off": ["0"], "on": ["1"]})
        self.assertTrue(checks[0]["winner_changed"])


if __name__ == "__main__":
    unittest.main()
