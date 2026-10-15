"""Power labels must survive assembly with literal evidence and safe ownership."""
import copy
import unittest
from unittest.mock import patch

from stage2 import assemble
from stage2.power_values import assign_power_values, published_power_value


class PowerValuePublicationTests(unittest.TestCase):
    def test_observed_value_and_literal_spelling_survive(self):
        for text in ('+5V', '3.3V', '3V3', '3v3', 'VCC', 'VDDA', 'VBAT', 'VDD_1.9V'):
            self.assertEqual(published_power_value('v', {'partname': {'text': text, 'conf': .9}}), text)
        self.assertEqual(published_power_value('gnd', {}), 'GND')
        for text in ('AGND', 'DGND', 'VSSA'):
            self.assertEqual(published_power_value('gnd', {'partname': {'text': text, 'conf': .9}}), text)

    def test_no_default_voltage_low_confidence_or_nonrail_text(self):
        self.assertIsNone(published_power_value('v', {}))
        for text, conf in (('3.3V', .79), ('C3', .99), ('MCU', .99), ('VCC', float('nan'))):
            self.assertIsNone(published_power_value('v', {'partname': {'text': text, 'conf': conf}}))
        self.assertIsNone(published_power_value('box', {'partname': {'text': 'VCC', 'conf': .99}}))

    def test_conflicting_observations_remain_unknown(self):
        info = {'partname': {'text': '3V3', 'conf': .99},
                'power_value': {'text': '+5V', 'conf': .9}}
        self.assertIsNone(published_power_value('v', info))
        self.assertIsNone(published_power_value('gnd', {'power_value_conflict': True}))

    def test_only_owned_label_roles_are_consumed(self):
        comps = [{'id': 's', 'type': 'v'}, {'id': 'ic', 'type': 'box'}]
        base = {'bbox': [10, 10, 40, 20], 'd': 3,
                'reads': {'net_label': ('+5V', .95)}}
        for extra in ({'designator_only': True}, {'pin_target': {'component_id': 's'}},
                      {'color_crop': {'component_id': 'ic'}}, {'wide': True}, {'d': 41}):
            assigned = {'s': {}, 'ic': {}}
            assign_power_values(comps, {'s': [dict(base, **extra)]}, assigned)
            self.assertIsNone(published_power_value('v', assigned['s']))
        assigned = {'s': {}, 'ic': {}}
        assign_power_values(comps, {'ic': [base]}, assigned)
        self.assertIsNone(published_power_value('v', assigned['s']))
        assign_power_values(comps, {'s': [base]}, assigned)
        self.assertEqual(published_power_value('v', assigned['s']), '+5V')
        self.assertEqual(assigned['s']['power_value']['box'], base['bbox'])

    def test_power_publication_changes_only_value_fields(self):
        stage = {'image': {'w': 200, 'h': 100}, 'components': [
            {'id': 's', 'type': 'v', 'bbox_json': [40, 30, 50, 40]},
            {'id': 'u', 'type': 'box', 'bbox_json': [100, 10, 150, 50]}],
            'pins': {'s': [[45, 50]], 'u': []}, 'nets': [],
            'wire_detail': {'segments': []}}
        assigned = {'comps': {'s': {'partname': {'text': '3V3', 'conf': .99}},
                              'u': {'designator': {'canon': 'U1'}, 'partname': {'text': 'CHIP'}}}}
        old_stage, old_assigned = copy.deepcopy(stage), copy.deepcopy(assigned)
        with patch.object(assemble, 'published_power_value', return_value=None):
            before = assemble.build_doc(stage, assigned, 'synthetic')
        after = assemble.build_doc(stage, assigned, 'synthetic')
        self.assertEqual(stage, old_stage)
        self.assertEqual(assigned, old_assigned)
        key = next(k for k, c in after['components'].items() if c['type'] == 'v')
        self.assertEqual(after['components'][key]['value'], '3V3')
        before['components'][key]['value'] = '3V3'
        self.assertEqual(after, before)


if __name__ == '__main__':
    unittest.main()
