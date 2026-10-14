"""A neighbouring symbol cannot own an IC numeral solely by centre distance."""
import copy
import importlib
import unittest
from unittest.mock import patch

assignment = importlib.import_module('stage2.assign')


def evidence(point, edge='top', anchor=None, monochrome=False):
    rec = dict(point_img=list(point), anchor_img=list(anchor or [130, 100]),
               edge=edge, trust='trusted', source='red_green_transition', frame_confirmed=True)
    if monochrome:
        rec.update(source='frame_normal_stroke', reason='frame_attached_normal_stroke',
                   frame_coverage=1., normal_ink_run=12., normal_ink_hits=12.,
                   first_ink_distance=0., text_overlap_fraction=0., branching_fraction=0.)
    return rec


class PinNumberRoleOwnerTests(unittest.TestCase):
    def setUp(self):
        self.comps = [dict(id='ic', type='box', bbox_img=[100, 100, 200, 200]),
                      dict(id='gnd', type='gnd', bbox_img=[85, 75, 103, 94])]
        self.pins = {'ic': [[130, 88]], 'gnd': [[103, 84]]}
        self.evidence = {'ic': [evidence(self.pins['ic'][0])]}
        self.digit = {'bbox': [113, 76, 123, 92], 'reads': {'pin_number': ('7', .993)}}

    def bind(self, texts=None):
        return assignment.assign(self.comps, self.pins, texts or [self.digit],
                                 pin_evidence=self.evidence)

    def test_top_number_uses_its_measured_ic_row_instead_of_closer_symbol(self):
        before = copy.deepcopy((self.comps, self.pins, self.evidence, self.digit))
        got = self.bind()['comps']
        self.assertEqual(got['ic']['pins'][0]['number']['text'], '7')
        self.assertIsNone(got['gnd']['pins'][0]['number'])
        self.assertIsNone(got['ic']['designator'])
        self.assertEqual((self.comps, self.pins, self.evidence, self.digit), before)

    def test_clean_current_monochrome_frame_evidence_also_qualifies(self):
        self.evidence['ic'] = [evidence(self.pins['ic'][0], monochrome=True)]
        self.assertEqual(self.bind()['comps']['ic']['pins'][0]['number']['text'], '7')

    def test_uncertain_stale_and_bare_trust_do_not_reclaim(self):
        changed = [dict(self.evidence['ic'][0], trust='uncertain'),
                   dict(self.evidence['ic'][0], point_img=[130, 89]),
                   dict(self.evidence['ic'][0], source='frame_normal_stroke')]
        for rec in changed:
            with self.subTest(rec=rec):
                self.evidence['ic'] = [rec]
                self.assertIsNone(self.bind()['comps']['ic']['pins'][0]['number'])

    def test_side_row_and_fully_external_rectangle_are_required(self):
        for rect in ([106, 50, 116, 66], [121, 97, 131, 109], [113, 110, 123, 126]):
            with self.subTest(rect=rect):
                self.digit['bbox'] = rect
                current = self.bind()
                with patch.object(assignment, 'PIN_NUMBER_ROLE_OWNER', False):
                    legacy = self.bind()
                self.assertEqual(current, legacy)

    def test_preexisting_ic_owner_is_not_transferred_between_ics(self):
        self.comps[1]['type'] = 'box'
        self.assertIsNone(self.bind()['comps']['ic']['pins'][0]['number'])

    def test_explicit_validated_target_remains_authoritative(self):
        point = self.pins['gnd'][0]
        self.evidence['gnd'] = [evidence(point, edge='right', anchor=[103, 84])]
        self.digit['pin_target'] = dict(component_id='gnd', pin_index=0, edge='right',
                                       point_img=point, anchor_img=[103, 84])
        self.assertIsNone(self.bind()['comps']['ic']['pins'][0]['number'])

    def test_invalid_explicit_target_is_not_rescued_by_owner_heuristic(self):
        self.digit['pin_target'] = dict(component_id='missing', pin_index=0, edge='top')
        got = self.bind()['comps']
        self.assertIsNone(got['ic']['pins'][0]['number'])
        self.assertIsNone(got['gnd']['pins'][0]['number'])

    def test_switch_off_and_non_numeric_text_preserve_original_assignment(self):
        with patch.object(assignment, 'PIN_NUMBER_ROLE_OWNER', False):
            got = self.bind()['comps']
        self.assertIsNone(got['ic']['pins'][0]['number'])
        self.assertEqual(got['gnd']['pins'][0]['number']['text'], '7')
        for number in ('0', 'U7', '1000', ''):
            self.digit['reads'] = {'pin_number': (number, .993)}
            self.assertIsNone(self.bind()['comps']['ic']['pins'][0]['number'])

    def test_reclaim_changes_only_number_role_of_a_multi_role_record(self):
        self.digit['reads'].update(value=('7V', .94), pin_name=('LABEL', .91))
        current = self.bind()
        with patch.object(assignment, 'PIN_NUMBER_ROLE_OWNER', False):
            legacy = self.bind()
        for result in (current, legacy):
            for comp in result['comps'].values():
                for pin in comp['pins'].values():
                    pin.pop('number')
        self.assertEqual(current, legacy)

    def test_complete_independent_gray_digit_can_beat_shifted_high_confidence_read(self):
        point, anchor = [212, 160], [200, 160]
        self.pins = {'ic': [point]}
        self.evidence = {'ic': [evidence(point, 'right', anchor, monochrome=True)]}
        shifted = {'bbox': [213, 148, 219, 160], 'reads': {'pin_number': ('1', .98)}}
        larger = {'bbox': [211, 146, 218, 158], 'reads': {'pin_number': ('4', .52),
                    'designator': ('4', .98), 'value': ('4', .80), 'net_label': ('4', .87)}}
        clean = {'bbox': [211, 146, 218, 158], 'geometry_bbox': [211, 146, 218, 158],
                 'number_only': True, 'source': 'mono_pin_number_ppocr',
                 'reads': {'pin_number': ('4', .94)},
                 'pin_target': dict(component_id='ic', pin_index=0, edge='right',
                                    point_img=point, anchor_img=anchor),
                 'mono_pin_ocr': dict(role='pin_number', backend='ppocr_v4_direct', accepted=True,
                     held_reason=None, flags=[], selected_views=[
                         dict(padding=2, text='4', confidence=.94),
                         dict(padding=4, text='4', confidence=.96)])}
        base = self.bind([shifted, larger])['comps']['ic']['pins'][0]
        self.assertEqual(base['number']['text'], '1')
        changed = self.bind([shifted, larger, clean])['comps']['ic']['pins'][0]
        self.assertEqual(changed['number']['text'], '4')
        self.assertTrue(changed['number_independently_read'])
        held = copy.deepcopy(clean)
        held['mono_pin_ocr']['flags'] = ['glyph_integrity_unresolved']
        self.assertEqual(self.bind([shifted, larger, held])['comps']['ic']['pins'][0]['number']['text'], '1')


if __name__ == '__main__':
    unittest.main()
