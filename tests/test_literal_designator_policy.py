"""Literal printed suffixes and caption observations must keep their scope."""
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from stage2 import run

assignment = importlib.import_module('stage2.assign')


class LiteralIdentityTests(unittest.TestCase):
    @staticmethod
    def fresh_caption(value, confidence=.91):
        box = [100, 90, 130, 98]
        return {'bbox': box, 'designator_only': True, 'designator_target': 'ic',
                'source': 'ic_caption_ppocr', 'reads': {'designator': (value, confidence)},
                'designator_ocr': {'accepted': True, 'unchanged_source_pixels': True,
                    'owner_candidates': ['ic'], 'bbox': box,
                    'raw_reads': [{'padding': p, 'text': value, 'confidence': confidence}
                                  for p in (2, 4)]}}

    def test_observed_separators_and_suffixes_survive(self):
        for value in ('U_A', 'U_B', 'U12_A', 'R14', 'R104'):
            ctype = 'r' if value.startswith('R') else 'box'
            self.assertEqual(assignment.canon_designator(value, ctype), (value, 1.0))

    def test_separator_does_not_enable_prefix_or_spelling_repair(self):
        for value in ('P_31', 'V_REF', 'U_', '_A', 'U__A', 'U_ A', 'U_A!', 'U_A_B'):
            self.assertEqual(assignment.canon_designator(value, 'box'), ('', 0.))
        # Preserve established numeric correction, without inserting '_'.
        self.assertEqual(assignment.canon_designator('Rl4', 'r'), ('R14', .9))
        self.assertEqual(assignment.canon_designator('UD', 'box'), ('U0', .9))
        self.assertNotEqual(assignment.canon_designator('UB', 'box')[0], 'U_B')
        for value, expected in (('R1O', 'R10'), ('R1B', 'R18'), ('R1S', 'R15')):
            self.assertEqual(assignment.canon_designator(value, 'r'), (expected, .9))
        for value in ('U21A', 'U21B', 'U21O'):
            self.assertEqual(assignment.canon_designator(value, 'box', literal=True), (value, 1.))

    def test_caption_cannot_be_stolen_or_used_as_other_text_role(self):
        components = [{'id': 'left', 'type': 'box', 'bbox_img': [20, 30, 100, 100]},
                      {'id': 'right', 'type': 'box', 'bbox_img': [105, 30, 200, 100]}]
        caption = {'bbox': [90, 15, 118, 25], 'designator_only': True,
                   'designator_target': 'right',
                   'reads': {'designator': ('U_A', .99), 'partname': ('FAKE_MODEL', .99),
                             'pin_number': ('7', .99)}}
        result = assignment.assign(components, {}, [caption], global_match=False)
        self.assertIsNone(result['comps']['left']['designator'])
        self.assertEqual(result['comps']['right']['designator']['canon'], 'U_A')
        self.assertIsNone(result['comps']['right']['partname'])
        self.assertFalse(result['comps']['right']['pins'])
        caption['designator_target'] = 'missing'
        result = assignment.assign(components, {}, [caption], global_match=False)
        self.assertTrue(all(c['designator'] is None for c in result['comps'].values()))

    def test_printed_unit_identity_competes_with_literal_digits_and_beats_letter_fallback(self):
        component = {'id': 'ic', 'type': 'box', 'bbox_img': [100, 100, 200, 200]}
        reads = [{'bbox': [145, 145, 155, 155], 'reads': {'designator': ('UD', .99)}},
                 {'bbox': [100, 90, 130, 98], 'reads': {'designator': ('U_A', .99)}}]
        result = assignment.assign([component], {}, reads, global_match=False)
        self.assertEqual(result['comps']['ic']['designator']['canon'], 'U_A')
        self.assertEqual([s['canon'] for s in result['desig_slots']['ic']], ['U_A'])

    def test_consensus_replaces_same_glyph_old_channel_even_with_high_confidence_borrowing(self):
        component = {'id': 'ic', 'type': 'box', 'bbox_img': [100, 100, 200, 200]}
        reads = [{'bbox': [100, 90, 130, 98], 'reads': {'designator': ('U3', .999),
                  'pin_name': ('U3', .999), 'net_label': ('U3', .999)}},
                 self.fresh_caption('U_B')]
        result = assignment.assign([component], {}, reads, global_match=False)
        self.assertEqual(result['comps']['ic']['designator']['canon'], 'U_B')
        self.assertEqual([s['canon'] for s in result['desig_slots']['ic']], ['U_B'])
        # Incomplete or weak views cannot suppress the old observation.
        for fresh in (self.fresh_caption('U_B', .89), self.fresh_caption('U_B')):
            if fresh['reads']['designator'][1] >= .9:
                fresh['designator_ocr']['raw_reads'][1]['text'] = 'U_A'
            result = assignment.assign([component], {}, [reads[0], fresh], global_match=False)
            self.assertEqual(result['comps']['ic']['designator']['canon'], 'U3')

    def test_conflicting_fresh_observations_of_one_caption_are_withheld(self):
        component = {'id': 'ic', 'type': 'box', 'bbox_img': [100, 100, 200, 200]}
        result = assignment.assign([component], {},
            [self.fresh_caption('U12_A'), self.fresh_caption('U12_B', .99)], global_match=False)
        self.assertIsNone(result['comps']['ic']['designator'])
        self.assertIsNone(result['comps']['ic']['partname'])
        self.assertFalse(result['net_labels'])

    def test_same_physical_caption_cannot_create_two_ic_identities(self):
        components = [{'id': 'ic', 'type': 'box', 'bbox_img': [100, 100, 200, 200]},
                      {'id': 'next', 'type': 'box', 'bbox_img': [210, 100, 300, 200]}]
        old = {'bbox': [100, 90, 130, 98], 'reads': {'designator': ('U3', .99)}}
        result = assignment.assign(components, {}, [old, self.fresh_caption('U_A')], global_match=False)
        self.assertEqual(result['comps']['ic']['designator']['canon'], 'U_A')
        self.assertIsNone(result['comps']['next']['designator'])

    def test_literal_suffix_is_only_used_by_a_qualified_caption_observation(self):
        component = {'id': 'ic', 'type': 'box', 'bbox_img': [100, 100, 200, 200]}
        fresh = self.fresh_caption('U21B')
        result = assignment.assign([component], {}, [fresh], global_match=False)
        self.assertEqual(result['comps']['ic']['designator']['canon'], 'U21B')
        ordinary = {'bbox': fresh['bbox'], 'reads': {'designator': ('U21B', .99)}}
        result = assignment.assign([component], {}, [ordinary], global_match=False)
        self.assertEqual(result['comps']['ic']['designator']['canon'], 'U218')

    def test_independent_caption_preserves_ic_prefix_and_printed_case(self):
        component = {'id': 'ic', 'type': 'box', 'bbox_img': [100, 100, 200, 200]}
        for value in ('IC2', 'ic07', 'u1', 'u12_A'):
            with self.subTest(value=value):
                result = assignment.assign([component], {}, [self.fresh_caption(value)], global_match=False)
                self.assertEqual(result['comps']['ic']['designator']['canon'], value)
        self.assertEqual(assignment.canon_designator('IC2', 'box'), ('', 0.))
        for value in ('UD', 'U_A!', 'J1', 'IC2!'):
            self.assertEqual(assignment.canon_designator(value, 'box', literal=True), ('', 0.))
        internal={'bbox':[140,140,165,153],'reads':{'designator':('U0O',.999)}}
        result=assignment.assign([component],{},[internal,self.fresh_caption('IC2')],global_match=False)
        self.assertEqual(result['comps']['ic']['designator']['canon'],'IC2')

    def test_wrong_prefix_consensus_only_rejects_duplicated_caption_channel(self):
        components = [{'id': 'ic', 'type': 'box', 'bbox_img': [100, 100, 200, 200]},
                      {'id': 'next', 'type': 'box', 'bbox_img': [300, 100, 400, 200]}]
        old = {'bbox': [100, 90, 130, 98], 'reads': {'designator': ('U1', .999)}}
        real = {'bbox': [300, 90, 330, 98], 'reads': {'designator': ('U1', .999)}}
        fresh = self.fresh_caption('J1')
        fresh['designator_ocr'].update(conflicting_existing_identity=True, rejected_designator_prefix=True)
        result = assignment.assign(components, {}, [old, real, fresh], global_match=False)
        self.assertIsNone(result['comps']['ic']['designator'])
        self.assertEqual(result['comps']['next']['designator']['canon'], 'U1')
        self.assertFalse(result['comps']['ic']['pins'])
        self.assertIsNone(result['comps']['ic']['partname'])
        # A stale flag or a disagreement cannot invalidate the old channel.
        for changed in ('conflicting_existing_identity', 'raw_reads'):
            invalid = self.fresh_caption('J1')
            invalid['designator_ocr'].update(conflicting_existing_identity=True, rejected_designator_prefix=True)
            if changed == 'raw_reads':
                invalid['designator_ocr']['raw_reads'][1]['text'] = 'J2'
            else:
                invalid['designator_ocr'][changed] = False
            result = assignment.assign([components[0]], {}, [old, invalid], global_match=False)
            self.assertEqual(result['comps']['ic']['designator']['canon'], 'U1')

    def test_ambiguous_wrong_caption_is_rejected_for_all_overlapping_owners(self):
        components=[{'id':'ic','type':'box','bbox_img':[100,100,200,200]},
                    {'id':'near','type':'box','bbox_img':[105,100,205,200]}]
        old={'bbox':[100,90,130,98],'reads':{'designator':('U1',.999)}}
        fresh=self.fresh_caption('J1')
        fresh['designator_ocr'].update(owner_candidates=['ic','near'],rejection_only=True,
            rejected_designator_prefix=True,conflicting_existing_identity=True)
        result=assignment.assign(components,{},[old,fresh],global_match=False)
        self.assertTrue(all(c['designator'] is None for c in result['comps'].values()))

    def test_shifted_duplicate_ocr_rectangle_cannot_reintroduce_rejected_identity(self):
        component={'id':'ic','type':'box','bbox_img':[100,100,200,200]}
        fresh=self.fresh_caption('J1')
        fresh['designator_ocr'].update(conflicting_existing_identity=True,rejected_designator_prefix=True)
        old={'bbox':[101,91,131,99],'reads':{'designator':('U1',.999)}}
        result=assignment.assign([component],{},[old,fresh],global_match=False)
        self.assertIsNone(result['comps']['ic']['designator'])


class CaptionPipelineTests(unittest.TestCase):
    def pipeline(self, *, mode='auto', roles=None, do_ocr=True, min_conf=0):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            doc = {'image': {'path': 'synthetic.png', 'w': 200, 'h': 200},
                   'components': [{'id': 'ic', 'type': 'box',
                       'bbox_img': [20, 30, 180, 180], 'bbox_json': [20, 20, 180, 170]}],
                   'pins': {}, 'pin_evidence': {}, 'texts': [[20, 10, 48, 22]], 'nets': []}
            (base / 's.stage1.json').write_text(json.dumps(doc), encoding='utf8')
            initial = [{'bbox': [20, 10, 48, 22], 'reads': {'designator': ('U', .99)}}]
            extra = [{'bbox': [20, 10, 48, 22], 'designator_only': True,
                      'designator_target': 'ic', 'reads': {'designator': ('U_A', .96)}}]
            engine = SimpleNamespace(read=Mock(return_value=initial))
            with patch.multiple(run, PINCOL_WIN_TEXTS=False, COLOR_NUMBER_CROP=False,
                PIN_TEXT_CROP=False, PIN_LANE_CROP=False, PIN_NUMBER_REPAIR=False,
                PIN_NAME_OCR='off', MONO_PIN_OCR='off', DESIGNATOR_OCR=mode), \
                patch.object(run, 'imread_unicode', return_value=np.full((200, 200, 3), 255, np.uint8)), \
                patch('stage2.designator_reading.read_designator_captions',
                      return_value=(extra, [{'accepted': True}])) as reader:
                run.process('s', base, base / 'out', engine, roles=roles,
                            do_ocr=do_ocr, min_conf=min_conf, texts='stage1')
            final = json.loads((base / 'out' / 's.json').read_text(encoding='utf8'))
            assigned = json.loads((base / 'out' / '_assigned' / 's.json').read_text(encoding='utf8'))
            return final, assigned, reader, engine

    def test_process_uses_literal_identity_and_keeps_audit_out_of_official_json(self):
        final, assigned, reader, engine = self.pipeline()
        self.assertIn('U_A', final['components'])
        self.assertEqual(assigned['designator_ocr_audit'], [{'accepted': True}])
        self.assertNotIn('designator_ocr_audit', final)
        reader.assert_called_once()
        engine.read.assert_called_once()

    def test_no_ocr_off_and_unrequested_roles_skip_caption_provider(self):
        for kwargs in ({'do_ocr': False}, {'mode': 'off'}, {'roles': ['pin_name']}, {'roles': []}):
            _, assigned, reader, _ = self.pipeline(**kwargs)
            reader.assert_not_called()
            self.assertEqual(assigned['designator_ocr_audit'], [])

    def test_global_minimum_confidence_applies_to_caption_observations(self):
        final, _, reader, _ = self.pipeline(min_conf=.98)
        self.assertNotIn('U_A', final['components'])
        reader.assert_called_once()


if __name__ == '__main__':
    unittest.main()
