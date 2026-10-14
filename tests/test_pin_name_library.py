"""Pin-name correction must preserve identity, evidence, and legal custom names."""
import copy
import unittest

from stage2.assemble import build_doc
from stage2.pin_name_library import normalize_assigned_pin_names, propose_names


class PinNameLibraryTests(unittest.TestCase):
    def fixture(self, names, *, ctype='box'):
        points = [[90, 130 + 25*i] for i in range(len(names))]
        component = dict(id='ic', type=ctype, bbox_img=[100, 100, 300, 100+25*len(names)],
                         bbox_json=[100, 100, 300, 300])
        doc = dict(image=dict(height=1000), components=[component], pins={'ic': points},
                   pin_evidence={'ic': [dict(point_img=p, anchor_img=[100, p[1]], edge='left',
                        trust='trusted', source='red_green_transition', frame_confirmed=True) for p in points]},
                   nets=[], wire_detail={})
        pins = {i: dict(name=dict(text=text, conf=.96, box=[110, points[i][1]-5, 165, points[i][1]+5]),
                       number=dict(text=str(i+1), conf=.97, box=[70, points[i][1]-8, 82, points[i][1]-1]))
                for i, text in enumerate(names)}
        assigned = dict(comps={'ic': dict(designator=dict(text='U1', conf=.99), pins=pins)}, net_labels=[])
        return doc, assigned

    def visual(self, doc, assigned, text, *, index=0, flags=None, held='unresolved_complete_word_character_ambiguity'):
        return dict(source='mono_pin_name_ppocr', name_only=True, reads={},
            bbox=assigned['comps']['ic']['pins'][index]['name']['box'],
            pin_target=dict(component_id='ic', pin_index=index, edge='left',
                            point_img=doc['pins']['ic'][index], anchor_img=[100, doc['pins']['ic'][index][1]]),
            mono_pin_ocr=dict(backend='ppocr_v4_direct', accepted=False, flags=flags or [], held_reason=held,
                selected_views=[dict(text=text, confidence=.95, padding=p, rotation=0,
                    sampling=dict(model_input_sha256='model-'+str(p), original_gray_input_sha256='gray-'+str(p)))
                    for p in (2, 4)]))

    def corrected(self, names, reads=None, **kwargs):
        doc, assigned = self.fixture(names, **kwargs)
        return normalize_assigned_pin_names(doc, assigned, reads or [])

    def text(self, result, index=0):
        return result['comps']['ic']['pins'][index]['name']['text']

    def test_gpio_stem_correction_preserves_the_entire_observed_numeric_suffix(self):
        for original, expected in [('GPI016', 'GPIO16'), ('GPI010', 'GPIO10'), ('GPI05', 'GPIO5')]:
            with self.subTest(original=original):
                self.assertEqual(self.text(self.corrected([original, 'GPIO2', 'GPIO13'])), expected)

    def test_gpio_zero_requires_observed_gpio_context(self):
        self.assertEqual(self.text(self.corrected(['GPIOO', 'GPIO2', 'GPIO13'])), 'GPIO0')
        self.assertEqual(self.text(self.corrected(['GPIOO', 'GND', 'VCC'])), 'GPIOO')
        self.assertEqual(self.text(self.corrected(['GPIOO', 'GPIO2', 'GPIO2'])), 'GPIOO')

    def test_numbered_and_custom_originals_stay_intact(self):
        names = ['GPIO10', 'GPIO16', 'GPIO100', 'gpio0', 'TXD', 'RXD7', 'GPIO0_EN',
                 'USER_OUTPUT', 'STDB+', 'GND', 'PC6/RESET', 'PB8', 'IO', 'DO', 'OSCO', 'OSCI']
        result = self.corrected(names)
        self.assertEqual([self.text(result, i) for i in range(len(names))], names)

    def test_uart_output_spelling_is_legal_without_clean_numeric_confirmation(self):
        self.assertEqual(self.text(self.corrected(['RXDO', 'GPIO2', 'GPIO13'])), 'RXDO')
        doc, assigned = self.fixture(['RXDO', 'GND', 'VCC'])
        read = self.visual(doc, assigned, 'RXD0')
        self.assertEqual(self.text(normalize_assigned_pin_names(doc, assigned, [read])), 'RXDO')

    def test_actual_dual_uart_reads_can_resolve_a_character_only_replacement_hold(self):
        doc, assigned = self.fixture(['RXDO', 'TXDO', 'GPIO2', 'GPIO13'])
        reads = [self.visual(doc, assigned, 'RXD0'), self.visual(doc, assigned, 'TXD0', index=1)]
        before_reads, before_assigned = copy.deepcopy(reads), copy.deepcopy(assigned)
        result = normalize_assigned_pin_names(doc, assigned, reads)
        self.assertEqual([self.text(result, i) for i in (0, 1)], ['RXD0', 'TXD0'])
        self.assertEqual(reads, before_reads)
        self.assertEqual(assigned, before_assigned)
        pin = result['comps']['ic']['pins'][0]
        self.assertEqual(pin['number'], before_assigned['comps']['ic']['pins'][0]['number'])
        self.assertEqual(pin['number_selection_name'], before_assigned['comps']['ic']['pins'][0]['name'])
        self.assertEqual(pin['name']['conf'], .96)
        self.assertEqual(pin['name']['raw_ocr_text'], 'RXDO')
        self.assertTrue(pin['name_normalization']['accepted'])

    def test_two_crops_are_correlated_visual_support_not_a_synthetic_confidence_boost(self):
        doc, assigned = self.fixture(['RXDO', 'GPIO2', 'GPIO13'])
        for change in ('disagree', 'low_confidence', 'one_view', 'same_tensor', 'same_padding', 'different_rotation', 'missing_sampling'):
            with self.subTest(change=change):
                read = self.visual(doc, assigned, 'RXD0')
                views = read['mono_pin_ocr']['selected_views']
                if change == 'disagree': views[1]['text'] = 'RXDO'
                elif change == 'low_confidence': views[1]['confidence'] = .89
                elif change == 'one_view': views.pop()
                elif change == 'same_tensor': views[1]['sampling'] = copy.deepcopy(views[0]['sampling'])
                elif change == 'same_padding': views[1]['padding'] = views[0]['padding']
                elif change == 'different_rotation': views[1]['rotation'] = 90
                elif change == 'missing_sampling': views[1]['sampling'] = {}
                self.assertEqual(self.text(normalize_assigned_pin_names(doc, assigned, [read])), 'RXDO')

    def test_incomplete_contaminated_or_rejected_partner_crops_cannot_authorize_a_repair(self):
        doc, assigned = self.fixture(['RXDO', 'GPIO2', 'GPIO13'])
        for flags, reason in [(['glyph_integrity_unresolved'], None), (['multiple_pin_rows'], None),
                              ([], 'incomplete_name_replacement'), ([], 'unresolved_dirty_partner_number_geometry')]:
            with self.subTest(flags=flags, reason=reason):
                read = self.visual(doc, assigned, 'RXD0', flags=flags, held=reason)
                self.assertEqual(self.text(normalize_assigned_pin_names(doc, assigned, [read])), 'RXDO')

    def test_other_pin_and_stale_target_cannot_confirm_a_name(self):
        doc, assigned = self.fixture(['RXDO', 'GPIO2', 'GPIO13'])
        for change in ('other_pin', 'other_component', 'stale_point', 'other_box'):
            with self.subTest(change=change):
                read = self.visual(doc, assigned, 'RXD0')
                if change == 'other_pin': read['pin_target']['pin_index'] = 1
                elif change == 'other_component': read['pin_target']['component_id'] = 'other'
                elif change == 'stale_point': read['pin_target']['point_img'] = [80, 950]
                else: read['bbox'] = [200, 400, 240, 415]
                self.assertEqual(self.text(normalize_assigned_pin_names(doc, assigned, [read])), 'RXDO')

    def test_distinct_interpretations_stay_ambiguous_even_with_one_supported_alternative(self):
        self.assertEqual({p['text'] for p in propose_names('DI0')}, {'D10', 'DIO'})
        doc, assigned = self.fixture(['DI0', 'D2', 'D3', 'DO'])
        read = self.visual(doc, assigned, 'DIO')
        result = normalize_assigned_pin_names(doc, assigned, [read])
        self.assertEqual(self.text(result), 'DI0')
        self.assertEqual(result['pin_name_library_audit']['entries'][0]['reason'], 'multiple_legal_candidates_preserved')

    def test_letter_digit_suffix_needs_visual_support(self):
        doc, assigned = self.fixture(['GPIOI6', 'GPIO2', 'GPIO13'])
        self.assertEqual(self.text(normalize_assigned_pin_names(doc, assigned, [])), 'GPIOI6')
        read = self.visual(doc, assigned, 'GPIO16')
        self.assertEqual(self.text(normalize_assigned_pin_names(doc, assigned, [read])), 'GPIO16')

    def test_port_letter_confusion_cannot_be_inferred_from_numbering_order(self):
        doc, assigned = self.fixture(['XTAL1/P86', 'PB4', 'PB5'])
        self.assertEqual(self.text(normalize_assigned_pin_names(doc, assigned, [])), 'XTAL1/P86')
        read = self.visual(doc, assigned, 'XTAL1/PB6')
        self.assertEqual(self.text(normalize_assigned_pin_names(doc, assigned, [read])), 'XTAL1/PB6')

    def test_slash_alias_polarity_and_separators_are_preserved(self):
        doc, assigned = self.fixture(['~RXDO#/GPIO2', 'GPIO13', 'GPIO5'])
        read = self.visual(doc, assigned, '~RXD0#/GPIO2')
        self.assertEqual(self.text(normalize_assigned_pin_names(doc, assigned, [read])), '~RXD0#/GPIO2')
        self.assertEqual(self.text(self.corrected(['GPI0_16#', 'GPIO_2', 'GPIO_13'])), 'GPIO_16#')

    def test_no_character_insertion_or_number_guessing(self):
        for text in ['GP05', 'GPO9', 'GPI', 'RX', 'MIG3', 'C1', '0', '17', 'STDB+']:
            with self.subTest(text=text):
                self.assertEqual(self.text(self.corrected([text, 'GPIO2', 'GPIO13'])), text)

    def test_literal_numeric_name_is_not_reinterpreted_as_io(self):
        doc, assigned = self.fixture(['100', 'IO2', 'IO3'])
        read = self.visual(doc, assigned, 'IO0')
        self.assertEqual(self.text(normalize_assigned_pin_names(doc, assigned, [read])), '100')

    def test_corrections_do_not_cascade_from_new_names(self):
        result = self.corrected(['GPI02', 'GPI013', 'GPI016'])
        self.assertEqual([self.text(result, i) for i in range(3)], ['GPI02', 'GPI013', 'GPI016'])

    def test_low_confidence_and_invalid_confidence_are_preserved(self):
        doc, assigned = self.fixture(['GPI016', 'GPIO2', 'GPIO13'])
        for confidence in (.8, float('nan'), float('inf'), 'invalid'):
            with self.subTest(confidence=confidence):
                assigned['comps']['ic']['pins'][0]['name']['conf'] = confidence
                self.assertEqual(self.text(normalize_assigned_pin_names(doc, assigned, [])), 'GPI016')

    def test_non_ic_fields_and_explicit_missing_number_rank_are_preserved(self):
        self.assertEqual(self.text(self.corrected(['GPI016', 'GPIO2', 'GPIO13'], ctype='other')), 'GPI016')
        doc, assigned = self.fixture(['GPI016', 'GPIO2', 'GPIO13'])
        assigned['comps']['ic']['pins'][0]['number_selection_name'] = None
        result = normalize_assigned_pin_names(doc, assigned, [])
        self.assertIsNone(result['comps']['ic']['pins'][0]['number_selection_name'])

    def test_normalization_is_idempotent(self):
        doc, assigned = self.fixture(['GPI016', 'GPIO2', 'GPIO13'])
        once = normalize_assigned_pin_names(doc, assigned, [])
        self.assertEqual(normalize_assigned_pin_names(doc, once, []), once)

    def test_export_structure_and_duplicate_number_winner_do_not_change(self):
        doc, assigned = self.fixture(['GPI016', 'GPIO2', 'GPIO13'])
        assigned['comps']['ic']['pins'][1]['number']['text'] = '1'
        assigned['comps']['ic']['pins'][1]['name']['conf'] = .91
        before = build_doc(doc, assigned, 'fixture')
        result = normalize_assigned_pin_names(doc, assigned, [])
        after = build_doc(doc, result, 'fixture')
        self.assertEqual(before['components'], after['components'])
        self.assertEqual(before['__pin_index'], after['__pin_index'])
        self.assertEqual(before['nets'], after['nets'])
        self.assertEqual(after['pins']['U1']['pin_1']['pinname'], 'GPIO16')
        after['pins']['U1']['pin_1']['pinname'] = before['pins']['U1']['pin_1']['pinname']
        self.assertEqual(after, before)


if __name__ == '__main__':
    unittest.main()
