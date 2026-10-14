"""Image-confirmed anonymous symbols retain their real network membership."""
import copy
import unittest
from unittest.mock import patch
from stage2 import assemble
from schematic_parser.data import parse_hypergraph


def fixture(ctype='gnd'):
    stage = {'image': {'w': 200, 'h': 100},
             'components': [{'id': 'r', 'type': 'r', 'bbox_json': [10,40,30,50]},
                            {'id': 's', 'type': ctype, 'bbox_json': [60,30,80,40]}],
             'pins': {'r': [[40,55],[0,55]], 's': [[70,50]]},
             'passive_pin_report': {'s': {'type': ctype, 'after': [[70,50]],
                 'sources': ['body_attached_red_green_transition'], 'coordinate_rule': 'red_green_pixel_boundary'}},
             'wire_detail': {'segments': [[[40,55],[70,50]]]},
             'nets': [{'pins': [['r',0],['s',0]], 'seg_index': [0]}]}
    assigned = {'comps': {'r': {'designator': {'text':'R44','canon':'R44'}}}}
    return stage, assigned


class ConfirmedPowerNetPublicationTests(unittest.TestCase):
    def build(self, stage, assigned, **flags):
        with patch.multiple(assemble, DROP_UNKEYED=False, NET_DROP_UNKEYED=True,
                            KEEP_CONFIRMED_POWER_PINS=True, SYM_MERGE=False, **flags):
            return assemble.build_doc(stage, assigned, 'synthetic')

    def test_anonymous_confirmed_ground_has_a_valid_pin_reference(self):
        stage, assigned = fixture()
        old = copy.deepcopy(stage)
        doc = self.build(stage, assigned)
        self.assertEqual(stage, old)
        symbol = next(k for k,v in doc['components'].items() if v['type']=='gnd')
        refs = parse_hypergraph(next(iter(doc['nets'].values()))['hyperGraph'])
        self.assertIn((symbol,'1'), refs)
        self.assertIn('pin_1', doc['pins'][symbol])

    def test_confirmed_supply_is_retained_too(self):
        stage, assigned = fixture('v')
        doc = self.build(stage, assigned)
        symbol = next(k for k,v in doc['components'].items() if v['type']=='v')
        self.assertTrue(any((symbol,'1') in parse_hypergraph(n['hyperGraph']) for n in doc['nets'].values()))

    def test_multi_candidate_symbols_are_not_collapsed_or_exempted(self):
        stage, assigned = fixture(); stage['pins']['s'].append([70,70])
        doc = self.build(stage, assigned)
        self.assertTrue(all(not k.startswith('p') for n in doc['nets'].values() for k,v in parse_hypergraph(n['hyperGraph'])))

    def test_stale_or_unconfirmed_report_does_not_exempt_a_key(self):
        for change in ({'after':[[70,70]]}, {'sources':['fallback_legacy']},
                       {'type':'v'}, {'coordinate_rule':'monochrome'}):
            with self.subTest(change=change):
                stage, assigned = fixture(); stage['passive_pin_report']['s'].update(change)
                doc = self.build(stage, assigned)
                self.assertTrue(all(not k.startswith('p') for n in doc['nets'].values() for k,v in parse_hypergraph(n['hyperGraph'])))

    def test_fallback_ic_and_passive_keys_do_not_get_the_power_exemption(self):
        for ctype in ('box','r','other'):
            stage, assigned = fixture(ctype)
            self.assertFalse(assemble._confirmed_power_terminal(stage,'s',ctype,stage['pins']['s']))

    def test_nonfinite_observation_is_rejected(self):
        stage, assigned = fixture(); stage['passive_pin_report']['s']['after']=[[float('nan'),50]]
        self.assertFalse(assemble._confirmed_power_terminal(stage,'s','gnd',stage['pins']['s']))

    def test_power_only_network_survives_when_a_real_terminal_is_confirmed(self):
        stage, assigned = fixture(); stage['nets'][0]['pins']=[['s',0]]
        doc=self.build(stage,assigned)
        self.assertEqual(len(doc['nets']),1)

    def test_legacy_symbol_merging_keeps_the_confirmed_reference(self):
        stage, assigned = fixture()
        with patch.multiple(assemble,DROP_UNKEYED=False,NET_DROP_UNKEYED=True,
                            KEEP_CONFIRMED_POWER_PINS=True,SYM_MERGE=True):
            doc=assemble.build_doc(stage,assigned)
        self.assertTrue(any(k.startswith('p') for n in doc['nets'].values() for k,v in parse_hypergraph(n['hyperGraph'])))

    def test_opt_out_retains_legacy_filter(self):
        stage, assigned = fixture()
        with patch.multiple(assemble,DROP_UNKEYED=False,NET_DROP_UNKEYED=True,
                            KEEP_CONFIRMED_POWER_PINS=False,SYM_MERGE=False):
            doc=assemble.build_doc(stage,assigned)
        self.assertTrue(all(not k.startswith('p') for n in doc['nets'].values() for k,v in parse_hypergraph(n['hyperGraph'])))


if __name__ == '__main__': unittest.main()
