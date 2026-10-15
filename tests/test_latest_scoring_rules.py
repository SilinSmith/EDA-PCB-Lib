"""Official rule changes must affect evaluation without changing inference."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from schematic_parser import metrics as m
from tools.debug.ledger import build_ledger


def component(typ='r', value='1k', x=0):
    return {'type': typ, 'value': value, 'Name': 'CHIP' if typ == 'box' else None,
            'bbox': [x, 0, x + 10, 10]}


def pin(name='IN', x=0):
    return {'pinname': name, 'point': {'x': x, 'y': 0}}


def net(refs, x=0):
    return {'hyperGraph': '(' + refs + ')',
            'edges': {'e': [{'x': x, 'y': 0}, {'x': x + 100, 'y': 0}]}}


def doc(comps=None, pins=None, nets=None):
    return {'components': comps or {}, 'pins': pins or {}, 'nets': nets or {}}


class LatestScoringRulesTests(unittest.TestCase):
    def test_profiles_keep_explicit_historical_defaults(self):
        latest = m.EvalConfig.for_profile()
        self.assertEqual((latest.bbox_tol, latest.pin_tol, latest.line_tol), (20, 20, 20))
        self.assertTrue(latest.net_pin_collapse_nonbox)
        self.assertTrue(latest.pin_box_only)
        for name in ('protocol2', 'protocol'):
            cfg = m.EvalConfig.for_profile(name)
            self.assertEqual((cfg.bbox_tol, cfg.pin_tol, cfg.line_tol), (20, 5, 5))
            self.assertFalse(cfg.power_symbol_match)
            self.assertEqual(cfg.net_soft, name == 'protocol2')
            self.assertEqual(cfg.net_pin_collapse_nonbox, name == 'protocol')
        with self.assertRaises(ValueError):
            m.EvalConfig.for_profile('typo')

    def test_all_three_spatial_metrics_include_20_and_reject_beyond(self):
        gt = doc({'U1': component('box')}, {'U1': {'pin_1': pin()}}, {'n': net('U1.1')})
        for offset, expected in ((10, 1), (20, 1), (20.01, 0)):
            pred = copy.deepcopy(gt)
            pred['components']['U1']['bbox'] = [offset, 0, offset + 10, 10]
            pred['pins']['U1']['pin_1'] = pin(x=offset)
            pred['nets']['n'] = net('U1.1', offset)
            score = m.evaluate_case(gt, pred)
            for field in ('component', 'pin', 'net_line'):
                with self.subTest(offset=offset, field=field):
                    self.assertEqual(getattr(score, field).f1, expected)
        self.assertEqual(m.evaluate_case(gt, gt).score, 100)

    def test_nonbox_name_number_position_do_not_change_score(self):
        gt = doc({'R1': component()}, {'R1': {'pin_1': pin('1')}}, {'n': net('R1.1')})
        pred = copy.deepcopy(gt)
        pred['pins']['R1'] = {'pin_random': pin('WRONG', 500)}
        pred['nets']['n'] = net('R1.random')
        self.assertEqual(m.evaluate_case(gt, pred).score, 100)
        self.assertLess(m.evaluate_case(gt, pred, cfg=m.EvalConfig.for_profile('protocol2')).score, 100)

    def test_ic_name_number_still_score_independently(self):
        gt = doc({'U1': component('box')}, {'U1': {'pin_1': pin()}}, {'n': net('U1.1')})
        pred = copy.deepcopy(gt)
        pred['pins']['U1']['pin_1']['pinname'] = 'WRONG'
        self.assertEqual(m.evaluate_case(gt, pred).pin.f1, 0)
        pred['pins']['U1'] = {'pin_2': pin()}
        pred['nets']['n'] = net('U1.2')
        score = m.evaluate_case(gt, pred)
        self.assertEqual((score.pin.f1, score.net_hypergraph.f1), (0, 0))

    def test_two_level_network_does_not_short_two_ends(self):
        comps = {'R1': component(), 'U1': component('box')}
        gt = doc(comps, nets={'a': net('R1.1,U1.1'), 'b': net('R1.2,U1.2')})
        pred = copy.deepcopy(gt)
        pred['nets'] = {'a': net('R1.2,U1.1'), 'b': net('R1.1,U1.2')}
        self.assertEqual(m.evaluate_case(gt, pred).net_hypergraph.f1, 1)
        pred['nets'] = {'shorted': net('R1.1,R1.2,U1.1,U1.2')}
        self.assertLess(m.evaluate_case(gt, pred).net_hypergraph.f1, 1)
        pred['nets'] = {'a': net('R1.1,R1.2,U1.1'), 'b': net('U1.2')}
        self.assertLess(m.evaluate_case(gt, pred).net_hypergraph.f1, 1)

    def test_unknown_power_keys_and_equal_swapped_keys_use_geometry(self):
        gt = doc({'GND_1': component('gnd', 'GND', 0),
                  'GND_2': component('gnd', 'GND', 100)},
                 nets={'a': net('GND_1.119'), 'b': net('GND_2.202')})
        pred = doc({'GND_2': component('gnd', 'GND', 0),
                    'GND_1': component('gnd', 'GND', 100)},
                   nets={'a': net('GND_2.1'), 'b': net('GND_1.1')})
        score, keymap = m.assign_components(gt, pred, m.EvalConfig())
        self.assertEqual(score.f1, 1)
        self.assertEqual(keymap, {'GND_1': 'GND_2', 'GND_2': 'GND_1'})
        self.assertEqual(m.evaluate_case(gt, pred).net_hypergraph.f1, 1)
        pred['components'] = {'arbitrary': pred['components']['GND_2']}
        pred['nets'] = {'a': net('arbitrary.1')}
        self.assertEqual(m.assign_components(gt, pred, m.EvalConfig())[0].tp, 1)

    def test_wrong_power_value_type_or_position_cannot_match_even_same_key(self):
        gt = doc({'GND_1': component('v', '3V3')}, nets={'n': net('GND_1.1')})
        for replacement in (component('v', '5V'), component('gnd', '3V3'),
                            component('r', '3V3'), component('v', '3V3', 21)):
            pred = doc({'GND_1': replacement}, nets={'n': net('GND_1.1')})
            score = m.evaluate_case(gt, pred)
            with self.subTest(replacement=replacement):
                self.assertEqual(score.component.f1, 0)
                self.assertEqual(score.net_hypergraph.f1, 0)

    def test_voltage_text_is_not_silently_normalized(self):
        gt = doc({'a': component('v', '3V3')})
        for value in ('3.3V', '3v3', None):
            self.assertEqual(m.evaluate_case(gt, doc({'b': component('v', value)})).component.f1, 0)

    def test_power_pairing_is_one_to_one_and_does_not_steal_named_parts(self):
        gt = doc({'a': component('gnd', 'GND'), 'R1': component()})
        pred = doc({'b': component('gnd', 'GND'), 'c': component('gnd', 'GND')})
        score = m.evaluate_case(gt, pred).component
        self.assertEqual((score.tp, score.fp, score.fn), (1, 1, 1))
        self.assertEqual(m.evaluate_case(doc({'R1': component()}), doc({'R2': component()})).component.f1, 0)

    def test_power_assignment_finds_two_pairs_when_nearest_greedy_would_lose_one(self):
        gt = doc({'a': component('gnd', 'GND', 0), 'b': component('gnd', 'GND', 30)})
        pred = doc({'p': component('gnd', 'GND', 15), 'q': component('gnd', 'GND', 0)})
        score, mapping = m.assign_components(gt, pred, m.EvalConfig())
        self.assertEqual((score.tp, score.fp, score.fn), (2, 0, 0))
        self.assertEqual(mapping, {'p': 'b', 'q': 'a'})

    def test_partial_auto_key_permutation_does_not_overwrite_unmatched_symbol(self):
        gt = doc({'a': component('gnd', 'GND', 0), 'b': component('gnd', 'GND', 100)})
        pred = doc({'b': component('gnd', 'GND', 0), 'a': component('gnd', 'GND', 1000)})
        _, keymap = m.assign_components(gt, pred, m.EvalConfig())
        rewritten = m.rewrite_keys(pred, keymap)
        self.assertEqual(len(rewritten['components']), 2)
        self.assertEqual(rewritten['components']['a']['bbox'], [0, 0, 10, 10])

    def test_extra_ic_pins_not_hidden_and_known_ic_type_is_authoritative(self):
        gt = doc({'U1': component('box')}, {'U1': {'pin_1': pin()}}, {'n': net('U1.1')})
        pred = copy.deepcopy(gt)
        pred['components']['U2'] = component('box')
        pred['pins']['U2'] = {'pin_1': pin()}
        self.assertEqual(m.evaluate_case(gt, pred).pin.fp, 1)
        pred['components']['U1']['type'] = 'r'
        pred['nets']['n'] = net('U1.2')
        self.assertEqual(m.evaluate_case(gt, pred).net_hypergraph.f1, 0)

    def test_reports_share_key_mapping_filtering_matching_and_soft_counts(self):
        gt = doc({'GND_1': component('gnd', 'GND'), 'U1': component('box')},
                 {'GND_1': {'pin_119': pin('119')}, 'U1': {'pin_1': pin()}},
                 {'n': net('GND_1.119,U1.1')})
        pred = doc({'GND_9': component('gnd', 'GND'), 'U1': component('box')},
                   {'GND_9': {'pin_1': pin('WRONG', 999)}, 'U1': {'pin_1': pin(x=10)}},
                   {'n': net('GND_9.1,U1.2')})
        original = copy.deepcopy(pred)
        for profile in m.RULE_PROFILES:
            led = build_ledger(gt, pred, cfg=m.EvalConfig.for_profile(profile))
            self.assertTrue(all(c['ok'] for c in led['selfcheck']), led['selfcheck'])
            self.assertEqual(led['config']['pin_tol'], 20 if profile == 'latest' else 5)
        self.assertEqual(pred, original)


class ScoringEntryPointTests(unittest.TestCase):
    def test_entry_points_use_their_own_checkout_from_another_directory(self):
        root = Path(__file__).resolve().parents[1]
        script = '''import runpy,sys,json
sys.argv=[sys.argv[1], '--help']
try: runpy.run_path(sys.argv[0],run_name='__main__')
except SystemExit as exc:
 if exc.code: raise
import schematic_parser.metrics as m
print(json.dumps({'metrics': m.__file__}))
'''
        with tempfile.TemporaryDirectory() as temp:
            for path in ('tools/eval/fast.py', 'tools/eval/stage2_eval.py',
                         'tools/debug/dbg.py', 'tools/eval/pin_network_ab.py',
                         'tools/eval/compare_manifest.py'):
                with self.subTest(path=path):
                    result = subprocess.run([sys.executable, '-B', '-c', script, str(root / path)],
                        cwd=temp, capture_output=True, text=True, encoding='utf-8',
                        env=dict(os.environ, PYTHONIOENCODING='utf-8'))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(Path(json.loads(result.stdout.splitlines()[-1])['metrics']),
                                     root / 'schematic_parser' / 'metrics.py')


if __name__ == '__main__':
    unittest.main()
