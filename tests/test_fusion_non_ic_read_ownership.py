"""Observed name ownership and geometry evidence across conflict resolution."""
import unittest
from unittest.mock import patch
from fusion import claims, priors, solve


def pool_with(*slots):
    pool = claims.ClaimPool('synthetic', {}, {})
    for cid, ctype, candidates in slots:
        sid = f'comp.{cid}.key'
        for value, score, source in candidates:
            pool.add(sid, value, score, source, meta={'type': ctype})
    return pool


class ObservedNameOwnershipTests(unittest.TestCase):
    def test_lower_duplicate_read_cannot_displace_stronger_owner(self):
        pool = pool_with(('a', 'r', [('R44', .91, 'ocr:designator'), ('R144', .90, 'prior:run')]),
                         ('b', 'r', [('R44', .70, 'ocr:designator'), ('R45', .65, 'prior:run')]))
        result = solve.solve(pool, do_gap=False)['assign']
        self.assertEqual(result['comp.a.key'][0], 'R44')
        self.assertEqual(result['comp.b.key'][0], 'R45')

    def test_prior_cannot_start_a_chain_that_displaces_two_observed_names(self):
        pool = pool_with(('a', 'c', [('C4', .95, 'ocr:designator'), ('C14', .8, 'prior:run')]),
                         ('b', 'c', [('C14', .90, 'ocr:designator'), ('C24', .5, 'prior:run')]),
                         ('c', 'c', [('C4', .4, 'prior:run'), ('C34', .3, 'prior:run')]))
        result = solve.solve(pool, do_gap=False)['assign']
        self.assertEqual([result[f'comp.{cid}.key'][0] for cid in 'abc'], ['C4', 'C14', 'C34'])

    def test_stronger_independent_axis_candidate_is_still_eligible(self):
        pool = pool_with(('a', 'r', [('R47', .45, 'ocr:designator'), ('R7', .50, 'geo:axis')]))
        self.assertEqual(solve.solve(pool, do_gap=False)['assign']['comp.a.key'][0], 'R7')

    def test_direct_negative_read_is_reserved_before_stronger_derived_offer(self):
        pool = pool_with(('a', 'r', [('R44', -.9, 'ocr:designator'), ('R144', -.91, 'prior:run')]),
                         ('b', 'r', [('R44', -.4, 'geo:axis'), ('R45', -.5, 'ocr:designator')]))
        result = solve.solve(pool, do_gap=False)['assign']
        self.assertEqual(result['comp.a.key'][0], 'R44')
        self.assertEqual(result['comp.b.key'][0], 'R45')

    def test_axis_correction_can_use_an_old_key_released_by_its_original_slot(self):
        pool = pool_with(('a', 'c', [('C14', -.8, 'ocr:designator'), ('C24', -.75, 'geo:axis')]),
                         ('b', 'c', [('C13', -.9, 'ocr:designator'), ('C14', -.85, 'geo:axis')]))
        result = solve.solve(pool, do_gap=False)['assign']
        self.assertEqual([result[f'comp.{cid}.key'][0] for cid in 'ab'], ['C24', 'C14'])

    def test_gap_filling_cannot_outrank_axis_evidence(self):
        pool = pool_with(('a', 'c', [('C1', .8, 'ocr:designator')]),
                         ('b', 'c', [('C2', .82, 'ocr:designator')]),
                         ('c', 'c', [('C4', .5, 'geo:axis')]),
                         ('d', 'c', [('C5', .81, 'ocr:designator')]))
        result = solve.solve(pool)
        self.assertGreater(result['stats']['gap_candidates'], 0)
        self.assertEqual(result['assign']['comp.c.key'][0], 'C4')
        self.assertTrue(all(score <= .49 for value, score, source in result['cands']['comp.c.key']
                            if source == 'solve:gapfill'))

    def test_negative_axis_score_does_not_allow_a_different_prior_to_win(self):
        pool = pool_with(('a', 'c', [('C44', -.8, 'ocr:designator'), ('C144', -.75, 'geo:axis')]))
        self.assertAlmostEqual(priors._cap(pool, 'comp.a.key', 'C1144', -.675), -.76)
        self.assertEqual(priors._cap(pool, 'comp.a.key', 'C144', -.70), -.70)

    def test_ic_prior_path_is_unchanged(self):
        pool = pool_with(('a', 'box', [('U44', -.8, 'ocr:designator'), ('U144', -.75, 'geo:axis')]))
        self.assertEqual(priors._cap(pool, 'comp.a.key', 'U1144', -.675), -.675)

    def test_ic_matching_path_is_unchanged(self):
        pool = pool_with(('a', 'box', [('U44', .91, 'ocr:designator'), ('U144', .90, 'prior:run')]),
                         ('b', 'box', [('U44', .70, 'ocr:designator'), ('U45', .65, 'prior:run')]))
        enabled = solve.solve(pool, do_gap=False)['assign']
        with patch.object(solve, 'NONIC_READ_OWNERSHIP', False):
            legacy = solve.solve(pool, do_gap=False)['assign']
        self.assertEqual(enabled, legacy)

    def test_disabled_guard_reproduces_existing_matching_behavior(self):
        pool = pool_with(('a', 'r', [('R44', .91, 'ocr:designator'), ('R144', .90, 'prior:run')]),
                         ('b', 'r', [('R44', .70, 'ocr:designator'), ('R45', .65, 'prior:run')]))
        with patch.object(solve, 'NONIC_READ_OWNERSHIP', False):
            result = solve.solve(pool, do_gap=False)['assign']
        self.assertEqual(result['comp.a.key'][0], 'R144')


if __name__ == '__main__': unittest.main()
