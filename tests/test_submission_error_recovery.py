"""A failed first OCR case must not prevent a later valid case being exported."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from submission import run


class SubmissionErrorRecoveryTests(unittest.TestCase):
    def test_ocr_failure_is_logged_and_later_case_is_delivered(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            inputs, outputs = folder / 'inputs', folder / 'outputs'
            inputs.mkdir()
            for name in ('01_bad', '02_good'):
                encoded, png = cv2.imencode('.png', np.full((16, 16, 3), 255, np.uint8))
                self.assertTrue(encoded)
                (inputs / (name + '.png')).write_bytes(png.tobytes())
            prediction = {'components': {}, 'pins': {}, 'nets': {}}

            def stage1(cases, output, **kwargs):
                output.mkdir(parents=True, exist_ok=True)
                rows = []
                for case in cases:
                    data = {'case': case.case_id, 'image': {'path': case.png_path, 'width': 16, 'height': 16},
                            'components': [], 'pins': {}, 'nets': [], 'wires': {}, 'errors': []}
                    (output / (case.case_id + '.stage1.json')).write_text(json.dumps(data), encoding='utf-8')
                    rows.append({'case': case.case_id, 'ok': True, 'stats': {}, 'timing': {}})
                return rows

            def stage2(case, source, output, engine, **kwargs):
                if case == '01_bad':
                    raise ValueError('simulated first-case OCR failure')
                output.mkdir(parents=True, exist_ok=True)
                (output / (case + '.json')).write_text(json.dumps(prediction), encoding='utf-8')
                return {'case': case, 'n_components': 0, 'n_designator': 0, 't_total': 0}

            def stage3(case, first, second, output, **kwargs):
                output.mkdir(parents=True, exist_ok=True)
                (output / (case + '.json')).write_text(json.dumps(prediction), encoding='utf-8')
                return {'case': case, 'n_slots_total': 0, 'n_named': 0, 't': 0}

            with patch.dict(os.environ, {'EDA_GPU_CAP': '0', 'EDA_CASE_WORKERS': '1',
                                         'CUDA_VISIBLE_DEVICES': '', 'FUSION_LLM': '0'}), \
                    patch('gpu_mem.cap_vram', return_value='test CPU'), \
                    patch.object(run.assets, 'describe', return_value={}), \
                    patch('stage1.pipeline.run_cases', side_effect=stage1), \
                    patch('stage2.run.process', side_effect=stage2), \
                    patch('stage2.text.TextEngine') as engine, \
                    patch('stage3.run.process', side_effect=stage3), \
                    contextlib.redirect_stdout(io.StringIO()):
                engine.return_value.describe.return_value = 'test OCR'
                status = run.main(['-input', str(inputs), '-output', str(outputs),
                                   '-device', 'cpu', '-mode', 'off', '-vlm', 'off', '-fusion', 'off'])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads((outputs / 'final_results/02_good_Pred.json').read_text(encoding='utf-8')),
                             prediction)
            log = (outputs / 'final_results/run.log').read_text(encoding='utf-8')
            self.assertIn('simulated first-case OCR failure', log)
            self.assertNotIn('UnboundLocalError', log)
            decision = json.loads((outputs / 'decision_log.json').read_text(encoding='utf-8'))
            good = next(row for row in decision['cases'] if row['case'] == '02_good')
            self.assertTrue(good['exported'])


if __name__ == '__main__':
    unittest.main()
