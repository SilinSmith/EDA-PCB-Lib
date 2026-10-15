"""Imports and diagnostic entry points must stay inside their actual checkout."""
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CheckoutPortabilityTests(unittest.TestCase):
    def test_library_imports_do_not_inject_another_checkout(self):
        code = '''import cv2, numpy, sys, json
before = list(sys.path)
import pipeline.config
import stage1.pintrace
import tools.eval.stage1_wires_eval
print(json.dumps({'same': before == sys.path, 'repo': pipeline.config.REPO}))
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                capture_output=True, text=True, encoding='utf-8',
                                env=dict(os.environ, PYTHONIOENCODING='utf-8'), check=True)
        data = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(data['same'])
        self.assertEqual(Path(data['repo']), ROOT)

    def test_config_follows_a_moved_checkout_with_spaces(self):
        with tempfile.TemporaryDirectory() as temp:
            moved = Path(temp) / 'alternate checkout' / 'eda project'
            config = moved / 'pipeline' / 'config.py'
            config.parent.mkdir(parents=True)
            shutil.copyfile(ROOT / 'pipeline' / 'config.py', config)
            values = runpy.run_path(str(config))
            self.assertEqual(Path(values['REPO']), moved.resolve())
            self.assertEqual(Path(values['EDA_ROOT']), moved.parent.resolve())
            self.assertEqual(Path(values['DATA_ROOT']), moved / 'yolo_part' / 'Dataset(waitforpreprocess)')

    def test_pintrace_help_runs_outside_the_project_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, str(ROOT / 'stage1' / 'pintrace.py'), '--help'],
                                    cwd=temp, capture_output=True, text=True, encoding='utf-8',
                                    env=dict(os.environ, PYTHONIOENCODING='utf-8'))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--stage1', result.stdout)
            self.assertIn('--out', result.stdout)


if __name__ == '__main__':
    unittest.main()
