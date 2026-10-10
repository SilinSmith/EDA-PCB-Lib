import copy
import hashlib
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from stage2.isolated_ocr import prepare_isolated_sample, read_isolated_many
from textrec import config as C
from textrec.dataset import crop_gray


class TensorRecorder:
    img_h = 32
    max_w = 384

    def __init__(self):
        self.calls = []

    def read_many(self, gray, items):
        self.calls.append((gray.copy(), copy.deepcopy(items)))
        out = []
        for bbox, role in items:
            tensor = np.asarray(crop_gray(Image.fromarray(gray), bbox, self.img_h, self.max_w))
            out.append((hashlib.sha256(tensor.tobytes()).hexdigest(), .5))
        return out


class IsolatedOCRTests(unittest.TestCase):
    def sample(self):
        gray = np.full((50, 60), 255, np.uint8)
        gray[15:25, 20:25] = 0
        gray[15:25, 30:38] = 80
        return {"gray": gray, "bbox": [20, 15, 38, 25], "role": "pin_name"}

    def test_adjacent_source_pixels_are_never_copied(self):
        sample = self.sample()
        safe = prepare_isolated_sample(sample)
        contaminated = sample["gray"].copy()
        contaminated[:15] = 0
        contaminated[25:] = 0
        contaminated[15:25, :20] = 0
        contaminated[15:25, 38:] = 0
        altered = prepare_isolated_sample(dict(sample, gray=contaminated))
        self.assertTrue(np.array_equal(safe[0], altered[0]))
        self.assertEqual(safe[2]["model_input_sha256"], altered[2]["model_input_sha256"])

    def test_border_glyph_has_same_input_as_translated_interior_glyph(self):
        source = self.sample()
        glyph = source["gray"][15:25, 20:38].copy()
        border = dict(source, gray=glyph, bbox=[0, 0, 18, 10])
        a = prepare_isolated_sample(source)
        b = prepare_isolated_sample(border)
        self.assertTrue(np.array_equal(a[0], b[0]))
        self.assertEqual(a[2]["model_input_sha256"], b[2]["model_input_sha256"])

    def test_rotation_changes_only_read_image_not_source_geometry(self):
        sample = self.sample()
        canvas, box, meta = prepare_isolated_sample(dict(sample, rotation=-90, padding=0))
        self.assertEqual(meta["source_bbox"], sample["bbox"])
        self.assertEqual(meta["sampled_bbox"], sample["bbox"])
        self.assertEqual(meta["source_glyph_shape"], [10, 18])
        self.assertEqual(meta["rotated_glyph_shape"], [18, 10])
        self.assertTrue(np.array_equal(canvas[box[1]:box[3], box[0]:box[2]],
                                       np.rot90(sample["gray"][15:25, 20:38], -1)))

    def test_unrelated_batch_width_and_order_do_not_change_sampling(self):
        sample = self.sample()
        long = dict(sample, gray=np.zeros((10, 180), np.uint8), bbox=[0, 0, 180, 10])
        rec = TensorRecorder()
        a = read_isolated_many(rec, [sample])[0]
        b = read_isolated_many(rec, [long, sample, long])[1]
        self.assertEqual(a, b)
        self.assertTrue(all(len(items) == 1 for _, items in rec.calls))
        self.assertEqual(a["text"], a["sampling"]["model_input_sha256"])

    def test_source_array_and_caller_geometry_remain_unchanged(self):
        sample = self.sample()
        before = copy.deepcopy(sample)
        read_isolated_many(TensorRecorder(), [dict(sample, rotation=90)])
        self.assertTrue(np.array_equal(sample["gray"], before["gray"]))
        self.assertEqual(sample["bbox"], before["bbox"])

    def test_adaptive_ink_expansion_stays_within_white_guard(self):
        sample = self.sample()
        with patch.object(C, "INK_YNORM", True), patch.object(C, "INK_FRAME", 4.):
            a = prepare_isolated_sample(sample)
            padded = np.pad(a[0], 50, constant_values=255)
            bbox = [v + 50 for v in a[1]]
            b = np.asarray(crop_gray(Image.fromarray(padded), bbox, 32, 384))
            self.assertEqual(a[2]["model_input_sha256"], hashlib.sha256(b.tobytes()).hexdigest())

    def test_invalid_input_is_rejected_and_empty_sequence_is_valid(self):
        self.assertEqual(read_isolated_many(TensorRecorder(), []), [])
        sample = self.sample()
        for fields in ({"bbox": [20, 15, 20, 25]}, {"bbox": [0, 0, float("nan"), 2]},
                       {"bbox": [100, 100, 110, 110]}, {"rotation": 45},
                       {"padding": -1}, {"padding": True},
                       {"gray": np.zeros((4, 4, 3), np.uint8)}):
            with self.subTest(fields=list(fields)), self.assertRaises(ValueError):
                prepare_isolated_sample(dict(sample, **fields))

    def test_import_does_not_require_the_optional_torch_runtime(self):
        script = """
import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch intentionally absent')
sys.meta_path.insert(0, BlockTorch())
import stage2.isolated_ocr
assert 'torch' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
