import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from stage2.pin_name_ocr import PinNameOCR


class Recorder:
    rec_image_shape = [3, 48, 320]
    rec_batch_num = 1

    def __init__(self):
        self.colors, self.batches, self.options = [], [], []

    def resize_norm_img(self, image, ratio):
        self.colors.append(image.copy())
        tensor = np.zeros((3, 48, int(48 * ratio)), np.float32)
        h, w = min(image.shape[0], 48), min(image.shape[1], tensor.shape[2])
        tensor[:, :h, :w] = image[:h, :w].transpose(2, 0, 1) / 127.5 - 1
        return tensor

    def session(self, batch):
        self.batches.append(batch.copy())
        return [np.ones((1, 2, 3), np.float32)]

    def postprocess_op(self, predictions, word_box, **options):
        self.options.append((word_box, options))
        return [("6", .97)]


class PinNumberOCRAdapterTests(unittest.TestCase):
    def test_direct_original_pixels_and_actual_session_hash(self):
        glyph = np.arange(70, dtype=np.uint8).reshape(10, 7)
        before = glyph.copy()
        rec = Recorder()
        result = PinNameOCR(rec, {"model_sha256": "verified"}).read_glyph(glyph)
        np.testing.assert_array_equal(glyph, before)
        self.assertEqual(rec.colors[0].shape, (10, 7, 3))
        for c in range(3):
            np.testing.assert_array_equal(rec.colors[0][:, :, c], before)
        self.assertEqual(result["sampling"]["role"], "pin_number")
        self.assertEqual(result["sampling"]["version"], "ppocr-v4-direct-glyph-single-v1")
        self.assertEqual(result["sampling"]["original_gray_input_shape"], [10, 7])
        self.assertEqual(result["sampling"]["original_gray_input_sha256"], hashlib.sha256(glyph.tobytes()).hexdigest())
        self.assertEqual(result["sampling"]["model_input_sha256"], hashlib.sha256(rec.batches[0].tobytes()).hexdigest())
        self.assertEqual(rec.batches[0].shape, (1, 3, 48, 320))
        self.assertEqual(rec.options[0], (False, {"wh_ratio_list": [.7], "max_wh_ratio": 320 / 48}))

    def test_direct_api_does_not_relax_established_name_api(self):
        provider = PinNameOCR(Recorder())
        glyph = np.full((10, 7), 255, np.uint8)
        self.assertEqual(provider.read_glyph(glyph)["text"], "6")
        with self.assertRaises(ValueError):
            provider.read_normalized(glyph)
        name = provider.read_normalized(np.full((32, 7), 255, np.uint8))
        self.assertEqual(name["sampling"]["role"], "pin_name")
        self.assertIn("shared_gray_input_shape", name["sampling"])
        self.assertNotIn("original_gray_input_shape", name["sampling"])

    def test_direct_accepts_noncontiguous_pixels_and_keeps_singleton_context(self):
        rec = Recorder()
        provider = PinNameOCR(rec)
        glyph = np.arange(70, dtype=np.uint8).reshape(10, 7)[:, ::-1]
        first = provider.read_glyph(glyph)
        provider.read_glyph(np.full((12, 300), 255, np.uint8))
        last = provider.read_glyph(glyph)
        self.assertEqual(first, last)
        self.assertTrue(all(a.shape[0] == 1 for a in rec.batches))
        np.testing.assert_array_equal(rec.colors[0][:, :, 0], glyph)

    def test_invalid_source_rejected_before_recognizer(self):
        rec = Recorder()
        provider = PinNameOCR(rec)
        for gray in (None, [], np.zeros((0, 4), np.uint8), np.zeros((4, 0), np.uint8),
                     np.zeros((385, 4), np.uint8), np.zeros((4, 385), np.uint8),
                     np.zeros((4, 4), np.float32), np.zeros((4, 4, 1), np.uint8)):
            with self.subTest(shape=getattr(gray, "shape", None)):
                with self.assertRaises(ValueError):
                    provider.read_glyph(gray)
        self.assertEqual(rec.colors, [])

    def test_direct_nonfinite_tensor_or_output_is_rejected(self):
        rec = Recorder()
        provider = PinNameOCR(rec)
        gray = np.full((10, 7), 255, np.uint8)
        tensor = np.zeros((3, 48, 320), np.float32)
        tensor[0, 0, 0] = np.nan
        with patch.object(rec, "resize_norm_img", return_value=tensor):
            with self.assertRaises(ValueError):
                provider.read_glyph(gray)
        self.assertEqual(rec.batches, [])
        with patch.object(rec, "session", return_value=[np.array([np.inf])]):
            with self.assertRaises(ValueError):
                provider.read_glyph(gray)


if __name__ == "__main__":
    unittest.main()
