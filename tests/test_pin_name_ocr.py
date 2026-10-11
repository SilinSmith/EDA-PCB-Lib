import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from stage2 import pin_name_ocr as P


class FakeRecognizer:
    rec_image_shape = [3, 48, 320]
    rec_batch_num = 1

    def __init__(self, text="FB", confidence=.98):
        self.text, self.confidence = text, confidence
        self.colors, self.batches, self.options = [], [], []
        self.nonfinite_tensor = False
        self.nonfinite_output = False

    def resize_norm_img(self, image, ratio):
        self.colors.append(image.copy())
        # Tests assert the hash of this exact observed input; actual resize
        # equivalence is checked separately with the real optional model.
        width = int(48 * ratio)
        tensor = np.zeros((3, 48, width), np.float32)
        tensor[:, :32, :image.shape[1]] = image.transpose(2, 0, 1) / 127.5 - 1.
        if self.nonfinite_tensor:
            tensor[0, 0, 0] = np.nan
        return tensor

    def session(self, batch):
        self.batches.append(batch.copy())
        output = np.ones((1, 2, 3), np.float32)
        if self.nonfinite_output:
            output[0, 0, 0] = np.nan
        return [output]

    def postprocess_op(self, preds, word_box, **options):
        self.options.append((word_box, options))
        return [(self.text, self.confidence)]


class PinNameOCRTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((32, 45), 255, np.uint8)
        self.gray[9:24, 10:30] = 80

    def test_actual_session_tensor_hash_and_shared_gray_hash(self):
        rec = FakeRecognizer("ADJ(GND)", .97)
        provider = P.PinNameOCR(rec, {"model_sha256": "verified-hash"})
        result = provider.read_normalized(self.gray)
        batch = rec.batches[0]
        self.assertEqual(result["text"], "ADJ(GND)")
        self.assertEqual(result["confidence"], .97)
        self.assertEqual(result["sampling"]["model_input_shape"], list(batch.shape))
        self.assertEqual(result["sampling"]["model_input_sha256"], hashlib.sha256(batch.tobytes()).hexdigest())
        self.assertEqual(result["sampling"]["shared_gray_input_sha256"], hashlib.sha256(self.gray.tobytes()).hexdigest())
        self.assertEqual(result["sampling"]["model_sha256"], "verified-hash")
        self.assertEqual(batch.dtype, np.float32)

    def test_input_unmodified_three_equal_channels_and_singleton(self):
        rec = FakeRecognizer()
        source = self.gray.copy()
        P.PinNameOCR(rec).read_normalized(source)
        np.testing.assert_array_equal(source, self.gray)
        for channel in range(3):
            np.testing.assert_array_equal(rec.colors[0][:, :, channel], self.gray)
        self.assertEqual(len(rec.batches), 1)
        self.assertEqual(rec.batches[0].shape, (1, 3, 48, 320))
        self.assertEqual(rec.options[0], (False, {"wh_ratio_list": [45/32], "max_wh_ratio": 320/48}))

    def test_other_crop_cannot_change_this_crop(self):
        rec = FakeRecognizer()
        provider = P.PinNameOCR(rec)
        first = provider.read_normalized(self.gray)
        provider.read_normalized(np.zeros((32, 360), np.uint8))
        second = provider.read_normalized(self.gray)
        self.assertEqual(first, second)
        self.assertTrue(all(batch.shape[0] == 1 for batch in rec.batches))
        self.assertEqual(rec.batches[1].shape, (1, 3, 48, 540))

    def test_noncontiguous_input_is_supported(self):
        value = self.gray[:, ::-1]
        self.assertFalse(value.flags.c_contiguous)
        result = P.PinNameOCR(FakeRecognizer()).read_normalized(value)
        self.assertEqual(result["sampling"]["shared_gray_input_sha256"], hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())

    def test_invalid_shared_shapes_and_dtype_are_rejected_before_model(self):
        rec = FakeRecognizer()
        provider = P.PinNameOCR(rec)
        for value in (None, self.gray.tolist(), self.gray.astype(np.float32), np.zeros((31, 45), np.uint8),
                      np.zeros((32, 0), np.uint8), np.zeros((32, 385), np.uint8), np.zeros((32, 45, 1), np.uint8)):
            with self.subTest(shape=getattr(value, "shape", None)):
                with self.assertRaises(ValueError):
                    provider.read_normalized(value)
        self.assertEqual(rec.colors, [])

    def test_invalid_recognizer_config_is_rejected(self):
        rec = FakeRecognizer()
        rec.rec_image_shape = [3, 32, 320]
        with self.assertRaises(ValueError):
            P.PinNameOCR(rec)
        rec.rec_image_shape, rec.rec_batch_num = [3, 48, 320], 2
        with self.assertRaises(ValueError):
            P.PinNameOCR(rec)

    def test_nonfinite_tensor_is_rejected_before_session(self):
        rec = FakeRecognizer()
        rec.nonfinite_tensor = True
        with self.assertRaises(ValueError):
            P.PinNameOCR(rec).read_normalized(self.gray)
        self.assertEqual(rec.batches, [])

    def test_wrong_normalized_shape_is_rejected_before_session(self):
        rec = FakeRecognizer()
        with patch.object(rec, "resize_norm_img", return_value=np.zeros((3, 48, 321), np.float32)):
            with self.assertRaises(ValueError):
                P.PinNameOCR(rec).read_normalized(self.gray)
        self.assertEqual(rec.batches, [])

    def test_nonfinite_model_output_is_rejected(self):
        rec = FakeRecognizer()
        rec.nonfinite_output = True
        with self.assertRaises(ValueError):
            P.PinNameOCR(rec).read_normalized(self.gray)

    def test_invalid_confidences_are_rejected(self):
        for confidence in (np.nan, np.inf, -.1, 1.1, "bad"):
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    P.PinNameOCR(FakeRecognizer(confidence=confidence)).read_normalized(self.gray)

    def test_no_string_repair_or_case_normalization(self):
        for text in ("VIn", "Vin", "V/INI", "", "NC25"):
            self.assertEqual(P.PinNameOCR(FakeRecognizer(text)).read_normalized(self.gray)["text"], text)

    def test_off_never_loads_or_imports_optional_package(self):
        with patch.object(P, "_load_provider", side_effect=AssertionError("must not load")):
            self.assertIsNone(P.get_pin_name_ocr("off"))
        code = """
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.startswith(('rapidocr_onnxruntime','onnxruntime')):
            raise AssertionError('optional package imported')
sys.meta_path.insert(0, BlockOptional())
from stage2.pin_name_ocr import get_pin_name_ocr
assert get_pin_name_ocr('off') is None
"""
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_auto_missing_load_warns_once_and_on_reports_cached_failure(self):
        error = importlib.metadata.PackageNotFoundError("rapidocr-onnxruntime")
        with patch.object(P, "_provider", P._UNSET), patch.object(P, "_load_error", None), patch.object(P, "_warning_emitted", False), patch.object(P, "_load_provider", side_effect=error) as loader:
            with self.assertWarns(RuntimeWarning):
                self.assertIsNone(P.get_pin_name_ocr("auto"))
            with patch.object(P.warnings, "warn", side_effect=AssertionError("warn only once")):
                self.assertIsNone(P.get_pin_name_ocr("auto"))
            with self.assertRaisesRegex(P.PinNameOCRError, "unavailable"):
                P.get_pin_name_ocr("on")
            self.assertEqual(loader.call_count, 1)

    def test_success_is_cached_and_off_does_not_remove_it(self):
        provider = object()
        with patch.object(P, "_provider", P._UNSET), patch.object(P, "_load_provider", return_value=provider) as loader:
            self.assertIs(P.get_pin_name_ocr("auto"), provider)
            self.assertIsNone(P.get_pin_name_ocr("off"))
            self.assertIs(P.get_pin_name_ocr("on"), provider)
            self.assertEqual(loader.call_count, 1)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            P.get_pin_name_ocr("enabled")

    def test_on_loader_failure_is_explicit(self):
        with patch.object(P, "_provider", P._UNSET), patch.object(P, "_load_error", None), patch.object(P, "_load_provider", side_effect=RuntimeError("session error")):
            with self.assertRaisesRegex(P.PinNameOCRError, "session error"):
                P.get_pin_name_ocr("on")

    def test_wrong_optional_package_version_is_rejected_before_import(self):
        with patch.object(P.importlib.metadata, "version", return_value="9.9.9"), patch.object(P.importlib, "import_module", side_effect=AssertionError("must not import")):
            with self.assertRaisesRegex(P.PinNameOCRError, "1.4.4"):
                P._load_provider()

    def test_missing_and_mismatched_models_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = types.SimpleNamespace(__file__=str(root/"__init__.py"))
            with self.assertRaisesRegex(P.PinNameOCRError, "missing"):
                P._verified_model_path(package)
            (root/"models").mkdir()
            model = root/"models/ch_PP-OCRv4_rec_infer.onnx"
            model.write_bytes(b"wrong-model")
            with self.assertRaisesRegex(P.PinNameOCRError, "SHA256 mismatch"):
                P._verified_model_path(package)
            with patch.object(P, "EXPECTED_MODEL_SHA256", hashlib.sha256(b"wrong-model").hexdigest()):
                self.assertEqual(P._verified_model_path(package), model)


if __name__ == "__main__":
    unittest.main()
