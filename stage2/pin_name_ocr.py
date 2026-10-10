"""Optional, verified PP-OCR recognition of isolated pin text.

The provider receives the exact uint8 gray image already normalized to the
CRNN's 32px input height. It does not extract glyphs, choose a direction, bind
pins or repair strings. Numbers use a separate explicit original-glyph API.
Detection/classification models and
network downloads are never used here.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import math
from pathlib import Path
import warnings

import numpy as np


EXPECTED_PACKAGE_VERSION = "1.4.4"
EXPECTED_MODEL_SHA256 = "48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b"
REC_IMAGE_SHAPE = (3, 48, 320)
SHARED_GRAY_HEIGHT = 32
SHARED_GRAY_MAX_WIDTH = 384
SAMPLING_VERSION = "ppocr-v4-shared-gray-single-v1"
_UNSET = object()
_provider = _UNSET
_load_error = None
_warning_emitted = False


class PinNameOCRError(RuntimeError):
    """The explicitly requested optional name recognizer is unavailable."""


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _model_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _verified_model_path(package):
    model = Path(package.__file__).resolve().parent / "models/ch_PP-OCRv4_rec_infer.onnx"
    if not model.is_file():
        raise PinNameOCRError(f"PP-OCR name model is missing: {model}")
    found = _model_hash(model)
    if found != EXPECTED_MODEL_SHA256:
        raise PinNameOCRError(f"PP-OCR name model SHA256 mismatch: {found}")
    return model


class PinNameOCR:
    """Single-crop adapter; injected recognizers allow dependency-free tests."""

    def __init__(self, recognizer, metadata=None):
        if tuple(recognizer.rec_image_shape) != REC_IMAGE_SHAPE:
            raise ValueError("pin name OCR requires recognition shape [3, 48, 320]")
        if recognizer.rec_batch_num != 1:
            raise ValueError("pin name OCR requires single-crop recognition")
        self.recognizer = recognizer
        self.metadata = dict(metadata or {})

    def read_normalized(self, gray):
        """Recognize one unchanged shared gray32 image and record real input.

        This follows TextRecognizer 1.4.4's singleton resize/session/postprocess
        calls exactly. The tensor whose hash is recorded is the tensor passed
        to the ONNX session, not a separately recreated approximation.
        """
        if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8:
            raise ValueError("shared gray input must be a uint8 numpy array")
        if gray.ndim != 2 or gray.shape[0] != SHARED_GRAY_HEIGHT:
            raise ValueError("shared gray input must have shape [32, width]")
        if not 1 <= gray.shape[1] <= SHARED_GRAY_MAX_WIDTH:
            raise ValueError("shared gray input width must be in [1, 384]")
        return self._read_gray(gray, role="pin_name", version=SAMPLING_VERSION,
                               input_prefix="shared_gray_input")

    def read_glyph(self, gray):
        """Read original number pixels, without CRNN's intermediate resampling.

        The caller supplies the complete isolated glyph plus synthetic white
        margins. PP performs its own one-step 48px normalization. This API
        deliberately cannot change the established gray32 name path.
        """
        if (not isinstance(gray, np.ndarray) or gray.dtype != np.uint8
                or gray.ndim != 2 or not all(1 <= n <= 384 for n in gray.shape)):
            raise ValueError("number glyph must be a nonempty uint8 gray array, at most 384px per axis")
        return self._read_gray(gray, role="pin_number",
                               version="ppocr-v4-direct-glyph-single-v1",
                               input_prefix="original_gray_input")

    def _read_gray(self, gray, *, role, version, input_prefix):
        # Replicate grayscale channels without introducing source pixels or
        # sharing writable storage with the caller.
        color = np.repeat(gray[:, :, None], 3, axis=2)
        ratio = gray.shape[1] / float(gray.shape[0])
        max_ratio = max(REC_IMAGE_SHAPE[2] / REC_IMAGE_SHAPE[1], ratio)
        normalized = self.recognizer.resize_norm_img(color, max_ratio)
        normalized = np.asarray(normalized)
        expected_shape = (REC_IMAGE_SHAPE[0], REC_IMAGE_SHAPE[1], int(REC_IMAGE_SHAPE[1] * max_ratio))
        if (normalized.shape != expected_shape
                or not np.isfinite(normalized).all()):
            raise ValueError("PP-OCR produced an invalid normalized tensor")
        batch = np.ascontiguousarray(normalized[np.newaxis, :], dtype=np.float32)
        model_input_hash = _sha(batch)
        outputs = self.recognizer.session(batch)
        if not isinstance(outputs, (tuple, list)) or not outputs:
            raise ValueError("PP-OCR session returned no recognition output")
        preds = outputs[0]
        if not np.isfinite(np.asarray(preds)).all():
            raise ValueError("PP-OCR returned non-finite recognition probabilities")
        result = self.recognizer.postprocess_op(
            preds, False, wh_ratio_list=[ratio], max_wh_ratio=max_ratio)
        if not isinstance(result, (tuple, list)) or len(result) != 1:
            raise ValueError("PP-OCR must return one recognition result")
        try:
            text, confidence = result[0]
            confidence = float(confidence)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("PP-OCR returned an invalid recognition result") from exc
        if not isinstance(text, str) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("PP-OCR text/confidence must be finite and valid")
        return {
            "text": text,
            "confidence": confidence,
            "sampling": {
                "version": version,
                "strategy": "single",
                "role": role,
                input_prefix + "_shape": list(gray.shape),
                input_prefix + "_sha256": _sha(gray),
                "model_input_shape": list(batch.shape),
                "model_input_dtype": str(batch.dtype),
                "model_input_sha256": model_input_hash,
                "recognition_tensor_shape": list(normalized.shape),
                "max_wh_ratio": max_ratio,
                "model_sha256": self.metadata.get("model_sha256"),
                "model_source": self.metadata.get("model_source"),
                "confidence_definition": "arithmetic mean of emitted nonblank character probabilities",
            },
        }


def _load_provider():
    # Both the version lookup and heavy package/session imports happen only
    # when a caller requests this provider. No network or alternate hash is
    # accepted when a different install happens to be present.
    version = importlib.metadata.version("rapidocr-onnxruntime")
    if version != EXPECTED_PACKAGE_VERSION:
        raise PinNameOCRError(f"pin name OCR requires rapidocr-onnxruntime {EXPECTED_PACKAGE_VERSION}; found {version}")
    package = importlib.import_module("rapidocr_onnxruntime")
    module = importlib.import_module("rapidocr_onnxruntime.ch_ppocr_rec")
    model_path = _verified_model_path(package)
    config = {
        "model_path": str(model_path),
        "rec_batch_num": 1,
        "rec_img_shape": list(REC_IMAGE_SHAPE),
        "intra_op_num_threads": 4,
        "inter_op_num_threads": 1,
        "use_cuda": False,
        "use_dml": False,
    }
    recognizer = module.TextRecognizer(config)
    chars = recognizer.postprocess_op.character
    required = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_()"
    if not all(char in chars for char in required):
        raise PinNameOCRError("PP-OCR name model dictionary lacks required ASCII characters")
    metadata = {
        "model_source": "rapidocr_onnxruntime package/models/ch_PP-OCRv4_rec_infer.onnx",
        "model_path": str(model_path),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "upstream": "https://github.com/RapidAI/RapidOCR",
        "package_version": version,
        "dependency_versions": {name: _dependency_version(name) for name in
                                ("rapidocr-onnxruntime", "onnxruntime", "numpy", "opencv-python", "opencv-python-headless")},
        "dictionary_source": "ONNX model metadata character list",
        "dictionary_count": len(chars),
        "dictionary_sha256": hashlib.sha256("\n".join(chars).encode("utf8")).hexdigest(),
        "character_support": {text: all(char in chars for char in text) for text in
                              ("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789", "_", "(", ")")},
        "config": config,
        "detector_used": False,
        "angle_classifier_used": False,
        "providers": recognizer.session.session.get_providers(),
    }
    return PinNameOCR(recognizer, metadata)


def get_pin_name_ocr(mode="auto"):
    """Return a verified name provider, None on auto fallback, or fail on on.

    ``off`` returns without importing the optional OCR package. Failed auto
    loads are cached and warned once. An explicit ``on`` always reports a
    prior or current load failure instead of silently dropping the provider.
    """
    global _provider, _load_error, _warning_emitted
    if mode not in ("off", "auto", "on"):
        raise ValueError("pin name OCR mode must be off, auto or on")
    if mode == "off":
        return None
    if _provider is _UNSET:
        try:
            _provider = _load_provider()
        except Exception as exc:
            _load_error = exc
            _provider = None
    if _provider is None:
        message = f"Optional pin name OCR is unavailable; using existing OCR: {_load_error}"
        if mode == "on":
            raise PinNameOCRError(message) from _load_error
        if not _warning_emitted:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            _warning_emitted = True
    return _provider
