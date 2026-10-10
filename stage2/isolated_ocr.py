"""Read a complete observed glyph without borrowing pixels from neighbouring text.

Only new pin-lane reads use this adapter.  Original-image geometry, recognition
rotation and synthetic white padding are separate.  Each sample is read alone:
CRNN's bidirectional recurrent context otherwise depends on the longest crop in
its batch, even when CTC decoding stops at the sample's own effective width.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np
from PIL import Image

from textrec import config as C


DEFAULT_PADDING = 3
SAMPLING_VERSION = "isolated-glyph-single-v1"


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _source_gray(value):
    if isinstance(value, Image.Image):
        value = np.asarray(value.convert("L"))
    value = np.asarray(value)
    if value.ndim != 2 or value.size == 0 or value.dtype != np.uint8:
        raise ValueError("gray must be a nonempty uint8 grayscale image")
    return value


def prepare_isolated_sample(sample, *, img_h=None, max_w=None):
    """Return ``(white_canvas, local_read_bbox, sampling_metadata)``.

    ``sample.gray`` is a grayscale source or a complete glyph crop, and ``bbox``
    uses top-left pixels in that array.  The caller must include *all* observed
    character strokes in this bbox.  Pixels outside it are never copied.
    ``rotation`` is counter-clockwise degrees, in 90 degree increments.  Fixed
    synthetic ``padding`` defaults to 3 pixels; it never samples the source.
    """
    # Keep the adapter importable when OCR weights/runtime are unavailable.
    # Preparation runs only after the caller has a usable recognizer.
    from textrec.dataset import crop_gray

    if not isinstance(sample, Mapping):
        raise TypeError("sample must be a mapping")
    gray = _source_gray(sample["gray"])
    try:
        bbox = [float(v) for v in sample["bbox"]]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bbox must contain four finite coordinates") from exc
    if (len(bbox) != 4 or not all(math.isfinite(v) for v in bbox)
            or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
        raise ValueError("bbox must contain four finite ordered coordinates")
    rotation = sample.get("rotation", 0)
    if isinstance(rotation, bool) or rotation not in (0, 90, -90, 180, -180, 270, -270):
        raise ValueError("rotation must be a multiple of 90 degrees")
    padding = sample.get("padding", DEFAULT_PADDING)
    if (isinstance(padding, bool) or not isinstance(padding, (int, np.integer))
            or not 0 <= padding <= 64):
        raise ValueError("padding must be an integer between 0 and 64")
    padding = int(padding)
    x0, y0 = max(0, math.floor(bbox[0])), max(0, math.floor(bbox[1]))
    x1, y1 = min(gray.shape[1], math.ceil(bbox[2])), min(gray.shape[0], math.ceil(bbox[3]))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox does not intersect gray")
    glyph = np.ascontiguousarray(gray[y0:y1, x0:x1].copy())
    source_shape = list(glyph.shape)
    glyph = np.ascontiguousarray(np.rot90(glyph, int(rotation) // 90))
    padded = np.pad(glyph, padding, constant_values=255)

    # crop_gray expands its bbox and optionally probes/reframes the ink band.
    # Reserve enough WHITE space even for that second adaptive expansion; the
    # isolated canvas contains no neighbouring source pixels at any distance.
    h = padded.shape[0]
    my = max(0., float(C.PAD_Y_FRAC)) * h
    mx = max(0., float(C.PAD_X_FRAC)) * h
    probe = max(float(C.INK_PROBE_PAD), float(C.INK_PROBE_PAD_FRAC) * (h + 2 * my))
    frame = max(1., float(C.INK_FRAME))
    guard = max(16, int(math.ceil(max(mx + 2, my + probe + frame * (h + 2 * my + 2 * probe) / 2 + 2))))
    canvas = np.pad(padded, guard, constant_values=255)
    ph, pw = padded.shape
    read_bbox = [guard, guard, guard + pw, guard + ph]
    normalized = np.asarray(crop_gray(Image.fromarray(canvas), read_bbox,
                                     img_h or C.IMG_H, max_w or C.MAX_W))
    meta = {
        "version": SAMPLING_VERSION,
        "strategy": "single",
        "source_bbox": bbox,
        "sampled_bbox": [x0, y0, x1, y1],
        "source_glyph_shape": source_shape,
        "rotated_glyph_shape": list(glyph.shape),
        "rotation": int(rotation),
        "padding": padding,
        "guard": guard,
        "read_bbox": read_bbox,
        "glyph_sha256": _sha(glyph),
        "model_input_shape": list(normalized.shape),
        "model_input_sha256": _sha(normalized),
        "preprocessing": {key: getattr(C, key) for key in (
            "PAD_X_FRAC", "PAD_Y_FRAC", "INK_YNORM", "INK_PROBE_PAD",
            "INK_PROBE_PAD_FRAC", "INK_FRAME")},
    }
    if "variant" in sample:
        meta["variant"] = str(sample["variant"])
    return canvas, read_bbox, meta


def read_isolated_many(recognizer, samples):
    """Read samples independently; return ordered text/confidence/metadata dicts.

    A sample contains ``gray``, ``bbox``, ``role`` and optional ``rotation`` and
    ``padding``.  No highest-confidence selection or expected-text correction
    occurs here.  Empty input returns an empty list.  The singleton calls are
    intentional: adding or reordering unrelated long/short samples cannot alter
    the tensor width or recurrent context of an existing sample.
    """
    out = []
    for sample in samples:
        canvas, bbox, meta = prepare_isolated_sample(
            sample, img_h=getattr(recognizer, "img_h", C.IMG_H),
            max_w=getattr(recognizer, "max_w", C.MAX_W))
        role = sample["role"]
        result = recognizer.read_many(canvas, [(bbox, role)])
        if len(result) != 1:
            raise ValueError("isolated recognizer must return one result per sample")
        text, confidence = result[0]
        meta["role"] = role
        out.append({"text": str(text), "confidence": float(confidence), "sampling": meta})
    return out
