# -*- coding: utf-8 -*-
"""Extract symbol pin geometry from Altium ``.SchLib`` (and ``.IntLib``).

Format reference: docs/SCHLIB_FORMAT (FileHeader + per-symbol ``Data`` streams;
text records are pipe-delimited, pins are usually *binary* type-1 records).

Writes ``work/_symlib_altium.json``::

    {"<lib file>": {"<symbol>": {"pins": [{"num": "7", "name": "GPIO0",
                                           "len": 30, "orient": "R",
                                           "spot_x": 160, "spot_y": -250,
                                           "body_x": 130, "body_y": -250}],
                                 "body": [x1, y1, x2, y2] | null,
                                 "desc": "..."}}}

Raw integer coordinates (1 unit = 1/10 grid) are kept: the page-space scale and
placement are calibrated later against ground-truth pin points, never assumed.

Usage:
    python -X utf8 eda-pcb/tools/dev/schlib_extract.py [src_dir ...]
"""
from __future__ import annotations

import glob
import json
import os
import struct
import sys

import olefile

HERE = os.path.dirname(os.path.abspath(__file__))
EDA = os.path.abspath(os.path.join(HERE, "..", ".."))
WS = os.path.dirname(EDA)
OUT = os.path.join(WS, "work", "_symlib_altium.json")

ORIENT = {(0, 0): "R", (0, 1): "L", (1, 0): "U", (1, 1): "D"}
DIRV = {"R": (1, 0), "L": (-1, 0), "U": (0, 1), "D": (0, -1)}


def _num(fields: dict, key: str, default: float = 0.0) -> float:
    v = fields.get(key)
    if v in (None, "", "%UTF8%"):
        return default
    try:
        val = float(v)
    except ValueError:
        return default
    frac = fields.get(key + "_Frac")
    if frac:
        try:
            val += float(frac) / 100000.0
        except ValueError:
            pass
    return val


def _text(fields: dict, key: str) -> str:
    v = fields.get(key, "") or ""
    return v.replace("%UTF8%", "") if isinstance(v, str) else v


def _decode(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", "ignore")
    txt = raw.decode("latin-1", "ignore")
    if txt.count("\x00") > max(1, len(txt) // 3):
        try:
            txt = raw.decode("utf-16-le", "ignore")
        except UnicodeDecodeError:
            pass
    return txt


def _fields(payload: bytes) -> dict:
    txt = _decode(payload).replace("\x00", "").replace("\r", "").replace("\n", "")
    out = {}
    for part in txt.split("|"):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
    return out


def _pascal(buf: bytes, off: int):
    n = buf[off]
    return buf[off + 1: off + 1 + n].decode("cp1252", "ignore"), off + 1 + n


def _parse_binary_pin(buf: bytes) -> dict | None:
    """Type-1 pin record; see SCHLIB_FORMAT § Binary Pin Records."""
    if len(buf) < 16:
        return None
    try:
        desc, o = _pascal(buf, 12)                     # description (N bytes)
        formal = buf[o]
        elec = buf[o + 1]
        flags = buf[o + 2]
        length, xx, yy = struct.unpack_from("<hhh", buf, o + 3)
        o += 9
        colour = struct.unpack_from("<I", buf, o)[0]
        o += 4
        name, o = _pascal(buf, o)
        num, o = _pascal(buf, o)
    except (IndexError, struct.error):
        return None
    orient = ORIENT[(1 if flags & 0x01 else 0, 1 if flags & 0x02 else 0)]
    dx, dy = DIRV[orient]
    return {
        "num": num, "name": name, "len": length, "orient": orient,
        "spot_x": xx, "spot_y": yy,
        "body_x": xx - dx * length, "body_y": yy - dy * length,
        "elec": elec, "formal": formal, "flags": flags, "colour": colour,
    }


def _rect_of(fields: dict):
    for xk, yk, xk2, yk2 in (("Location.X", "Location.Y", "Corner.X", "Corner.Y"),):
        if all(k in fields for k in (xk, yk, xk2, yk2)):
            x1, y1 = _num(fields, xk), _num(fields, yk)
            x2, y2 = _num(fields, xk2), _num(fields, yk2)
            if x1 != x2 and y1 != y2:
                return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return None


def parse_data_stream(raw: bytes) -> dict:
    sym = {"pins": [], "body": None, "desc": ""}
    off = 0
    n = len(raw)
    while off + 4 <= n:
        word = struct.unpack_from("<I", raw, off)[0]
        ln = word & 0xFFFFFF
        kind = (word >> 24) & 0xFF
        off += 4
        if ln == 0 or off + ln > n:
            break
        payload = raw[off: off + ln]
        off += ln
        if kind == 0x01:
            pin = _parse_binary_pin(payload)
            if pin:
                sym["pins"].append(pin)
            continue
        fields = _fields(payload)
        rec = fields.get("RECORD", "")
        if rec == "14":
            box = _rect_of(fields)
            if box is None:                    # zero-height/width rects are lines
                continue
            sym["body"] = box if sym["body"] is None else sym["body"]
        elif rec == "34" and not sym["desc"]:
            sym["desc"] = _text(fields, "Text")
        elif rec == "41" and not sym["desc"]:
            sym["desc"] = _text(fields, "Name")
    return sym


def _section_map(ole) -> dict:
    """SectionKeys stream maps long symbol names to short storage names."""
    out = {}
    try:
        raw = ole.openstream("SectionKeys").read()
    except OSError:
        return out
    cur = {}
    for chunk in raw.split(b"\x00"):
        txt = chunk.decode("latin-1", "ignore")
        if "|" not in txt:
            continue
        for part in txt.split("|"):
            if "=" in part:
                k, _, v = part.partition("=")
                cur[k.strip()] = v.strip()
        if "LibRef" in cur and "SectionKey" in cur:
            out[cur["SectionKey"]] = cur["LibRef"]
            cur = {}
    return out


def parse_file(path: str) -> dict | None:
    if not olefile.isOleFile(path):
        return None
    ole = olefile.OleFileIO(path)
    try:
        names = ole.listdir()
        secmap = _section_map(ole)
        out = {}
        for entry in names:
            if entry[-1] != "Data" or len(entry) < 2:
                continue
            key = entry[0]
            try:
                raw = ole.openstream(entry).read()
            except OSError:
                continue
            sym = parse_data_stream(raw)
            if not sym["pins"]:
                continue
            out[secmap.get(key, key)] = sym
        return out or None
    finally:
        ole.close()


def collect(srcs) -> dict:
    files: list[str] = []
    for s in srcs:
        if os.path.isdir(s):
            for ext in ("*.SchLib", "*.IntLib", "*.schlib"):
                files += glob.glob(os.path.join(s, "**", ext), recursive=True)
        else:
            files.append(s)
    out = {}
    for path in sorted(set(files)):
        try:
            sym = parse_file(path)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  FAIL {os.path.basename(path)}: {exc}")
            continue
        if not sym:
            print(f"  --   {os.path.basename(path)}: no symbol data")
            continue
        npins = sum(len(v["pins"]) for v in sym.values())
        print(f"  {os.path.basename(path)}: {len(sym)} symbols / {npins} pins")
        out[os.path.basename(path)] = sym
    return out


def main():
    srcs = sys.argv[1:] or [os.path.join(WS, "symbol_lib")]
    print(f"sources: {srcs}")
    data = collect(srcs)
    if not data:
        print("no symbol data found")
        return 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
