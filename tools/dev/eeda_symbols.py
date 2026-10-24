# -*- coding: utf-8 -*-
"""Build a symbol store (name -> pin geometry) from 嘉立创EDA / EasyEDA Pro files.

Sources accepted under ``symbol_lib/``:
  * ``*.eprj2``            SQLite project (``components.dataStr`` = symbol JSON-lines)
  * ``*.epro`` / ``*.zip`` ZIP bundle (``project.json`` + ``SYMBOL/<uuid>.esym``)
  * ``*.epro`` single symbol export (``SYMBOL`` at root, no project.json)

Output ``work/_eda_symbols.json``: {name_lower: [variant, ...]} where variant is
``{src, title, lcsc, npins, pins: {num: [x, y]}, rect: [x1, y1, x2, y2] | null}``.
Library units are EasyEDA's 10-mil units; pin (x, y) is the *connection* end.
"""
from __future__ import annotations

import base64
import glob
import gzip
import json
import os
import sqlite3
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
EDA = os.path.abspath(os.path.join(HERE, "..", ".."))
WS = os.path.dirname(EDA)
LIB = os.path.join(WS, "symbol_lib")
OUT = os.path.join(WS, "work", "_eda_symbols.json")


def _body(s: str) -> str:
    if s.startswith("base64"):
        s = s[6:]
    raw = base64.b64decode(s)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")


def parse_symbol_text(body: str) -> dict:
    """EasyEDA symbol JSON-lines -> {pins, rect, title}.

    Handles both the Pro (``1.1``) and old (``1.0``) line formats.
    """
    raw_pins, num_of, rect, title, ppm = {}, {}, None, None, None
    for line in body.replace("\r", "").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        tag = rec[0]
        if tag == "PIN":
            raw_pins[rec[1]] = (float(rec[4]), float(rec[5]))
        elif tag == "RECT" and rect is None:
            rect = [float(rec[2]), float(rec[3]), float(rec[4]), float(rec[5])]
        elif tag == "ATTR":
            if rec[3] == "NUMBER":
                num_of[rec[2]] = str(rec[4])
            elif rec[3] == "Symbol" and len(rec) > 4:
                title = title or rec[4]
    pins = {}
    for eid, xy in raw_pins.items():
        num = num_of.get(eid)
        if num is not None:
            pins[num] = xy
    return {"pins": pins, "rect": rect, "title": title}


def from_eprj2(path: str) -> list[dict]:
    out = []
    try:
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = c.execute("select title, dataStr from components").fetchall()
        c.close()
    except sqlite3.Error as exc:
        print(f"   ! {os.path.basename(path)}: {exc}")
        return out
    for title, ds in rows:
        if not isinstance(ds, str):
            continue
        try:
            sym = parse_symbol_text(_body(ds))
        except Exception:                                      # noqa: BLE001
            continue
        if not sym["pins"]:
            continue
        out.append({"src": os.path.basename(path), "title": title, **sym})
    return out


def from_zip(path: str) -> list[dict]:
    out = []
    try:
        z = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile):
        return out
    names = z.namelist()
    meta = {}
    if "project.json" in names:
        try:
            pj = json.loads(z.read("project.json").decode("utf-8", "replace"))
        except json.JSONDecodeError:
            pj = {}
        for sec, key in (("symbols", "title"), ("devices", "title")):
            for uid, m in (pj.get(sec) or {}).items():
                if isinstance(m, dict) and m.get(key):
                    meta[uid] = m[key]
        # device attributes carry the LCSC code
        for uid, m in (pj.get("devices") or {}).items():
            attrs = m.get("attributes") or {}
            code = attrs.get("Supplier Part")
            if code and uid in meta:
                meta[uid] = (meta[uid], code)
    for n in names:
        if not n.upper().startswith("SYMBOL/") or "." not in n:
            continue
        uid = os.path.splitext(os.path.basename(n))[0]
        try:
            sym = parse_symbol_text(z.read(n).decode("utf-8", "replace"))
        except Exception:                                      # noqa: BLE001
            continue
        if not sym["pins"]:
            continue
        mt = meta.get(uid)
        title, lcsc = (mt if isinstance(mt, tuple) else (mt, None))
        title = title or sym.get("title") or uid
        out.append({"src": os.path.basename(path), "title": title, "lcsc": lcsc, **sym})
    z.close()
    return out


def dedupe(variants: list[dict]) -> list[dict]:
    seen, out = set(), []
    for v in variants:
        sig = (tuple(sorted((k, round(x, 2), round(y, 2)) for k, (x, y) in v["pins"].items())),
               tuple(round(q, 2) for q in v["rect"]) if v["rect"] else None)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(v)
    return out


def main() -> int:
    files = sorted(glob.glob(os.path.join(LIB, "**", "*"), recursive=True))
    proj = [p for p in files if p.lower().endswith(".eprj2")]
    zips = [p for p in files if p.lower().endswith((".epro", ".zip", ".eprj_backup"))]
    store: dict[str, list[dict]] = {}
    n_sym = 0
    for p in proj + zips:
        got = from_eprj2(p) if p.lower().endswith(".eprj2") else from_zip(p)
        n_sym += len(got)
        for v in got:
            t = (v.get("title") or "").strip().lower()
            if not t:
                continue
            store.setdefault(t, []).append(v)
    for t in list(store):
        store[t] = dedupe(store[t])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)
    print(f"files: {len(proj)} proj + {len(zips)} zip")
    print(f"symbols parsed: {n_sym}; unique names: {len(store)}; "
          f"total variants: {sum(len(v) for v in store.values())}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
