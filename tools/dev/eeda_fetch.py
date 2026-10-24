# -*- coding: utf-8 -*-
"""Fetch EasyEDA/LCSC symbols by part name and merge them into a local store.

The competition pages were drawn in 嘉立创EDA, and GT pin points equal the
library pin connection ends exactly (verified: scale 2.0 px per 10-mil unit,
residual <= 0.7 px).  So the pipeline needs the symbol for every IC name.

Cache/merge file: ``work/_eda_online.json`` -> {name_lower: [variant, ...]}.

Usage:
    python -X utf8 eda-pcb/tools/dev/eeda_fetch.py --names work/_need_names.json
    python -X utf8 eda-pcb/tools/dev/eeda_fetch.py AMS1117-3.3 MT2492
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
EDA = os.path.abspath(os.path.join(HERE, "..", ".."))
WS = os.path.dirname(EDA)
OUT = os.path.join(WS, "work", "_eda_online.json")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    " (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}


def http_json(url, tries=3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as exc:                               # noqa: BLE001
            last = exc
            time.sleep(1.0 + i)
    raise last


def search(mpn, limit=6):
    u = ("https://easyeda.com/api/eda/product/list?keyword="
         f"{urllib.parse.quote(mpn)}&currPage=1&pageSize=20")
    try:
        res = http_json(u).get("result") or {}
    except Exception as exc:                                  # noqa: BLE001
        print(f"    search fail {mpn}: {exc}")
        return []
    exact, other = [], []
    low = mpn.strip().lower()
    for it in res.get("productList") or []:
        code = it.get("number")
        if not code:
            continue
        m = (it.get("mpn") or "").strip().lower()
        (exact if m == low else other).append((code, it.get("mpn"), it.get("package")))
    return (exact + other)[:limit]


def symbol_of(code):
    data = http_json(f"https://easyeda.com/api/products/{code}/components")
    res = data.get("result") or {}
    ds = res.get("dataStr")
    if isinstance(ds, str):
        import base64
        import gzip
        s = ds[6:] if ds.startswith("base64") else ds
        raw = base64.b64decode(s)
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        ds = json.loads(raw.decode("utf-8", "replace"))
    pins, rect = {}, None
    for sh in ds.get("shape") or []:
        if not isinstance(sh, str):
            continue
        if sh.startswith("P~"):
            f = sh.split("~")
            try:
                pins[str(f[3])] = [float(f[4]), float(f[5])]
            except (IndexError, ValueError):
                continue
        elif sh.startswith("R~") and rect is None:
            f = sh.split("~")
            try:
                rect = [float(f[1]), float(f[2]),
                        float(f[1]) + float(f[5]), float(f[2]) + float(f[6])]
            except (IndexError, ValueError):
                rect = None
    return {"src": f"lcsc:{code}", "lcsc": code, "title": res.get("title"),
            "pins": pins, "rect": rect}


def load_store():
    if os.path.isfile(OUT):
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_store(store):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)


def main(argv):
    names = []
    if len(argv) >= 2 and argv[0] == "--names":
        with open(argv[1], encoding="utf-8-sig") as f:
            names = json.load(f)
    else:
        names = argv
    names = [n.strip() for n in names if isinstance(n, str) and n.strip()]
    store = load_store()
    for name in names:
        low = name.lower()
        if store.get(low):
            print(f"  = {name}: cached ({len(store[low])} variants)")
            continue
        cands = search(name)
        print(f"  + {name}: {len(cands)} candidates {[c[0] for c in cands]}")
        variants = []
        for code, mpn, pkg in cands:
            try:
                sym = symbol_of(code)
            except Exception as exc:                          # noqa: BLE001
                print(f"      {code}: fetch fail {type(exc).__name__}")
                continue
            if not sym["pins"]:
                continue
            sym["mpn"] = mpn
            sym["package"] = pkg
            variants.append(sym)
            time.sleep(0.2)
        if variants:
            store[low] = variants
            save_store(store)
        print(f"    kept {len(variants)} symbol variants")
    print(f"store: {len(store)} names -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
