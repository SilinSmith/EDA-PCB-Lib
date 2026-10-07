#!/usr/bin/env python3
"""通过阿里云盘匿名分享接口，列出/下载 PCBnet Dataset300 全量数据。

分享来源：https://www.alipan.com/s/HYuvtbNZf3q  密码 89di
（PCBnet 官方 README 给出的镜像链接，见 https://github.com/ECNUACRush/PCBnet/）

用法：
    python alipan_download.py --list                 # 只列出清单并写 manifest
    python alipan_download.py --download             # 下载 png/json/eds/net
    python alipan_download.py --download --only-png-json   # 只下 png + CVJsonStd.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SHARE_ID = "HYuvtbNZf3q"
SHARE_PWD = "89di"
API = "https://api.aliyundrive.com"
OUT_ROOT = Path(r"Y:\workspace\EDA\PCB训练集\_raw\Dataset300")
MANIFEST = Path(r"Y:\workspace\EDA\work\recon\dataset300_manifest.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
    "Referer": "https://www.alipan.com/",
    "Origin": "https://www.alipan.com",
}


def get_share_token() -> str:
    r = requests.post(
        f"{API}/v2/share_link/get_share_token",
        headers=HEADERS,
        json={"share_id": SHARE_ID, "share_pwd": SHARE_PWD},
        timeout=30,
    )
    r.raise_for_status()
    tok = r.json().get("share_token")
    if not tok:
        raise RuntimeError(f"no share_token: {r.text[:300]}")
    return tok


def list_dir(session: requests.Session, share_token: str, parent_file_id: str) -> list[dict]:
    items: list[dict] = []
    marker = ""
    while True:
        body = {
            "share_id": SHARE_ID,
            "parent_file_id": parent_file_id,
            "limit": 100,
            "order_by": "name",
            "order_direction": "ASC",
            "fields": "*",
        }
        if marker:
            body["marker"] = marker
        r = session.post(
            f"{API}/v2/file/list",
            headers={**HEADERS, "x-share-token": share_token},
            json=body,
            timeout=30,
        )
        r.raise_for_status()
        js = r.json()
        items.extend(js.get("items", []))
        marker = js.get("next_marker") or ""
        if not marker:
            break
    return items


CANDIDATE_DL_ENDPOINTS = (
    "/v2/file/get_share_link_download_url",
    "/v2/file/get_download_url",
    "/adrive/v3/file/get_share_link_download_url",
)


def get_download_url(session: requests.Session, share_token: str, file_id: str) -> str:
    last = ""
    for ep in CANDIDATE_DL_ENDPOINTS:
        try:
            r = session.post(
                f"{API}{ep}",
                headers={**HEADERS, "x-share-token": share_token},
                json={"share_id": SHARE_ID, "file_id": file_id, "expire_sec": 900},
                timeout=30,
            )
            if r.status_code == 200:
                url = r.json().get("url")
                if url:
                    return url
            last = f"{ep} -> {r.status_code} {r.text[:160]}"
        except Exception as e:  # noqa: BLE001
            last = f"{ep} -> {e}"
    raise RuntimeError(f"no download url; last={last}")


def build_manifest(session: requests.Session, share_token: str) -> list[dict]:
    root = list_dir(session, share_token, "root")
    print(f"[root] {len(root)} item(s): " + ", ".join(i['name'] for i in root))
    case_dirs: list[dict] = []
    for top in root:
        if top["type"] != "folder":
            continue
        subs = list_dir(session, share_token, top["file_id"])
        print(f"[{top['name']}] {len(subs)} case folders")
        for case in subs:
            if case["type"] != "folder":
                case_dirs.append({"case": case["name"], "kind": "file", **case})
                continue
            files = list_dir(session, share_token, case["file_id"])
            case_dirs.append(
                {
                    "case": case["name"],
                    "kind": "dir",
                    "file_id": case["file_id"],
                    "files": [
                        {"name": f["name"], "size": f.get("size"), "file_id": f["file_id"]}
                        for f in files
                    ],
                }
            )
    return case_dirs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--only-png-json", action="store_true")
    args = ap.parse_args()

    session = requests.Session()
    tok = get_share_token()
    print(f"share_token acquired (len={len(tok)})")

    with_manifest = MANIFEST.exists()
    if with_manifest and not args.list:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        print(f"loaded manifest: {len(manifest)} cases")
    else:
        manifest = build_manifest(session, tok)
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"manifest written: {MANIFEST}  ({len(manifest)} cases)")

    total_bytes = 0
    per_case = []
    for ent in manifest:
        fs = ent.get("files", [])
        s = sum(f["size"] or 0 for f in fs)
        total_bytes += s
        per_case.append((ent["case"], len(fs), s))
    print(f"\ntotal files size = {total_bytes/1e6:.1f} MB over {len(per_case)} cases")
    for name, n, s in per_case[:8]:
        print(f"   {name}: {n} files, {s/1e6:.2f} MB")
    print("   ...")

    if not args.download:
        return 0

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    want_ext = (".png", ".json") if args.only_png_json else (".png", ".json", ".eds", ".net", ".jpg")
    ok = skipped = failed = 0
    for ent in manifest:
        if ent.get("kind") != "dir":
            continue
        cdir = OUT_ROOT / ent["case"]
        cdir.mkdir(parents=True, exist_ok=True)
        for f in ent.get("files", []):
            if want_ext and Path(f["name"]).suffix.lower() not in want_ext:
                continue
            dst = cdir / f["name"]
            if dst.exists() and dst.stat().st_size == (f["size"] or -1):
                skipped += 1
                continue
            try:
                url = get_download_url(session, tok, f["file_id"])
                with session.get(url, headers=HEADERS, timeout=180, stream=True) as r:
                    r.raise_for_status()
                    tmp = dst.with_suffix(dst.suffix + ".part")
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                    tmp.replace(dst)
                ok += 1
                print(f"  ok   {ent['case']}/{f['name']}  {(f['size'] or 0)/1e6:.2f} MB")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  FAIL {ent['case']}/{f['name']}: {e}")
                time.sleep(1)
    print(f"\ndownload done: ok={ok} skipped={skipped} failed={failed} -> {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
