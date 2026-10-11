"""Prepare a read-only, source-traceable pin OCR review set.

Labels copied from CVJsonStd are candidates, never certified ground truth.
All duplicate rules are conservative heuristics: no old model training manifest
is available, so model-training independence must remain unknown.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_image(path):
    a = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if a is None:
        raise ValueError(f"Cannot load image: {path}")
    return a


def fingerprint(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = gray < 220
    yy, xx = np.nonzero(ink)
    if xx.size:
        gray = gray[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(thumb)[:8, :8].ravel()[1:]
    bits = dct > np.median(dct)
    phash = sum(int(b) << k for k, b in enumerate(bits))
    mask = cv2.resize(ink.astype(np.uint8), (128, 128), interpolation=cv2.INTER_AREA)
    return phash, mask


def label_evidence(doc):
    refs = collections.defaultdict(list)
    for cid, comp in (doc.get("components") or {}).items():
        for pid, pin in (comp.get("pins") or {}).items():
            for field, role in (("pinNameByOCR", "pin_name"), ("pinNumberByOCR", "pin_number")):
                tid = pin.get(field)
                if tid:
                    refs[tid].append({"component_id": cid, "pin_id": pid, "field": field, "role": role,
                                      "tip": pin.get("tip"), "base": pin.get("base")})
    rows = []
    for tid, text in (doc.get("texts") or {}).items():
        details = str(text.get("details") or "").strip().lower().replace("_", "")
        role = {"pinname": "pin_name", "pinnumber": "pin_number"}.get(details)
        if role is None:
            unique = {x["role"] for x in refs[tid]}
            if len(unique) == 1:
                role = unique.pop()
        if role is None:
            continue
        content = str(text.get("content") or "").strip()
        loc = text.get("location") or {}
        try:
            box = [float(loc[k]) for k in ("x1", "y1", "x2", "y2")]
        except (KeyError, ValueError, TypeError):
            continue
        if not content or len(content) > 24 or not all(math.isfinite(v) for v in box):
            continue
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        rows.append({"text_id": tid, "text": content, "role": role, "bbox": box,
                     "label_evidence": {"json_pointer": f"/texts/{tid}", "original_record": text,
                                        "component_pin_references": refs[tid]},
                     "label_status": "unverified_source_annotation",
                     "human_verified": False})
    return rows


def source_kind(path):
    stem = path.stem.lower()
    for name in ("datasheet", "altium", "kicad", "jlc", "other"):
        if name in stem:
            return name
    return "unreported_source"


def model_tokens(names):
    """Conservative model-family clues, not proof of same schematic."""
    found = set()
    for name in names:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{5,31}", name):
            norm = re.sub(r"[^A-Z0-9]", "", token.upper())
            if len(norm) >= 6 and sum(c.isdigit() for c in norm) >= 2:
                found.add(norm)
    return found


def counter_jaccard(a, b):
    keys = set(a) | set(b)
    denom = sum(max(a[k], b[k]) for k in keys)
    return sum(min(a[k], b[k]) for k in keys) / max(denom, 1)


class Groups:
    def __init__(self, n):
        self.parents = list(range(n))

    def get(self, n):
        while self.parents[n] != n:
            self.parents[n] = self.parents[self.parents[n]]
            n = self.parents[n]
        return n

    def join(self, a, b):
        self.parents[self.get(b)] = self.get(a)


def sample_rows(rows, count):
    selected = []
    for role in ("pin_name", "pin_number"):
        pool = [x for x in rows if x["role"] == role]
        pool.sort(key=lambda r: (r["bbox"][3] - r["bbox"][1], len(r["text"]), r["text_id"]))
        # Spread heights and string lengths, with a cap on repeated same labels.
        unique = []
        seen = collections.Counter()
        for item in pool:
            if seen[item["text"]] >= 2:
                continue
            unique.append(item)
            seen[item["text"]] += 1
        n = min(count // 2, len(unique))
        if n:
            selected.extend(unique[int(i)] for i in np.linspace(0, len(unique) - 1, n))
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--per-case", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    before = {}
    for jf in sorted(args.dataset.rglob("*CVJsonStd.json")):
        images = [p for p in jf.parent.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if len(images) != 1:
            continue
        image_path = images[0]
        doc = json.loads(jf.read_text(encoding="utf-8-sig"))
        image = load_image(image_path)
        phash, mask = fingerprint(image)
        text_bag = collections.Counter(str(t.get("content") or "").strip() for t in doc.get("texts", {}).values()
                                       if str(t.get("content") or "").strip())
        search_names = [str(t.get("content")) for t in doc.get("texts", {}).values()
                        if t.get("details") == "searchName"]
        entry = {"case": jf.parent.name, "json": str(jf), "image": str(image_path),
                 "json_sha256": sha(jf), "image_sha256": sha(image_path),
                 "width": image.shape[1], "height": image.shape[0], "phash": phash,
                 "source_kind": source_kind(jf), "search_names": search_names,
                 "rows": label_evidence(doc), "text_bag": text_bag, "mask": mask}
        entries.append(entry)
        before[str(jf)] = entry["json_sha256"]
        before[str(image_path)] = entry["image_sha256"]
    gt_models = {}
    for p in sorted(args.gt.rglob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8-sig"))
        gt_models[p.parent.name] = model_tokens([str(c.get("Name") or "") for c in doc.get("components", {}).values()
                                               if c.get("type") == "box"])
        before[str(p)] = sha(p)
    gt = []
    for p in sorted(args.gt.rglob("*")):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            im = load_image(p)
            phash, _ = fingerprint(im)
            gt.append({"case": p.parent.name, "image": str(p), "sha256": sha(p), "phash": phash})
            before[str(p)] = sha(p)
    groups = Groups(len(entries))
    duplicate_links = []
    for i, a in enumerate(entries):
        for k in range(i):
            b = entries[k]
            hd = (a["phash"] ^ b["phash"]).bit_count()
            text_sim = counter_jaccard(a["text_bag"], b["text_bag"])
            reasons = []
            if a["image_sha256"] == b["image_sha256"]:
                reasons.append("exact_image_sha256")
            if hd <= 8:
                reasons.append("cropped_foreground_phash_hamming_le8")
            if len(a["text_bag"]) >= 12 and len(b["text_bag"]) >= 12 and text_sim >= 0.80:
                reasons.append("multiset_source_text_jaccard_ge080")
            if reasons:
                groups.join(i, k)
                duplicate_links.append({"a": a["case"], "b": b["case"], "reasons": reasons,
                                        "phash_hamming": hd, "text_jaccard": round(text_sim, 4)})
    excluded_groups = set()
    gt_links = []
    for i, a in enumerate(entries):
        for b in gt:
            hd = (a["phash"] ^ b["phash"]).bit_count()
            source_models = model_tokens(a["search_names"])
            possible_models = sorted({x for x in source_models for y in gt_models.get(b["case"], set())
                                      if x == y or (min(len(x), len(y)) >= 6 and (x.startswith(y) or y.startswith(x)))})
            if a["image_sha256"] == b["sha256"] or hd <= 12 or possible_models:
                excluded_groups.add(groups.get(i))
                gt_links.append({"dataset_case": a["case"], "gt_case": b["case"],
                                 "exact_image": a["image_sha256"] == b["sha256"], "phash_hamming": hd,
                                 "conservative_shared_model_family": possible_models})
    members = collections.defaultdict(list)
    for i, e in enumerate(entries):
        e["group"] = f"family_{groups.get(i):03d}"
        e["excluded_due_to_gt_group"] = groups.get(i) in excluded_groups
        members[e["group"]].append(e["case"])
    # Do not select the first fifty previously examined official examples.
    # This cannot undo unknown use in the checkpoint's historical training.
    candidates = [e for e in entries if not e["excluded_due_to_gt_group"] and
                  int(e["case"]) > 50 and len(sample_rows(e["rows"], args.per_case)) >= 12]
    chosen = []
    chosen_groups = set()
    sources = collections.Counter()
    while candidates and len(chosen) < args.cases:
        candidates.sort(key=lambda e: (sources[e["source_kind"]],
                         -min(len(e["rows"]), 50), int(e["case"])))
        item = candidates.pop(0)
        if item["group"] in chosen_groups:
            continue
        chosen.append(item)
        chosen_groups.add(item["group"])
        sources[item["source_kind"]] += 1
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    manifest = []
    page_links = []
    for entry in chosen:
        case_dir = args.out / entry["case"]
        case_dir.mkdir(exist_ok=True)
        image = Image.open(entry["image"]).convert("RGB")
        selected = sample_rows(entry["rows"], args.per_case)
        sheet = Image.new("RGB", (1200, 180 + 200 * math.ceil(len(selected) / 3)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 15), f"{entry['case']}  {entry['source_kind']}  {entry['width']}x{entry['height']}  SOURCE LABELS UNVERIFIED", font=font, fill="black")
        draw.text((20, 48), "左侧：原始文本框 ×4；右侧：邻域保留符号，红框为原始标注", font=font, fill="black")
        overlay = image.copy()
        od = ImageDraw.Draw(overlay)
        trs = []
        for idx, row in enumerate(selected):
            row = dict(row)
            x1, y1, x2, y2 = row["bbox"]
            bbox = [max(0, math.floor(x1)), max(0, math.floor(y1)),
                    min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2))]
            crop = image.crop(bbox)
            crop_path = case_dir / f"{row['text_id']}_{row['role']}.png"
            crop.save(crop_path)
            context_box = [max(0, bbox[0] - 12), max(0, bbox[1] - 12),
                           min(image.width, bbox[2] + 12), min(image.height, bbox[3] + 12)]
            context = image.crop(context_box)
            cd = ImageDraw.Draw(context)
            cd.rectangle([bbox[0] - context_box[0], bbox[1] - context_box[1],
                          bbox[2] - context_box[0] - 1, bbox[3] - context_box[1] - 1], outline="red", width=1)
            context_path = case_dir / f"{row['text_id']}_context.png"
            context.save(context_path)
            od.rectangle(bbox, outline="red", width=2)
            od.text((bbox[0], max(0, bbox[1] - 18)), row["text_id"], fill="red", font=font)
            row.update({"image": entry["image"], "case": entry["case"], "src": "cvjson_unverified",
                        "split": "review_candidate", "role_id": {"pin_name": 2, "pin_number": 3}[row["role"]],
                        "family_group": entry["group"], "source_json": entry["json"],
                        "source_image_sha256": entry["image_sha256"], "source_json_sha256": entry["json_sha256"],
                        "crop_path": str(crop_path), "context_path": str(context_path), "crop_bbox": bbox,
                        "coordinate_system": "top_left_image_pixels_xyxy",
                        "old_checkpoint_training_independence": "unknown_no_training_image_manifest"})
            manifest.append(row)
            cx, cy = (idx % 3) * 400, 110 + (idx // 3) * 200
            draw.text((cx + 8, cy), f"{row['text_id']} {row['role']}: {row['text']}", font=font, fill="black")
            scaled = crop.resize((crop.width * 4, crop.height * 4), Image.Resampling.NEAREST)
            scaled.thumbnail((180, 130), Image.Resampling.NEAREST)
            cc = context.resize((context.width * 3, context.height * 3), Image.Resampling.NEAREST)
            cc.thumbnail((195, 145), Image.Resampling.NEAREST)
            sheet.paste(scaled, (cx + 8, cy + 35))
            sheet.paste(cc, (cx + 196, cy + 35))
            draw.text((cx + 8, cy + 175), str(bbox), font=font, fill="gray")
            trs.append(f"<tr><td>{html.escape(row['text_id'])}</td><td>{row['role']}</td><td>{html.escape(row['text'])}</td><td><img src='{crop_path.name}'></td><td><img src='{context_path.name}'></td><td>{bbox}</td><td>未核验</td></tr>")
        sheet.save(case_dir / "contact.png")
        overlay.save(case_dir / "full-overlay.png")
        (case_dir / "review.html").write_text("<!doctype html><meta charset='utf-8'><style>td{padding:12px;border:1px solid #ddd}img{image-rendering:pixelated;min-height:48px;max-width:500px}body{font-family:sans-serif}</style>" +
            f"<h1>{entry['case']} 原始标注核验</h1><p>所有标签仅抄录原始JSON，尚未人工核验；不可直接当准确率真值。</p><p>原图 {html.escape(entry['image'])}</p><a href='full-overlay.png'>完整原图及裁片位置</a><table><tr><th>ID</th><th>角色</th><th>原标签</th><th>原框裁片</th><th>邻域</th><th>像素xyxy</th><th>状态</th></tr>" + "".join(trs) + "</table>", encoding="utf-8")
        page_links.append(f"<li><a href='{entry['case']}/review.html'>{entry['case']} {entry['source_kind']} 原始图/裁片/证据</a>；同源组 {entry['group']}</li>")
    (args.out / "manifest-candidate.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest), encoding="utf-8")
    inventory = [{k: v for k, v in e.items() if k not in ("mask", "text_bag", "rows")} |
                 {"pin_role_counts": dict(collections.Counter(r["role"] for r in e["rows"]))} for e in entries]
    verification = {"all_source_images_and_json_unchanged": all(sha(p) == h for p, h in before.items()),
                    "source_hashes": before}
    report = {"dataset_images": len(entries), "gt_images": len(gt),
              "role_counts": dict(collections.Counter(r["role"] for e in entries for r in e["rows"])),
              "family_count": len(members), "family_members": dict(members), "duplicate_links": duplicate_links,
              "gt_exclusions": gt_links, "excluded_dataset_cases": [e["case"] for e in entries if e["excluded_due_to_gt_group"]],
              "selected_cases": [e["case"] for e in chosen], "selected_role_counts": dict(collections.Counter(r["role"] for r in manifest)),
              "selected_family_count": len(chosen_groups), "heuristic_limit": "phash and source-text similarity do not prove absence of partial or redrawn same-source circuits; shared model family excluded conservatively even if layouts differ",
              "label_boundary": "All source labels remain unverified; no independent OCR accuracy may be computed until visual review",
              "model_training_boundary": "Historical checkpoint image manifest missing; chosen cases are not guaranteed unseen by existing model",
              "inventory": inventory, "source_verification": verification}
    (args.out / "inventory-and-grouping.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "index.html").write_text("<!doctype html><meta charset='utf-8'><h1>独立候选裁片核验包</h1><p>待核验标签，不是已确认验证集。已排除10GT像素近重复及其候选同源组；旧OCR权重是否训练过这些图未知。来源文本相似和感知哈希分组都是保守线索，不能证明所有同源图已排除。</p><ul>" + "".join(page_links) + "</ul>", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("dataset_images", "family_count", "selected_cases", "selected_role_counts", "excluded_dataset_cases")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
