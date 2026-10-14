"""Read vertical pin labels in an exact clockwise view of original pixels.

The view is local to text recognition. It preserves pin indices and does not
change the stage1 document, source image, endpoints or network identities.
Only measured top/bottom normal strokes can open this additional path.
"""
from __future__ import annotations

import copy

import cv2

from .mono_pin_crops import _finite, measured_text_contact
from .structmatch import trusted_frame_pin_evidence


class ClockwisePinView:
    EDGES = {"top": "right", "right": "bottom", "bottom": "left", "left": "top"}
    SIDES = {"T": "R", "R": "B", "B": "L", "L": "T"}

    def __init__(self, height, width):
        self.height, self.width = int(height), int(width)
        if min(self.height, self.width) <= 0:
            raise ValueError("source image dimensions must be positive")

    def point(self, point):
        return [self.height - 1 - point[1], point[0]]

    def inverse_point(self, point):
        return [point[1], self.height - 1 - point[0]]

    def rectangle(self, rect):
        """Transform half-open sampling extents, rather than pixel centres."""
        x0, y0, x1, y1 = rect
        return [self.height - y1, x0, self.height - y0, x1]

    def inverse_rectangle(self, rect):
        x0, y0, x1, y1 = rect
        return [y0, self.height - x1, y1, self.height - x0]

    def restore_frame_band(self, band):
        result = copy.deepcopy(band)
        side = result['side']
        result['side'] = {0:3,1:0,2:1,3:2}[side]
        if side in (0,2):
            a,b = result['persistent_band']
            result['persistent_band'] = [self.height-b,self.height-a]
            result['inner_search_boundary'] = self.height-1-result['inner_search_boundary']
        else:
            a,b = result['tangent_span']
            result['tangent_span'] = [self.height-b,self.height-a]
        return result

    def frame(self, rect):
        """Frame sides represent measured pixel coordinates, not slice edges."""
        x0, y0, x1, y1 = rect
        return [self.height - 1 - y1, x0, self.height - 1 - y0, x1]

    def evidence(self, record):
        result = copy.deepcopy(record)
        if not isinstance(result, dict):
            return result
        edge = result.get("edge")
        for key in ("point_img", "anchor_img"):
            point = _finite(result.get(key), 2)
            if point is not None:
                result[key] = self.point(point)
        if edge in self.EDGES:
            result["edge"] = self.EDGES[edge]
            value = result.get("frame_outer_img")
            if isinstance(value, (int, float)) and edge in ("top", "bottom"):
                result["frame_outer_img"] = self.height - 1 - value
        # Hollow-marker measurements are directional and presently supported
        # on horizontal frame sides only. A rotation must not promote them.
        if result.get("source") == "frame_hollow_marker":
            result["trust"] = "rejected"
        return result

    def document(self, doc):
        result = {key: copy.deepcopy(doc.get(key) or ([] if key == "components" else {}))
                  for key in ("components", "pins", "pin_evidence", "pin_mono", "pincol")}
        result["pin_text_view_rotation"] = 90
        result["pin_text_source_shape"] = [self.height, self.width]
        for component in result.get("components") or []:
            frame = _finite(component.get("bbox_img"), 4)
            if frame is not None:
                component["bbox_img"] = self.frame(frame)
            centre = _finite(component.get("center_img"), 2)
            if centre is not None:
                component["center_img"] = self.point(centre)
        for cid, points in (result.get("pins") or {}).items():
            result["pins"][cid] = [self.point(_finite(p, 2)) if _finite(p, 2) is not None else p
                                   for p in points]
        for cid, records in (result.get("pin_evidence") or {}).items():
            result["pin_evidence"][cid] = [self.evidence(r) for r in records]
        for metadata in (result.get("pin_mono") or {}).values():
            for row in metadata.get("stroke_rows") or []:
                edge = row.get("edge")
                if edge not in self.EDGES:
                    continue
                row["edge"] = self.EDGES[edge]
                for key in ("point_img", "anchor_img"):
                    p = _finite(row.get(key), 2)
                    if p is not None:
                        row[key] = self.point(p)
                if isinstance(row.get("evidence"), dict):
                    row["evidence"] = self.evidence(row["evidence"])
                if edge in ("left", "right"):
                    if isinstance(row.get("tangent_center"), (int, float)):
                        row["tangent_center"] = self.height - 1 - row["tangent_center"]
                    band = _finite(row.get("tangent_band"), 2)
                    if band is not None:
                        row["tangent_band"] = [self.height-1-band[1], self.height-1-band[0]]
                elif isinstance(row.get("frame_outer_img"), (int, float)):
                    row["frame_outer_img"] = self.height - 1 - row["frame_outer_img"]
        for records in (result.get("pincol") or {}).values():
            for record in records:
                if "x" in record and "y" in record:
                    record["x"], record["y"] = self.point([record["x"], record["y"]])
                record["side"] = self.SIDES.get(record.get("side"), record.get("side"))
                record["axis"] = {"x": "y", "y": "x"}.get(record.get("axis"), record.get("axis"))
                for key in ("in_win", "out_win"):
                    rect = _finite(record.get(key), 4)
                    if rect is not None:
                        record[key] = self.rectangle(rect)
        return result

    def reads(self, reads):
        output = copy.deepcopy(reads)
        for record in output:
            for key in ("bbox", "geometry_bbox", "glyph_bbox"):
                rect = _finite(record.get(key), 4)
                if rect is not None:
                    record[key] = self.rectangle(rect)
            if _finite(record.get("center"), 2) is not None:
                record["center"] = self.point(record["center"])
            target = record.get("pin_target")
            if isinstance(target, dict):
                target["edge"] = self.EDGES.get(target.get("edge"), target.get("edge"))
                for key in ("point_img", "anchor_img"):
                    point = _finite(target.get(key), 2)
                    if point is not None:
                        target[key] = self.point(point)
        return output

    def restore(self, record, doc):
        result = copy.deepcopy(record)
        restored_boxes=set()
        def original_box(holder,key):
            identity=(id(holder),key)
            if identity not in restored_boxes and _finite(holder.get(key),4) is not None:
                holder[key]=self.inverse_rectangle(holder[key])
                restored_boxes.add(identity)
        for key in ("bbox", "geometry_bbox", "glyph_bbox"):
            original_box(result,key)
        target = result["pin_target"]
        cid, index = target["component_id"], target["pin_index"]
        evidence = doc["pin_evidence"][cid][index]
        target.update(edge=evidence["edge"], point_img=list(doc["pins"][cid][index]),
                      anchor_img=list(evidence["anchor_img"]))
        meta = result["mono_pin_ocr"]
        canonical = meta["cleanup_evidence"]
        # Keep the complete directional audit under an explicit coordinate
        # label. Top-level geometry and measured exclusions use original pixels.
        meta["directional_cleanup_evidence"] = dict(canonical,
            coordinate_space="clockwise_rotated_source_pixels")
        cleanup = {key: canonical[key] for key in
                   ("method", "integrity_unresolved", "black_pixels_removed",
                    "pitch_pixels", "word_gap_pixels", "independent_of_ocr_text",
                    "same_side_rows") if key in canonical}
        cleanup["physical_evidence"] = copy.deepcopy(evidence)
        if canonical.get('sampling_color_mode'):
            cleanup['sampling_color_mode']=canonical['sampling_color_mode']
        protection=canonical.get('chromatic_frame_text_protection')
        if isinstance(protection,dict):
            protection=copy.deepcopy(protection)
            protection['source_pixels']=[self.inverse_point(p) for p in protection['source_pixels']]
            cleanup['chromatic_frame_text_protection']=protection
        proof = canonical.get('vertical_frame_crossing_number')
        if isinstance(proof,dict):
            proof = copy.deepcopy(proof)
            proof.update(sampling_bbox=self.inverse_rectangle(proof['sampling_bbox']),
                         point_img=list(target['point_img']),anchor_img=list(target['anchor_img']),edge=target['edge'])
            proof['measured_frame_band']=self.restore_frame_band(proof['measured_frame_band'])
            cleanup['vertical_frame_crossing_number']=proof
        fringe = canonical.get('name_fringe')
        if isinstance(fringe,dict):
            fringe=copy.deepcopy(fringe)
            for key in ('bbox','core_bbox','physical_search_region'):
                fringe[key]=self.inverse_rectangle(fringe[key])
            cleanup['name_fringe']=fringe
        cleanup["known_line_exclusions"] = [self.inverse_point(p) for p in
                                             canonical.get("known_line_exclusions", [])]
        region = _finite(canonical.get("physical_search_region"), 4)
        if region is not None:
            cleanup["physical_search_region"] = self.inverse_rectangle(region)
        stroke = canonical.get("stroke_probe") or {}
        cleanup["stroke_probe"] = {key: stroke[key] for key in
            ("method", "stroke_top", "stroke_bottom") if key in stroke}
        cleanup["stroke_probe"].update(tangent_axis="x", rotation=90)
        probe = _finite(stroke.get("probe"), 4)
        if probe is not None:
            cleanup["stroke_probe"]["probe"] = self.inverse_rectangle(probe)
        meta.update(cleanup_evidence=cleanup, rotation=90,
                    orientation_source="vertical_frame_normal_stroke_clockwise_original_pixels",
                    source_image_shape=[self.height, self.width])
        if _finite(meta.get("region"), 4) is not None:
            meta["region"] = self.inverse_rectangle(meta["region"])
        visited = set()
        for views in (meta.get("selected_views", []), meta.get("raw_reads", []),
                      meta.get('initial_core_views',[]),meta.get('fringe_retry_views',[]),
                      *meta.get("orientation_trials", [])):
            for view in views:
                if id(view) not in visited:
                    view["rotation"] = (90 + view.get("rotation", 0)) % 360
                    visited.add(id(view))
        if meta.get('selected_views'):
            meta['glyph_reading_rotation']=meta['selected_views'][0]['rotation']
        meta['pin_geometry_rotation']=90
        collision=meta.get('number_collision_guard')
        if isinstance(collision,dict) and _finite(collision.get('existing_box'),4) is not None:
            collision['existing_box']=self.inverse_rectangle(collision['existing_box'])
        for key in ("previous_name", "previous_number", "previous_number_selection_name"):
            previous = meta.get(key)
            if isinstance(previous,dict):
                original_box(previous,'box')
        guard = meta.get("name_replacement_guard")
        if isinstance(guard, dict):
            for key in ("previous_box", "proposed_box"):
                if _finite(guard.get(key), 4) is not None:
                    guard[key] = self.inverse_rectangle(guard[key])
        partner = (meta.get("partner_guard") or {}).get("winning_number_after_rebinding")
        if isinstance(partner,dict):
            original_box(partner,'box')
        return result


def _has_measured_vertical_pin(doc):
    for component in doc.get("components") or []:
        if component.get("type") != "box":
            continue
        cid, frame = component.get("id"), _finite(component.get("bbox_img"), 4)
        if frame is None:
            continue
        points = (doc.get("pins") or {}).get(cid) or []
        records = (doc.get("pin_evidence") or {}).get(cid) or []
        for raw, record in zip(points, records):
            point = _finite(raw, 2)
            if (point is not None and isinstance(record, dict)
                    and record.get("edge") in ("top", "bottom")
                    and record.get("source") != "frame_hollow_marker"
                    and measured_text_contact(record, frame, point)):
                return True
    return False


def read_vertical_pin_texts(doc, reads, engine, bgr, horizontal_reader, **options):
    if (bgr is None or getattr(bgr, "ndim", 0) != 3 or bgr.shape[2] != 3
            or not _has_measured_vertical_pin(doc)):
        return []
    view = ClockwisePinView(*bgr.shape[:2])
    directional = view.document(doc)
    rotated = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    observed = horizontal_reader(directional, view.reads(reads), engine, rotated, **options)
    return [view.restore(record, doc) for record in observed]
