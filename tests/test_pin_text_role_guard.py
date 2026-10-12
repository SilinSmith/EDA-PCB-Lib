"""Protect observed pin names without claiming model text or stale OCR crops."""
import copy
import importlib
import unittest
from unittest.mock import patch

assign = importlib.import_module("stage2.assign")

import _unmerged_gate

# ── 门（细粒度）：本模块绝大多数用例测的是队友分支未并入 main 的绑定路径 ──
# 缺失 ⇒ 只跳过打在该主体上的那些用例（`requires()`，见 tests/_unmerged_gate.py）；
# 例外见文件尾注释（`_best_number_read` 那条在 main 上本来就能过）。
_GUARD = [("stage2.assign", ("PIN_TEXT_ROLE_GUARD", "PIN_DIGIT_ENTITY"))]


def text(rect, value, roles=("pin_name", "partname"), conf=.98, **metadata):
    return dict(bbox=list(rect), reads={role: (value, conf) for role in roles}, **metadata)


def evidence(point, trust="trusted", source="red_green_transition"):
    return {"point_img": list(point), "anchor_img": [100, point[1]], "edge": "left",
            "trust": trust, "source": source, "frame_confirmed": True}


def target(cid, index, point):
    return {"component_id": cid, "pin_index": index, "edge": "left", "point_img": list(point)}


class PinTextRoleGuardTests(unittest.TestCase):
    box = [100, 100, 220, 240]
    points = [[90, 145], [90, 185]]

    def assigned(self, texts, records=None, guard=True, components=None, pins=None):
        components = components or [{"id": "ic", "type": "box", "bbox_img": self.box}]
        pins = pins or {"ic": self.points}
        if records is None:
            records = {"ic": [evidence(p) for p in self.points]}
        with patch.object(assign, "PIN_TEXT_ROLE_GUARD", guard), \
                patch.object(assign, "PINCOL_GUIDE", False), \
                patch.object(assign, "PIN_DIGIT_ENTITY", True):
            return assign.assign(components, pins, copy.deepcopy(texts), pin_evidence=records)

    @_unmerged_gate.requires(_GUARD)
    def test_vout_and_fault_keep_name_role_before_partname(self):
        for name in ("VOUT", "FAULT"):
            with self.subTest(name=name):
                reads = [text([110, 140, 150, 150], name)]
                fixed = self.assigned(reads)["comps"]["ic"]
                old = self.assigned(reads, guard=False)["comps"]["ic"]
                self.assertEqual(old["partname"]["text"], name)
                self.assertIsNone(old["pins"][0]["name"])
                self.assertIsNone(fixed["partname"])
                self.assertEqual(fixed["pins"][0]["name"]["text"], name)

    @_unmerged_gate.requires(_GUARD)
    def test_central_or_external_model_is_still_available(self):
        for rect in ([130, 205, 205, 215], [110, 245, 185, 255]):
            with self.subTest(rect=rect):
                reads = [text([110, 140, 150, 150], "VOUT"),
                         text(rect, "TPS54331", roles=("partname", "pin_name"))]
                fixed = self.assigned(reads)["comps"]["ic"]
                self.assertEqual(fixed["partname"]["text"], "TPS54331")
                self.assertEqual(fixed["pins"][0]["name"]["text"], "VOUT")
                self.assertIsNone(fixed["pins"][1]["name"])

    @_unmerged_gate.requires(_GUARD)
    def test_monochrome_or_stale_evidence_preserves_old_role_behavior(self):
        reads = [text([110, 140, 150, 150], "VOUT")]
        records = [
            {"ic": [evidence(p, source="gray_frame") for p in self.points]},
            {"ic": [evidence([p[0], p[1]+1]) for p in self.points]},
            {"ic": [evidence(p, trust="uncertain") for p in self.points]},
            {},
        ]
        for rec in records:
            with self.subTest(records=rec):
                self.assertEqual(self.assigned(reads, rec), self.assigned(reads, rec, guard=False))

    @_unmerged_gate.requires(_GUARD)
    def test_pin_name_only_reads_cannot_become_designator_partname_or_value(self):
        for name in ("U1", "TPS54331", "1.0uF"):
            reads = [text([110, 140, 150, 150], name, roles=("pin_name",), name_only=True)]
            rec = self.assigned(reads)["comps"]["ic"]
            self.assertIsNone(rec["designator"])
            self.assertIsNone(rec["partname"])
            self.assertIsNone(rec["value"])
            self.assertEqual(rec["pins"][0]["name"]["text"], name)
            self.assertIsNone(rec["pins"][0]["number"])

    @_unmerged_gate.requires(_GUARD)
    def test_valid_pin_target_overrides_equal_nearest_component(self):
        components = [{"id": cid, "type": "box", "bbox_img": self.box} for cid in ("other", "ic")]
        pins = {cid: self.points for cid in ("other", "ic")}
        records = {cid: [evidence(p) for p in self.points] for cid in ("other", "ic")}
        reads = [text([110, 140, 150, 150], "VOUT", roles=("pin_name",),
                      name_only=True, pin_target=target("ic", 0, self.points[0]))]
        out = self.assigned(reads, records, components=components, pins=pins)["comps"]
        self.assertEqual(out["ic"]["pins"][0]["name"]["text"], "VOUT")
        self.assertIsNone(out["other"]["pins"][0]["name"])

    @_unmerged_gate.requires(_GUARD)
    def test_targeted_name_cannot_jump_to_another_row(self):
        reads = [text([110, 140, 150, 150], "VOUT", roles=("pin_name",),
                      pin_target=target("ic", 1, self.points[1]))]
        slots = self.assigned(reads)["comps"]["ic"]["pins"]
        self.assertTrue(all(slot["name"] is None for slot in slots.values()))

    @_unmerged_gate.requires(_GUARD)
    def test_targeted_name_cannot_jump_when_other_contact_forces_legacy_phase(self):
        reads = [text([110, 180, 150, 190], "VOUT", roles=("pin_name",),
                      pin_target=target("ic", 0, self.points[0]))]
        records = {"ic": [evidence(self.points[0]), evidence(self.points[1], source="gray_frame")]}
        records["ic"][1]["frame_confirmed"] = False
        slots = self.assigned(reads, records)["comps"]["ic"]["pins"]
        self.assertTrue(all(slot["name"] is None for slot in slots.values()))

    @_unmerged_gate.requires(_GUARD)
    def test_malformed_or_expired_targets_are_skipped_entirely(self):
        targets = [None, "ic", {}, target("missing", 0, self.points[0]),
                   target("ic", -1, self.points[0]), target("ic", True, self.points[0]),
                   target("ic", 99, self.points[0]), target("ic", "0", self.points[0]),
                   target("ic", 0, [90, 146]), target("ic", 0, [float("nan"), 145]),
                   dict(target("ic", 0, self.points[0]), edge="right")]
        # None is absence of metadata and intentionally keeps an ordinary read.
        for item in targets[1:]:
            with self.subTest(target=item):
                reads = [text([110, 140, 150, 150], "VOUT", roles=("pin_name",), pin_target=item)]
                slots = self.assigned(reads)["comps"]["ic"]["pins"]
                self.assertTrue(all(slot["name"] is None for slot in slots.values()))

    @_unmerged_gate.requires(_GUARD)
    def test_targeted_crop_with_stale_contact_evidence_is_skipped(self):
        reads = [text([110, 140, 150, 150], "VOUT", roles=("pin_name",),
                      pin_target=target("ic", 0, self.points[0]))]
        records = {"ic": [evidence([90, 146]), evidence(self.points[1])]}
        slots = self.assigned(reads, records)["comps"]["ic"]["pins"]
        self.assertTrue(all(slot["name"] is None for slot in slots.values()))

    def test_strong_truncated_number_uses_three_strong_same_box_votes_only_on_color(self):
        # `_best_number_read` 的「同框三票一致才改号」在主分支上存在，但它是 opt-in：
        # `stage2/assign.py:58 PINNUM_CONSENSUS = env_on("STAGE2_PINNUM_CONSENSUS", "0")`
        # （队友分支默认 "1"），函数体两边逐字节相同 ⇒ 测试按本仓既有惯例把这个开关打开
        # （同 tests/test_pin_number_consensus.py:14/26/36/44/55）。
        digits = {"pin_number": ("1", .999), "designator": ("14", .987),
                  "value": ("14", .939), "net_label": ("14", .980), "pin_name": ("N", .9)}
        with patch.object(assign, "PINNUM_CONSENSUS", True):
            self.assertEqual(assign._best_number_read(digits, fresh_color=True)[1], "14")
            self.assertEqual(assign._best_number_read(digits, fresh_color=False)[1], "1")
            for bad in ({"value": ("14", .7)}, {"net_label": ("12", .99)},
                        {"designator": ("4", .99), "value": ("4", .99), "net_label": ("4", .99)}):
                revised = dict(digits, **bad)
                self.assertEqual(assign._best_number_read(revised, fresh_color=True)[1], "1")

    @_unmerged_gate.requires(_GUARD)
    def test_number_ink_geometry_keeps_padding_out_of_candidate_ranking(self):
        # These are two readings of exactly the same digit ink; the clean
        # re-read has a wider padded sampling box but should not lose by row.
        old = {"bbox": [85, 132, 95, 145], "reads": {"pin_number": ("13", .7)}}
        new = {"bbox": [82, 129, 98, 148], "geometry_bbox": [86, 134, 94, 145],
               "reads": {"pin_number": ("8", .95)}, "number_only": True,
               "color_crop": {"component_id": "ic", "pin_index": 0, "entity_id": "ic:0:test",
                              "edge": "left", "anchor_img": [100, 145],
                              "source_boxes": [{"bbox": old["bbox"], "color_contamination_ratio": 0}],
                              "glyph_touches_search_boundary": False}}
        reads = [text([110, 140, 150, 150], "IS", roles=("pin_name",)), old, new]
        slots = self.assigned(reads)["comps"]["ic"]["pins"]
        self.assertEqual(slots[0]["number"]["text"], "8")
        self.assertEqual(slots[0]["name"]["text"], "IS")

    @_unmerged_gate.requires(_GUARD)
    def test_trusted_color_on_other_row_cannot_override_uncertain_rows_strong_digit(self):
        digits = {"pin_number": ("1", .999), "designator": ("14", .987),
                  "value": ("14", .939), "net_label": ("14", .980), "pin_name": ("N", .9)}
        reads = [text([110, 140, 150, 150], "A", roles=("pin_name",)),
                 text([110, 180, 150, 190], "B", roles=("pin_name",)),
                 {"bbox": [85, 174, 98, 185], "reads": digits}]
        records = {"ic": [evidence(self.points[0]), evidence(self.points[1], trust="uncertain")]}
        slots = self.assigned(reads, records)["comps"]["ic"]["pins"]
        self.assertEqual(slots[1]["number"]["text"], "1")

    def _digit_fixture(self):
        old = {"bbox": [85, 134, 98, 145], "reads": {"pin_number": ("24", .999)}}
        clean = {"bbox": [82, 131, 98, 147], "geometry_bbox": [86, 135, 96, 145],
                 "reads": {"pin_number": ("4", .95)}, "number_only": True,
                 "color_crop": {"component_id": "ic", "pin_index": 0,
                                "edge": "left", "anchor_img": [100, 145],
                                "entity_id": "ic:0:86:135:96:145",
                                "source_boxes": [{"bbox": old["bbox"], "color_contamination_ratio": .3}],
                                "glyph_touches_search_boundary": False}}
        return old, clean

    @_unmerged_gate.requires(_GUARD)
    def test_symbol_contaminated_digit_is_replaced_only_by_same_clean_glyph(self):
        old, clean = self._digit_fixture()
        reads = [text([110, 140, 150, 150], "IS", roles=("pin_name",)), old, clean]
        slots = self.assigned(reads)["comps"]["ic"]["pins"]
        self.assertEqual(slots[0]["number"]["text"], "4")
        self.assertEqual(slots[0]["name"]["text"], "IS")

    @_unmerged_gate.requires(_GUARD)
    def test_old_numeric_read_is_preserved_when_clean_glyph_hits_search_boundary(self):
        old, clean = self._digit_fixture()
        clean["color_crop"]["glyph_touches_search_boundary"] = True
        slots = self.assigned([old, clean])["comps"]["ic"]["pins"]
        self.assertEqual(slots[0]["number"]["text"], "24")

    @_unmerged_gate.requires(_GUARD)
    def test_stale_digit_entity_anchor_or_wrong_side_cannot_suppress_original_read(self):
        for changes in ({"anchor_img": [100, 185]}, {"anchor_img": [float("nan"), 145]},
                        {"edge": "right"}, {"component_id": "missing"},
                        {"pin_index": 99}, {"pin_index": True}):
            with self.subTest(changes=changes):
                old, clean = self._digit_fixture()
                clean["color_crop"].update(changes)
                slots = self.assigned([old, clean])["comps"]["ic"]["pins"]
                self.assertEqual(slots[0]["number"]["text"], "24")

    @_unmerged_gate.requires(_GUARD)
    def test_nonfinite_digit_geometry_or_bad_source_container_cannot_crash_or_suppress(self):
        for field, value in (("geometry_bbox", [86, 135, float("nan"), 145]),
                             ("source_boxes", None), ("source_boxes", "not boxes"),
                             ("source_boxes", [{}])):
            with self.subTest(field=field, value=value):
                old, clean = self._digit_fixture()
                if field == "geometry_bbox":
                    clean[field] = value
                else:
                    clean["color_crop"][field] = value
                slots = self.assigned([old, clean])["comps"]["ic"]["pins"]
                self.assertEqual(slots[0]["number"]["text"], "24")

    @_unmerged_gate.requires(_GUARD)
    def test_numeric_entity_cannot_claim_ink_from_another_pin_row(self):
        old, clean = self._digit_fixture()
        # The stale crop's stored anchor is current, but its alleged glyph
        # comes from a different row. It cannot rewrite row 0's old numeral.
        clean["geometry_bbox"] = [86, 175, 96, 185]
        clean["bbox"] = [82, 171, 98, 187]
        slots = self.assigned([old, clean])["comps"]["ic"]["pins"]
        self.assertEqual(slots[0]["number"]["text"], "24")


if __name__ == "__main__":
    unittest.main()


# ── 门（细粒度）：见文件头 `_GUARD` 与各用例上的 `@_unmerged_gate.requires(...)` ──
# 本模块**不再**整模块跳过。17 例里只有 `test_strong_truncated_number_uses_three_strong_
# same_box_votes_only_on_color` 在 main 上本来就能过（它测的 `_best_number_read` 存在，
# 只是 opt-in），已按本仓惯例在用例内打开 `PINNUM_CONSENSUS`；其余 16 例断言打在队友分支
# 未并入的 `PIN_TEXT_ROLE_GUARD` / `PIN_DIGIT_ENTITY` 上（`patch.object` 会因属性不存在
# 抛 `AttributeError`），按缺失前置条件逐例跳过。
