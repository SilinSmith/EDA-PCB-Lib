"""Only fresh measured monochrome rows can use a consensus OCR replacement."""
import copy
import unittest

from stage2.assign import assign
from stage2.assemble import build_doc


def contact(point, trust="trusted"):
    return dict(point_img=list(point), anchor_img=[100, point[1]], edge="left",
                trust=trust, reason="frame_attached_normal_stroke", frame_confirmed=True,
                frame_coverage=1.0, normal_ink_run=12, normal_ink_hits=12,
                first_ink_distance=1, text_overlap_fraction=0, branching_fraction=0)


def provider(role, value, box, index=0, row=180):
    return dict(bbox=box, geometry_bbox=box, reads={role: (value, .95)},
                name_only=role == "pin_name", number_only=role == "pin_number",
                source="mono_" + role + "_ppocr",
                pin_target=dict(component_id="ic", pin_index=index, edge="left",
                                point_img=[90, row], anchor_img=[100, row]),
                mono_pin_ocr=dict(backend="ppocr_v4_direct", accepted=True, flags=[],
                                  selected_views=[dict(padding=p, text=value, confidence=.95)
                                                  for p in (2, 4)]))


class MonoPinBindingTests(unittest.TestCase):
    def setUp(self):
        self.comps = [dict(id="ic", type="box", bbox_img=[100, 100, 300, 300])]
        self.pins = {"ic": [[90, 180]]}
        self.evidence = {"ic": [contact(self.pins["ic"][0])]}

    def bind(self, reads):
        return assign(self.comps, self.pins, reads, pin_evidence=self.evidence)["comps"]["ic"]

    def test_measured_name_is_reserved_before_model_selection(self):
        raw = dict(bbox=[111, 174, 151, 188], reads={"pin_name": ("VBA1", .94),
                                                    "partname": ("VBA1", .95)})
        out = self.bind([raw])
        self.assertIsNone(out["partname"])
        self.assertEqual(out["pins"][0]["name"]["text"], "VBA1")

    def test_unmeasured_gray_label_cannot_reserve_model_text(self):
        self.evidence["ic"][0]["normal_ink_hits"] = 1
        raw = dict(bbox=[111, 174, 151, 188], reads={"pin_name": ("VBA1", .94),
                                                    "partname": ("VBA1", .95)})
        self.assertEqual(self.bind([raw])["partname"]["text"], "VBA1")

    def test_independent_complete_name_beats_only_its_old_observation(self):
        old = dict(bbox=[105, 174, 145, 188], name_only=True, reads={"pin_name": ("VBA1", .999)})
        new = provider("pin_name", "VBAT", [111, 174, 151, 188])
        before = copy.deepcopy([old, new])
        out = self.bind([old, new])
        self.assertEqual(out["pins"][0]["name"]["text"], "VBAT")
        self.assertEqual([old, new], before)
        self.assertIsNone(out["partname"])

    def test_complete_number_override_is_local_even_when_legacy_is_confident(self):
        old = dict(bbox=[79, 167, 96, 179], number_only=True,
                   reads={"pin_number": ("13", .999)})
        new = provider("pin_number", "113", [77, 167, 96, 179])
        self.assertEqual(self.bind([old, new])["pins"][0]["number"]["text"], "113")

    def test_disagreeing_and_held_views_cannot_override(self):
        old = dict(bbox=[79, 167, 96, 179], number_only=True,
                   reads={"pin_number": ("13", .60)})
        for update in (dict(text="13"), dict(confidence=.89), dict(padding=8)):
            new = provider("pin_number", "113", [77, 167, 96, 179])
            new["mono_pin_ocr"]["selected_views"][1].update(update)
            self.assertEqual(self.bind([old, new])["pins"][0]["number"]["text"], "13")
        new = provider("pin_number", "113", [77, 167, 96, 179])
        new["mono_pin_ocr"].update(accepted=False, held_reason="ambiguous")
        self.assertEqual(self.bind([old, new])["pins"][0]["number"]["text"], "13")

    def test_stale_target_or_frame_cannot_create_a_replacement(self):
        for patch in (dict(point_img=[90, 181]), dict(anchor_img=[100, 181]), dict(edge="right")):
            new = provider("pin_name", "VBAT", [111, 174, 151, 188])
            new["pin_target"].update(patch)
            self.assertIsNone(self.bind([new])["pins"][0]["name"])
        self.evidence["ic"][0]["anchor_img"] = [170, 180]
        new = provider("pin_name", "VBAT", [171, 174, 211, 188])
        self.assertIsNone(self.bind([new])["pins"][0]["name"])

    def test_uncertain_contact_never_gains_mono_priority(self):
        self.evidence["ic"][0]["trust"] = "uncertain"
        self.assertIsNone(self.bind([provider("pin_name", "VBAT", [111, 174, 151, 188])])["pins"][0]["name"])

    def test_override_does_not_change_another_pin_or_its_name(self):
        self.pins["ic"].append([90, 220])
        self.evidence["ic"].append(contact(self.pins["ic"][1]))
        reads = [provider("pin_number", "113", [77, 167, 96, 179]),
                 dict(bbox=[79, 207, 96, 219], number_only=True, reads={"pin_number": ("13", .99)}),
                 dict(bbox=[111, 174, 151, 188], name_only=True, reads={"pin_name": ("VBAT", .99)}),
                 dict(bbox=[111, 214, 151, 228], name_only=True, reads={"pin_name": ("VDD", .99)})]
        pins = self.bind(reads)["pins"]
        self.assertEqual([pins[i]["number"]["text"] for i in range(2)], ["113", "13"])
        self.assertEqual([pins[i]["name"]["text"] for i in range(2)], ["VBAT", "VDD"])

    def duplicate_fixture(self, *, independent_number=True, invalid_views=False):
        self.pins["ic"].append([90, 220])
        self.evidence["ic"].append(contact(self.pins["ic"][1]))
        name = provider("pin_name", "VBAT", [111, 174, 151, 188])
        name["mono_pin_ocr"].update(previous_name=None,
                                   partner_guard={"new_number_confirmed": True})
        reads = [dict(bbox=[111, 72, 132, 85], reads={"designator": ("U1", .99)}),
                 name,
                 dict(bbox=[79, 167, 96, 179], number_only=True,
                      reads={"pin_number": ("113", .5)}),
                 dict(bbox=[79, 207, 96, 219], number_only=True,
                      reads={"pin_number": ("113", .6)}),
                 dict(bbox=[111, 214, 151, 228], name_only=True,
                      reads={"pin_name": ("VDD", .6)})]
        if independent_number:
            number = provider("pin_number", "113", [77, 167, 96, 179])
            if invalid_views:
                number["mono_pin_ocr"]["selected_views"][1]["text"] = "13"
            reads.append(number)
        assigned = assign(self.comps, self.pins, reads, pin_evidence=self.evidence)
        stage = dict(image={"w": 350, "h": 330}, components=self.comps,
                     pins=self.pins, pin_evidence=self.evidence,
                     wire_detail={"segments": [[[90, row], [50, row]] for row in (180, 220)]},
                     nets=[{"pins": [["ic", i]], "seg_index": [i]} for i in range(2)])
        return assigned, build_doc(stage, assigned, "synthetic")

    def test_independent_number_winner_releases_missing_name_freeze_for_duplicate_selection(self):
        assigned, final = self.duplicate_fixture()
        physical = assigned["comps"]["ic"]["pins"][0]
        self.assertEqual(physical["number"]["text"], "113")
        self.assertTrue(physical["number_independently_read"])
        self.assertNotIn("number_selection_name", physical)
        self.assertEqual(final["__pin_index"], {"ic": {"0": "pin_113"}})
        self.assertEqual(final["pins"]["U1"]["pin_113"], {
            "pinname": "VBAT", "point": {"x": 90., "y": 150.}})
        self.assertEqual(next(iter(final["nets"].values()))["hyperGraph"], "(U1.113)")
        self.assertEqual(next(iter(next(iter(final["nets"].values()))["edges"].values()))[0]["y"], 150.)

    def test_meta_claim_without_actual_independent_number_retains_original_name_freeze(self):
        assigned, final = self.duplicate_fixture(independent_number=False)
        physical = assigned["comps"]["ic"]["pins"][0]
        self.assertNotIn("number_independently_read", physical)
        self.assertIn("number_selection_name", physical)
        self.assertIsNone(physical["number_selection_name"])
        self.assertEqual(final["__pin_index"], {"ic": {"1": "pin_113"}})

    def test_unqualified_new_digit_does_not_release_missing_name_freeze(self):
        assigned, final = self.duplicate_fixture(invalid_views=True)
        physical = assigned["comps"]["ic"]["pins"][0]
        self.assertNotIn("number_independently_read", physical)
        self.assertIsNone(physical["number_selection_name"])
        self.assertEqual(final["__pin_index"], {"ic": {"1": "pin_113"}})

    def test_name_and_number_keep_the_same_measured_row_despite_a_frame_fringe_duplicate(self):
        self.pins['ic']=[[90,176.5],[90,180]]
        self.evidence['ic']=[contact(p) for p in self.pins['ic']]
        name=provider('pin_name','READY',[111,174,151,188],1,180)
        number=provider('pin_number','A2',[78,168,96,177],1,180)
        for observation in (name,number):
            observation['mono_pin_ocr']['cleanup_evidence']={
                'stroke_probe':{'stroke_top':179,'stroke_bottom':182}}
        before=copy.deepcopy((self.pins,self.evidence,name,number))
        reads=[dict(bbox=[111,72,132,85],reads={'designator':('U1',.99)}),name,number,
               dict(bbox=[111,174,151,188],reads={'pin_name':('READY',.99)})]
        assigned=assign(self.comps,self.pins,reads,pin_evidence=self.evidence)
        pair=assigned['comps']['ic']['pins'][1]
        self.assertEqual(pair['name']['text'],'READY')
        self.assertEqual(pair['number']['text'],'A2')
        self.assertIsNone(assigned['comps']['ic']['pins'][0]['name'])
        self.assertEqual((self.pins,self.evidence,name,number),before)
        stage=dict(image={'w':350,'h':330},components=self.comps,pins=self.pins,pin_evidence=self.evidence,
                   wire_detail={'segments':[[[90,180],[50,180]]]},nets=[{'pins':[['ic',1]],'seg_index':[0]}])
        final=build_doc(stage,assigned,'synthetic')
        self.assertEqual(final['pins']['U1']['pin_A2']['pinname'],'READY')
        self.assertEqual(final['__pin_index']['ic'],{'1':'pin_A2'})
        self.assertEqual(next(iter(final['nets'].values()))['hyperGraph'],'(U1.A2)')

    def test_missing_stroke_measurement_or_another_real_row_cannot_bypass_name_ambiguity(self):
        for other_row,stroke in [(176.5,{}),(174,{'stroke_top':179,'stroke_bottom':182}),
                                 (180.5,{'stroke_top':179,'stroke_bottom':182}),
                                 (176.5,{'stroke_top':185,'stroke_bottom':188})]:
            with self.subTest(other_row=other_row,stroke=stroke):
                self.pins['ic']=[[90,other_row],[90,180]]
                self.evidence['ic']=[contact(p) for p in self.pins['ic']]
                name=provider('pin_name','READY',[111,170,151,188],1,180)
                name['mono_pin_ocr']['cleanup_evidence']={'stroke_probe':stroke}
                result=self.bind([name])['pins']
                self.assertIsNone(result[1]['name'])


if __name__ == "__main__":
    unittest.main()
