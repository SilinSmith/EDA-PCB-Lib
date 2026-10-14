"""Measured ball numbers survive binding, duplicate selection and net export."""
import copy
import unittest

from stage2.assign import assign
from stage2.assemble import build_doc
from stage2.pin_identifier import pin_identifier_key
from test_mono_pin_binding import contact, provider


class AlphanumericIdentifierTests(unittest.TestCase):
    def fixture(self, numbers, *, independent=True):
        rows = [180 + 40 * i for i in range(len(numbers))]
        comps = [dict(id="ic", type="box", bbox_img=[100, 100, 300, 400])]
        pins = {"ic": [[90, y] for y in rows]}
        evidence = {"ic": [contact(p) for p in pins["ic"]]}
        reads = [dict(bbox=[111, 72, 132, 85], reads={"designator": ("U1", .99)})]
        for i, (value, y) in enumerate(zip(numbers, rows)):
            reads.append(provider("pin_name", "SIGNAL%d" % i, [111, y-6, 171, y+8], i, y))
            number = provider("pin_number", value, [77, y-13, 96, y-1], i, y)
            if not independent:
                number = dict(bbox=number["bbox"], number_only=True,
                              reads={"pin_number": (value, .99)})
            reads.append(number)
        stage = dict(image={"w": 350, "h": 450}, components=comps, pins=pins,
                     pin_evidence=evidence,
                     wire_detail={"segments": [[[90,y], [50,y]] for y in rows]},
                     nets=[dict(pins=[["ic",i]], seg_index=[i]) for i in range(len(rows))])
        snapshot = copy.deepcopy((stage, reads))
        assigned = assign(comps, pins, reads, pin_evidence=evidence)
        final = build_doc(stage, assigned, "synthetic")
        self.assertEqual((stage, reads), snapshot)
        return stage, assigned, final

    def test_literal_key_grammar_and_legacy_numeric_duplicate_equivalence(self):
        self.assertEqual(pin_identifier_key("01"), pin_identifier_key("1"))
        for value in ("P9", "U18", "A01", "12B", "b3"):
            self.assertIsNone(pin_identifier_key(value))
            self.assertEqual(pin_identifier_key(value, allow_alphanumeric=True), value)
        for value in ("0", "1000", "VDD", "A1/B1", "U1.2", "1234B", "A1!", None):
            self.assertIsNone(pin_identifier_key(value, allow_alphanumeric=True))

    def test_independent_identifiers_keep_letters_case_and_zeroes_through_net_export(self):
        numbers = ["P9", "U18", "a01", "12B"]
        _, assigned, final = self.fixture(numbers)
        self.assertEqual([assigned["comps"]["ic"]["pins"][i]["number"]["text"]
                          for i in range(4)], numbers)
        self.assertTrue(all(p.get("number_independently_read")
                            for p in assigned["comps"]["ic"]["pins"].values()))
        self.assertEqual(set(final["pins"]["U1"]), {"pin_"+n for n in numbers})
        self.assertEqual(final["__pin_index"]["ic"], {str(i):"pin_"+n for i,n in enumerate(numbers)})
        self.assertEqual({n["hyperGraph"] for n in final["nets"].values()},
                         {"(U1.%s)" % n for n in numbers})

    def test_ordinary_letter_number_text_has_no_independent_export_qualification(self):
        _, _, final = self.fixture(["P9"], independent=False)
        self.assertFalse(final["pins"]["U1"])
        self.assertFalse(final["nets"])

    def test_duplicate_letter_number_keeps_one_physical_row_without_net_alias(self):
        _, _, final = self.fixture(["P9", "P9"])
        self.assertEqual(set(final["pins"]["U1"]), {"pin_P9"})
        index = final["__pin_index"]["ic"]
        self.assertEqual(len(index), 1)
        kept = int(next(iter(index)))
        self.assertEqual(len(final["nets"]), 1)
        net = next(iter(final["nets"].values()))
        self.assertEqual(net["hyperGraph"], "(U1.P9)")
        self.assertEqual(net["edges"]["edge_1"][0]["y"], 450.-(180+40*kept))

    def test_stale_geometry_removes_letter_identifier_qualification_at_export(self):
        stage, assigned, _ = self.fixture(["P9"])
        stage["pin_evidence"]["ic"][0]["anchor_img"] = [170,180]
        final = build_doc(stage, assigned, "synthetic")
        self.assertFalse(final["pins"]["U1"])

    def test_legacy_conflicting_read_of_same_glyph_cannot_create_an_extra_number_on_another_row(self):
        comps=[dict(id="ic",type="box",bbox_img=[100,100,300,300])]
        pins={"ic":[[90,171],[90,180],[90,220]]}
        evidence={"ic":[contact(p) for p in pins["ic"]]}
        evidence["ic"][0].update(trust="uncertain",normal_ink_run=0,normal_ink_hits=0)
        reads=[provider("pin_number","P9",[77,167,96,179],1,180),
               dict(bbox=[75,164,97,180],number_only=True,reads={"pin_number":("17",.99)}),
               dict(bbox=[78,207,96,219],number_only=True,reads={"pin_number":("7",.99)})]
        assigned=assign(comps,pins,reads,pin_evidence=evidence)["comps"]["ic"]
        self.assertIsNone(assigned["pins"][0]["number"])
        self.assertEqual(assigned["pins"][1]["number"]["text"],"P9")
        self.assertEqual(assigned["pins"][2]["number"]["text"],"7")
        self.assertEqual(assigned["number_glyph_aliases_held"][0]["owner_pin"],1)

    def test_frame_fringe_duplicate_does_not_block_a_verified_number_above_actual_stroke(self):
        comps=[dict(id='ic',type='box',bbox_img=[100,100,300,300])]
        pins={'ic':[[90,176.5],[90,180]]}
        evidence={'ic':[contact(p) for p in pins['ic']]}
        new=provider('pin_number','A2',[78,168,96,177],1,180)
        new['mono_pin_ocr']['cleanup_evidence']={'stroke_probe':{'stroke_top':179,'stroke_bottom':182}}
        result=assign(comps,pins,[new],pin_evidence=evidence)['comps']['ic']['pins']
        self.assertTrue(result[1]['number_independently_read'])
        self.assertEqual(result[1]['number']['text'],'A2')
        new['mono_pin_ocr']['cleanup_evidence']['stroke_probe']['stroke_top']=185
        result=assign(comps,pins,[new],pin_evidence=evidence)['comps']['ic']['pins']
        self.assertFalse(result[1].get('number_independently_read'))


if __name__ == "__main__":
    unittest.main()
