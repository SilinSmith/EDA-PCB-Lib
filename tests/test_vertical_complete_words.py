import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stage2.assign import assign
from stage2.mono_pin_crops import mono_pin_boxes
from stage2.mono_pin_reading import read_mono_pin_texts, _guard_vertical_number_collisions
from stage2.vertical_pin_reading import ClockwisePinView
from stage2.vertical_pin_crops import complete_vertical_pin_boxes, qualified_crossing_vertical_number


def color_fixture():
    horizontal=np.full((140,220,3),255,np.uint8)
    cv2.rectangle(horizontal,(30,10),(140,130),(0,0,190),1)
    cv2.line(horizontal,(140,70),(156,70),(0,0,190),1)
    cv2.putText(horizontal,'ABCD',(100,73),cv2.FONT_HERSHEY_SIMPLEX,.30,(0,0,0),1)
    cv2.putText(horizontal,'23',(134,68),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1)
    image=cv2.rotate(horizontal,cv2.ROTATE_90_COUNTERCLOCKWISE)
    point,anchor=[70,63],[70,79]
    evidence=dict(point_img=point[:],anchor_img=anchor[:],edge='top',trust='trusted',
        frame_confirmed=True,frame_coverage=1.,source='red_stub_endpoint',
        color_evidence=dict(red_hits=16,red_span=16,red_coverage=1.,stroke_width=1))
    doc=dict(components=[dict(id='ic',type='box',bbox_img=[10,79,130,189])],
             pins={'ic':[point]},pin_evidence={'ic':[evidence]})
    return doc,image,horizontal


class Provider:
    def __init__(self):self.count=0;self.samples=[]
    def read_glyph(self,image):
        self.samples.append(image.copy())
        text='ABCD' if self.count<2 else '23';self.count+=1
        return dict(text=text,confidence=.99)


def old_number():
    return dict(source='mono_pin_number_ppocr',number_only=True,bbox=[30,40,36,48],
        reads={'pin_number':('3',.99)},pin_target=dict(component_id='ic',pin_index=1),
        mono_pin_ocr=dict(backend='ppocr_v4_direct',accepted=True,held_reason=None,
                         selected_views=[dict(padding=p,text='3',confidence=.99) for p in (2,4)]))


class VerticalCompleteWordTests(unittest.TestCase):
    def test_crossing_number_retains_both_digits_and_binds_in_original_pixels(self):
        doc,image,_=color_fixture();before=copy.deepcopy(doc);pixels=image.copy();provider=Provider()
        with patch('stage2.mono_pin_reading.get_pin_name_ocr',return_value=provider):
            records=read_mono_pin_texts(doc,[],None,image)
        number=next(r for r in records if r.get('number_only'))
        self.assertEqual(number['reads']['pin_number'][0],'23')
        self.assertTrue(qualified_crossing_vertical_number(number['mono_pin_ocr'],number['bbox'],number['pin_target']))
        bound=assign(doc['components'],doc['pins'],records,pin_evidence=doc['pin_evidence'])
        self.assertEqual(bound['comps']['ic']['pins'][0]['number']['text'],'23')
        self.assertEqual(bound['comps']['ic']['pins'][0]['name']['text'],'ABCD')
        self.assertTrue(bound['comps']['ic']['pins'][0]['number_independently_read'])
        np.testing.assert_array_equal(image,pixels);self.assertEqual(doc,before)

    def test_frame_measurement_is_required_for_crossing_priority(self):
        doc,image,horizontal=color_fixture();view=ClockwisePinView(*image.shape[:2]);directional=view.document(doc)
        horizontal[:,140]=255
        candidates,_=mono_pin_boxes(directional,horizontal)
        records=complete_vertical_pin_boxes(directional,horizontal,candidates)
        self.assertFalse(any(c['cleanup_evidence'].get('vertical_frame_crossing_number') for c in records))

    def test_mixed_color_text_on_frame_keeps_original_dark_pixels(self):
        doc,image,horizontal=color_fixture()
        # Explicit narrow digit stroke crossing the frame: independent of the
        # font renderer's placement of the gap between the two fixture digits.
        horizontal[61:65,139:142]=(0,0,0)
        gray=cv2.cvtColor(horizontal,cv2.COLOR_BGR2GRAY)
        chroma=horizontal.max(2).astype(np.int16)-horizontal.min(2).astype(np.int16)
        text_rows=np.flatnonzero((gray[:,140]<190)&(chroma[:,140]<30))
        self.assertTrue(len(text_rows))
        horizontal[10:131,140]=(127,127,207)
        horizontal[text_rows,140]=(52,52,132)
        horizontal[70,140]=(63,63,177)
        image=cv2.rotate(horizontal,cv2.ROTATE_90_COUNTERCLOCKWISE)
        original=image.copy();provider=Provider()
        with patch('stage2.mono_pin_reading.get_pin_name_ocr',return_value=provider):
            records=read_mono_pin_texts(doc,[],None,image)
        number=next(r for r in records if r.get('number_only'))
        protected=number['mono_pin_ocr']['cleanup_evidence']['chromatic_frame_text_protection']['source_pixels']
        self.assertTrue(protected)
        value=int(cv2.cvtColor(np.uint8([[[52,52,132]]]),cv2.COLOR_BGR2GRAY)[0,0])
        self.assertTrue(any(np.any(sample==value) for sample in provider.samples))
        self.assertNotIn(doc['pin_evidence']['ic'][0]['anchor_img'],protected)
        np.testing.assert_array_equal(image,original)

    def test_chroma_changing_wire_crossing_is_excluded_from_text_protection(self):
        doc,image,horizontal=color_fixture();horizontal[10:131,140]=(127,127,207)
        horizontal[65,140]=(63,63,177)
        view=ClockwisePinView(*image.shape[:2]);directional=view.document(doc)
        candidates,_=mono_pin_boxes(directional,horizontal)
        records=complete_vertical_pin_boxes(directional,horizontal,candidates)
        for record in records:
            protected=(record['cleanup_evidence'].get('chromatic_frame_text_protection') or {}).get('source_pixels',[])
            self.assertNotIn([140,65],protected)

    def test_mismatched_box_anchor_and_frame_band_cannot_claim_crossing_priority(self):
        doc,image,_=color_fixture();provider=Provider()
        with patch('stage2.mono_pin_reading.get_pin_name_ocr',return_value=provider):
            records=read_mono_pin_texts(doc,[],None,image)
        number=next(r for r in records if r.get('number_only'))
        for change in ('box','anchor','band','extent'):
            meta=copy.deepcopy(number['mono_pin_ocr']);proof=meta['cleanup_evidence']['vertical_frame_crossing_number']
            if change=='box':proof['sampling_bbox'][0]+=1
            elif change=='anchor':proof['anchor_img'][0]+=1
            elif change=='band':proof['measured_frame_band']['side']=0
            else:proof['maximum_inside_pixels']=30
            self.assertFalse(qualified_crossing_vertical_number(meta,number['bbox'],number['pin_target']))

    def test_shared_view_audit_rotation_is_restored_once(self):
        doc,image,_=color_fixture();provider=Provider()
        with patch('stage2.mono_pin_reading.get_pin_name_ocr',return_value=provider):
            records=read_mono_pin_texts(doc,[],None,image)
        for record in records:
            meta=record['mono_pin_ocr']
            self.assertTrue(all(v['rotation']==90 for v in meta['selected_views']))
            self.assertTrue(all(v['rotation']==90 for v in meta['raw_reads']))

    def test_shared_prior_caption_box_is_restored_once(self):
        doc,image,_=color_fixture();view=ClockwisePinView(*image.shape[:2])
        expected=[35,45,50,52];shared=dict(text='READY',conf=.99,box=view.rectangle(expected))
        record=dict(bbox=view.rectangle(expected),pin_target=dict(component_id='ic',pin_index=0),
                    mono_pin_ocr=dict(cleanup_evidence={},previous_name=shared,
                                     previous_number_selection_name=shared))
        restored=view.restore(record,doc)
        self.assertEqual(restored['mono_pin_ocr']['previous_name']['box'],expected)
        self.assertEqual(restored['mono_pin_ocr']['previous_number_selection_name']['box'],expected)

    def run_collision(self,old,certainty=.95,independent=True):
        record=dict(number_only=True,reads={'pin_number':('3',certainty)},
                    pin_target=dict(component_id='ic',pin_index=0),mono_pin_ocr=dict(accepted=True))
        existing=dict(comps={'ic':dict(pins={1:dict(number_independently_read=independent,
                                                   number=dict(text='3',conf=.99,box=[30,40,36,48]))})})
        with patch('stage2.assign.assign',return_value=existing):
            _guard_vertical_number_collisions({},[old],[record])
        return record

    def test_weaker_same_provider_collision_preserves_established_digit(self):
        result=self.run_collision(old_number())
        self.assertEqual(result['reads'],{})
        self.assertEqual(result['mono_pin_ocr']['held_reason'],'stronger_existing_independent_number_collision')

    def test_legacy_confidence_cannot_reject_an_independent_new_digit(self):
        old=old_number();old['source']='legacy_crnn';old.pop('mono_pin_ocr')
        self.assertTrue(self.run_collision(old)['reads'])

    def test_stronger_color_path_of_same_provider_can_protect_its_digit(self):
        old=old_number();old['source']='color_pin_number_ppocr'
        old['pin_number_ocr']=old.pop('mono_pin_ocr')
        old['pin_number_ocr']['flags']=['mixed_frame_box','color_contamination']
        self.assertEqual(self.run_collision(old)['reads'],{})

    def test_stronger_new_digit_remains_eligible(self):
        self.assertTrue(self.run_collision(old_number(),certainty=.999)['reads'])

    def test_unvalidated_old_digit_cannot_reject_an_independent_new_digit(self):
        self.assertTrue(self.run_collision(old_number(),independent=False)['reads'])

    def test_current_horizontal_read_can_protect_its_digit_from_vertical_collision(self):
        doc,image,_=color_fixture();horizontal=old_number();captured=[]
        def vertical(doc,reads,*args,**options):
            captured.extend(reads)
            return []
        with patch('stage2.mono_pin_reading._read_horizontal_mono_pin_texts',return_value=[horizontal]), \
                patch('stage2.vertical_pin_reading.read_vertical_pin_texts',side_effect=vertical):
            result=read_mono_pin_texts(doc,[],None,image)
        self.assertEqual(result,[horizontal])
        self.assertIn(horizontal,captured)


if __name__=='__main__':unittest.main()
