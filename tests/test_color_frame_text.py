"""Behavioral checks for current colored strokes and original glyph pixels."""
import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stage2.mono_pin_crops import mono_pin_boxes, measured_text_contact
from stage2.mono_pin_reading import read_mono_pin_texts
from stage2.vertical_pin_reading import ClockwisePinView
from stage2.assign import assign


def fixture():
    image = np.full((120, 220, 3), 255, np.uint8)
    cv2.rectangle(image, (80, 10), (180, 110), (0, 0, 200), 2)
    cv2.line(image, (180, 50), (195, 50), (0, 0, 200), 1)
    cv2.line(image, (196, 50), (215, 50), (0, 130, 0), 1)
    point, anchor = [195, 50], [180, 50]
    evidence = dict(point_img=point[:], anchor_img=anchor, edge='right',
        source='red_green_transition', trust='trusted', frame_confirmed=True,
        frame_coverage=1., normal_ink_run=15, normal_ink_hits=15,
        color_evidence=dict(red_hits=15, red_span=15, red_coverage=1., stroke_width=1))
    doc = dict(components=[dict(id='ic', type='box', bbox_img=[80,10,180,110])],
               pins={'ic':[point]},pin_evidence={'ic':[evidence]})
    return doc, image


class Provider:
    def __init__(self):self.inputs=[]
    def read_glyph(self,gray):
        self.inputs.append(gray.copy())
        return dict(text='ABC' if len(self.inputs)<=2 else '16',confidence=.99)


class ColorFrameTextTests(unittest.TestCase):
    def get(self, boxes, role):
        return next(c for c in boxes if c['role']==role)

    def test_colored_name_keeps_exact_original_gray_and_body_boundary(self):
        doc,image=fixture()
        image[42:58,150:170]=(130,70,0)
        before=copy.deepcopy(doc);source=image.copy()
        boxes,gray=mono_pin_boxes(doc,image)
        name=self.get(boxes,'pin_name')
        self.assertEqual(name['bbox'],[150,42,170,58])
        self.assertFalse(name['flags'])
        self.assertEqual(name['cleanup_evidence']['sampling_color_mode'],'original_grayscale_of_color_contact')
        self.assertEqual(name['cleanup_evidence']['physical_evidence']['source'],'red_green_transition')
        np.testing.assert_array_equal(gray,cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
        np.testing.assert_array_equal(image,source);self.assertEqual(doc,before)

    def test_small_neutral_number_adjacent_to_colored_stroke_loses_no_body_pixel(self):
        doc,image=fixture()
        image[43:50,187:190]=50
        image[43:50,193:198]=100
        image[42,187:198]=215
        boxes,gray=mono_pin_boxes(doc,image)
        number=self.get(boxes,'pin_number')
        self.assertEqual(number['bbox'],[187,43,198,50])
        self.assertFalse(number['flags'])
        self.assertEqual(number['cleanup_evidence']['black_pixels_removed'],0)
        np.testing.assert_array_equal(gray[43:50,187:198],cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)[43:50,187:198])

    def test_bare_stale_uncertain_or_unmeasured_color_record_opens_no_crop(self):
        doc,image=fixture();image[42:58,150:170]=40
        changes=[dict(trust='uncertain'),dict(frame_confirmed=False),dict(source='unverified'),
                 dict(point_img=[195,51]),dict(anchor_img=[160,50]),dict(color_evidence={}),
                 dict(color_evidence=dict(red_hits=2,red_span=15,red_coverage=1.,stroke_width=1)),
                 dict(color_evidence=dict(red_hits=15,red_span=15,red_coverage=float('nan'),stroke_width=1))]
        for change in changes:
            with self.subTest(change=change):
                changed=copy.deepcopy(doc);changed['pin_evidence']['ic'][0].update(change)
                self.assertEqual(mono_pin_boxes(changed,image)[0],[])

    def test_cached_color_contact_with_no_current_chromatic_stroke_does_not_load_model(self):
        for replacement in ((0,0,0),(180,0,0)):
            doc,image=fixture();image[42:58,150:170]=40
            image[49:52,182:196]=replacement
            with patch('stage2.mono_pin_reading.get_pin_name_ocr') as loader:
                self.assertEqual(read_mono_pin_texts(doc,[],None,image),[])
            loader.assert_not_called()

    def test_narrow_one_and_its_disconnected_cap_stay_in_the_complete_number(self):
        doc,image=fixture()
        image[36:47,185:186]=0
        image[36:38,183:186]=0
        image[36:47,190:196]=40
        number=self.get(mono_pin_boxes(doc,image)[0],'pin_number')
        self.assertEqual(number['bbox'],[183,36,196,47])
        self.assertFalse(number['flags'])

    def test_complete_colored_inversion_bar_keeps_letters_inside_its_envelope(self):
        doc,image=fixture()
        image[42:58,145:148]=(150,70,0)
        image[42,145:169]=(150,70,0)
        image[42:58,167:170]=(150,70,0)
        image[44:58,151:155]=(150,70,0)
        image[44:58,159:163]=(150,70,0)
        name=self.get(mono_pin_boxes(doc,image)[0],'pin_name')
        self.assertEqual(name['bbox'],[145,42,170,58])
        self.assertFalse(name['flags'])
        self.assertEqual(len(name['cleanup_evidence']['text_row_selection']['enclosed_same_baseline_glyphs']),2)

    def test_a_neighboring_physical_row_cannot_supply_its_colored_name(self):
        doc,image=fixture()
        doc['pins']['ic'].append([195,70])
        other=copy.deepcopy(doc['pin_evidence']['ic'][0])
        other.update(point_img=[195,70],anchor_img=[180,70])
        doc['pin_evidence']['ic'].append(other)
        cv2.line(image,(180,70),(195,70),(0,0,200),1)
        image[42:58,150:170]=(150,70,0)
        image[62:78,149:170]=(150,70,0)
        names=[c for c in mono_pin_boxes(doc,image)[0] if c['role']=='pin_name']
        self.assertEqual([c['bbox'] for c in names],[[150,42,170,58],[149,62,170,78]])
        self.assertTrue(all(not c['flags'] for c in names))

    def test_vertical_color_view_reads_original_pixels_and_restores_same_target(self):
        doc,horizontal=fixture()
        horizontal[42:58,150:170]=(130,70,0)
        horizontal[36:47,187:203]=50
        image=cv2.rotate(horizontal,cv2.ROTATE_90_COUNTERCLOCKWISE)
        doc['components'][0]['bbox_img']=[10,39,110,139]
        doc['pins']['ic']=[[50,24]]
        ev=doc['pin_evidence']['ic'][0];ev.update(point_img=[50,24],anchor_img=[50,39],edge='top')
        before=copy.deepcopy(doc);source=image.copy();provider=Provider()
        with patch('stage2.mono_pin_reading.get_pin_name_ocr',return_value=provider):
            records=read_mono_pin_texts(doc,[],None,image)
        self.assertEqual(len(records),2)
        for record in records:
            self.assertEqual(record['pin_target']['edge'],'top')
            self.assertEqual(record['pin_target']['point_img'],[50,24])
            self.assertEqual(record['mono_pin_ocr']['rotation'],90)
        view=ClockwisePinView(*image.shape[:2]);gray=cv2.cvtColor(horizontal,cv2.COLOR_BGR2GRAY)
        for r,samples in zip(records,(provider.inputs[:2],provider.inputs[2:])):
            x0,y0,x1,y1=view.rectangle(r['bbox'])
            for margin,sample in zip((2,4),samples):
                np.testing.assert_array_equal(sample,np.pad(gray[y0:y1,x0:x1],margin,constant_values=255))
        result=assign(doc['components'],doc['pins'],records,pin_evidence=doc['pin_evidence'])
        pin=result['comps']['ic']['pins'][0]
        self.assertEqual(pin['number']['text'],'16');self.assertEqual(pin['name']['text'],'ABC')
        self.assertEqual(doc,before);np.testing.assert_array_equal(image,source)


if __name__=='__main__':unittest.main()
