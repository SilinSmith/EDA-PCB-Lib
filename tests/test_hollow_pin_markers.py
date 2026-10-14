"""Actual triangle and outer-line pixels are required before mono pin reading."""
import copy
import unittest
import cv2
import numpy as np

from stage1.pin_evidence import recover_frame_pins,build_pin_evidence
from stage1.pipeline import overlay_mono_evidence
from stage2.mono_pin_crops import mono_pin_boxes,mono_observations
from stage2.structmatch import trusted_frame_pin_evidence


class HollowPinMarkerTests(unittest.TestCase):
    box=[200.,40.,450.,380.]

    def image(self, *, bowtie=False, line=True, touching=False):
        image=np.full((420,600,3),255,np.uint8)
        cv2.rectangle(image,(200,40),(450,380),(0,0,0),1)
        triangle=np.array([(200,100),(184,94),(184,106)],np.int32)
        cv2.polylines(image,[triangle],True,(190,190,190),1)
        tip=184
        if bowtie:
            cv2.polylines(image,[np.array([(182,94),(182,106),(166,100)],np.int32)],True,(190,190,190),1)
            image[100,181:185]=0
            tip=166
        if line:
            cv2.line(image,(130,100),(tip,100),(0,0,0),1)
        cv2.putText(image,"B3",(150,89),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_8)
        cv2.putText(image,"INPUT",(212,104),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_8)
        if touching:
            cv2.line(image,(184,83),(184,95),(0,0,0),1)
        return image

    def recover(self, image, old=(), source="yolo"):
        comps=[dict(id="ic",type="box",source=source,bbox_img=self.box)]
        before=image.copy(); snapshot=copy.deepcopy(old)
        pins,mapping,audit=recover_frame_pins(comps,{"ic":list(old)},image)
        ev=build_pin_evidence(comps,pins,np.any(image<170,axis=2))
        ev=overlay_mono_evidence(ev,audit,pins)
        doc=dict(components=comps,pins=pins,pin_evidence=ev,pin_mono=audit)
        np.testing.assert_array_equal(image,before)
        self.assertEqual(old,snapshot)
        return doc,mapping,audit

    def test_single_and_opposing_triangles_open_separate_complete_number_and_name_crops(self):
        for bowtie in (False,True):
            with self.subTest(bowtie=bowtie):
                image=self.image(bowtie=bowtie)
                doc,_,_=self.recover(image)
                observations=list(mono_observations(doc))
                markers=[o for o in observations if o['evidence'].get('source')=='frame_hollow_marker']
                self.assertEqual(len(markers),1)
                self.assertTrue(trusted_frame_pin_evidence(markers[0]['evidence'],self.box,markers[0]['point_img']))
                boxes,_=mono_pin_boxes(doc,image)
                numbers=[c for c in boxes if c['role']=='pin_number']
                self.assertEqual(len(numbers),1)
                self.assertFalse(numbers[0]['flags'])
                self.assertLess(numbers[0]['bbox'][3],94)
                self.assertTrue(any(c['role']=='pin_name' for c in boxes))

    def test_triangular_hole_without_connected_outer_line_is_not_a_pin(self):
        doc,_,audit=self.recover(self.image(line=False))
        self.assertFalse(audit['ic']['marker_rows'])
        self.assertFalse(list(mono_observations(doc)))

    def test_shape_only_component_and_colored_marker_never_gain_recovery_trust(self):
        doc,_,_=self.recover(self.image(),source='cv:bigic')
        self.assertFalse(list(mono_observations(doc)))
        image=self.image()
        image[(image[:,:,0]==190)&(image[:,:,1]==190)]=[0,0,190]
        _,_,audit=self.recover(image)
        self.assertFalse(audit['ic']['marker_rows'])

    def test_marker_crop_rechecks_current_triangle_outer_line_and_frame(self):
        image=self.image(bowtie=True);doc,_,_=self.recover(image)
        for mutation in ('triangle','line','frame'):
            changed=image.copy()
            if mutation=='triangle':changed[93:108,165:201]=255
            elif mutation=='line':changed[99:102,125:166]=255
            else:changed[40:381,199:202]=255
            self.assertFalse(mono_pin_boxes(doc,changed)[0],mutation)

    def test_number_touching_marker_remains_held_instead_of_losing_its_stem(self):
        image=self.image(touching=True);doc,_,_=self.recover(image)
        numbers=[c for c in mono_pin_boxes(doc,image)[0] if c['role']=='pin_number']
        self.assertTrue(numbers)
        self.assertTrue(all('glyph_integrity_unresolved' in c['flags'] for c in numbers))

    def test_multiple_old_candidates_keep_indices_and_only_new_measured_row_has_marker_trust(self):
        old=[[188.,99.],[188.,101.]]
        doc,mapping,audit=self.recover(self.image(),old)
        self.assertEqual(doc['pins']['ic'][:2],old)
        self.assertEqual(mapping['ic'],{0:0,1:1})
        self.assertEqual(len(audit['ic']['added']),1)
        self.assertTrue(all(e['trust']!='trusted' for e in doc['pin_evidence']['ic'][:2]))
        self.assertEqual(len(list(mono_observations(doc))),1)

    def test_partial_measurement_or_stale_anchor_cannot_use_marker_trust_flag(self):
        doc,_,_=self.recover(self.image())
        marker=next(e for e in doc['pin_evidence']['ic'] if e.get('source')=='frame_hollow_marker')
        for field,value in [('outer_line_run',3),('connected_to_frame',False),('hole_vertices',[]),('hole_area',0)]:
            bad=copy.deepcopy(marker);bad['marker'][field]=value
            self.assertFalse(trusted_frame_pin_evidence(bad,self.box))
        bad=copy.deepcopy(marker);bad['anchor_img'][0]-=10
        self.assertFalse(trusted_frame_pin_evidence(bad,self.box))

    def test_previous_marker_in_number_cell_is_removed_without_erasing_black_text_core(self):
        image=self.image(bowtie=True)
        for vertices in ([(200,75),(184,69),(184,81)],[(182,69),(182,81),(166,75)]):
            cv2.polylines(image,[np.array(vertices,np.int32)],True,(190,190,190),1)
        image[75,181:185]=0
        cv2.line(image,(130,75),(166,75),(0,0,0),1)
        cv2.putText(image,"21",(150,64),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_8)
        doc,_,_=self.recover(image)
        numbers=[c for c in mono_pin_boxes(doc,image)[0] if c['role']=='pin_number']
        self.assertEqual(len(numbers),2)
        self.assertTrue(all(not c['flags'] for c in numbers))
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        for c in numbers:
            self.assertTrue(all(gray[y,x]>=100 for x,y in c['cleanup_evidence']['known_line_exclusions']))

    def test_pale_real_outline_needs_black_outer_wire_and_is_rechecked_at_its_measured_threshold(self):
        image=self.image()
        image[np.all(image==190,axis=2)]=238
        doc,_,_=self.recover(image)
        markers=[e for e in doc['pin_evidence']['ic'] if e.get('source')=='frame_hollow_marker']
        self.assertEqual(len(markers),1)
        self.assertEqual(markers[0]['marker']['ink_threshold'],245)
        self.assertTrue(mono_pin_boxes(doc,image)[0])
        changed=image.copy();changed[np.all(changed==238,axis=2)]=255
        self.assertFalse(mono_pin_boxes(doc,changed)[0])
        no_line=self.image(line=False);no_line[np.all(no_line==190,axis=2)]=238
        doc,_,_=self.recover(no_line)
        self.assertFalse(doc['pin_mono']['ic']['marker_rows'])

    def test_half_pixel_right_border_keeps_the_real_marker_row(self):
        from pinmarkers import hollow_marker_bands
        image=cv2.flip(self.image(),1)
        self.box=[149.,40.,399.,380.]
        mask=image.max(2)<230
        coverage=float(np.mean(mask[48:373,399]))
        bands=hollow_marker_bands(mask,self.box,1,(398.5,coverage>=.65,coverage))
        self.assertEqual(len(bands),1)
        band=bands[0];point=[410.5,band['anchor_img'][1]]
        marker=dict(point_img=point,anchor_img=band['anchor_img'],edge='right',
                    trust='trusted',frame_confirmed=True,frame_coverage=coverage,
                    source='frame_hollow_marker',reason='frame_attached_hollow_marker',marker=band['marker'])
        doc=dict(components=[dict(id='ic',type='box',bbox_img=self.box)],
                 pins={'ic':[point]},pin_evidence={'ic':[marker]})
        self.assertEqual(marker['anchor_img'][0],398.5)
        self.assertTrue(trusted_frame_pin_evidence(marker,self.box))
        self.assertTrue(mono_pin_boxes(doc,image)[0])
        detached=copy.deepcopy(marker)
        detached['marker']['connected_marker_region'][0]+=2
        self.assertFalse(trusted_frame_pin_evidence(detached,self.box))

    def test_black_line_core_is_measured_separately_from_gray_raster_fringe(self):
        image=self.image()
        image[99,130:184]=210
        image[101,130:184]=210
        doc,_,_=self.recover(image)
        boxes,_=mono_pin_boxes(doc,image)
        self.assertTrue(boxes)
        stroke=boxes[0]['cleanup_evidence']['stroke_probe']
        self.assertEqual(stroke['stroke_top'],100)
        self.assertEqual(stroke['stroke_bottom'],101)
        self.assertGreater(stroke['marker_outer_line_probe'][3]-stroke['marker_outer_line_probe'][1],1)
        changed=image.copy()
        changed[100,130:184]=210
        self.assertFalse(mono_pin_boxes(doc,changed)[0])

    def test_previous_number_contact_does_not_hold_a_separate_current_number_cell(self):
        image=self.image(bowtie=True)
        for vertices in ([(200,75),(184,69),(184,81)],[(182,69),(182,81),(166,75)]):
            cv2.polylines(image,[np.array(vertices,np.int32)],True,(190,190,190),1)
        image[75,181:185]=0
        cv2.line(image,(130,75),(166,75),(0,0,0),1)
        cv2.putText(image,'21',(150,64),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_8)
        # The older glyph's stem visibly meets the older contour, well above
        # the current number cell. It must be preserved and remain held there.
        image[63:72,184]=0
        source=image.copy();doc,_,_=self.recover(image)
        numbers=[c for c in mono_pin_boxes(doc,image)[0] if c['role']=='pin_number']
        previous=next(c for c in numbers if c['pin_target']['anchor_img'][1]==75)
        current=next(c for c in numbers if c['pin_target']['anchor_img'][1]==100)
        self.assertIn('glyph_integrity_unresolved',previous['flags'])
        self.assertFalse(current['flags'])
        self.assertGreater(sum(e.get('protected_core_pixels_outside_current_cell',0)
                               for e in current['cleanup_evidence']['measured_line_evidence']),0)
        np.testing.assert_array_equal(image,source)


if __name__=='__main__':
    unittest.main()
