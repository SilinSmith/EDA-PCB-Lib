"""Behavioral fixture: full word next to frame, unrelated ink farther inside."""
import unittest
import cv2
import numpy as np
from stage2.vertical_pin_reading import ClockwisePinView
from stage2.mono_pin_crops import mono_pin_boxes
from stage2.vertical_pin_crops import complete_vertical_pin_boxes

class VerticalNameBorderTests(unittest.TestCase):
    def test_complete_name_next_to_frame_excludes_separate_interior_word(self):
        horizontal=np.full((160,200,3),255,np.uint8)
        cv2.rectangle(horizontal,(30,10),(140,150),(0,0,190),1)
        cv2.line(horizontal,(140,100),(155,100),(0,0,190),1)
        # Three compact glyph bodies end immediately before the measured frame.
        for x in (124,130,136):
            horizontal[96:103,x:x+3]=0
        horizontal[96:103,111:115]=0
        image=cv2.rotate(horizontal,cv2.ROTATE_90_COUNTERCLOCKWISE)
        doc=dict(components=[dict(id='ic',type='box',bbox_img=[10,59,150,169])],
                 pins={'ic':[[100,44]]},pin_evidence={'ic':[dict(point_img=[100,44],
                 anchor_img=[100,59],edge='top',trust='trusted',frame_confirmed=True,
                 frame_coverage=1.,source='red_stub_endpoint',color_evidence=dict(
                 red_hits=15,red_span=15,red_coverage=1.,stroke_width=1))]})
        view=ClockwisePinView(*image.shape[:2]);directional=view.document(doc)
        candidates,_=mono_pin_boxes(directional,horizontal)
        found=complete_vertical_pin_boxes(directional,horizontal,candidates)
        names=[c for c in found if c['role']=='pin_name']
        self.assertEqual(len(names),1)
        self.assertEqual(names[0]['bbox'],[124,96,139,103])
        self.assertFalse(names[0]['flags'])

if __name__=='__main__':unittest.main()
