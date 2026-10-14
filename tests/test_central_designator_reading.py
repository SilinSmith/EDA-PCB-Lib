"""Only a literal, central identity with a current frame gains rereading."""
import copy
import unittest
import cv2
import numpy as np
from stage2.designator_reading import central_caption_candidates,read_designator_captions


class Provider:
    def __init__(self,values=('U23A','U23A')):
        self.values=iter(values);self.inputs=[]
    def read_glyph(self,gray):
        self.inputs.append(gray.copy())
        return dict(text=next(self.values),confidence=.98)


class CentralCaptionTests(unittest.TestCase):
    def setUp(self):
        self.image=np.full((700,700,3),255,np.uint8)
        cv2.rectangle(self.image,(100,100),(600,600),(0,0,0),1)
        self.image[300:312,320:354]=0
        self.components=[dict(id='ic',type='box',source='yolo',bbox_img=[100,100,600,600])]
        self.pins={'ic':[[90,300],[610,300]]}
        self.evidence={'ic':[dict(point_img=point,anchor_img=[100 if index==0 else 600,300],
                    edge='left' if index==0 else 'right',trust='trusted',frame_confirmed=True,
                    reason='frame_attached_normal_stroke',frame_coverage=1.,normal_ink_run=22,
                    normal_ink_hits=22,first_ink_distance=1,text_overlap_fraction=0,branching_fraction=0)
                    for index,point in enumerate(self.pins['ic'])]}
        self.reads=[dict(bbox=[320,300,354,312],reads={'designator':('U23A',.8)})]

    def candidates(self):
        return central_caption_candidates(self.image,self.components,self.reads,self.pins,self.evidence,{'U','IC'})

    def read(self,provider):
        return read_designator_captions(self.image,self.components,self.reads,provider=provider,
                                        pins=self.pins,pin_evidence=self.evidence)

    def test_actual_central_suffix_is_read_literally_from_two_original_views(self):
        snapshot=copy.deepcopy((self.components,self.reads,self.pins,self.evidence));source=self.image.copy()
        provider=Provider();output,audit=self.read(provider)
        self.assertEqual(output[0]['reads'],{'designator':('U23A',.98)})
        self.assertEqual(output[0]['designator_target'],'ic')
        self.assertTrue(audit[0]['measured_central_caption'])
        for padding,pixels in zip((4,6),provider.inputs):
            np.testing.assert_array_equal(pixels,np.pad(self.image[300:312,320:354,0],padding,constant_values=255))
        np.testing.assert_array_equal(self.image,source)
        self.assertEqual((self.components,self.reads,self.pins,self.evidence),snapshot)

    def test_strong_old_suffix_still_needs_literal_confirmation(self):
        self.reads[0]['reads']['designator']=('U23A',.99)
        output,_=self.read(Provider())
        self.assertEqual(output[0]['reads']['designator'][0],'U23A')
        output,_=self.read(Provider(('U23A','U23B')))
        self.assertEqual(output,[])

    def test_missing_current_frame_ambiguous_owner_and_shape_only_body_are_rejected(self):
        for variant in ('frame','owner','cv','evidence'):
            with self.subTest(variant=variant):
                self.setUp()
                if variant=='frame':self.image[100:601,100]=255
                elif variant=='owner':self.components.append(dict(self.components[0],id='other'))
                elif variant=='cv':self.components[0]['source']='cv:bigic'
                else:self.evidence['ic'][0]['anchor_img']=[170,300]
                self.assertEqual(self.candidates(),[])

    def test_near_border_pin_word_and_unobserved_identity_do_not_open_a_caption(self):
        self.reads[0]['bbox']=[110,300,144,312]
        self.assertEqual(self.candidates(),[])
        self.setUp();self.reads[0]['reads']={'pin_name':('VDD',.99)}
        self.assertEqual(self.candidates(),[])
        self.setUp();self.reads[0]['reads']={'designator':('U23A',.2)}
        self.assertEqual(self.candidates(),[])

    def test_compact_identity_in_a_real_pin_corridor_is_not_an_internal_caption(self):
        self.image[:]=255
        cv2.rectangle(self.image,(100,100),(240,600),(0,0,0),1)
        self.components[0]['bbox_img']=[100,100,240,600]
        self.pins['ic'][1]=[250,300]
        self.evidence['ic'][1].update(point_img=[250,300],anchor_img=[240,300])
        self.reads[0]['bbox']=[135,300,169,312]
        self.assertEqual(self.candidates(),[])


if __name__=='__main__':unittest.main()
