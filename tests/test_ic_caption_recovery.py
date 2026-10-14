import copy,unittest
import cv2,numpy as np
from stage2.designator_reading import read_designator_captions,_pixel_caption_boxes,_small_caption_views,qualified_small_caption
from stage2.assign import assign

class Reader:
    def __init__(self,values):self.values=iter(values);self.samples=[]
    def read_glyph(self,image):
        self.samples.append(image.copy());word,certainty=next(self.values)
        return dict(text=word,confidence=certainty)

class CaptionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.image=np.full((160,200,3),255,np.uint8)
        self.image[22:29,40:44]=(190,0,0);self.image[22:29,46:50]=(190,0,0)
        self.components=[dict(id='ic',type='box',source='yolo',bbox_img=[40,35,150,120])]
        self.reads=[dict(bbox=[39,21,52,30],reads={'designator':('U2',.95)})]

    def test_complete_strong_blue_identity_is_preserved_without_rereading(self):
        provider=Reader([])
        output,audit=read_designator_captions(self.image,self.components,self.reads,provider=provider)
        self.assertEqual(output,[]);self.assertEqual(provider.samples,[])

    def test_enclosing_layout_box_cannot_take_the_verified_local_caption(self):
        self.components.append(dict(id='container',type='box',source='cv:bigic',bbox_img=[0,0,180,150]))
        provider=Reader([('U2',.99),('U2',.98)]*2)
        extra,audit=read_designator_captions(self.image,self.components,self.reads,provider=provider)
        self.assertEqual(extra[0]['designator_target'],'ic')
        self.assertTrue(extra[0]['designator_ocr']['enclosing_container_conflict'])
        result=assign(self.components,{},self.reads+extra)
        self.assertEqual(result['comps']['ic']['designator']['canon'],'U2')
        self.assertIsNone(result['comps']['container']['designator'])

    def test_blue_word_missing_from_detector_requires_opposing_pin_frame_evidence(self):
        none=_pixel_caption_boxes(self.image,self.components,[])
        self.assertEqual(none,[])
        evidence={'ic':[dict(edge=side,trust='trusted',frame_confirmed=True) for side in ('left','right')]}
        found=_pixel_caption_boxes(self.image,self.components,[],evidence)
        self.assertTrue(any(r.get('caption_ink_family')=='blue' and r['bbox']==[39,21,51,30] for r in found))

    def test_cross_role_disagreement_rereads_original_caption_without_digit_guessing(self):
        self.reads[0]['reads']={'designator':('U22',.99),'value':('U202',.88)}
        provider=Reader([('U202',.99),('U202',.98)]*2)
        extra,audit=read_designator_captions(self.image,self.components,self.reads,provider=provider)
        original=next(r for r in extra if r['bbox']==self.reads[0]['bbox'])
        self.assertEqual(original['reads']['designator'][0],'U202')
        self.assertTrue(original['designator_ocr']['conflicting_role_reads'])

    def test_font_relative_margin_uses_complete_source_and_is_assignment_checked(self):
        glyph=np.full((7,11),255,np.uint8);glyph[1:6,1:5]=30;glyph[1:6,7:10]=30
        before=glyph.copy();box=[20,10,31,17];fragments=[[21,11,25,16],[27,11,30,16]]
        provider=Reader([('U2',.98),('U2',.97)])
        result=_small_caption_views(glyph,box,fragments,provider,[],{'U'})
        self.assertEqual(result['paddings'],[1,2]);np.testing.assert_array_equal(glyph,before)
        for p,pixels in zip((1,2),provider.samples):np.testing.assert_array_equal(pixels,np.pad(glyph,p,constant_values=255))
        meta=dict(bbox=box,pixel_caption_recovery=True,pixel_caption_fragments=fragments,
                  raw_reads=result['views'],paddings=result['paddings'],font_relative_caption=result['proof'])
        self.assertEqual(qualified_small_caption(meta,'U2',.97),'U2')
        for key in ('core_height','source_bbox','natural_dark_margin'):
            changed=copy.deepcopy(meta)
            changed['font_relative_caption'][key]=9 if key=='core_height' else [21,10,31,17] if key=='source_bbox' else False
            self.assertIsNone(qualified_small_caption(changed,'U2',.97))

    def test_font_relative_margin_rejects_cut_ink_and_strong_contradictory_read(self):
        glyph=np.full((7,11),255,np.uint8);glyph[1:6,1:5]=30;glyph[1:6,7:10]=30
        box=[20,10,31,17];fragments=[[21,11,25,16],[27,11,30,16]]
        cut=glyph.copy();cut[0,2]=30
        self.assertIsNone(_small_caption_views(cut,box,fragments,Reader([]),[],{'U'}))
        result=_small_caption_views(glyph,box,fragments,Reader([('U2',.98),('U2',.97)]),
                                   [dict(text='U3',confidence=.99)],{'U'})
        self.assertIsNone(result)

    def test_caption_budget_prioritizes_duplicate_identities_over_early_noise(self):
        components=[dict(id='a',type='box',bbox_img=[40,35,95,120]),
                    dict(id='b',type='box',bbox_img=[120,35,180,120])]
        reads=[dict(bbox=[x,22,x+10,29],reads={'designator':('NOISE',.8)}) for x in (45,60,75)]
        reads.extend([dict(bbox=[39,21,52,30],reads={'designator':('U1',.97)}),
                      dict(bbox=[120,21,134,30],reads={'designator':('U1',.98)})])
        extra,_=read_designator_captions(self.image,components,reads,limit=2,
            provider=Reader([('U1',.99),('U1',.98),('U2',.99),('U2',.98)]))
        observed={r['designator_target']:r['reads']['designator'][0] for r in extra}
        self.assertEqual(observed,{'a':'U1','b':'U2'})
        final=assign(components,{},reads+extra)
        self.assertEqual(final['comps']['a']['designator']['canon'],'U1')
        self.assertEqual(final['comps']['b']['designator']['canon'],'U2')

if __name__=='__main__':unittest.main()
