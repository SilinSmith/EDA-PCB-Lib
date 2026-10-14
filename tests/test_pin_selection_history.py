import copy
import unittest
from unittest.mock import patch

import numpy as np

from stage2.mono_pin_reading import _guard_recovered_names
from stage2.assign import assign


class PinSelectionHistoryTests(unittest.TestCase):
    def test_repeated_name_reads_keep_the_first_duplicate_ranking_observation(self):
        old_name=dict(text='READY',conf=.99,box=[111,174,151,188])
        frozen=dict(text='RDY',conf=.55,box=[111,174,151,188])
        for initial in (frozen,None):
            old=dict(name=old_name,number=None,number_selection_name=initial)
            existing=dict(comps={'ic':dict(pins={0:old})})
            doc=dict(components=[],pins={},pin_evidence={})
            record=dict(reads={'pin_name':('READY',.98)},name_only=True,
                pin_target=dict(component_id='ic',pin_index=0,edge='left',point_img=[90,180],anchor_img=[100,180]),
                mono_pin_ocr=dict(accepted=True))
            with patch('stage2.assign.assign',return_value=existing):
                _guard_recovered_names(doc,[],[record],[],np.full((330,350,3),255,np.uint8))
            self.assertEqual(record['mono_pin_ocr']['previous_number_selection_name'],initial)
            self.assertEqual(old['name'],old_name)

    def test_assignment_preserves_frozen_rank_until_number_is_independently_read(self):
        comp=dict(id='ic',type='box',bbox_img=[100,100,300,300])
        point=[90,180]
        ev=dict(point_img=point,anchor_img=[100,180],edge='left',trust='trusted',
            frame_confirmed=True,reason='frame_attached_normal_stroke',frame_coverage=1.,
            normal_ink_run=12,normal_ink_hits=12,first_ink_distance=1,text_overlap_fraction=0.,branching_fraction=0.)
        legacy=dict(bbox=[78,168,96,177],number_only=True,reads={'pin_number':('1',.99)})
        new=dict(bbox=[111,174,151,188],name_only=True,reads={'pin_name':('READY',.98)},
            source='mono_pin_name_ppocr',pin_target=dict(component_id='ic',pin_index=0,edge='left',point_img=point,anchor_img=[100,180]),
            mono_pin_ocr=dict(backend='ppocr_v4_direct',accepted=True,flags=[],
                selected_views=[dict(padding=p,text='READY',confidence=.98) for p in (2,4)],
                previous_name=dict(text='READY',conf=.99,box=[111,174,151,188]),
                previous_number_selection_name=dict(text='RDY',conf=.55,box=[111,174,151,188])))
        source=copy.deepcopy(new)
        result=assign([comp],{'ic':[point]},[legacy,new],pin_evidence={'ic':[ev]})
        pin=result['comps']['ic']['pins'][0]
        self.assertEqual(pin['name']['text'],'READY')
        self.assertEqual(pin['number_selection_name']['text'],'RDY')
        self.assertEqual(pin['number_selection_name']['conf'],.55)
        self.assertEqual(new,source)


if __name__=='__main__':unittest.main()
