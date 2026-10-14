"""Shared red ink must not assign a neighbour's terminal to this component."""
import unittest

import cv2
import numpy as np

import test_passive_pins as fixtures
from stage1.passive_pins import refine_pins


class TerminalBodyOwnershipTests(unittest.TestCase):
    def chain(self):
        image, comp, old = fixtures.PassivePinTests().resistor()
        image[111:181, 98:105] = 255
        cv2.line(image, (100,110),(100,130),(0,0,200),1)
        cv2.rectangle(image,(96,130),(104,140),(0,0,200),1)
        cv2.line(image,(100,140),(100,155),(0,0,200),1)
        cv2.line(image,(100,156),(100,180),(0,140,0),1)
        peer=dict(id='peer',type='r',bbox_img=[96,130,104,140],bbox_raw_img=[96,130,104,140])
        old['peer']=[(100,145)]
        return image,comp,peer,old

    def test_red_connected_resistor_does_not_supply_its_far_terminal(self):
        image,comp,peer,old=self.chain()
        result,report=refine_pins([comp,peer],old,image)
        self.assertEqual(result['r'],[(100,74.5)])
        self.assertEqual(report['r']['after'],[(100,74.5)])
        self.assertEqual(result['peer'],[(100,155.5)])

    def test_inset_peer_box_still_blocks_a_path_through_its_body(self):
        image,comp,peer,old=self.chain()
        peer.update(bbox_img=[98,132,102,138],bbox_raw_img=[98,132,102,138])
        result,_=refine_pins([comp,peer],old,image)
        self.assertEqual(result['r'],[(100,74.5)])

    def test_ownership_guard_survives_a_quarter_turn(self):
        image,comp,peer,old=self.chain()
        rotate=lambda p:(p[1],219-p[0])
        turned=[]
        for c in (comp,peer):
            corners=np.asarray([rotate(c['bbox_img'][:2]),rotate(c['bbox_img'][2:])])
            c=dict(c,bbox_img=[*corners.min(axis=0),*corners.max(axis=0)])
            c['bbox_raw_img']=c['bbox_img']
            turned.append(c)
        result,_=refine_pins(turned,{cid:[rotate(p)for p in points]for cid,points in old.items()},np.rot90(image))
        np.testing.assert_allclose(result['r'],[rotate((100,74.5))])

    def test_supply_uses_the_direct_contact_not_the_series_resistors_far_terminal(self):
        image=np.full((220,220,3),255,np.uint8)
        cv2.line(image,(96,90),(104,90),(0,0,200),1)
        cv2.line(image,(100,90),(100,110),(0,0,200),1)
        cv2.rectangle(image,(96,110),(104,130),(0,0,200),1)
        cv2.line(image,(100,130),(100,135),(0,0,200),1)
        cv2.line(image,(100,136),(100,180),(0,140,0),1)
        supply=dict(id='v',type='v',bbox_img=[96,88,104,94],bbox_raw_img=[96,88,104,94])
        resistor=dict(id='r',type='r',bbox_img=[96,110,104,130],bbox_raw_img=[96,110,104,130])
        old={'v':[(100,100)],'r':[(100,136)]}
        result,report=refine_pins([supply,resistor],old,image)
        # The newly authorised opposed-length rule can locate the resistor's
        # near end and share it with its directly attached supply. It must
        # never give that supply the far end beyond the resistor body.
        self.assertEqual(result['v'],[(100.0,104.5)])
        self.assertEqual(result['r'],[(100.0,104.5),(100.0,135.5)])
        self.assertEqual(report['v']['sources'],['peer_shared_terminal'])
        self.assertEqual(report['v']['connection_evidence'][0]['peer_id'],'r')
        self.assertNotIn((100.0,135.5),result['v'])

    def test_lateral_neighbour_does_not_block_a_real_terminal(self):
        image,comp,old=fixtures.PassivePinTests().resistor()
        peer=dict(id='peer',type='r',bbox_img=[115,115,125,135],bbox_raw_img=[115,115,125,135])
        result,_=refine_pins([comp,peer],old,image)
        self.assertEqual(result['r'],[(100,74.5),(100,125.5)])

    def test_body_beyond_the_measured_green_boundary_does_not_block_it(self):
        image,comp,old=fixtures.PassivePinTests().resistor()
        peer=dict(id='peer',type='r',bbox_img=[96,40,104,60],bbox_raw_img=[96,40,104,60])
        result,_=refine_pins([comp,peer],old,image)
        self.assertEqual(result['r'],[(100,74.5),(100,125.5)])

    def test_overlapping_detector_proposal_is_not_a_separate_obstacle(self):
        image,comp,old=fixtures.PassivePinTests().resistor()
        peer=dict(id='peer',type='d',bbox_img=[96,108,104,118],bbox_raw_img=[96,108,104,118])
        result,_=refine_pins([comp,peer],old,image)
        self.assertEqual(result['r'],[(100,74.5),(100,125.5)])

    def test_detector_edge_near_the_lead_does_not_block_a_real_boundary(self):
        image,comp,old=fixtures.PassivePinTests().resistor()
        peer=dict(id='peer',type='gnd',bbox_img=[99.75,112,110,122],bbox_raw_img=[99.75,112,110,122])
        result,_=refine_pins([comp,peer],old,image)
        self.assertEqual(result['r'],[(100,74.5),(100,125.5)])

    def test_clean_offset_terminal_keeps_axis_support_without_publishing_neighbour_tip(self):
        image,comp,peer,old=self.chain()
        comp.update(bbox_img=[102,90,110,110],bbox_raw_img=[102,90,110,110])
        result,report=refine_pins([comp,peer],old,image)
        self.assertEqual(result['r'],[(100,74.5)])
        self.assertEqual(report['r']['connection_evidence'][0]['axis_support'],
                         'aligned_red_chain_with_intervening_body')


if __name__=='__main__':
    unittest.main()
