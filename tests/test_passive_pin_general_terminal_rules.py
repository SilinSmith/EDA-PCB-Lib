"""Connection roles, measurable symmetry and non-red node regressions."""
import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stage1.passive_pins import refine_pins
from stage2 import assemble

RED=(0,0,190)
GREEN=(0,140,0)
BLUE=(180,0,0)
BLACK=(0,0,0)


def capacitor(body=RED, lead=RED, wire=GREEN):
    image=np.full((220,220,3),255,np.uint8)
    for x in (96,104):cv2.line(image,(x,90),(x,110),body,1)
    cv2.line(image,(80,100),(95,100),lead,1)
    cv2.line(image,(105,100),(120,100),lead,1)
    cv2.line(image,(30,100),(79,100),wire,1)
    cv2.line(image,(121,100),(195,100),wire,1)
    comp=dict(id='c',type='c',bbox_img=[93,86,108,114],bbox_raw_img=[93,86,108,114],
              center_img=[100.5,100],pending=False)
    return image,comp,{'c':[(81,100),(119,100)]}


def direct_ground():
    image,comp,pins=capacitor()
    image[98:103,121:196]=255
    cv2.line(image,(105,100),(150,100),RED,1)
    for x,half in ((150,10),(154,6),(158,2)):
        cv2.line(image,(x,100-half),(x,100+half),RED,1)
    ground=dict(id='g',type='gnd',bbox_img=[148,87,161,113],bbox_raw_img=[148,87,161,113],
                center_img=[154.5,100],pending=False)
    pins['g']=[(132,100)]
    return image,[comp,ground],pins


class GeneralTerminalRulesTests(unittest.TestCase):
    def test_unmarked_contact_uses_observed_body_symmetry_and_is_shared_with_ground(self):
        image,comps,pins=direct_ground()
        result,report=refine_pins(comps,pins,image)
        self.assertEqual(result['c'],[(79.5,100.0),(120.5,100.0)])
        self.assertEqual(result['g'],[(120.5,100.0)])
        self.assertEqual(report['c']['sources'][-1],'symmetric_opposed_terminal')
        self.assertEqual(report['g']['sources'],['peer_shared_terminal'])

    def test_detector_centre_offset_does_not_move_the_reflected_contact(self):
        image,comps,pins=direct_ground()
        comps[0].update(bbox_raw_img=[91,88,106,113],bbox_img=[91,88,106,113])
        result,report=refine_pins(comps,pins,image)
        self.assertEqual(result['c'][-1],(120.5,100.0))
        self.assertEqual(report['c']['connection_evidence'][-1]['body_geometry']['low'],95.5)

    def test_quarter_turn_keeps_the_same_physical_symmetric_contact(self):
        image,comps,pins=direct_ground()
        rotate=lambda pt:(pt[1],219-pt[0])
        for comp in comps:
            x0,y0,x1,y1=comp['bbox_raw_img']
            comp.update(bbox_img=[y0,219-x1,y1,219-x0],bbox_raw_img=[y0,219-x1,y1,219-x0])
        got,_=refine_pins(comps,{cid:[rotate(pt)for pt in pts]for cid,pts in pins.items()},np.rot90(image))
        np.testing.assert_allclose(sorted(got['c']),sorted([rotate((79.5,100)),rotate((120.5,100))]))
        self.assertEqual(got['g'],[rotate((120.5,100))])

    def test_both_unmarked_ends_do_not_invent_a_measured_length(self):
        image,comps,pins=direct_ground()
        cv2.line(image,(30,100),(95,100),RED,1)
        _,report=refine_pins(comps,pins,image)
        self.assertNotIn('symmetric_opposed_terminal',report.get('c',{}).get('sources',[]))

    def test_symmetry_requires_a_continuous_facing_peer(self):
        image,comps,pins=direct_ground()
        image[99:102,135:139]=255
        _,report=refine_pins(comps,pins,image)
        self.assertNotIn('symmetric_opposed_terminal',report.get('c',{}).get('sources',[]))
        self.assertNotIn('peer_shared_terminal',report.get('g',{}).get('sources',[]))

    def test_a_nonsymmetric_body_cannot_supply_equal_length_evidence(self):
        image,comps,pins=direct_ground()
        image[90:111,104]=255
        cv2.line(image,(104,97),(104,103),RED,1)
        _,report=refine_pins(comps,pins,image)
        self.assertNotIn('symmetric_opposed_terminal',report.get('c',{}).get('sources',[]))

    def test_blue_body_black_stem_blue_wire_uses_the_external_boundary(self):
        image,comp,pins=capacitor(BLUE,BLACK,BLUE)
        result,report=refine_pins([comp],pins,image)
        self.assertEqual(result['c'],[(79.5,100.0),(120.5,100.0)])
        self.assertEqual(report['c']['sources'],['lead_wire_colour_boundary']*2)
        self.assertEqual(report['c']['connection_evidence'][0]['lead_pigment'],'neutral')

    def test_black_symbol_red_wire_is_a_colour_image_and_uses_the_black_red_boundary(self):
        image,comp,pins=capacitor(BLACK,BLACK,RED)
        result,report=refine_pins([comp],pins,image)
        self.assertEqual(result['c'],[(79.5,100.0),(120.5,100.0)])
        self.assertEqual(report['c']['connection_evidence'][0]['wire_pigment'],'red')

    def test_monochrome_and_ic_inputs_remain_unchanged(self):
        image,comp,pins=capacitor(BLACK,BLACK,BLACK)
        self.assertEqual(refine_pins([comp],pins,image),(pins,{}))
        image,comp,pins=capacitor(BLACK,BLACK,BLUE)
        ic=dict(comp,type='box')
        self.assertEqual(refine_pins([ic],pins,image),(pins,{}))

    def test_coloured_annotations_do_not_enable_grayscale_node_optimization(self):
        image,comp,pins=capacitor(BLACK,BLACK,BLACK)
        cv2.circle(image,(125,100),3,BLACK,-1)
        cv2.putText(image,'C1',(90,65),cv2.FONT_HERSHEY_SIMPLEX,.6,BLUE,1)
        self.assertEqual(refine_pins([comp],pins,image),(pins,{}))

    def test_direct_blue_filled_node_uses_its_centre(self):
        image,comp,pins=capacitor()
        cv2.line(image,(105,100),(124,100),RED,1)
        cv2.line(image,(129,100),(195,100),GREEN,1)
        cv2.circle(image,(125,100),3,BLUE,-1)
        result,report=refine_pins([comp],pins,image)
        self.assertEqual(result['c'][-1],(125.0,100.0))
        self.assertEqual(report['c']['sources'][-1],'first_filled_junction_center')

    def test_an_earlier_colour_boundary_has_priority_over_a_later_node(self):
        image,comp,pins=capacitor(RED,RED,(160,0,160))
        cv2.circle(image,(160,100),3,BLUE,-1)
        cv2.line(image,(160,104),(160,180),GREEN,1)
        result,report=refine_pins([comp],pins,image)
        self.assertEqual(result['c'][-1],(120.5,100.0))
        self.assertEqual(report['c']['sources'][-1],'lead_wire_colour_boundary')

    def test_a_blue_text_block_off_the_axis_does_not_supply_a_junction(self):
        image,comp,pins=capacitor()
        image[98:103,121:196]=255
        cv2.line(image,(105,100),(140,100),RED,1)
        cv2.putText(image,'8',(142,98),cv2.FONT_HERSHEY_SIMPLEX,.6,BLUE,2)
        _,report=refine_pins([comp],pins,image)
        self.assertNotIn('first_filled_junction_center',report.get('c',{}).get('sources',[]))

    def test_coloured_body_extension_to_black_pin_is_an_internal_change(self):
        image,comp,pins=capacitor(BLUE,BLACK,BLUE)
        cv2.line(image,(91,100),(95,100),BLUE,1)
        cv2.line(image,(105,100),(109,100),BLUE,1)
        result,report=refine_pins([comp],pins,image)
        self.assertEqual(result['c'],[(79.5,100.0),(120.5,100.0)])
        for proof in report['c']['connection_evidence']:
            self.assertEqual(proof['lead_pigment'],'neutral')
            self.assertTrue(proof['internal_body_stem_pixels_img'])

    def test_a_weak_lateral_mixed_mark_blocks_the_symmetry_fallback(self):
        image,comps,pins=direct_ground()
        image[101,119:121]=(207,219,220)
        _,report=refine_pins(comps,pins,image)
        self.assertNotIn('symmetric_opposed_terminal',report.get('c',{}).get('sources',[]))
        self.assertNotIn('peer_shared_terminal',report.get('g',{}).get('sources',[]))

    def test_internal_body_pin_antialiasing_does_not_become_an_external_contact(self):
        image,comp,pins=capacitor(BLUE,BLACK,BLUE)
        cv2.line(image,(91,100),(95,100),BLUE,1)
        cv2.line(image,(105,100),(109,100),BLUE,1)
        image[100,90]=(180,200,205)
        image[100,110]=(180,200,205)
        result,report=refine_pins([comp],pins,image)
        self.assertEqual(result['c'],[(79.5,100.0),(120.5,100.0)])
        self.assertEqual(report['c']['connection_evidence'][0]['lead_pigment'],'neutral')

    def test_final_assembly_preserves_inferred_and_shared_coordinates(self):
        image,comps,pins=direct_ground()
        result,report=refine_pins(comps,pins,image)
        stage=dict(components=comps,pins=result,passive_pin_report=report,
                   image=dict(w=220,h=220),nets=[])
        with patch.object(assemble,'DROP_UNKEYED',False):doc=assemble.build_doc(stage,{},'synthetic')
        for index,comp in enumerate(comps):
            cid=comp['id']
            for i,point in enumerate(result[cid]):
                pk=doc['__pin_index'][cid][str(i)]
                self.assertEqual(doc['pins'][f'p{index:03d}'][pk]['point'],dict(x=point[0],y=220-point[1]))

    def test_refinement_does_not_mutate_inputs(self):
        image,comps,pins=direct_ground()
        before=copy.deepcopy((image,comps,pins))
        refine_pins(comps,pins,image)
        np.testing.assert_array_equal(image,before[0])
        self.assertEqual(comps,before[1]);self.assertEqual(pins,before[2])


if __name__=='__main__':unittest.main()
