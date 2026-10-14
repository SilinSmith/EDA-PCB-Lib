"""RGB package ownership and four-terminal delivery, independent of OCR labels."""
import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stage1.rgb_led import group_components,clip_external_rays
from stage1 import wires,pintrace,nets
from stage2 import assemble

RED, GREEN, BLUE = (0, 0, 190), (0, 140, 0), (180, 0, 0)


def scene():
    image = np.full((220, 330, 3), 255, np.uint8)
    triangle = lambda points, colour: cv2.fillPoly(image, [np.array(points, np.int32)], colour)
    triangle([(190, 100), (190, 120), (210, 110)], BLUE)
    triangle([(190, 150), (190, 170), (210, 160)], GREEN)
    triangle([(110, 160), (130, 150), (130, 170)], RED)
    for a, b in [((81, 110), (189, 110)), ((165, 110), (165, 160)),
                 ((131, 160), (189, 160)), ((211, 110), (260, 110)),
                 ((211, 160), (260, 160)), ((70, 160), (109, 160))]:
        cv2.line(image, a, b, RED, 1)
    for a, b in [((20, 110), (80, 110)), ((261, 110), (310, 110)),
                 ((261, 160), (310, 160)), ((20, 160), (69, 160))]:
        cv2.line(image, a, b, GREEN, 1)
    comps, pins, report = [], {}, {}
    for cid, box, point in [('b', [190, 100, 210, 120], (260.5, 110)),
                            ('g', [190, 150, 210, 170], (260.5, 160)),
                            ('r', [110, 150, 130, 170], (69.5, 160))]:
        comps.append(dict(id=cid, type='led', pending=False, bbox_img=box, bbox_raw_img=box,
                          center_img=[(box[0]+box[2])/2, (box[1]+box[3])/2]))
        pins[cid] = [point]
        report[cid] = dict(type='led', after=[point], sources=['body_attached_red_green_transition'],
                           connection_evidence=[dict(kind='synthetic_observed_boundary')], axis_y=False)
    return image, comps, pins, report


class RGBGroupingTests(unittest.TestCase):
    def test_three_internal_subsymbols_become_one_four_terminal_package(self):
        image, comps, pins, report = scene()
        got, points, records, groups = group_components(comps, pins, report, image)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]['terminal_group'], 'rgb_led')
        self.assertEqual(points[got[0]['id']], [(80.5, 110.0), (260.5, 110), (260.5, 160), (69.5, 160)])
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0]['member_ids']), {'r', 'g', 'b'})
        self.assertEqual(len(records[got[0]['id']]['after']), 4)

    def test_same_colour_leds_are_not_an_rgb_package(self):
        image, comps, pins, report = scene()
        image[np.all(image == GREEN, axis=2)] = RED
        image[np.all(image == BLUE, axis=2)] = RED
        self.assertEqual(group_components(comps, pins, report, image), (comps, pins, report, []))

    def test_green_external_interconnect_does_not_define_internal_ownership(self):
        image, comps, pins, report = scene()
        cv2.line(image, (165, 121), (165, 149), GREEN, 1)
        self.assertEqual(group_components(comps, pins, report, image), (comps, pins, report, []))

    def test_internal_blank_does_not_get_bridged(self):
        image, comps, pins, report = scene()
        image[132:135, 165] = 255
        self.assertEqual(group_components(comps, pins, report, image), (comps, pins, report, []))

    def test_missing_fourth_terminal_blocks_grouping(self):
        image, comps, pins, report = scene()
        image[110, 20:81] = 255
        self.assertEqual(group_components(comps, pins, report, image), (comps, pins, report, []))

    def test_real_outline_area_does_not_overrule_green_or_blue_fill(self):
        image, comps, pins, report = scene()
        for comp in comps[:2]:
            a,b,c,d=comp['bbox_img']
            cv2.rectangle(image,(a,b),(c,d),RED,2)
        self.assertEqual(len(group_components(comps,pins,report,image)[3]),1)

    def test_rotation_keeps_all_physical_terminals(self):
        image, comps, pins, report = scene()
        width=image.shape[1]
        rotate=lambda pt:(pt[1],width-1-pt[0])
        for comp in comps:
            a,b,c,d=comp['bbox_img']
            comp.update(bbox_img=[b,width-1-c,d,width-1-a],bbox_raw_img=[b,width-1-c,d,width-1-a])
        for cid in pins:
            pins[cid]=[rotate(pt)for pt in pins[cid]]
            report[cid]['after']=list(pins[cid])
        got,points,_,groups=group_components(comps,pins,report,np.rot90(image))
        self.assertEqual(len(groups),1)
        self.assertEqual(sorted(points[got[0]['id']]),sorted(map(rotate,[(80.5,110),(260.5,110),(260.5,160),(69.5,160)])))

    def test_ic_and_inputs_are_preserved_and_original_parts_stay_in_ledger(self):
        image, comps, pins, report = scene()
        ic=dict(id='ic',type='box',bbox_img=[270,20,310,60],bbox_raw_img=[270,20,310,60])
        comps.append(ic);pins['ic']=[(270,40)]
        before=copy.deepcopy((image,comps,pins,report))
        got,points,_,groups=group_components(comps,pins,report,image)
        self.assertEqual(next(c for c in got if c['id']=='ic'),ic)
        self.assertEqual(points['ic'],pins['ic'])
        self.assertEqual(len(groups[0]['original_components']),3)
        np.testing.assert_array_equal(image,before[0])
        self.assertEqual((comps,pins,report),before[1:])

    def test_assembly_keeps_four_points_and_does_not_apply_two_pin_stamp(self):
        image, comps, pins, report = scene()
        comps,pins,report,groups=group_components(comps,pins,report,image)
        stage=dict(components=comps,pins=pins,passive_pin_report=report,
                   image=dict(w=330,h=220),nets=[])
        with patch.object(assemble,'DROP_UNKEYED',False):doc=assemble.build_doc(stage,{},'synthetic')
        self.assertEqual(len(doc['components']),1)
        self.assertEqual(len(doc['pins']['p000']),4)
        for index,point in enumerate(pins[comps[0]['id']]):
            key=doc['__pin_index'][comps[0]['id']][str(index)]
            self.assertEqual(doc['pins']['p000'][key]['point'],dict(x=point[0],y=220-point[1]))

    def test_recovered_package_mask_keeps_four_external_networks_separate(self):
        image,comps,pins,report=scene()
        comps,pins,_,groups=group_components(comps,pins,report,image)
        geometry,_=wires.extract_wires(image,occlude=[c['bbox_img']for c in comps],
                                       drop_frame=False,min_extent=0,erase=[])
        ink=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<pintrace.INK).astype(np.uint8)
        segments=geometry['segments']+pintrace.trace_from_boxes(ink,[c['bbox_img']for c in comps])
        network,_=nets.build_nets(pins,segments,ink=ink,components=comps)
        cid=groups[0]['id']
        memberships=[n['id']for n in network if any(p[0]==cid for p in n['pins'])]
        self.assertEqual(len(set(memberships)),4)

    def test_a_neighbours_ray_is_split_before_it_can_short_the_package(self):
        group=dict(bbox_img=[110,100,210,170])
        segments=[[[20,110],[310,110]],[[20,190],[310,190]]]
        before=copy.deepcopy(segments)
        clipped=clip_external_rays(segments,[group])
        self.assertEqual(clipped,[[[20,110],[109.5,110.0]],[[210.5,110.0],[310,110]],[[20,190],[310,190]]])
        self.assertEqual(segments,before)


if __name__=='__main__':
    unittest.main()
