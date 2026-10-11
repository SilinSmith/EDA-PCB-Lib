"""Post-inference scoring only; labels never feed an image inference worker.
Final physical-row association uses exported pin.point, never stage1 indexes.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from scipy.optimize import linear_sum_assignment


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(x, y):
    w = max(0, min(x[2], y[2]) - max(x[0], y[0]))
    h = max(0, min(x[3], y[3]) - max(x[1], y[1]))
    q = w * h
    return q / max(1, (x[2] - x[0]) * (x[3] - x[1])
                   + (y[2] - y[0]) * (y[3] - y[1]) - q)


def edge(point, box):
    # A terminal left of the frame and within its vertical span is a left
    # terminal, even if its Euclidean distance to the top edge is smaller.
    # This is essential for short symbols with long terminal stubs.
    x, y = point
    if box[1] <= y <= box[3]:
        if x < box[0]:
            return 'left'
        if x > box[2]:
            return 'right'
    if box[0] <= x <= box[2]:
        if y < box[1]:
            return 'top'
        if y > box[3]:
            return 'bottom'
    distances = dict(left=abs(point[0] - box[0]),
                     top=abs(point[1] - box[1]),
                     right=abs(point[0] - box[2]),
                     bottom=abs(point[1] - box[3]))
    return min(distances, key=distances.get)


def export_to_image(point, height):
    if isinstance(point, dict):
        x, y = point['x'], point['y']
    else:
        x, y = point
    x, y = float(x), float(y)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError('nonfinite exported point')
    return [x, height - y]


def number(pin_key):
    return pin_key[4:].split('_')[0] if pin_key.startswith('pin_') else None


def boxes_match(reviewed, candidates):
    """One-to-one geometry; threshold fixed before candidate evaluation."""
    if not reviewed or not candidates:
        return {}
    costs = [[1 - iou(r, c) for _, c in candidates] for r in reviewed]
    rr, cc = linear_sum_assignment(costs)
    return {int(r): candidates[int(c)][0]
            for r, c in zip(rr, cc) if 1 - costs[r][c] > .5}


def row_assignment(rows, items, side):
    axis = 1 if side in ('left', 'right') else 0
    rr = [r for r in rows if r['side'] == side]
    ii = [(i, o) for i, o in enumerate(items) if o.get('side') == side]
    if not rr or not ii:
        return []
    costs = [[abs(o['point'][axis] - r['frame_point_for_row_matching_only'][axis])
              if abs(o['point'][axis] - r['frame_point_for_row_matching_only'][axis])
              <= r['physical_row_tolerance'] else 100000 for _, o in ii] for r in rr]
    ridx, oidx = linear_sum_assignment(costs)
    return [(rr[int(r)], ii[int(o)][0], ii[int(o)][1], costs[r][o])
            for r, o in zip(ridx, oidx) if costs[r][o] < 100000]


def set_tolerances(rows):
    for side in ('left', 'right', 'top', 'bottom'):
        axis = 1 if side in ('left', 'right') else 0
        side_rows = sorted((r for r in rows if r['side'] == side),
                           key=lambda r: r['frame_point_for_row_matching_only'][axis])
        locs = [r['frame_point_for_row_matching_only'][axis] for r in side_rows]
        for i, row in enumerate(side_rows):
            gaps = [abs(locs[i] - locs[j]) for j in (i-1, i+1)
                    if 0 <= j < len(locs)]
            row['physical_row_tolerance'] = .45 * min(gaps) if gaps else 8


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--stage1-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or base/'inference'/args.arm
    report = args.report or base/'inference'/f'{args.arm}-physical-results.json'
    labels = load(args.labels)
    all_rows, extras, components, provenance = [], [], [], []
    totals = defaultdict(int)
    cases = sorted({c['case'] for c in labels['components']})
    for case in cases:
        stage1_path = args.stage1_dir/f'{case}.stage1.json'
        output_path = output_dir/f'{case}.json'
        assigned_path = output_dir/'_assigned'/f'{case}.json'
        index_path = output_dir/f'{case}.pinindex.json'
        stage1, pred = load(stage1_path), load(output_path)
        assigned = load(assigned_path).get('comps', {}) if assigned_path.exists() else {}
        indexes = load(index_path) if index_path.exists() else {}
        height = stage1['image']['h']
        case_labels = [c for c in labels['components'] if c['case'] == case]
        reviewed_boxes = [[c['body_bbox'][k] for k in ('x1', 'y1', 'x2', 'y2')]
                          for c in case_labels]
        stage1_candidates = [(c['id'], c['bbox_img']) for c in stage1['components']]
        export_candidates = [(key, [c['bbox'][0], height-c['bbox'][3],
                                    c['bbox'][2], height-c['bbox'][1]])
                             for key, c in pred['components'].items()]
        stage1_match = boxes_match(reviewed_boxes, stage1_candidates)
        export_match = boxes_match(reviewed_boxes, export_candidates)
        stage1_boxes = dict(stage1_candidates)
        export_boxes = dict(export_candidates)
        provenance.append(dict(case=case, stage1=str(stage1_path),
                               stage1_sha256=sha(stage1_path), output=str(output_path),
                               output_sha256=sha(output_path)))
        for component_index, label in enumerate(case_labels):
            box = reviewed_boxes[component_index]
            cid = stage1_match.get(component_index)
            key = export_match.get(component_index)
            identity_scored = label.get('identity_scored', True)
            identity = key == label['designator'] if identity_scored else None
            components.append(dict(case=case, source_component=label['component_id'],
                                   inference_component=cid, expected_designator=label['designator'],
                                   output_key=key, identity_correct=identity,
                                   identity_scored=identity_scored,
                                   stage1_geometric_iou=iou(stage1_boxes[cid], box) if cid else 0,
                                   final_geometric_iou=iou(export_boxes[key], box) if key else 0))
            rows = []
            for expected in labels['rows']:
                if expected['case'] != case or expected['component_id'] != label['component_id']:
                    continue
                row = dict(expected)
                row.update(identity_scored=identity_scored, inference_component=cid,
                           output_component_key=key, identity_correct=identity,
                           bound_number=None, bound_name=None, emitted=False,
                           output_number=None, output_name=None, number_correct=False,
                           name_correct=False, both_correct=False, all_three_correct=False,
                           stage1_row_candidate=None)
                rows.append(row)
            set_tolerances(rows)
            all_rows.extend(rows)
            raw_items = [dict(index=i, point=list(point), side=edge(point, box))
                         for i, point in enumerate(stage1['pins'].get(cid, []))] if cid else []
            raw_bindings = assigned.get(cid, {}).get('pins', {}) if cid else {}
            pinindex = indexes.get(cid, {}) if cid else {}
            reverse_indexes = defaultdict(list)
            for index, pin_key in pinindex.items():
                reverse_indexes[pin_key].append(int(index))
            output = []
            if key is not None:
                for pin_key, pin in pred['pins'].get(key, {}).items():
                    item = dict(pk=pin_key, stage1_index_candidates=reverse_indexes.get(pin_key, []),
                                number=number(pin_key), name=pin.get('pinname'))
                    try:
                        point = export_to_image(pin['point'], height)
                        item.update(point=point, side=edge(point, box))
                    except (KeyError, ValueError, TypeError):
                        item.update(point=None, side=None, invalid_point=True)
                    output.append(item)
            matched_output = set()
            for side in ('left', 'right', 'top', 'bottom'):
                for row, _, item, _ in row_assignment(rows, raw_items, side):
                    row['stage1_row_candidate'] = item['index']
                    binding = raw_bindings.get(str(item['index']), {})
                    row['bound_number'] = (binding.get('number') or {}).get('text')
                    row['bound_name'] = (binding.get('name') or {}).get('text')
                for row, output_index, item, delta in row_assignment(rows, output, side):
                    matched_output.add(output_index)
                    row.update(emitted=True, output_pin_key=item['pk'],
                               output_point=item['point'], output_number=item['number'],
                               output_name=item['name'], physical_row_delta=delta,
                               number_correct=item['number'] == row['expected_number'],
                               name_correct=item['name'] == row['expected_name'])
                    row['both_correct'] = row['number_correct'] and row['name_correct']
                    row['all_three_correct'] = bool(row['both_correct'] and identity) if identity_scored else False
            for i, item in enumerate(output):
                if i not in matched_output:
                    extras.append(dict(case=case, source_component=label['component_id'],
                                       expected_designator=label['designator'], output_component_key=key,
                                       inference_component=cid,
                                       reason='No reviewed physical row on the same side within fixed row tolerance',
                                       **item))
            totals['gt'] += len(rows)
            totals['predicted_in_reviewed_boxes'] += len(output)
    for row in all_rows:
        for field in ('emitted', 'number_correct', 'name_correct', 'both_correct',
                      'all_three_correct', 'identity_correct'):
            totals[field] += int(bool(row[field]))
        totals['raw_bound_number_correct'] += int(row['bound_number'] == row['expected_number'])
        totals['raw_bound_name_correct'] += int(row['bound_name'] == row['expected_name'])
        totals['has_stage1_physical_candidate'] += int(row['stage1_row_candidate'] is not None)
    totals['emitted_identity_correct'] = sum(int(bool(r['emitted'] and r['identity_correct'])) for r in all_rows)
    totals['component_identity_correct'] = sum(int(bool(c['identity_correct'])) for c in components)
    totals['identity_evaluable_components'] = sum(int(c['identity_scored']) for c in components)
    totals['identity_evaluable_rows'] = sum(int(r['identity_scored']) for r in all_rows)
    totals['reviewed_component_count'] = len(components)
    totals['extra_outputs_outside_reviewed_rows'] = len(extras)
    by_case = {}
    for case in cases:
        rr = [r for r in all_rows if r['case'] == case]
        cc = [c for c in components if c['case'] == case]
        ee = [e for e in extras if e['case'] == case]
        by_case[case] = dict(gt=len(rr), reviewed_components=len(cc),
                            predicted_in_reviewed_boxes=sum(r['emitted'] for r in rr)+len(ee),
                            extra_outputs=len(ee),
                            **{k:sum(int(bool(r[k])) for r in rr) for k in
                               ('emitted', 'number_correct', 'name_correct', 'both_correct',
                                'all_three_correct', 'identity_correct')})
    result = dict(arm=args.arm, totals=dict(totals), by_case=by_case,
                  components=components, rows=all_rows, extra_outputs=extras,
                  provenance=dict(labels=str(args.labels), labels_sha256=sha(args.labels),
                                  stage1_dir=str(args.stage1_dir), output_dir=str(output_dir), cases=provenance),
                  method='Stage1 and final components matched one-to-one to reviewed boxes by geometric IoU > 0.5. Final outputs use exported pin.point converted from lower-left to image coordinates, never old stage1 index coordinates. Final physical rows matched one-to-one on the same reviewed box side; fixed orthogonal tolerance is 0.45 the smallest adjacent reviewed pin pitch, or 8px on single-row sides. This is physical-row association, not endpoint coordinate scoring. Missing output is wrong. Name/number scoring ignores component identity; all-three requires exact printed identity. Unprinted identity is not guessed. Extra outputs in reviewed boxes are listed separately.',
                  limits=labels.get('independence', '')+' Labels are evaluation-only and assistant-reviewed unless their source says otherwise. Only selected IC boxes are fully reviewed; no full-image false-positive rate is claimed.')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['totals'], ensure_ascii=False))


if __name__ == '__main__':
    main()
