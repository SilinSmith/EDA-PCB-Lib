"""Conservative semantic correction of already assigned IC pin names.

The lexicon proposes spellings, never pin identities. Independent OCR records
remain unchanged, and the original name used to rank duplicate numbers is
frozen before a correction. The original OCR confidence is not increased.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import re


_TOKEN = re.compile(r"([~!]*)([A-Za-z0-9_]+)([#~!+\-]*)\Z")
_DIGITS = re.compile(r"(_?)([0-9OolI]{1,4})\Z")
_ALLOWED_HELD = {None, '', 'unresolved_complete_word_character_ambiguity'}


@lru_cache(maxsize=1)
def load_library():
    path = Path(__file__).with_name('pin_name_library.json')
    raw = path.read_bytes()
    library = json.loads(raw)
    if library.get('schema_version') != 1:
        raise ValueError('Unsupported pin-name library schema')
    library['sha256'] = hashlib.sha256(raw).hexdigest()
    return library


def _confidence(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and 0 <= result <= 1 else None


def _changes(raw, candidate, library):
    if len(raw) != len(candidate):
        return None
    pairs = {frozenset(pair) for pair in library['confusion_pairs']}
    changes = []
    for index, (old, new) in enumerate(zip(raw, candidate)):
        if old == new:
            continue
        key = frozenset((old.upper(), new.upper()))
        if old == 'l' and new == '1':
            key = frozenset(('I', '1'))
        if key not in pairs:
            return None
        changes.append(dict(index=index, original=old, corrected=new))
    return changes if 0 < len(changes) <= library['policy']['max_character_changes'] else None


def _stem(raw, standard, library):
    """Retain all original letters/case except an explicitly confusable glyph."""
    if len(raw) != len(standard):
        return None
    output = ''.join(old if old.upper() == new else
                     (new.lower() if raw.islower() else new)
                     for old, new in zip(raw, standard))
    return output if output == raw or _changes(raw, output, library) is not None else None


def _token_forms(token, library, *, corrected=False):
    match = _TOKEN.fullmatch(token)
    if not match:
        return []
    leading, body, trailing = match.groups()
    forms = []
    for group, names in library['fixed_names'].items():
        for standard in names:
            value = _stem(body, standard, library)
            if value is not None and (corrected or value == body):
                forms.append(dict(text=leading + value + trailing, group=group,
                                  stem=standard, numeric=False, repair='visual_context'))
    for family in library['families']:
        for standard in family['stems']:
            if len(body) < len(standard):
                continue
            prefix = _stem(body[:len(standard)], standard, library)
            if prefix is None:
                continue
            tail = body[len(standard):]
            numeric = _DIGITS.fullmatch(tail)
            if not tail and family['suffix'] == 'optional':
                suffix, is_numeric = '', False
            elif numeric:
                separator, digits = numeric.groups()
                if not corrected and not digits.isdigit():
                    continue
                digits = ''.join('0' if c in 'Oo' else '1' if c in 'Il' else c for c in digits)
                suffix, is_numeric = separator + digits, True
            else:
                continue
            value = leading + prefix + suffix + trailing
            if corrected or value == token:
                forms.append(dict(text=value, group=family['group'], stem=standard,
                                  numeric=is_numeric, repair=family['repair']))
    return forms


def propose_names(text, library=None):
    """Return exact-length, explicit-confusion candidates, without accepting one."""
    library = library or load_library()
    parts = text.split('/')
    choices = []
    for part in parts:
        original = dict(text=part, unchanged=True)
        forms = [f for f in _token_forms(part, library, corrected=True) if f['text'] != part]
        choices.append([original] + forms)
    if math.prod(len(c) for c in choices) > 64:
        return []
    result = {}
    for variant in itertools.product(*choices):
        candidate = '/'.join(f['text'] for f in variant)
        changes = _changes(text, candidate, library)
        if changes is None:
            continue
        changed = [f for f in variant if not f.get('unchanged')]
        row = dict(text=candidate, changes=changes, forms=changed)
        # Prefer a numeric grammar over an identical fixed-name candidate.
        old = result.get(candidate)
        if old is None or sum(f.get('numeric', False) for f in changed) > sum(f.get('numeric', False) for f in old['forms']):
            result[candidate] = row
    return [result[key] for key in sorted(result)]


def _component_context(pins, library):
    context = {}
    for index, pin in pins.items():
        name = pin.get('name') or {}
        conf = _confidence(name.get('conf'))
        if conf is None or conf < library['policy']['minimum_context_confidence']:
            continue
        # Repaired names must never become new evidence for later repairs.
        text = name.get('raw_ocr_text', name.get('text', ''))
        if not isinstance(text, str):
            continue
        for part in text.split('/'):
            for form in _token_forms(part, library):
                context.setdefault(form['group'], []).append(dict(
                    pin_index=str(index), text=part, stem=form['stem'], numeric=form['numeric'], confidence=conf))
    return context


def _supporting_context(form, context, pin_index):
    peers = [p for p in context.get(form['group'], []) if p['pin_index'] != str(pin_index)]
    if form['numeric']:
        peers = [p for p in peers if p['numeric']]
    # A GPIO controller may expose just one UART; actual alternative OCR still
    # has to read its numeric suffix. GPIO does not decide a UART index.
    if form['group'] == 'uart' and form['numeric'] and not peers:
        gpio = [p for p in context.get('gpio', []) if p['numeric'] and p['pin_index'] != str(pin_index)]
        if len({p['text'] for p in gpio}) >= 2:
            peers = gpio
    return peers


def _same_box(a, b):
    try:
        return len(a) == len(b) == 4 and all(math.isfinite(float(x)) and math.isfinite(float(y))
            and abs(float(x)-float(y)) <= 1 for x, y in zip(a, b))
    except (TypeError, ValueError, OverflowError):
        return False


def _clean_dual_evidence(doc, reads, cid, index, name, candidate, library):
    """Only current, complete, isolated observations can resolve ambiguity.

    A read held *solely* for character ambiguity can be reconsidered here. A
    crop rejected for clipping, partner geometry, contamination, or missing
    glyphs cannot authorize a dictionary replacement.
    """
    from .assign import _target_index
    threshold = library['policy']['minimum_visual_confidence']
    evidence = []
    for read_index, rec in enumerate(reads):
        if rec.get('source') != 'mono_pin_name_ppocr' or not rec.get('name_only'):
            continue
        target = rec.get('pin_target')
        if not isinstance(target, dict) or target.get('component_id') != cid:
            continue
        actual = _target_index(rec, cid, doc.get('pins', {}).get(cid, []),
                               doc.get('pin_evidence', {}).get(cid))
        if actual in (None, -1) or str(actual) != str(index) or not _same_box(rec.get('bbox'), name.get('box')):
            continue
        meta = rec.get('mono_pin_ocr') or {}
        if (meta.get('backend') != 'ppocr_v4_direct' or meta.get('flags')
                or meta.get('held_reason') not in _ALLOWED_HELD):
            continue
        views = meta.get('selected_views') or []
        if not isinstance(views, list) or len(views) != 2:
            continue
        valid = []
        for view in views:
            conf = _confidence(view.get('confidence'))
            sampling = view.get('sampling') or {}
            if (view.get('text') != candidate or conf is None or conf < threshold
                    or not sampling.get('model_input_sha256') or not sampling.get('original_gray_input_sha256')):
                break
            valid.append(dict(text=view['text'], confidence=conf, padding=view.get('padding'),
                              rotation=view.get('rotation'), sampling=copy.deepcopy(sampling)))
        if (len(valid) == 2 and len({v['padding'] for v in valid}) == 2
                and len({v['sampling']['model_input_sha256'] for v in valid}) == 2
                and len({v['rotation'] for v in valid}) == 1):
            evidence.append(dict(read_index=read_index, source=rec['source'],
                                 original_held_reason=meta.get('held_reason'), views=valid))
    return evidence


def normalize_assigned_pin_names(doc, assigned, reads):
    """Return a copy with an auditable name decision; no other fields are readjusted."""
    library = load_library()
    result = copy.deepcopy(assigned)
    components = {c.get('id'): c for c in doc.get('components', []) if c.get('type') == 'box'}
    audit = []
    for cid, comp in result.get('comps', {}).items():
        if cid not in components:
            continue
        pins = comp.get('pins') or {}
        context = _component_context(pins, library)
        for index, pin in pins.items():
            name = pin.get('name') or {}
            raw = name.get('text')
            if not isinstance(raw, str) or not raw:
                continue
            if pin.get('name_normalization'):
                audit.append(copy.deepcopy(pin['name_normalization']))
                continue
            recognized = all(not part or _token_forms(part, library) for part in raw.split('/'))
            row = dict(component_id=cid, pin_index=str(index), original_text=raw, output_text=raw,
                       original_confidence=name.get('conf'), accepted=False, candidates=[])
            if raw.isdigit():
                row['reason'] = 'literal_numeric_name_preserved'
                audit.append(row)
                continue
            ambiguous_original = any((_TOKEN.fullmatch(part).group(2).upper() if _TOKEN.fullmatch(part)
                                      else part.upper()) in library['policy']['ambiguous_exact_names']
                                     for part in raw.split('/'))
            if recognized and not ambiguous_original:
                row['reason'] = 'recognized_original_preserved'
                audit.append(row)
                continue
            proposals = propose_names(raw, library)
            conf = _confidence(name.get('conf'))
            for proposal in proposals:
                candidate = copy.deepcopy(proposal)
                peers = [_supporting_context(f, context, index) for f in candidate['forms']]
                visual = _clean_dual_evidence(doc, reads, cid, index, name, candidate['text'], library)
                candidate.update(context=peers, visual_evidence=visual, eligible=False)
                if conf is None or conf < library['policy']['minimum_original_confidence']:
                    candidate['reason'] = 'original_read_below_threshold'
                elif any(f['group'] in library['policy']['read_only_groups'] for f in candidate['forms']):
                    candidate['reason'] = 'read_only_name_group'
                elif any(not p for p in peers):
                    candidate['reason'] = 'no_current_chip_context'
                else:
                    # Only GPIO O/0 changes have a strong enough *format*
                    # prior to proceed without independent numeric reads.
                    format_only = all(f['repair'] == 'gpio_format' and len({p['text'] for p in support}) >= 2
                                      for f, support in zip(candidate['forms'], peers))
                    o_zero_only = all({c['original'].upper(), c['corrected'].upper()} == {'O', '0'}
                                      for c in candidate['changes'])
                    if visual or (format_only and o_zero_only):
                        candidate.update(eligible=True, reason='clean_dual_reads_and_chip_context' if visual
                                         else 'gpio_grammar_and_multiple_observed_peers')
                    else:
                        candidate['reason'] = 'independent_visual_confirmation_required'
                row['candidates'].append(candidate)
            eligible = [c for c in row['candidates'] if c['eligible']]
            if len(eligible) == 1 and len(row['candidates']) == 1:
                decision = eligible[0]
                row.update(accepted=True, output_text=decision['text'], reason=decision['reason'])
                # Freeze even an explicit None: semantic names must not change
                # duplicate-number ranking or the physical row that survives.
                if 'number_selection_name' not in pin:
                    pin['number_selection_name'] = copy.deepcopy(pin.get('name'))
                name['raw_ocr_text'] = raw
                name['text'] = decision['text']
                pin['name_normalization'] = copy.deepcopy(row)
            else:
                row['reason'] = ('multiple_legal_candidates_preserved' if len(row['candidates']) > 1 else
                                 'insufficient_evidence_preserved' if proposals else 'outside_library_preserved')
            audit.append(row)
    result['pin_name_library_audit'] = dict(revision=library['revision'], sha256=library['sha256'],
        confidence_definition='original OCR confidence, not a probability of the semantic correction',
        corrections=sum(row['accepted'] for row in audit), entries=audit)
    return result
