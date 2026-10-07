# -*- coding: utf-8 -*-
"""Schematic-parsing toolkit for the 2026 EDA-Elite Challenge, problem 6.

Sub-modules:
    data     -- dataset discovery / pairing / JSON parsing helpers
    metrics  -- the four competition F1 metrics (ComponentF1, PinF1,
                NetHypergraphF1, NetLineF1) plus the weighted case score
"""
from .data import (Case, DEFAULT_DATA_ROOT, SPLITS, discover, find_case,
                   load_json, save_json, parse_hypergraph, build_hypergraph,
                   normalise_doc, describe_doc, pin_key_to_number,
                   number_to_pin_key, net_pin_set)
from .metrics import (EvalConfig, F1, CaseScore, evaluate_case, aggregate,
                      WEIGHTS)

__all__ = [
    'Case', 'DEFAULT_DATA_ROOT', 'SPLITS', 'discover', 'find_case',
    'load_json', 'save_json', 'parse_hypergraph', 'build_hypergraph',
    'normalise_doc', 'describe_doc', 'pin_key_to_number', 'number_to_pin_key',
    'net_pin_set', 'EvalConfig', 'F1', 'CaseScore', 'evaluate_case',
    'aggregate', 'WEIGHTS',
]
