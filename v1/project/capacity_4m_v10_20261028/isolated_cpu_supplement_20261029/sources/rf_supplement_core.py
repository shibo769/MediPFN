"""Pure RF supplement calculations and outcome-blind subset proposal.

This module does not access a test or register a protocol. Final users must
bind this source, the case schedule and all random streams before evaluation.
"""
import math
from statistics import NormalDist
import numpy as np

LEVELS = (0.90, 0.95, 0.99)
ALPHAS = {0.90:0.1, 0.95:0.05, 0.99:0.01}


def select_subset(cases, expected_cells, expected_id_cells=0):
    """Four rep-0 mechanisms per (stress scenario,n) / (real source,n).

    Real-X: the first two smooth, first tree, first combinations mechanism.
    Stress: the first four mechanisms. Sort solely by cluster identity. The
    rule depends exclusively on preregistration metadata, not outcomes.
    """
    cells = {};id_cells={}
    for c in cases:
        kind = c['kind']
        if kind == 'stress': key = (kind, c['scenario'], int(c['n']))
        elif kind == 'realx': key = (kind, c['source'], int(c['n']))
        elif kind == 'id':
            if int(c['rep'])==0:id_cells.setdefault((c['family'],int(c['n']),int(c['p'])),[]).append(c)
            continue
        else: raise ValueError('unknown case kind')
        if int(c['rep']) == 0: cells.setdefault(key, []).append(c)
    if len(cells) != expected_cells: raise ValueError('unexpected predefined cells')
    indices = []
    for key, rows in sorted(cells.items()):
        rows.sort(key=lambda c: (str(c['cluster_id']), int(c['index'])))
        chosen = []
        quotas = {'smooth': 2, 'tree': 1, 'combinations': 1} if key[0]=='realx' else {None: 4}
        for family, count in quotas.items():
            candidates = [c for c in rows if family is None or c['family']==family]
            distinct=[]
            for c in candidates:
                if c['cluster_id'] not in {v['cluster_id'] for v in distinct}:distinct.append(c)
                if len(distinct)==count:break
            if len(distinct)!=count:raise ValueError('insufficient distinct mechanisms in family '+str(family))
            chosen.extend(distinct)
        if len({c['cluster_id'] for c in chosen})!=4:raise ValueError('duplicate selected mechanisms')
        indices.extend(int(c['index']) for c in chosen)
    if len(id_cells)!=expected_id_cells:raise ValueError('unexpected ID cell count')
    for key,rows in sorted(id_cells.items()):
        rows.sort(key=lambda c:(str(c['cluster_id']),int(c['index'])))
        if not rows:raise ValueError('missing ID mechanism')
        indices.append(int(rows[0]['index']))
    if len(indices) != 4*expected_cells+expected_id_cells or len(set(indices)) != len(indices): raise ValueError('subset identity')
    return sorted(indices)


def endpoints_from_replicates(values, level):
    """Match evaluate.bootstrap_refit_interval ranks/failure tails exactly."""
    if len(values) != 100 or level not in LEVELS: raise ValueError('invalid B100')
    arr = [None if v is None else float(v) for v in values]
    if any(v is not None and not math.isfinite(v) for v in arr):
        raise ValueError('nonfinite replicate must be encoded as failure')
    alpha = ALPHAS[level]
    lo = sorted(-math.inf if v is None else v for v in arr)
    hi = sorted(math.inf if v is None else v for v in arr)
    return (lo[max(0, int(math.floor(100*alpha/2))-1)],
            hi[min(99, int(math.ceil(100*(1-alpha/2)))-1)])


def normal_interval(estimate, se, level):
    if level not in LEVELS or not math.isfinite(estimate) or not math.isfinite(se) or se <= 0:
        raise ValueError('invalid normal interval input')
    z = NormalDist().inv_cdf(1-ALPHAS[level]/2)
    return estimate-z*se, estimate+z*se


def encode_interval(lo, hi, truth):
    """Keep infinity flags explicit; invalid/unbounded intervals count failure."""
    a, b = float(lo), float(hi)
    if math.isnan(a) or math.isnan(b) or a > b: raise ValueError('invalid endpoint ordering')
    finite = math.isfinite(a) and math.isfinite(b)
    return {'ok':finite, 'lo':a if math.isfinite(a) else None,
            'hi':b if math.isfinite(b) else None,
            'lower_unbounded':a == -math.inf, 'upper_unbounded':b == math.inf,
            'covered':bool(a <= truth <= b) if finite else False,
            'length':b-a if finite else None}
