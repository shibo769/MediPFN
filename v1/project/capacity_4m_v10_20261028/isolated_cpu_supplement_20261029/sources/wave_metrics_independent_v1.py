"""Independent, in-memory metric audit; does not read or authorize test data.

Inputs are to be supplied by a future independently verified loader: this
module is not a scheduler, a truth reconstruction, or a production final audit.
Failures count as uncovered; lengths and their paired contrasts use only
finite, ordered successful intervals. Cross-method pairs require byte identity
when methods come from the two different workers.
"""
import math


def _se(records):
    if not records:
        return None
    grouped = {}
    for group, value in records:
        grouped.setdefault(group, []).append(value)
    if len(grouped) < 2:
        return None
    n = len(records)
    mean = sum(value for _, value in records) / n
    total = sum(sum(value - mean for value in values) ** 2 for values in grouped.values())
    return math.sqrt((len(grouped) / (len(grouped) - 1)) * total) / n


def _close(actual, expected, name):
    if expected is None:
        if actual is not None:
            raise ValueError(name + ': expected null')
    elif isinstance(expected, (int, bool)):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(name + ': integer mismatch')
    elif not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(name + ': numeric mismatch')


def _check(actual, expected, label):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError(label + ': output schema mismatch')
    for key, value in expected.items():
        if isinstance(value, str):
            if actual[key] != value:
                raise ValueError(label + ': ' + key)
        else:
            _close(actual[key], value, label + ': ' + key)


def audit(cases, pfn, rf, truths, selected, groups, results, pairs):
    """Recompute coverage, lengths, failures, pairing, and cluster SE.

    The caller must separately check all worker accounting and hashes,
    exact case identity, source parity, independent truths, and each input's
    evidence. This function assumes those inputs and checks only reported
    metrics against them; it does not claim any such evidence was checked.
    """
    n = len(cases)
    if any(set(table) != set(range(n)) for table in (pfn, rf, truths)):
        raise ValueError('incomplete attempt identities')
    if any(not math.isfinite(float(truths[i])) for i in range(n)):
        raise ValueError('nonfinite truth')
    selected = set(selected)
    computed_results, computed_pairs = [], []
    for name, indices, methods, reference in groups:
        if len(indices) != len(set(indices)) or not set(indices).issubset(set(range(n))) or not indices:
            raise ValueError('invalid group')
        if reference not in methods or len(methods) != len(set(methods)):
            raise ValueError('invalid methods')
        if any(method.startswith('rf_bootstrap_') for method in methods) and not set(indices).issubset(selected):
            raise ValueError('bootstrap denominator')
        values = {}
        for method in methods:
            by_index = {}
            for i in indices:
                row = rf[i] if method.startswith('rf_') else pfn[i]
                common = row.get('common_ok', row.get('ok', False))
                if type(common) is not bool:
                    raise ValueError('invalid common status')
                method_value = row.get('methods', {}).get(method)
                if common and (not isinstance(method_value, dict) or type(method_value.get('ok')) is not bool):
                    raise ValueError('missing/malformed registered method')
                item = method_value if isinstance(method_value, dict) else {}
                if common and item['ok']:
                    lo, hi = float(item['lo']), float(item['hi'])
                    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                        raise ValueError('invalid interval')
                    covered, length = int(lo <= truths[i] <= hi), hi - lo
                else:
                    covered, length = 0, None
                by_index[i] = (covered, length, bool(item.get('lower_unbounded') or item.get('upper_unbounded')))
            units = [(cases[i]['cluster_id'], by_index[i][0]) for i in indices]
            lens = [by_index[i][1] for i in indices if by_index[i][1] is not None]
            computed_results.append(dict(group=name, method=method, attempts=len(indices), valid=len(lens),
                failures=len(indices)-len(lens), unbounded=sum(by_index[i][2] for i in indices),
                clusters=len({cases[i]['cluster_id'] for i in indices}),
                coverage=sum(by_index[i][0] for i in indices)/len(indices), cluster_se=_se(units),
                mean_length_valid=sum(lens)/len(lens) if lens else None))
            values[method] = by_index
        for method in methods:
            if method == reference:
                continue
            units, lengths = [], []
            cross_worker = method.startswith('rf_') != reference.startswith('rf_')
            for i in indices:
                if cross_worker:
                    a, b = pfn[i], rf[i]
                    fields = ((a.get('observation'), b.get('observed')), (a.get('fold'), b.get('fold')),
                              (a.get('population_sha256'), b.get('population_sha256')))
                    if any(x is not None and y is not None and x != y for x,y in fields):
                        raise ValueError('different observed data identities')
                    if not all(x is not None and y is not None for x,y in fields):
                        continue
                ca, la, _ = values[reference][i]
                cb, lb, _ = values[method][i]
                units.append((cases[i]['cluster_id'], ca-cb))
                if la is not None and lb is not None:
                    lengths.append((cases[i]['cluster_id'], la-lb))
            computed_pairs.append(dict(group=name, reference=reference, method=method, attempts=len(indices),
                paired_attempts=len(units), identity_unavailable=len(indices)-len(units),
                clusters=len({c for c,_ in units}),
                coverage_difference=sum(v for _,v in units)/len(units) if units else None,
                cluster_se=_se(units), length_paired_valid=len(lengths),
                length_excluded=len(indices)-len(lengths),
                length_difference=sum(v for _,v in lengths)/len(lengths) if lengths else None,
                length_cluster_se=_se(lengths)))
    if len(results) != len(computed_results) or len(pairs) != len(computed_pairs):
        raise ValueError('missing or additional summary rows')
    for i, (actual, expected) in enumerate(zip(results, computed_results)):
        _check(actual, expected, 'metric row '+str(i))
    for i, (actual, expected) in enumerate(zip(pairs, computed_pairs)):
        _check(actual, expected, 'pair row '+str(i))
    return {'status': 'PASS_METRICS_ONLY', 'metric_rows': len(results), 'paired_rows': len(pairs)}
