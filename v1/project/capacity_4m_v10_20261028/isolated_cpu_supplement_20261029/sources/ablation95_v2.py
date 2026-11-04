"""Frozen inference-stage comparisons for the 4M 95% NIE interval.

This module performs no registration, model fitting, selection, data generation,
or file writes.  The caller owns the protocol and must verify all bound source,
model, data, and accounting hashes before invoking this module.  All returned
intervals use exactly the same observed table and cross-fitted PFN features.

``matched_width_confidence`` uses the final method's per-table width, centered
at the temperature-confidence error median.  It diagnoses shape/centering at
equal length; it is not an independently trained, lower-cost deployment method.
``confidence_expansion`` instead applies the frozen calibration coefficient
directly to the temperature-confidence error interval; it is not fitted to
match final length or chosen on the test bank.
"""
import numpy as np
import torch

METHODS = (
    'sft',
    'confidence_temperature',
    'quantile_uncalibrated',
    'selected_calibrated',
    'confidence_expansion',
    'matched_width_confidence',
)


def evaluate(x, q, est, se, valid, *, cal, training, confidence, confidence_state,
             selection, chosen, only=None):
    """Return per-method endpoints (and failures) for one extracted table.

    ``q`` and ``x`` are the outputs of the original ``cal.make_features``;
    ``chosen['head']`` must be the separately loaded, frozen selected head.
    A caller must persist exact observed/fold hashes and determine coverage from
    the returned parameter endpoints and independently reconstructed truth.
    """
    result = {}
    if only is not None and only not in METHODS:
        raise ValueError('unregistered single-method timing request')
    if len(est) != 1 or len(se) != 1 or len(valid) != 1 or not bool(valid[0]):
        raise ValueError('one valid table with positive estimated scale required')
    est = np.asarray(est, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    q = np.asarray(q)
    if q.shape != (1, 3) or not np.isfinite(est).all() or not np.isfinite(se).all() or np.any(se <= 0):
        raise ValueError('invalid feature/quantile/scale shape or value')

    def save(name, lower_error, upper_error):
        lower = float(est[0] - se[0] * upper_error[0])
        upper = float(est[0] - se[0] * lower_error[0])
        if not np.isfinite([lower, upper]).all() or lower > upper:
            raise ValueError('nonfinite or reversed parameter interval')
        result[name] = {'ok': True, 'lo': lower, 'hi': upper, 'length': upper-lower}

    def failed(name, error):
        result[name] = {'ok': False, 'error': repr(error)}

    try:
        save('sft', q[:, 0], q[:, 2])
    except Exception as exc:
        failed('sft', exc)
    if only == 'sft': return {'sft':result['sft']}

    temperature_q = None
    try:
        mid = q[:, 1]
        scale = cal.scale_choice(confidence, x, valid, confidence_state['temperature'])
        temperature_q = np.column_stack((mid + scale*(q[:, 0]-mid), mid,
                                         mid + scale*(q[:, 2]-mid)))
        save('confidence_temperature', temperature_q[:, 0], temperature_q[:, 2])
    except Exception as exc:
        failed('confidence_temperature', exc)
    if only == 'confidence_temperature': return {'confidence_temperature':result['confidence_temperature']}

    correction = float(chosen['calibration_relative_expansion'])
    if not np.isfinite(correction) or correction < 0:
        raise ValueError('invalid frozen calibration coefficient')
    if temperature_q is None:
        for method in METHODS[2:]:
            failed(method, RuntimeError('temperature confidence dependency failed'))
        return {only:result[only]} if only is not None else result

    if only is None or only == 'confidence_expansion':
        try:
            # Same frozen scalar c, applied to the confidence stage.
            width = temperature_q[:, 2] - temperature_q[:, 0]
            save('confidence_expansion', temperature_q[:, 0] - correction*width,
                 temperature_q[:, 2] + correction*width)
        except Exception as exc:
            failed('confidence_expansion', exc)
        if only == 'confidence_expansion': return {'confidence_expansion':result['confidence_expansion']}

    final_lo = final_hi = None
    try:
        qscaled = temperature_q.astype(np.float32)
        if chosen['selected'] == 'confidence':
            qlo, qhi = qscaled[:, 0], qscaled[:, 2]
        else:
            norm = np.clip((x-selection['normalization_mean']) /
                           selection['normalization_sd'], -8, 8).astype(np.float32)
            with torch.no_grad():
                qlo, qhi, _, _ = training.endpoints(
                    chosen['head'](torch.from_numpy(norm)), torch.from_numpy(qscaled),
                    chosen['selected'], torch.from_numpy(se.astype(np.float32)),
                    selection['tau_unit'])
            qlo, qhi = qlo.numpy(), qhi.numpy()
        save('quantile_uncalibrated', qlo, qhi)
        if only == 'quantile_uncalibrated':return {'quantile_uncalibrated':result['quantile_uncalibrated']}
        width = qhi-qlo
        final_lo, final_hi = qlo-correction*width, qhi+correction*width
        save('selected_calibrated', final_lo, final_hi)
    except Exception as exc:
        if 'quantile_uncalibrated' not in result:
            failed('quantile_uncalibrated', exc)
        failed('selected_calibrated', exc)
    if only in ('quantile_uncalibrated','selected_calibrated'):
        return {only:result[only]}

    try:
        if final_lo is None or final_hi is None:
            raise RuntimeError('selected method failed; width matching unavailable')
        half = (final_hi-final_lo)/2
        mid = temperature_q[:, 1]
        save('matched_width_confidence', mid-half, mid+half)
    except Exception as exc:
        failed('matched_width_confidence', exc)
    return {only:result[only]} if only is not None else {name: result[name] for name in METHODS}
