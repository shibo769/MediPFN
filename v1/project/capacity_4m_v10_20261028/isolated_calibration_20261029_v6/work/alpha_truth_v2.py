"""Independent finite-population mediation truth for the registered cal families.

This function does not call SCM.nuisance, SCM._components, or SCM.theta_NIE.
Its inputs are the already reconstructed population and structural parameters.
"""
import numpy as np
from scipy.special import expit

FAMILIES = {'linear', 'smooth', 'heterosked', 'weak_overlap', 'covariate_shift', 'combinations'}


def independent_truth(s):
    if s.family not in FAMILIES:
        raise ValueError('unregistered calibration family: ' + str(s.family))
    x = np.asarray(s.population, dtype=np.float64)
    z = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-6)
    p = s.parameters
    linear = z @ p['weights'].T
    interaction = np.zeros(len(x))
    if len(p['pairs']):
        pairs = p['pairs']
        interaction = np.mean(np.tanh(z[:, pairs[:, 0]]) * np.tanh(z[:, pairs[:, 1]]), axis=1)
    nonlinear = np.zeros(len(x))
    if s.family in ('smooth', 'heterosked', 'covariate_shift'):
        nonlinear = .45 * (np.sin(z @ p['smooth_direction']) + .4 * np.tanh(z[:, p['active'][0]])) + .3 * interaction
    if s.family == 'combinations':
        nonlinear += 1.4 * np.mean(np.where(z[:, p['tree_columns']] > p['tree_thresholds'], p['tree_values'], -p['tree_values']), axis=1)
        nonlinear += np.sin(z @ p['rff_frequencies'].T + p['rff_phase']) @ p['rff_weights'] / np.sqrt(8)
        nonlinear += np.tanh(z @ p['mlp_weights'].T + p['mlp_bias']) @ p['mlp_output'] / np.sqrt(12)
    varying = s.family != 'linear'
    l0 = p['intercepts'][1] + linear[:, 1] + .8 * nonlinear + (.2 * interaction if varying else 0.)
    delta = p['gamma'] + (.35 * np.tanh(linear[:, 2]) if varying else 0.)
    q0 = p['floor'] + (1 - 2 * p['floor']) * expit(l0)
    q1 = p['floor'] + (1 - 2 * p['floor']) * expit(l0 + delta)
    effect = p['mediator_effect'] + p['am_effect']
    if varying:
        effect = effect + .35 * np.tanh(linear[:, 2]) + .25 * nonlinear + .25 * np.tanh(linear[:, 4])
    value = float(np.mean((q1 - q0) * effect))
    if not np.isfinite(value):
        raise ValueError('nonfinite independent truth')
    return value
