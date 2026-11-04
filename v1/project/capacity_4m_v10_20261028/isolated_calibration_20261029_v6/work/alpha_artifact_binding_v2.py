"""Validate the immutable 4M artifacts before any model deserialization."""
import hashlib
import json
from pathlib import Path

AUDITED_PROTOCOL_SHA = '5ed7802fa02a90d5748fc7f24b9bce3a25d45394ce9c164725b587d5cbb5cc08'
AUDITED_FINAL_SHA = '5edf21312af948eb98b39163516ba257aa0a29734506d5d87fb282a59ec0ccb8'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate(registration, guard, training_freeze, base, capacity, confidence, new):
    # Registration and the independently accounted guard must agree on all bytes.
    artifacts = registration['artifact_sha256']
    if guard['artifact_sha256'] != artifacts:
        raise RuntimeError('registration/guard model bindings differ')
    old = capacity/'robust_partial_composition_retry_20261028'
    protocol = old/'independent_test_20261028/protocol.json'
    audit = old/'independent_test_20261028/audit/final_independent_corrected_audit.json'
    required = [base/'capacity_full/large_seed0/checkpoint/model.pt',
                capacity/'mixture_top2/mixture.pt', confidence/'fitted/confidence.pt',
                confidence/'fitted/freeze.json', new/'trained_heads/head_90.pt',
                new/'trained_heads/head_99.pt',
                old/'registration.json', old/'trained_v1/freeze.json', protocol, audit]
    if set(artifacts) != {str(p.resolve()) for p in required}:
        raise RuntimeError('model artifact map missing or unexpected paths')
    for path in required:
        if sha(path) != artifacts[str(path.resolve())]:
            raise RuntimeError('model artifact hash mismatch: ' + str(path))
    if sha(protocol) != AUDITED_PROTOCOL_SHA or sha(audit) != AUDITED_FINAL_SHA:
        raise RuntimeError('audited 95% metadata changed')
    prior_audit = json.loads(audit.read_text())
    prior_protocol = json.loads(protocol.read_text())
    if prior_audit['status'] != 'PASS' or prior_audit['protocol_sha256'] != sha(protocol):
        raise RuntimeError('95% audit not bound to original protocol')
    for key, path in (('gpu_checkpoint_sha256', required[0]), ('mixture_sha256', required[1]),
                      ('confidence_sha256', required[2])):
        if prior_protocol[key] != artifacts[str(path.resolve())]:
            raise RuntimeError('model disagrees with audited 95% protocol: ' + key)
    conf_freeze = json.loads((confidence/'fitted/freeze.json').read_text())
    if conf_freeze['model_sha256'] != artifacts[str((confidence/'fitted/confidence.pt').resolve())]:
        raise RuntimeError('confidence model disagrees with its original freeze')
    if conf_freeze['base_sha256'] != artifacts[str((capacity/'mixture_top2/mixture.pt').resolve())]:
        raise RuntimeError('confidence freeze references a different mixture')
    prior_reg_path = old/'registration.json'
    prior = json.loads(prior_reg_path.read_text())
    prior_freeze = json.loads((capacity/'robust_partial_composition_retry_20261028/trained_v1/freeze.json').read_text())
    if (registration['prior_registration_sha256'] != sha(prior_reg_path)
            or prior_freeze['registration_sha256'] != sha(prior_reg_path)):
        raise RuntimeError('previous model registration mismatch')
    for path in (base/'capacity_full/large_seed0/checkpoint/model.pt',
                 capacity/'mixture_top2/mixture.pt', confidence/'fitted/confidence.pt'):
        if prior['source_sha256'][str(path.resolve())] != artifacts[str(path.resolve())]:
            raise RuntimeError('model differs from the prior registered artifact: ' + str(path))
    if prior_freeze['confidence_sha256'] != artifacts[str((confidence/'fitted/confidence.pt').resolve())]:
        raise RuntimeError('old robust freeze disagrees with confidence model')
    if guard['training_freeze_sha256'] != registration['training_freeze_sha256']:
        raise RuntimeError('training freeze guard differs')
    for level in (0.9, 0.99):
        desc = training_freeze['models'][str(level)]
        path = new/'trained_heads'/('head_%d.pt' % round(100*level))
        if desc['model'] != str(path) or desc['sha256'] != artifacts[str(path.resolve())]:
            raise RuntimeError('trained level head disagrees with its freeze')
    if (not isinstance(registration['training_job'], int) or registration['training_job'] <= 0
            or guard['training_job'] != registration['training_job']):
        raise RuntimeError('training job identity differs')
    return artifacts
