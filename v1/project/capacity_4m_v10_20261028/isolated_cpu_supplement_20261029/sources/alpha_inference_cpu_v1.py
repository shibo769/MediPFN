"""Frozen level-specific interval inference; no test labels or selection.

Per-method entrypoints permit independent end-to-end timing. The old confidence
network is always used with its frozen 95% decision threshold as a feature.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import alpha_heads as ah


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_context_level(freeze_path, expected_freeze_sha, confidence_path, expected_confidence_sha, calibration, requested_level):
    """Load only previously audited files; caller must enforce real qacct/guard."""
    if sha(freeze_path)!=expected_freeze_sha or sha(confidence_path)!=expected_confidence_sha:
        raise RuntimeError('frozen artifact hash mismatch')
    freeze=json.loads(Path(freeze_path).read_text())
    if freeze['status']!='PASS' or not freeze['fixed_95_confidence_anchor_not_arbitrary_level_confidence']:
        raise RuntimeError('unrecognized multi-level artifact')
    cm=torch.load(confidence_path,map_location='cpu',weights_only=False)
    conf=calibration.Confidence(cm['dim']);conf.load_state_dict(cm['state']);conf.eval()
    out={'calibration':calibration,'confidence':conf,'temperature':cm['temperature'],'heads':{}}
    if float(requested_level) not in ah.LEVELS:raise ValueError('unregistered level')
    for level in (float(requested_level),):
        record=freeze['models'][str(level)];path=Path(record['head_path'])
        if sha(path)!=record['head_sha256'] or record['n']!=400 or record['k']!={.90:361,.99:397}[level]:
            raise RuntimeError('level artifact mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['level']!=level or saved['anchor']!='fixed pre-existing temperature-scaled 95% confidence':
            raise RuntimeError('level/anchor mismatch')
        head=ah.QuantileHead(saved['dimension']);head.load_state_dict(saved['state']);head.eval()
        out['heads'][level]={'head':head,'mean':saved['normalization_mean'],'sd':saved['normalization_sd'],
                             'expansion':record['relative_expansion']}
    return out


def predict_one_level(x, q95, estimates, ses, valid, q_target, context, level, method):
    """Return lower/upper arrays for one level and one method.

    q_target are the original SFT CDF's ordered target-specific residual
    quantiles (5%,95%) or (.5%,99.5%), not confidence-adjusted endpoints.
    CDF extraction and the frozen PFN path are performed by the caller and must
    be inside the timer for the selected method. Caller logs failures per method.
    """
    level=float(level)
    if level not in ah.LEVELS or method not in ('sft','posttrain'):
        raise ValueError('unregistered level or method')
    est=np.asarray(estimates,dtype=np.float64);se=np.asarray(ses,dtype=np.float64);valid=np.asarray(valid,dtype=bool)
    n=len(est)
    if (est.shape!=(n,) or se.shape!=(n,) or valid.shape!=(n,) or not valid.all()
            or not np.isfinite(est).all() or not np.isfinite(se).all() or np.any(se<=0)):
        raise ValueError('invalid inference inputs')
    if method=='sft':
        q_target=np.asarray(q_target)
        if (q_target.shape!=(n,2) or not np.isfinite(q_target).all()
                or np.any(q_target[:,0]>q_target[:,1])):
            raise ValueError('invalid target-specific SFT quantiles')
        return est-se*q_target[:,1],est-se*q_target[:,0]
    x=np.asarray(x,dtype=np.float32);q95=np.asarray(q95)
    if x.ndim!=2 or x.shape[0]!=n or q95.shape!=(n,3) or not np.isfinite(x).all() or not np.isfinite(q95).all():
        raise ValueError('invalid post-training anchor inputs')
    rec=context['heads'][level]
    # The 95% scale is a fixed input transform, not a nominal-level claim.
    scale=context['calibration'].scale_choice(context['confidence'],x,valid,context['temperature'])
    normalized=np.clip((x-rec['mean'])/rec['sd'],-8,8).astype(np.float32)
    return ah.prediction_intervals(rec['head'],normalized,q95,scale,est,se,rec['expansion'])
