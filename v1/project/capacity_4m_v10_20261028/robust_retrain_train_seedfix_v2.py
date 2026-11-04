"""Frozen 4M post-training: train/DEV/cal only, never touches test data.

Run only after every upstream array task exits successfully.  The caller supplies
the exact registration hash and upstream job ID; all records are checked before
any optimisation.  The output is a new, immutable directory.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

B = Path('/projectnb/tmd-ortho/shibo/nie-uq-full-20261026-v9')
D = B / 'capacity_4m_v10_20261028'
R = D / 'robust_partial_composition_retry_20261028'
C = D / 'confidence_calibration_4m_20261028'
sys.path[:0] = [str(C), str(B / 'code'), str(B / 'posttrain_v10_20261026/tabcon')]
import calibration as cal

GROUPS = ('id', 'tree_rff', 'tree_mlp', 'rff_mlp')
SPLITS = ('train', 'dev', 'calibration')
MODES = ('bias', 'floor', 'bias_floor', 'quantile_group')
ALPHA = 0.05
EPOCHS = 160
SEED = 202610281301
SPEC = {
    'baseline': 'temperature confidence',
    'final_test_comparators': ['original SFT', 'raw confidence', 'temperature confidence', 'selected post-training'],
    'candidates': ['bias', 'floor', 'bias_floor', 'quantile_group'],
    'architecture': '16-unit tanh head; zero last layer; floor outputs initialized at -2',
    'updates': 160, 'optimizer': 'AdamW lr=0.001 weight_decay=0.01 full batch grad_clip=1',
    'training_groups': 'four groups equally weighted per update',
    'bias_loss': 'smooth L1 on studentized error; bias_floor adds detached-center absolute error pinball',
    'floor': 'tau=median(train SE)*softplus(head); interval scale sqrt(SE^2+tau^2); 95% absolute-error pinball',
    'quantile': 'group-equal 2.5% and 97.5% pinball, ordered via exponential widths',
    'selection': 'ID coverage >= confidence minus 0.05 and ID length <= 1.5 times confidence; include baseline; maximize minimum shift-group coverage, then minimize mean shift-group length; baseline wins exact ties, then candidate list order',
    'calibration': 'after selection freeze, pooled independent calibration mechanisms, 95% finite-sample order statistic on nonnegative relative interval expansion; fail if order index exceeds sample size',
    'seed': SEED,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def project_imports():
    result={}
    for name,module in list(sys.modules.items()):
        if name == '__main__': continue
        file=getattr(module,'__file__',None)
        if not file: continue
        path=Path(file).resolve()
        if path.is_relative_to(B.resolve()) and not path.is_relative_to((B/'env').resolve()) and path != Path(__file__).resolve() and path.is_file():
            result[str(path)]=sha(path)
    return dict(sorted(result.items()))


def check_array(job, expected):
    r = subprocess.run(['qacct', '-j', str(job)], capture_output=True, text=True, timeout=120)
    if r.returncode:
        raise RuntimeError(f'qacct unavailable or failed for {job}: {r.stderr}')
    records = {}
    for block in re.split(r'\n={10,}\s*\n', '\n'+r.stdout):
        vals = {}
        for line in block.splitlines():
            x = line.split()
            if len(x) >= 2 and x[0] in ('failed', 'exit_status', 'taskid', 'jobnumber'):
                if x[0] in vals: raise RuntimeError('ambiguous qacct record')
                vals[x[0]] = x[1]
        if not vals: continue
        if vals.get('jobnumber') != str(job) or not vals.get('taskid', '').isdigit():
            raise RuntimeError(f'invalid qacct record: {vals}')
        task=int(vals['taskid'])
        if task in records: raise RuntimeError(f'duplicate qacct record for task {task}')
        records[task] = vals
    ignored_failed = (str(job) == '8314063' and expected == set(range(14, 41))
        and set(records) == set(range(8, 41))
        and all(records[i].get('failed') == '0' and records[i].get('exit_status') == '1' for i in range(8, 14)))
    if set(records) != expected and not ignored_failed:
        raise RuntimeError(f'qacct task mismatch: expected {sorted(expected)}, found {sorted(records)}')
    for task in expected:
        vals=records[task]
        if vals.get('failed') != '0' or vals.get('exit_status') != '0':
            raise RuntimeError(f'upstream {job}.{task} did not succeed: {vals}')
    return [records[i] for i in sorted(expected)]


def load_checked(jobs, regsha):
    p = R / 'registration.json'
    if sha(p) != regsha:
        raise RuntimeError('registration hash differs from externally supplied hash')
    reg = json.loads(p.read_text())
    tasks = reg['tasks']
    if len(tasks) != 400 or any(x['index'] != i for i, x in enumerate(tasks)):
        raise RuntimeError('unexpected registered task count/order')
    for path, digest in reg['source_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('registered source changed: ' + path)
    for path, digest in reg['previous_protocol_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('prior protocol changed: ' + path)
    if sha(D/'fit.npz') != reg['fit_sha256'] or sha(B/'capacity_full/large_seed0/checkpoint/model.training.json') != reg['pretraining_log_sha256']:
        raise RuntimeError('prior fit or pretraining provenance changed')
    arrays = []
    accounting = []
    for job, expected in jobs:
        accounting.extend(check_array(job, expected))
    for i in range(1, 41):
        npz = R / 'shards' / ('part_%03d.npz' % i)
        info = json.loads(npz.with_suffix('.json').read_text())
        if (info['task'] != i or info['count'] != 10 or info['npz_sha256'] != sha(npz)
                or info['registration_sha256'] != regsha):
            raise RuntimeError(f'shard {i} manifest mismatch')
        with np.load(npz, allow_pickle=False) as f:
            a = {k: f[k] for k in ('features', 'q', 'est', 'se', 'truth', 'split', 'group', 'seeds', 'n', 'p')}
        expected = tasks[(i-1)*10:i*10]
        if any(len(v) != 10 for v in a.values()) or a['q'].shape != (10, 3) or a['seeds'].shape != (10, 4):
            raise RuntimeError(f'shard {i} shapes')
        for j, x in enumerate(expected):
            if (str(a['split'][j]) != x['split'] or str(a['group'][j]) != x['group']
                or int(a['n'][j]) != x['n'] or int(a['p'][j]) != x['p']
                or list(map(int, a['seeds'][j])) != [x[k] for k in ('mechanism_seed', 'observation_seed', 'fold_seed', 'view_seed')]):
                raise RuntimeError(f'shard {i}, row {j}: registered identity mismatch')
        for k in ('features', 'q', 'est', 'se', 'truth'):
            if not np.isfinite(a[k]).all(): raise RuntimeError(f'invalid {k} in {i}')
        if np.any(a['se'] <= 0) or np.any(a['q'][:, 0] > a['q'][:, 1]) or np.any(a['q'][:, 1] > a['q'][:, 2]):
            raise RuntimeError(f'invalid se or quantiles in {i}')
        arrays.append(a)
    a = {k: np.concatenate([x[k] for x in arrays]) for k in arrays[0]}
    for name in SPLITS:
        for group in GROUPS:
            if np.sum((a['split'] == name) & (a['group'] == group)) == 0:
                raise RuntimeError(f'empty {name}/{group}')
    sets = [set(map(int, a['seeds'][a['split'] == s, 0])) for s in SPLITS]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i+1,3)) or len(set(map(int,a['seeds'][:,0]))) != len(a['seeds']):
        raise RuntimeError('mechanisms overlap across or within splits')
    return reg, a, accounting


class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x): return self.net(x)


def endpoints(raw, q, mode, se, tau_unit):
    center = q[:, 1]
    lw = (center - q[:, 0]).clamp_min(1e-5)
    rw = (q[:, 2] - center).clamp_min(1e-5)
    basewidth = (lw + rw) / 2
    if mode in ('bias', 'bias_floor'):
        center = center + raw[:, 0]
    if mode in ('floor', 'bias_floor'):
        # tau is in outcome units, not a multiple of this table's SE.
        tau = tau_unit * F.softplus(raw[:, 1])
        factor = torch.sqrt(se**2 + tau**2) / se
        lw, rw = lw * factor, rw * factor
    if mode == 'quantile_group':
        center = center + raw[:, 0]
        lw = lw * torch.exp(raw[:, 1].clamp(-3, 3))
        rw = rw * torch.exp(raw[:, 2].clamp(-3, 3))
    return center-lw, center+rw, center, basewidth


def pinball(residual, tau): return torch.maximum(tau*residual, (tau-1)*residual)


def objective(model, x, q, z, se, group, mode, tau_unit):
    raw = model(x)
    low, high, center, basewidth = endpoints(raw, q, mode, se, tau_unit)
    if mode == 'bias':
        losses = F.smooth_l1_loss(center, z, reduction='none')
    elif mode == 'floor':
        tau = tau_unit * F.softplus(raw[:, 1])
        halfwidth = basewidth * torch.sqrt(se**2 + tau**2)
        losses = pinball(se*(z-q[:, 1]).abs() - halfwidth, .95)/tau_unit
    elif mode == 'bias_floor':
        tau = tau_unit * F.softplus(raw[:, 1])
        halfwidth = basewidth * torch.sqrt(se**2 + tau**2)
        losses = F.smooth_l1_loss(center, z, reduction='none') + pinball(se*(z-center.detach()).abs()-halfwidth, .95)/tau_unit
    else:
        losses = pinball(z-low,.025) + pinball(z-high,.975)
    # Fixed group-equal weighting; all groups influence each update equally.
    return torch.stack([losses[group == i].mean() for i in range(len(GROUPS))]).mean() + .0001 * (raw**2).mean()


def metrics(low, high, z, se, groups, indices):
    result = {}
    for g in GROUPS:
        ix = indices[groups[indices] == g]
        if not len(ix): raise RuntimeError('empty evaluation group')
        result[g] = {'n': int(len(ix)), 'coverage': float(np.mean((low[ix] <= z[ix]) & (z[ix] <= high[ix]))),
                     'length': float(np.mean((high[ix]-low[ix])*se[ix]))}
    return result


def main(args):
    torch.set_num_threads(4); torch.manual_seed(SEED); np.random.seed(SEED % (2**32))
    if sha(__file__) != args.source_sha256:
        raise RuntimeError('training source hash differs from external binding')
    for path in R.rglob('*'):
        name=path.name.lower()
        if name.startswith(('test','independent_test')) or 'test_authorization' in name:
            raise RuntimeError('new test artifact exists before candidate freeze: '+str(path))
    plan_path=R/'training_plan.json'
    if sha(plan_path) != args.plan_sha256: raise RuntimeError('training plan hash mismatch')
    plan=json.loads(plan_path.read_text())
    amend_path=R/'training_accounting_amendment.json'
    if sha(amend_path) != os.environ['TRAIN_ACCOUNTING_AMEND_SHA']: raise RuntimeError('accounting amendment hash mismatch')
    amend=json.loads(amend_path.read_text())
    fix_path=R/'training_seedfix_amendment.json'
    if sha(fix_path) != os.environ['TRAIN_SEEDFIX_AMEND_SHA']: raise RuntimeError('seed amendment hash mismatch')
    fix=json.loads(fix_path.read_text())
    if (fix['previous_source_sha256'] != amend['new_source_sha256']
            or fix['previous_source_sha256'] != sha(D/'robust_retrain_train_retry_v2.py')
            or fix['new_source_sha256'] != args.source_sha256
            or fix['accounting_amendment_sha256'] != sha(amend_path)):
        raise RuntimeError('seed amendment source mismatch')
    if amend['old_plan_sha256'] != args.plan_sha256:
        raise RuntimeError('accounting amendment does not bind plan and source')
    if (amend['old_source_sha256'] != sha(D/'robust_retrain_train_v2.py')
            or amend['retry_provenance_sha256'] != sha(R/'retry_8_13_provenance.json')
            or amend['retry_job'] != 8314841):
        raise RuntimeError('accounting amendment upstream mismatch')
    if plan != {'registration_sha256':args.registration_sha256,'source_sha256':amend['old_source_sha256'],'spec':SPEC,'project_import_sha256':project_imports()}:
        raise RuntimeError('training plan does not match source and registration')
    jobs = []
    all_tasks = set()
    job_ids = set()
    for part in args.job_arrays.split(','):
        match=re.fullmatch(r'([1-9][0-9]*):([1-9][0-9]*)(?:-([1-9][0-9]*))?',part)
        if not match: raise RuntimeError('invalid job array specification')
        job, first, last = int(match[1]), int(match[2]), int(match[3] or match[2])
        if job in job_ids or last < first: raise RuntimeError('duplicate job or reversed range')
        tasks=set(range(first,last+1))
        if all_tasks & tasks: raise RuntimeError('upstream task ranges overlap')
        all_tasks.update(tasks); job_ids.add(job); jobs.append((job,tasks))
    if all_tasks != set(range(1,41)): raise RuntimeError('upstream task ranges must partition 1..40')
    if args.job_arrays != '8314060:7,8314062:1-6,8314063:14-40,8314841:8-13':
        raise RuntimeError('job mapping differs from accounting amendment')
    reg, a, accounting = load_checked(jobs, args.registration_sha256)
    provenance=json.loads((R/'retry_8_13_provenance.json').read_text())
    for name,digest in provenance['preserved_shards_sha256'].items():
        if sha(R/'shards'/name) != digest: raise RuntimeError('preserved original shard changed: '+name)
    feature = a['features'].astype(np.float32)
    q = a['q'].astype(np.float32)
    train = np.flatnonzero(a['split'] == 'train'); dev = np.flatnonzero(a['split'] == 'dev'); calibration = np.flatnonzero(a['split'] == 'calibration')
    z = np.full(len(a['est']), np.nan, dtype=np.float32)
    for idx in (train, dev):
        z[idx] = ((a['est'][idx] - a['truth'][idx]) / a['se'][idx]).astype(np.float32)
    # The original confidence model sees original, unmodified features.
    fit = C/'fitted/confidence.pt'; freeze = json.loads((C/'fitted/freeze.json').read_text())
    if sha(fit) != freeze['model_sha256']: raise RuntimeError('confidence model hash mismatch')
    base = torch.load(fit,map_location='cpu',weights_only=False)
    if base['dim'] != feature.shape[1] or base['base_sha256'] != sha(D/'mixture_top2/mixture.pt'):
        raise RuntimeError('baseline confidence dimensions/mixture mismatch')
    net = cal.Confidence(base['dim']); net.load_state_dict(base['state']); net.eval()
    scale = cal.scale_choice(net, feature, np.ones(len(feature),dtype=bool),base['temperature']).astype(np.float32)
    q = np.column_stack((q[:,1] + scale*(q[:,0]-q[:,1]),q[:,1],q[:,1] + scale*(q[:,2]-q[:,1]))).astype(np.float32)
    mu = feature[train].mean(0); sd = feature[train].std(0); sd = np.where(sd < .05, 1., sd)
    tau_unit=float(np.median(a['se'][train]))
    if not np.isfinite(tau_unit) or tau_unit <= 0: raise RuntimeError('invalid training scale')
    norm = np.clip((feature-mu)/sd,-8,8).astype(np.float32)
    xt = torch.from_numpy(norm[train]); qt = torch.from_numpy(q[train]); zt = torch.from_numpy(z[train]); st=torch.from_numpy(a['se'][train].astype(np.float32))
    gt = torch.as_tensor(np.array([GROUPS.index(str(g)) for g in a['group'][train]]))
    base_lo, base_hi = q[:,0], q[:,2]
    dev_results = {'confidence': metrics(base_lo,base_hi,z,a['se'],a['group'],dev)}
    candidates = {}
    for mode in MODES:
        torch.manual_seed(SEED)
        model = Head(norm.shape[1])
        if mode in ('floor', 'bias_floor'):
            with torch.no_grad(): model.net[-1].bias[1] = -2.
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        for _ in range(EPOCHS):
            model.train(); opt.zero_grad(); loss=objective(model,xt,qt,zt,st,gt,mode,tau_unit)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite training objective')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        model.eval()
        with torch.no_grad():
            lo, hi, _, _ = endpoints(model(torch.from_numpy(norm)),torch.from_numpy(q),mode,torch.from_numpy(a['se'].astype(np.float32)),tau_unit)
        lo,hi = lo.numpy(),hi.numpy()
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
            raise RuntimeError('invalid candidate interval')
        dev_results[mode] = metrics(lo,hi,z,a['se'],a['group'],dev)
        candidates[mode] = (model.state_dict(),lo,hi)
    ref=dev_results['confidence']; ref_min=min(ref[g]['coverage'] for g in GROUPS[1:])
    # Confidence participates in the same preregistered ranking.
    eligible=[(ref_min,-float(np.mean([ref[g]['length'] for g in GROUPS[1:]])),1,0,'confidence')]
    for mode in MODES:
        v=dev_results[mode]; worst=min(v[g]['coverage'] for g in GROUPS[1:])
        if (v['id']['coverage'] >= ref['id']['coverage']-.05-1e-12
            and v['id']['length'] <= 1.5*ref['id']['length']):
            eligible.append((worst, -float(np.mean([v[g]['length'] for g in GROUPS[1:]])),0,-MODES.index(mode),mode))
    selected=max(eligible)[4]
    if selected == 'confidence': lo,hi=base_lo,base_hi
    else: _,lo,hi=candidates[selected]
    # Persist immutable selection and candidate parameters BEFORE calibration.
    out=R/'trained_v1'; out.mkdir(exist_ok=False)
    selection_model={'selected':selected,'states':{k:v[0] for k,v in candidates.items()},
           'normalization_mean':mu,'normalization_sd':sd,'tau_unit':tau_unit,'confidence_sha256':sha(fit),
           'registration_sha256':args.registration_sha256,'source_sha256':sha(__file__),'seed':SEED,'epochs':EPOCHS}
    torch.save(selection_model,out/'selection.pt')
    selection={'selected':selected,'dev_only_metrics':dev_results,'selection_model_sha256':sha(out/'selection.pt'),
               'source_sha256':sha(__file__),'registration_sha256':args.registration_sha256}
    with (out/'selection.json').open('x') as f: json.dump(selection,f,indent=2,allow_nan=False)
    # Independent calibration, selected once, expanded in studentized units.
    # k may exceed n for too-small samples; fail rather than silently use max.
    n=len(calibration); k=math.ceil((n+1)*(1-ALPHA))
    if k < 1 or k > n: raise RuntimeError(f'not enough independent calibration units: n={n}, k={k}')
    z[calibration] = ((a['est'][calibration] - a['truth'][calibration]) / a['se'][calibration]).astype(np.float32)
    width=np.maximum(hi[calibration]-lo[calibration],1e-5)
    scores=np.maximum.reduce((np.zeros(n),(lo[calibration]-z[calibration])/width,(z[calibration]-hi[calibration])/width))
    correction=float(np.partition(scores,k-1)[k-1])
    if not np.isfinite(correction) or correction < 0: raise RuntimeError('invalid calibration correction')
    model={'selected':selected,'selection_model_sha256':sha(out/'selection.pt'),'selection_sha256':sha(out/'selection.json'),
           'calibration_relative_expansion':correction,'calibration_n':n,'calibration_k':k,
           'confidence_sha256':sha(fit),'registration_sha256':args.registration_sha256,
           'source_sha256':sha(__file__),'seed':SEED,'epochs':EPOCHS}
    torch.save(model,out/'model.pt')
    plan={'baseline':'temperature-calibrated confidence head; pretrained/SFT head unchanged',
          'candidates':['confidence',*MODES],'seed':SEED,'epochs':EPOCHS,'optimizer':'AdamW lr=.001 weight_decay=.01, full batch, gradient clip 1',
          'equal_group_weights':True,'selection':'ID coverage >= confidence minus .05 and ID interval length <= 1.5 times confidence; confidence also participates; maximize worst DEV shift-group coverage then minimize mean shift interval length; baseline wins exact ties',
          'calibration':'pooled independent mechanisms, 95% split order statistic, one nonnegative relative width expansion applied after DEV selection; no OOD guarantee',
          'calibration_n':n,'calibration_k':k,'selected':selected,'dev_only_metrics':dev_results,
          'upstream_job_arrays':args.job_arrays,'upstream_qacct':accounting,'registration_sha256':args.registration_sha256,
          'source_sha256':sha(__file__),'training_plan_sha256':args.plan_sha256,'model_sha256':sha(out/'model.pt'),'confidence_sha256':sha(fit)}
    with (out/'freeze.json').open('x') as f: json.dump(plan,f,indent=2,allow_nan=False)
    print('FROZEN',selected,'model',sha(out/'model.pt'),'calibration_n',n,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-arrays',required=True,help='e.g. JOB:7,JOB:1-6,JOB:8-40');p.add_argument('--registration-sha256',required=True);p.add_argument('--source-sha256',required=True);p.add_argument('--plan-sha256',required=True)
    args=p.parse_args()
    if not all(re.fullmatch(r'[0-9a-f]{64}', x) for x in (args.registration_sha256,args.source_sha256,args.plan_sha256)): raise SystemExit('invalid hash')
    main(args)
