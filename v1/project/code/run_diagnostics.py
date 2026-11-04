"""Read-only oracle replacement and SE diagnostics on prespecified fixed laws.

Oracle labels are evaluation-only. They never enter inference or model selection.
The base deployed estimator uses the full-n consensus score. This diagnostic
replaces g, q, or the two A=1 outcome regressions in that saved nuisance matrix.
"""
from pathlib import Path
import argparse,json,os,time
import numpy as np
from data import sample_mechanism
from evaluate import EvaluationGuard,canonical_nuisance,canonical_score,interval_stats,failed_stats,metrics,clean,sha
from merge_cache import entries,load_entry
from apply_posttrain import cache_failed

ARMS=('pfn_multiview','oracle_g','oracle_q','oracle_outcome','oracle_all')
COLS=((),(0,),(1,2),(3,4),(0,1,2,3,4))

def identity_binding(raw,cache):
    for left,right in [('truth','truth'),('mechanism_id','mechanism_id'),('observation_seed','seeds'),('fold_seed','fold_seeds')]:
        np.testing.assert_array_equal(raw[left],cache[right])

def se_diagnostics(stats,truth,mechanism_ids):
    rows=[]
    for mid in np.unique(mechanism_ids):
        selected=mechanism_ids==mid;values=stats[selected];target=truth[selected]
        if not np.all(target==target[0]):raise ValueError('Cannot estimate sampling SD across different laws')
        for j,name in enumerate(ARMS):
            current=values[:,j,1];valid=(current[:,5]==0)&np.isfinite(current[:,:4]).all(1)
            z=(current[valid,0]-target[valid])/current[valid,1]
            row=dict(mechanism_id=str(mid),method=name,**metrics(current,target,.05,fixed_mechanism=True))
            row.update(z_mean=float(np.mean(z)) if len(z) else None,z_rms=float(np.sqrt(np.mean(z*z))) if len(z) else None,
                       z_quantiles={str(q):float(np.quantile(z,q)) for q in (.005,.025,.05,.5,.95,.975,.995)} if len(z) else {},
                       valid_only_z_denominator=len(z),all_attempts_denominator=len(target))
            rows.append(row)
    return rows

def run_cell(group,cache_entry,args):
    guard=EvaluationGuard(args.protocol,args.authorization);guard.check('test')
    registration=json.loads(Path(args.registration).read_text())
    if not registration.get('frozen') or registration['protocol_sha256']!=sha(args.protocol):raise ValueError('Diagnostic registration required')
    for name,digest in registration['source_hashes'].items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Diagnostic source drift '+name)
    gate=json.loads(Path(args.authorization).read_text())
    if gate.get('diagnostics_registration_sha256')!=sha(args.registration):raise ValueError('Final release must bind diagnostic registration')
    if group['bank']!='fixed_mechanism' or group['split']!='test':raise ValueError('Only prespecified fixed-mechanism test bank')
    ph=sha(args.protocol);protocol=json.loads(Path(args.protocol).read_text());cells={cell['name']:cell for cell in protocol['cells']}
    if group['name'] not in cells:raise ValueError('Unregistered diagnostic cell')
    descriptor=cells[group['name']]
    for key in ('name','split','bank','seed_index','family','p','n','population_kind','mechanisms','repeats','mechanism_seed','observation_seed','fold_seed'):
        if group[key]!=descriptor[key]:raise ValueError('Diagnostic cohort descriptor drift '+key)
    if group['count']!=descriptor['mechanisms']*descriptor['repeats'] or group.get('protocol_sha256')!=ph:raise ValueError('Diagnostic cohort count/protocol drift')
    header,_=entries(args.cache_manifest)
    if header.get('protocol_sha256')!=ph or header.get('checkpoint_sha256')!=gate.get('checkpoint_sha256'):raise ValueError('Diagnostic cache header drift')
    for key in ('name','split','bank','p','n','count'):
        if cache_entry.get(key)!=group[key]:raise ValueError('Diagnostic cache/raw group mismatch '+key)
    if group['raw_sha256']!=sha(group['raw_path']) or group['metadata_sha256']!=sha(group['metadata_path']):raise ValueError('Raw data drift')
    cache_path,cache=load_entry(cache_entry,args.cache_manifest,sha(args.protocol))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);path=output/(group['name']+'.diagnostics.npz');ledger=path.with_suffix('.json')
    binding=dict(raw_sha256=group['raw_sha256'],raw_metadata_sha256=group['metadata_sha256'],cache_sha256=sha(cache_path),
                 cache_metadata_sha256=cache_entry['metadata_sha256'],checkpoint_sha256=header['checkpoint_sha256'],
                 registration_sha256=sha(args.registration),protocol_sha256=sha(args.protocol))
    if path.exists() and ledger.exists():
        report=json.loads(ledger.read_text())
        if report['bindings']!=binding or report['sha256']!=sha(path):raise ValueError('Diagnostic output drift')
        return dict(name=group['name'],resumed=True)
    if path.exists() or ledger.exists():raise FileExistsError('Interrupted diagnostic output retained')
    with np.load(group['raw_path'],allow_pickle=False) as z:raw={key:z[key] for key in z.files}
    if len(raw['truth'])!=group['count']:raise ValueError('Diagnostic attempt count mismatch')
    cache_binding=json.loads(Path(cache_entry['metadata_path']).read_text())['binding']
    for key,value in (('protocol_sha256',ph),('checkpoint_sha256',header['checkpoint_sha256'])):
        declared=str(np.asarray(cache[key]).item()) if key in cache else cache_binding.get(key)
        if declared!=value:raise ValueError('Diagnostic cache scalar/ledger drift '+key)
    if cache_binding.get('group')!=group or cache['nuisance'].shape!=(group['count'],4,group['n'],5):raise ValueError('Diagnostic cache group/nuisance-shape drift')
    identity_binding(raw,cache);metadata=json.loads(Path(group['metadata_path']).read_text());laws={};count=len(raw['truth']);start=time.perf_counter()
    if len(metadata['tables'])!=count or metadata['cell']!=descriptor:raise ValueError('Diagnostic raw metadata count/cell mismatch')
    for i,record in enumerate(metadata['tables']):
        for left,right in (('mechanism_id','mechanism_id'),('observation_seed','observation_seed'),('fold_seed','fold_seed')):
            if str(record[left])!=str(raw[right][i]):raise ValueError('Diagnostic raw metadata table identity drift '+left)
    stats=np.empty((count,len(ARMS),3,6));mse=np.full((count,3),np.nan);truth_delta=np.empty(count);errors=[];base_failed=np.zeros(count,bool)
    for i,record in enumerate(metadata['tables']):
        law=record['mechanism'];mid=str(raw['mechanism_id'][i])
        if mid not in laws:
            scm=sample_mechanism(law['seed'],law['p'],law['family'],law['population_size'])
            if scm.population_hash!=law['population_hash']:raise ValueError('Regenerated finite population changed')
            np.testing.assert_allclose(scm.theta_NIE,law['theta_NIE'],rtol=0,atol=1e-12)
            laws[mid]=scm
        scm=laws[mid];truth_delta[i]=scm.theta_NIE-raw['truth'][i]
        np.testing.assert_allclose(scm.theta_NIE,raw['truth'][i],rtol=0,atol=1e-12)
        table={key:raw[key][i] for key in ('x','a','m','y')};oracle=canonical_nuisance(scm.nuisance(table['x']));base=cache['nuisance'][i,3]
        base_failed[i]=cache_failed(cache,i,3)
        if np.isfinite(base).all():mse[i]=[float(np.mean((base[:,columns]-oracle[:,columns])**2)) for columns in ((0,),(1,2),(3,4))]
        row_errors=[]
        for j,columns in enumerate(COLS):
            if base_failed[i] and name_is_base(j):
                stats[i,j]=failed_stats();row_errors.append('Saved PFN failure retained in partial replacements');continue
            nu=np.array(base,copy=True)
            if columns:nu[:,columns]=oracle[:,columns]
            # Even oracle_all cannot silently recover an invalid inference
            # attempt: its scientific diagnostic uses the known oracle itself.
            summary=canonical_score(table,nu);row_errors.append(summary['error'])
            for ai,alpha in enumerate((.10,.05,.01)):stats[i,j,ai]=interval_stats(summary,alpha)
        errors.append(row_errors)
    with path.open('xb') as handle:np.savez_compressed(handle,stats=stats,arm_names=np.array(ARMS),alphas=np.array([.1,.05,.01]),
        nuisance_mse=mse,truth_regeneration_delta=truth_delta,base_failed=base_failed,oracle_all_recovery=base_failed&(stats[:,-1,1,5]==0),truth=raw['truth'],mechanism_id=raw['mechanism_id'],seeds=raw['observation_seed'],fold_seeds=raw['fold_seed'])
    report=dict(name=group['name'],sha256=sha(path),bindings=binding,tables=count,seconds=time.perf_counter()-start,errors=errors,
        nuisance_mse_columns=['g','q0_q1','mu10_mu11'],se_diagnostics=se_diagnostics(stats,raw['truth'],raw['mechanism_id']),
        base_failed_attempts=int(base_failed.sum()),oracle_all_recovered_attempts=int(np.sum(base_failed&(stats[:,-1,1,5]==0))),
        clipping='Same fixed0.01 canonical score clipping for all five arms; oracle_all is a clipped-oracle diagnostic, not an unbiased untrimmed EIF claim.',
        oracle_labels='Evaluation diagnostic only; no trained artifact or deployable method uses these labels.',
        failure_denominators='Every method retains all attempts. Oracle_all can be evaluated from known truth even if a saved PFN nuisance is invalid.',
        source_hashes=registration['source_hashes'])
    ledger.write_text(json.dumps(clean(report),indent=2,allow_nan=False));return dict(name=group['name'],tables=count,sha256=report['sha256'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--raw-manifest',required=True);p.add_argument('--cache-manifest',required=True);p.add_argument('--registration',required=True);p.add_argument('--authorization',required=True);p.add_argument('--output',required=True);p.add_argument('--task-index',type=int,default=0);p.add_argument('--task-count',type=int,default=1);args=p.parse_args()
    if not 0<=args.task_index<args.task_count:raise ValueError('Invalid shard')
    # Gate before manifest/bank access.
    EvaluationGuard(args.protocol,args.authorization).check('test');header,caches=entries(args.cache_manifest);gate=json.loads(Path(args.authorization).read_text())
    if gate.get('checkpoint_sha256')!=header['checkpoint_sha256']:raise ValueError('Released checkpoint identity mismatch')
    manifest=json.loads(Path(args.raw_manifest).read_text());lookup={entry['name']:entry for entry in caches}
    if manifest.get('protocol_sha256')!=sha(args.protocol) or header.get('protocol_sha256')!=sha(args.protocol):raise ValueError('Diagnostic manifest protocol mismatch')
    groups=[g for g in manifest['groups'] if g['bank']=='fixed_mechanism' and g['split']=='test']
    for group in groups[args.task_index::args.task_count]:print(json.dumps(run_cell(group,lookup[group['name']],args)),flush=True)

def name_is_base(index):
    return index<len(ARMS)-1

if __name__=='__main__':main()
