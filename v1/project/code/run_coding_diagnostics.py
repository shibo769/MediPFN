"""Evaluation-only paired coding stress test of already frozen CI methods.

No fitting, calibration updates, method selection, or coverage guarantee.
The original and recoded rows are paired observations, never new iid draws.
"""
from pathlib import Path
import argparse,json,os,time
import numpy as np
import torch
import posttrain as pt
import posttrain_heuristics as heur
import apply_posttrain as core
import build_cache as builder
from merge_cache import entries,load_entry,read,sha,resolve
from evaluate import EvaluationGuard,canonical_score,interval_stats,clean
from equivariance import transform_table,fixed_permutation

DEPENDENCIES=('apply_posttrain.py','posttrain.py','posttrain_heuristics.py','evaluate.py','merge_cache.py','equivariance.py','build_cache.py','build_equivariant_cache.py','inference.py','model.py')
REPLAY_KEYS=('stats','arm_names','alphas','point_index','quantiles','theta_quantiles','implied_theta_mean','implied_theta_sd','base_estimate','base_score_se','actions')

def sources():
    return {name:sha(Path(__file__).with_name(name)) for name in (Path(__file__).name,)+DEPENDENCIES}

def subset(a,index):
    count=len(a['truth'])
    return {key:(value[index] if np.ndim(value)>0 and len(value)==count else value) for key,value in a.items()}

def validate_registration(args):
    registration=read(args.registration);eq=read(args.equivariance_registration);ph=sha(args.protocol)
    if registration.get('frozen') is not True or registration.get('protocol_sha256')!=ph:raise ValueError('Frozen coding-diagnostic registration required')
    if registration.get('source_hashes')!=sources():raise ValueError('Coding diagnostic source/dependency drift')
    if registration.get('equivariance_registration_sha256')!=sha(args.equivariance_registration):raise ValueError('Equivariance registration drift')
    if eq.get('frozen') is not True or eq.get('protocol_sha256')!=ph:raise ValueError('Frozen equivariance registration required')
    for name,digest in eq['source_hashes'].items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Paired generator source drift')
    if int(registration.get('tables_per_group',0))!=10:raise ValueError('Diagnostic subset is fixed at ten tables per registered group')
    return registration,eq

def metadata(entry,manifest):
    return read(resolve(Path(manifest).parent,entry['metadata_path']))

def check_registration_release(gate,registration,equivariance_registration):
    extensions=gate.get('extension_registrations',{})
    if extensions.get('equivariance')!=sha(equivariance_registration):raise ValueError('Seed gate equivariance registration mismatch')
    if extensions.get('coding_diagnostics')!=sha(registration):raise ValueError('Seed gate coding-diagnostic registration mismatch')

def validate_cache_binding(entry,a,ledger,header,group,expected_sources):
    binding=ledger['binding']
    for key in ('protocol_sha256','checkpoint_sha256'):
        if binding.get(key)!=header[key]:raise ValueError('Cache ledger identity drift '+key)
        if key in a and str(np.asarray(a[key]).item())!=header[key]:raise ValueError('Cache scalar identity drift '+key)
    if binding.get('group')!=group:raise ValueError('Cache ledger raw group mismatch')
    if binding.get('source_hashes')!=expected_sources:raise ValueError('Cache source binding drift')
    if a['nuisance'].shape!=(group['count'],4,group['n'],5):raise ValueError('Nuisance attempt/shape drift')

def aligned_pair(original,paired,raw,group,pairing,eq,transform):
    count=len(paired['truth']);indices=np.asarray(paired['origindex'])
    expected=np.asarray(eq.get('test_indices',{}).get(group['name'],list(range(10))),dtype=np.int64) if transform=='m_flip' else np.arange(10)
    if not np.array_equal(indices,expected) or count!=10:raise ValueError('Pair subset differs from registered first ten tables')
    if pairing.get('origindex')!=indices.tolist() or pairing.get('original_name')!=group['name'] or pairing.get('transform')!=transform:raise ValueError('Pairing ledger mismatch')
    base=subset(original,indices)
    for key,raw_key in (('truth','truth'),('mechanism_id','mechanism_id'),('seeds','observation_seed'),('fold_seeds','fold_seed')):
        np.testing.assert_array_equal(base[key],paired[key]);np.testing.assert_array_equal(base[key],raw[raw_key][indices])
    for key in ('split','bank','p','n','seed_index'):
        np.testing.assert_array_equal(base[key],paired[key])
    np.testing.assert_array_equal(paired['original_group'],np.full(count,group['name']))
    np.testing.assert_array_equal(paired['transform'],np.full(count,transform))
    ids=np.asarray([group['name']+':'+str(int(seed)) for seed in paired['seeds']])
    np.testing.assert_array_equal(paired['base_table_id'],ids);np.testing.assert_array_equal(paired['pair_id'],np.char.add(ids,'|'+transform))
    permutation=fixed_permutation(group['p'],eq['permutation_seed'],group['name']) if transform=='x_permutation' else np.arange(group['p'])
    expected_perm=np.full((count,30),-1,np.int64);expected_perm[:,:group['p']]=permutation
    np.testing.assert_array_equal(paired['x_permutation'],expected_perm)
    if pairing.get('permutation')!=permutation.tolist():raise ValueError('Permutation metadata drift')
    return base,indices,permutation

def verify_scores(a,raw,indices,transform=None,permutation=None):
    """Recompute every full-n score directly from raw64 labels and saved nuisances."""
    delta=0.;verified=0;failed=0
    for i,index in enumerate(indices):
        table={key:raw[key][index] for key in ('x','a','m','y')}
        if transform:table=transform_table(table,transform,permutation)
        for view in range(4):
            if core.cache_failed(a,i,view):failed+=1;continue
            result=canonical_score(table,a['nuisance'][i,view])
            if result['failed']:raise ValueError('Successful saved nuisance cannot reproduce score')
            expected=interval_stats(result,.05)[:5]
            np.testing.assert_allclose(expected,a['stats'][i,view],rtol=2e-10,atol=2e-10)
            np.testing.assert_allclose(result['score'],a['scores'][i,view],rtol=2e-10,atol=2e-10)
            point=0 if view==0 else 1 if view==3 else None
            if point is not None:
                np.testing.assert_allclose([result['estimate'],result['se']],[a['estimates'][i,point],a['ses'][i,point]],rtol=2e-10,atol=2e-10)
            delta=max(delta,float(np.max(abs(expected-a['stats'][i,view]))));verified+=1
    return dict(verified_views=verified,retained_failed_views=failed,max_statistic_difference=delta,full_n=int(raw['a'].shape[1]))

def load_methods(args,ph,ck):
    art,ids,seeds,identity=core.verify_posttrain(args.core)
    if identity!={'protocol_sha256':ph,'checkpoint_sha256':ck}:raise ValueError('Core training bank identity drift')
    hart,hids,hseeds=heur.verify_candidate(args.heur,args.core);ids|=hids;seeds|=hseeds
    gate=read(args.auth)
    if gate.get('checkpoint_sha256')!=ck or gate.get('posttrain_artifacts_sha256')!=sha(Path(args.core)/'artifacts.json') or gate.get('heuristics_artifacts_sha256')!=sha(Path(args.heur)/'artifacts.json'):raise ValueError('Seed-scalar authorization must bind all unchanged methods')
    calibration,binding=core.calibration_arrays(args.cal,ph,ck)
    calibrators=core.fit_calibrators(calibration,ids,seeds)
    with np.load(Path(args.core)/'normalization.npz',allow_pickle=False) as z:norm={k:(z[k+'_center'],z[k+'_scale']) for k in ('x_single','x_multi')}
    heads=pt.load_heads(args.core,'cpu');policies={name:pt.load_policy(Path(args.core)/(file+'.pt'),'cpu') for name,file in core.POLICIES.items()}
    hheads={name:heur.load_head(Path(args.heur)/(name+'.pt'),'cpu') for name in heur.HEAD_ARMS}
    hpolicies={name:heur.load_expanded_policy(Path(args.heur)/(name+'.pt'),'cpu') for name in heur.EXPANDED_ARMS}
    pool=read(Path(args.heur)/'expanded_pool.json')['selection']
    return dict(heads=heads,policies=policies,hheads=hheads,hpolicies=hpolicies,pool=pool,norm=norm,calibrators=calibrators,
                train_ids=ids,train_seeds=seeds,cal_ids=set(calibration['mechanism_id'].astype(str)),
                cal_seeds=set(map(int,calibration['seeds']))|set(map(int,calibration['fold_seeds'])),
                bindings=dict(core_artifacts_sha256=sha(Path(args.core)/'artifacts.json'),heuristics_artifacts_sha256=sha(Path(args.heur)/'artifacts.json'),calibration=binding))

@torch.no_grad()
def replay(a,methods):
    c=core.apply_group(a,methods['heads'],methods['policies'],methods['norm'],methods['calibrators'])
    h=heur.apply_candidate_group(a,methods['heads'],methods['hheads'],methods['hpolicies'],methods['norm'],methods['pool'])
    names=np.concatenate((np.char.add('core:',c['arm_names']),np.char.add('heuristics:',h['arm_names'])))
    stats=np.concatenate((c['stats'],h['stats']),axis=1);points=np.concatenate((c['point_index'],h['point_index']),axis=1)
    quantiles=np.concatenate((c['quantiles'],h['quantiles']),axis=1)
    chosen=np.take_along_axis(a['estimates'],points,axis=1);ses=np.take_along_axis(a['ses'],points,axis=1)
    theta_q=chosen[:,:,None,None]-ses[:,:,None,None]*quantiles
    mean=np.full(chosen.shape,np.nan);sd=mean.copy()
    for j,name in enumerate(core.ARMS):
        action=c['actions'][:,j]
        for i in range(len(chosen)):
            k=int(action[i])
            if name.startswith('empirical'):continue
            if k<2:mean[i,j]=chosen[i,j];sd[i,j]=ses[i,j];continue
            prefix=('generic_error' if k==2 else 'mediation_error')+'_'
            w,mu,sigma=(np.asarray(c[prefix+field][i],float) for field in ('weights','means','scales'))
            ez=np.sum(w*mu);var=np.sum(w*(mu*mu+sigma*sigma))-ez*ez
            mean[i,j]=chosen[i,j]-ses[i,j]*ez;sd[i,j]=ses[i,j]*np.sqrt(max(0.,var))
    offset=len(core.ARMS);w,mu,sigma=(np.asarray(h['cdf_'+field],float) for field in ('weights','means','scales'))
    ez=np.sum(w*mu,-1);var=np.sum(w*(mu*mu+sigma*sigma),-1)-ez*ez
    mean[:,offset:]=chosen[:,offset:]-ses[:,offset:]*ez;sd[:,offset:]=ses[:,offset:]*np.sqrt(np.maximum(var,0))
    return dict(stats=stats,arm_names=names,alphas=c['alphas'],point_index=points,quantiles=quantiles,theta_quantiles=theta_q,
                implied_theta_mean=mean,implied_theta_sd=sd,base_estimate=chosen,base_score_se=ses,
                actions=np.concatenate((c['actions'],h['actions']),1))

def paired_summary(left,right,truth,mechanism_ids):
    records=[]
    mechanism_ids=np.asarray(mechanism_ids);laws=np.unique(mechanism_ids)
    if mechanism_ids.shape!=np.asarray(truth).shape:raise ValueError('Paired summary law identity shape mismatch')
    for j,name in enumerate(left['arm_names']):
        for ai,alpha in enumerate(left['alphas']):
            a,b=left['stats'][:,j,ai],right['stats'][:,j,ai]
            va=(a[:,5]==0)&np.isfinite(a[:,:4]).all(1);vb=(b[:,5]==0)&np.isfinite(b[:,:4]).all(1);both=va&vb
            ca=va&(a[:,2]<=truth)&(truth<=a[:,3]);cb=vb&(b[:,2]<=truth)&(truth<=b[:,3]);diff=cb.astype(float)-ca.astype(float)
            residuals=np.array([np.sum(diff[mechanism_ids==law]-diff.mean()) for law in laws])
            cluster_se=float(np.sqrt(len(laws)/(len(laws)-1)*np.sum(residuals**2))/len(diff)) if len(laws)>1 else None
            def avg(value):return float(np.mean(value)) if len(value) else None
            def finite_diff(x,y):
                mask=both&np.isfinite(x)&np.isfinite(y)
                return avg((y-x)[mask])
            td=right['theta_quantiles'][:,j,ai]-left['theta_quantiles'][:,j,ai]
            records.append(dict(method=str(name),alpha=float(alpha),attempts=len(truth),original_failures=int((~va).sum()),paired_failures=int((~vb).sum()),both_valid=int(both.sum()),
                original_coverage=float(ca.mean()),paired_coverage=float(cb.mean()),coverage_difference=float(diff.mean()),paired_law_cluster_MCSE=cluster_se,law_clusters=len(laws),
                original_length_mean=avg((a[:,3]-a[:,2])[va]),paired_length_mean=avg((b[:,3]-b[:,2])[vb]),
                paired_valid_length_difference=avg(((b[:,3]-b[:,2])-(a[:,3]-a[:,2]))[both]),
                paired_valid_primary_point_difference=avg((b[:,0]-a[:,0])[both]),
                paired_valid_interval_endpoint_abs_shift=avg(abs(b[both,2:4]-a[both,2:4]).ravel()),
                paired_valid_theta_quantile_RMS_shift=float(np.sqrt(np.mean(td[both]**2))) if both.any() else None,
                implied_theta_mean_difference=finite_diff(left['implied_theta_mean'][:,j],right['implied_theta_mean'][:,j]),
                implied_theta_SD_difference=finite_diff(left['implied_theta_sd'][:,j],right['implied_theta_sd'][:,j]),
                distribution_definition='Six central error quantiles mapped to theta with each coding own canonical estimate/SE; moment comparisons only when a full normal/GMM distribution is available.',
                failure_denominator='Coverage treats all failed attempts as uncovered; length/shift diagnostics explicitly condition on valid outputs.'))
    return records

def run(args):
    if args.cpu_threads is not None:torch.set_num_threads(args.cpu_threads)
    registration,eq=validate_registration(args);ph=sha(args.protocol);guard=EvaluationGuard(args.protocol,args.auth)
    guard.check('test' if args.transform=='m_flip' else 'dev')
    gate=read(args.auth)
    check_registration_release(gate,args.registration,args.equivariance_registration)
    oh,original_entries=entries(args.cache_manifest);pair_header,pair_entries=entries(args.paired_cache_manifest);raw_manifest=read(args.raw_manifest)
    ck=oh['checkpoint_sha256']
    if any(h.get('protocol_sha256')!=ph or h.get('checkpoint_sha256')!=ck for h in (oh,pair_header)):raise ValueError('Original/paired manifest identity mismatch')
    if raw_manifest.get('protocol_sha256')!=ph:raise ValueError('Raw manifest protocol drift')
    if pair_header.get('registration_sha256',sha(args.equivariance_registration))!=sha(args.equivariance_registration):raise ValueError('Paired manifest registration mismatch')
    model_seed=int(args.model_seed_index);lookup={g['name']:g for g in original_entries};raw_lookup={g['name']:g for g in raw_manifest['groups']};cells={g['name']:g for g in read(args.protocol)['cells']}
    names=eq['test_groups']['m_flip'] if args.transform=='m_flip' else [name for name in eq['diagnostic_groups']['x_permutation'] if f'_s{model_seed}_' in name]
    if len(names)!=(15 if args.transform=='m_flip' else 4):raise ValueError('Registered diagnostic group denominator differs')
    selected=[g for g in pair_entries if g.get('transform')==args.transform]
    if {g.get('original_group') for g in selected}!=set(names) or len(selected)!=len(names):raise ValueError('Paired manifest missing/duplicate/unregistered groups')
    methods=load_methods(args,ph,ck);output=Path(args.out);output.mkdir(parents=True,exist_ok=True);reports=[];all_left=[];all_right=[];all_truth=[];all_laws=[]
    for pe in selected:
        name=pe['original_group'];group=raw_lookup[name];oe=lookup[name]
        if group['split']!=('test' if args.transform=='m_flip' else 'dev') or group['seed_index'] not in (-1,model_seed):raise ValueError('Coding subset bank/seed drift')
        for key in ('name','split','bank','seed_index','family','p','n','population_kind','mechanisms','repeats','mechanism_seed','observation_seed','fold_seed'):
            if group[key]!=cells[name][key]:raise ValueError('Registered raw descriptor drift '+key)
        if group['count']!=cells[name]['mechanisms']*cells[name]['repeats']:raise ValueError('Raw attempt count drift')
        op,original=load_entry(oe,args.cache_manifest,ph);pp,paired=load_entry(pe,args.paired_cache_manifest,ph)
        ol=metadata(oe,args.cache_manifest);pl=metadata(pe,args.paired_cache_manifest)
        expected_sources=builder.source_hashes();validate_cache_binding(oe,original,ol,oh,group,expected_sources)
        derived=dict(group);derived.update(name=pe['name'],count=10,equivariance=pl['pairing'])
        validate_cache_binding(pe,paired,pl,pair_header,derived,{**expected_sources,**eq['source_hashes']})
        if pl['pairing'].get('registration_sha256')!=sha(args.equivariance_registration):raise ValueError('Pair ledger registration drift')
        raw,raw_binding=builder.raw_group(group,args.raw_manifest,args.protocol,ph)
        for source in (ol,pl):
            for key in ('raw_sha256','raw_metadata_sha256'):
                if source['raw_provenance'].get(key)!=raw_binding[key]:raise ValueError('Paired/original raw provenance drift '+key)
        base,indices,permutation=aligned_pair(original,paired,raw,group,pl['pairing'],eq,args.transform)
        # Fold labels are never transformed; pairing retains their exact source
        # array through the immutable raw overlay and the registered generator.
        ids=set(base['mechanism_id'].astype(str));seeds=set(map(int,base['seeds']))|set(map(int,base['fold_seeds']))
        if ids&methods['cal_ids'] or seeds&methods['cal_seeds']:raise ValueError('Original-only calibration overlaps diagnostic rows')
        if args.transform=='m_flip' and (ids&methods['train_ids'] or seeds&methods['train_seeds']):raise ValueError('Final diagnostic rows overlap training')
        budget=pl['additional_inference_budget']
        if budget.get('tables')!=10 or budget.get('views')!=3 or budget.get('outer_folds')!=2:raise ValueError('Paired budget denominator mismatch')
        for key in ('encode_forward_calls','support_table_encodings','query_forward_calls','query_patient_evaluations'):
            if not isinstance(budget.get(key),int) or budget[key]<0:raise ValueError('Missing actual extra encoding budget '+key)
        if np.any(paired['success']) and (budget['support_table_encodings']==0 or budget['query_patient_evaluations']==0):raise ValueError('Successful recoded cache with zero extra encoding budget')
        locks=dict(protocol_sha256=ph,checkpoint_sha256=ck,source_hashes=sources(),registration_sha256=sha(args.registration),equivariance_registration_sha256=sha(args.equivariance_registration),
            authorization_sha256=sha(args.auth),raw_manifest_sha256=sha(args.raw_manifest),original_manifest_sha256=sha(args.cache_manifest),paired_manifest_sha256=sha(args.paired_cache_manifest),
            raw_binding=raw_binding,original_cache_sha256=sha(op),paired_cache_sha256=sha(pp),original_metadata_sha256=oe['metadata_sha256'],paired_metadata_sha256=pe['metadata_sha256'],**methods['bindings'])
        path=output/(name+'__'+args.transform+'.coding.npz');ledger=path.with_suffix('.json')
        if path.exists() or ledger.exists():
            if not(path.exists() and ledger.exists()):raise FileExistsError('Interrupted coding diagnostic retained')
            saved=read(ledger)
            if saved['locks']!=locks or saved['sha256']!=sha(path):raise ValueError('Coding diagnostic resume drift')
            with np.load(path,allow_pickle=False) as z:
                left={k:z['original_'+k] for k in REPLAY_KEYS};right={k:z['paired_'+k] for k in REPLAY_KEYS}
            reports.append(saved)
        else:
            score_checks=dict(original=verify_scores(base,raw,indices),paired=verify_scores(paired,raw,indices,args.transform,permutation))
            left=replay(base,methods);right=replay(paired,methods)
            payload={**{'original_'+k:v for k,v in left.items()},**{'paired_'+k:v for k,v in right.items()},'truth':base['truth'],'origindex':indices,
                     'mechanism_id':base['mechanism_id'],'seeds':base['seeds'],'fold_seeds':base['fold_seeds'],'fold_ids':raw['fold_ids'][indices],
                     'original_failed':base['failed'],'paired_failed':paired['failed'],'original_base_estimates':base['estimates'],'paired_base_estimates':paired['estimates'],
                     'original_base_ses':base['ses'],'paired_base_ses':paired['ses']}
            with path.open('xb') as f:np.savez_compressed(f,**payload)
            saved=dict(name=name,transform=args.transform,tables=10,sha256=sha(path),locks=locks,score_checks=score_checks,additional_inference_budget=budget,
                       summary=paired_summary(left,right,base['truth'],base['mechanism_id']),all_attempts_retained=True,evaluation_only=True,no_coverage_guarantee=True)
            pt.save(ledger,saved);reports.append(saved)
        all_left.append(left);all_right.append(right);all_truth.append(base['truth']);all_laws.append(base['mechanism_id']);print('CODING_DIAGNOSTIC',name,flush=True)
    def concatenate(values):
        return {k:(values[0][k] if k in ('arm_names','alphas') else np.concatenate([v[k] for v in values])) for k in values[0]}
    combined=paired_summary(concatenate(all_left),concatenate(all_right),np.concatenate(all_truth),np.concatenate(all_laws))
    pt.save(output/'summary.json',dict(protocol_sha256=ph,checkpoint_sha256=ck,source_hashes=sources(),registration_sha256=sha(args.registration),authorization_sha256=sha(args.auth),
        transform=args.transform,model_seed_index=model_seed,groups=len(reports),paired_tables=sum(r['tables'] for r in reports),records=combined,
        additional_inference_budget={k:sum(r['additional_inference_budget'][k] for r in reports) for k in ('encode_forward_calls','support_table_encodings','query_forward_calls','query_patient_evaluations')},
        group_ledgers=[str(output/(r['name']+'__'+args.transform+'.coding.json')) for r in reports],
        interpretation='Prespecified coding stress diagnostic only; same-row paired shifts, not independent extra test draws. Original CI evaluation remains authoritative. No optimization or method selection; no formal coverage guarantee.'))

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','cache-manifest','paired-cache-manifest','raw-manifest','core','heur','cal','auth','out','registration','equivariance-registration'):p.add_argument('--'+name,required=True)
    p.add_argument('--model-seed-index',type=int,choices=(0,1,2),required=True);p.add_argument('--transform',choices=('m_flip','x_permutation'),default='m_flip')
    p.add_argument('--cpu-threads',type=int,default=int(os.environ.get('NSLOTS',os.environ.get('SLURM_CPUS_PER_TASK','4'))));return p

if __name__=='__main__':run(parser().parse_args())
