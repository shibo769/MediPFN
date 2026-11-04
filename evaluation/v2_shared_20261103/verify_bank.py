"""Independent model-free admission checks for the frozen shared table bank."""
import argparse
from collections import Counter,defaultdict
from itertools import product
import json
from pathlib import Path
import numpy as np
import bank


def independent_truth(source,c):
    sigmoid=lambda x:np.exp(-np.logaddexp(0.,-np.asarray(x,float)))
    z,w=np.polynomial.hermite.hermgauss(256);z=np.sqrt(2.)*z;w=w/np.sqrt(np.pi)
    if source=='he_song_xu':return c['alpha_s']*c['beta_m']
    if source.startswith('jones_') and source!='jones_correlated':return float(np.dot(c['alpha0'],c['beta0']))
    if source=='jones_correlated':return len(set(c['a_to_m_indices'])&set(c['m_to_y_indices']))*c['a_to_m_coefficient']*c['m_to_y_coefficient']
    if source=='miles':return c['mediator_exposure']*(c['outcome_mediator']+c['outcome_exposure_mediator'])
    if source=='shan':return (c['mediator_treatment']+c['mediator_treatment_x1_squared']/3)*(c['outcome_mediator']+c['outcome_treatment_mediator'])
    if source=='zuo':return c['alpha_t']*(c['beta_m']+c['beta_mt'])
    if source=='xia_continuous':return c['m_a']*(c['y_m']+c['y_am'])
    if source=='xia_binary':
        baseline=c['m_logit_intercept']+c['m_logit_x']*c['x_sd']*z
        return float(w@(sigmoid(baseline+c['m_logit_a'])-sigmoid(baseline)))*(c['y_m']+c['y_am'])
    if source=='cheng':
        baseline=c['mediator_intercept']+c['covariate_sd']*np.linalg.norm(c['mediator_coefficients'])*z
        return float(w@(sigmoid(baseline+c['mediator_treatment_coefficient'])-sigmoid(baseline)))*c['outcome_mediator_coefficient']
    if source=='eric':
        result=0.;mc=c['mediator_coefficients'];yc=c['outcome_coefficients'];xc=c['x3_coefficients']
        for x1,x2 in product((0.,1.),repeat=2):
            p2=c['x2_coefficients'][0]+c['x2_coefficients'][1]*x1
            mass=(c['x1_probability'] if x1 else 1-c['x1_probability'])*(p2 if x2 else 1-p2)
            x3=xc[0]+xc[1]*x1+xc[2]*x2+c['x3_sd']*z
            b=mc[0]+mc[1]*x1+mc[2]*x2+mc[3]*x3+mc[4]*x1*x3
            result+=mass*float(w@(sigmoid(b+c['mediator_treatment_coefficient'])-sigmoid(b)))*(yc[5]+yc[6])
        return result
    if source in ('diaz_hejazi','diaz_intermediate'):
        result=0.
        for ww in product((0.,1.),repeat=3):
            x1,x2,x3=ww
            if source=='diaz_hejazi':
                mass=float(np.prod([p if value else 1-p for value,p in zip(ww,c['baseline_probabilities'])]))
                def probs(a):return np.array([1-sigmoid((a+x1)/(a+x1+.5)),sigmoid((a-1+x2)/(x3+3)),sigmoid((a-2+2*x1)/(2*x1+.5))])
                difference=(probs(c['mediation_path_scale'])-probs(0.))@c['outcome_mediator_coefficients']
            else:
                p3=c['p_w3_intercept']+c['p_w3_slope']*(x1+x2)
                mass=float(np.prod([p if value else 1-p for value,p in zip(ww,[c['p_w1'],c['p_w2'],p3])]))
                b=c['m_w1']*x1+c['m_w2']*x2;den=x1+x2+x3+c['y_denominator_intercept']
                difference=(sigmoid(b+c['m_a']+c['m_aw2']*x2)-sigmoid(b))*(sigmoid((c['y_a']+c['y_m'])/den)-sigmoid(c['y_a']/den))
            result+=mass*float(difference)
        return result
    raise ValueError(source)


def walk_ids(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if k.endswith('mechanism_ids') and isinstance(v,list):
                for x in v:
                    if isinstance(x,str):yield x
            elif k in ('mechanism_id','native_mechanism_id') and isinstance(v,str):yield v
            else:yield from walk_ids(v)
    elif isinstance(value,list):
        for v in value:yield from walk_ids(v)


def verify(root,dgp_root,archive_root=None,fixture_root=None,reproduce=True):
    root=Path(root);m=bank.load_manifest(root);dgp=bank.import_dgp(dgp_root)
    if bank.dgp_bindings(dgp_root)!=m['dgp_source_sha256']:raise ValueError('DGP source changed')
    if bank.file_sha256(root/'protocol.json')!=m['protocol_sha256']:raise ValueError('Frozen protocol changed')
    if bank.file_sha256(Path(bank.__file__))!=m['bank_program_sha256']:raise ValueError('Bank generator source changed')
    vendor=Path(dgp_root)/'dgp/vendor';source_manifest=json.loads((Path(dgp_root)/'dgp/source_manifest.json').read_text())
    for row in source_manifest['sources']:
        raw=(vendor/(row['name']+'.py')).read_bytes()
        import hashlib
        if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=row['sha']:raise ValueError('Upstream vendor blob changed')
    blocks=defaultdict(list);populations=defaultdict(set);weights=defaultdict(float);native_ids=set();seeds=set();rows=set();max_truth_error=0.
    for case in m['cases']:
        t=bank.load_case(root,case);meta=t['metadata'];obs=t['observed']
        if int(meta['mechanism_seed']).bit_length()<=256 or int(meta['table_seed']).bit_length()<=256:raise ValueError('Test seed left new namespace')
        if meta['table_seed'] in rows:raise ValueError('Observation seed reused')
        rows.add(meta['table_seed']);seeds.add(meta['mechanism_seed']);native_ids.add(case['mechanism_id'])
        blocks[case['mechanism_block_id']].append((case,t));populations[case['stratum']].add(case['cluster_id'])
        weights[(case['stratum'],case['n'])]+=case['analysis_weight_within_n']
        if obs['x'].ndim!=2 or obs['m'].ndim!=2 or obs['a'].ndim!=1 or obs['y'].ndim!=1:raise ValueError('Observed shape mismatch')
        if len(obs['x_types'])!=obs['x'].shape[1] or len(obs['mediator_types'])!=obs['m'].shape[1]:raise ValueError('Role types mismatch')
        if not np.isin(obs['a'],[0,1]).all():raise ValueError('Invalid treatment values')
        truth=independent_truth(case['source_id'],meta['config']);error=abs(truth-case['truth']);max_truth_error=max(max_truth_error,error)
        if error>1e-6*(1+abs(truth)):raise ValueError('Independent structural truth disagrees')
        if reproduce:
            r=dgp.sample_task(case['source_id'],case['n'],meta['table_seed'],mechanism_seed=meta['mechanism_seed'],config=meta['config'],compute_truth=True)
            observed={k:np.asarray(r[k],np.float32) for k in ('x','a','m','y')}
            observed.update(x_types=np.asarray(r['x_types'],dtype=bank._TYPE_DTYPE),mediator_types=np.asarray(r['mediator_types'],dtype=bank._TYPE_DTYPE))
            if bank.observed_hash(observed)!=case['observed_sha256']:raise ValueError('Independent table regeneration differs')
        if set(bank.load_observed_case(root,case))!=set(bank.OBSERVED_KEYS):raise ValueError('Inference loader exposes metadata/truth')
    formal=m['status']=='FROZEN_SCIENTIFIC_TEST_TABLE_BANK'
    if formal:
        if dict(Counter(c['stratum'] for c in m['cases']))!={'id':768,'source_ood':288,'composition_ood':96}:raise ValueError('Wrong prespecified case counts')
        for k,value in weights.items():
            if not np.isclose(value,1.):raise ValueError('Analysis weights do not sum to one')
        for block,entries in blocks.items():
            if len(entries)!=2:raise ValueError('Not two observation replicates per block')
            (a,ta),(b,tb)=entries
            if a['mechanism_id']!=b['mechanism_id'] or a['cluster_id']!=b['cluster_id'] or ta['metadata']['config']!=tb['metadata']['config']:raise ValueError('Replicates do not share one structural mechanism')
            if a['observation_seed']==b['observation_seed'] or a['observed_sha256']==b['observed_sha256']:raise ValueError('Replicate observations reused')
    overlap=dict(status='namespace and available archived fingerprint audit',prior_seed_max_bits=164,test_seed_min_bits=min(x.bit_length() for x in seeds),
                 exact_seed_and_row_overlap_possible=False,archived_files_checked=0,archived_unique_mechanism_ids=0,archived_native_id_overlaps=0)
    if archive_root:
        known=set();bindings={}
        for path in sorted(Path(archive_root).rglob('*.json')):
            doc=json.loads(path.read_text(encoding='utf-8'));found=set(walk_ids(doc))
            if found:known.update(found);bindings[path.relative_to(archive_root).as_posix()]=bank.file_sha256(path)
        overlap.update(archived_files_checked=len(bindings),archived_unique_mechanism_ids=len(known),archived_native_id_overlaps=len(known&native_ids),archive_file_sha256=bindings)
        # Same true population with fresh rows is a valid ID case, not automatically
        # a leakage failure. Here any actual fingerprint overlap is reported.
    if fixture_root and formal:
        f=bank.load_manifest(fixture_root)
        if {v['observation_seed'] for v in f['cases']}&rows:raise ValueError('Validation fixture/formal observation overlap')
        if {v['mechanism_seed'] for v in f['cases']}&seeds:raise ValueError('Validation fixture/formal mechanism seed overlap')
        overlap['fixture_formal_populations_overlap']=len({v['cluster_id'] for v in f['cases']}&{v['cluster_id'] for v in m['cases']})
    report=dict(status='PASS_MODEL_FREE_BANK_ADMISSION',manifest_sha256=bank.file_sha256(root/'manifest.json'),
         tables=len(m['cases']),counts=m['counts'],actual_unique_populations_by_stratum={k:len(v) for k,v in populations.items()},
         actual_unique_native_configurations=len(native_ids),independent_row_blocks=len(blocks),
         regenerated_observed_tables=len(m['cases']) if reproduce else 0,upstream_exact_blob_hashes_checked=15,
         independent_truth_max_absolute_difference=max_truth_error,weight_sums={str(k):v for k,v in weights.items()},
         training_overlap_audit=overlap,scientific_model_access='none; no scientific performance evaluated')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--dgp-root',required=True)
    p.add_argument('--archive-root');p.add_argument('--fixture-root');p.add_argument('--output',required=True);p.add_argument('--no-regenerate',action='store_true')
    a=p.parse_args();r=verify(a.bank,a.dgp_root,a.archive_root,a.fixture_root,not a.no_regenerate)
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('weight_sums','training_overlap_audit')}),flush=True)
