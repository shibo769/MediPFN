"""Strict frozen-index result aggregation; readiness never reads predictions.

Three fitted seeds share the same observation tables. Paired MC uncertainty
averages seed-specific differences within each table BEFORE estimating MCSE.
"""
from pathlib import Path
import argparse,csv,json,math
import numpy as np
from evaluate import sha,clean,metrics,score_vectors,EvaluationGuard

ALPHAS=(.1,.05,.01)
KINDS=('core','heuristics','classical','teacher','legacy','bootstrap','diagnostics')
MODES=('latent_theta','FSP_oracle_score')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),indent=2,allow_nan=False),encoding='utf-8')
def source_hashes():
    root=Path(__file__).resolve().parents[1]
    return {n:sha(root/n) for n in ('code/summarize_results.py','audit/audit_results.py','audit/independent_audit.py','code/evaluate.py')}
def roster():
    result=[]
    for kind in ('core','heuristics','diagnostics'):
        result.extend(dict(id=kind+':s'+str(i),kind=kind,seed_index=i) for i in range(3))
    for kind in ('classical','bootstrap'):
        result.extend(dict(id=kind+':'+b,kind=kind,baseline=b,seed_index=-1) for b in ('linear','gbm','rf','oracle'))
    result.extend(dict(id='teacher:'+m+':s'+str(i),kind='teacher',mode=m,seed_index=i) for m in MODES for i in range(3))
    result.append(dict(id='legacy',kind='legacy',seed_index=-1));return result
def methods(bundle):
    k=bundle['kind']
    if k=='core':
        from apply_posttrain import ARMS
        return tuple(ARMS)
    if k=='heuristics':
        from posttrain_heuristics import APPLICATION_ARMS
        return tuple(APPLICATION_ARMS)
    if k=='teacher':return ('native_wald','empirical_error','generic_error','generic_mean_corrected')
    if k=='classical':return (bundle['baseline'],)
    if k=='legacy':return ('old_v1','native_fsp')
    if k=='bootstrap':return (bundle['baseline']+'_score_bootstrap',bundle['baseline']+'_refit_bootstrap')
    if k=='diagnostics':return ('pfn_multiview','oracle_g','oracle_q','oracle_outcome','oracle_all')
    raise ValueError('Unknown bundle kind')
def selected_cells(protocol,bundle):
    tests=[c for c in protocol['cells'] if c['split']=='test'];kind=bundle['kind']
    if kind=='diagnostics':return [c for c in tests if c['bank']=='fixed_mechanism']
    if kind=='legacy':return [c for c in tests if c['bank'] in ('synthetic','semisynthetic') and int(c['p'])==3]
    if kind=='bootstrap':return [c for c in tests if c['bank']=='synthetic' and c['population_kind']=='synthetic' and c['family'] in ('linear','weak_overlap','combinations') and int(c['p']) in (3,10,30) and int(c['n']) in (600,1000)]
    return tests
def expected_count(protocol,cell,bundle):
    full=int(cell['mechanisms'])*int(cell['repeats'])
    if bundle['kind']=='bootstrap':return min(full,int(protocol.get('bootstrap',{}).get('tables_per_selected_cell',30)))
    return full
def verify_raw_descriptor(cell,group,protocol_hash):
    if any(group.get(k)!=v for k,v in cell.items()) or group.get('protocol_sha256')!=protocol_hash or int(group['count'])!=int(cell['mechanisms'])*int(cell['repeats']):raise ValueError('Raw manifest differs from registered cell '+cell['name'])
def resolve(root,path):
    p=Path(path);return p if p.is_absolute() else root/p
def paths(root,bundle,cell):
    substitutions=dict(bundle,name=cell['name']);out={}
    for key,value in bundle.items():
        if key.endswith('_template'):out[key[:-9]]=resolve(root,value.format(**substitutions))
    for required in ('prediction','ledger','audit'):
        if required not in out:raise ValueError('Missing '+required+'_template for '+bundle['id'])
    return out
def validate_index(root,protocol_path,index_path):
    p=read(protocol_path);i=read(index_path)
    if not i.get('frozen') or i.get('protocol_sha256')!=sha(protocol_path) or i.get('source_hashes')!=source_hashes():raise ValueError('Frozen result index/source binding required')
    expected={r['id']:r for r in roster()};actual={b['id']:b for b in i['bundles']}
    if len(actual)!=len(i['bundles']) or set(expected)!=set(actual):raise ValueError('Final index requires the complete exact bundle roster')
    for key,definition in expected.items():
        if any(actual[key].get(k)!=v for k,v in definition.items()):raise ValueError('Bundle descriptor mismatch '+key)
    selection=i['primary_selection'];selection_path=resolve(root,selection['path'])
    if selection.get('split')!='dev' or not selection.get('method') or sha(selection_path)!=selection['sha256']:raise ValueError('Original DEV-locked primary selection required')
    dev_lock=read(selection_path)
    if dev_lock.get('frozen',dev_lock.get('complete')) is not True or dev_lock.get('protocol_sha256')!=sha(protocol_path) or dev_lock.get('primary_family')!=selection['method']:raise ValueError('Result primary must exactly equal original DEV-locked primary_family')
    allowed={method_id(b,m) for b in i['bundles'] for m in methods(b)}
    if selection['method'] not in allowed:raise ValueError('DEV primary method is outside the fixed method roster')
    if any(c.get(side) not in allowed for c in i.get('comparisons',[]) for side in ('left','right')):raise ValueError('Comparison names a method outside the fixed roster')
    return p,i
def readiness(root,protocol_path,index_path):
    p,i=validate_index(root,protocol_path,index_path);missing=[];expected=0
    for b in i['bundles']:
        for cell in selected_cells(p,b):
            pp=paths(root,b,cell);expected+=1
            for role,path in pp.items():
                if not path.exists():missing.append(dict(bundle=b['id'],cell=cell['name'],role=role,path=str(path)))
    return dict(ready=not missing,final=False,test_metrics_read=False,readiness_scope='Path presence and frozen index/source/DEV lock only; scientific certificate validation still required for final.',expected_cell_bundles=expected,missing=missing,missing_count=len(missing),protocol_sha256=sha(protocol_path),result_index_sha256=sha(index_path))
def summary_gate(protocol_path,index_path,authorization,summary_authorization):
    EvaluationGuard(protocol_path,authorization).check('test');g=read(summary_authorization)
    if g.get('final_summary_authorized') is not True or g.get('protocol_sha256')!=sha(protocol_path) or g.get('result_index_sha256')!=sha(index_path) or g.get('source_hashes')!=source_hashes():raise ValueError('Explicit globally complete final-summary authorization required')
    if read(authorization).get('dev_selection_sha256')!=read(index_path)['primary_selection']['sha256']:raise ValueError('Global release and frozen original DEV choice mismatch')
def verify_certificate(certificate,protocol_path,authorization,prediction,ledger,raw,expected):
    c=read(certificate)
    for key,value in (('passed',True),('protocol_sha256',sha(protocol_path)),('authorization_sha256',sha(authorization)),('predictions_sha256',sha(prediction)),('ledger_sha256',sha(ledger)),('raw_sha256',sha(raw)),('attempted_tables',expected)):
        if c.get(key)!=value:raise ValueError('Independent audit certificate mismatch '+key+' '+str(certificate))
    if c.get('source_hashes')!=source_hashes():raise ValueError('Audit orchestration source binding drift')
    return c
def read_stats(path,bundle):
    with np.load(path,allow_pickle=False) as z:
        if bundle['kind'] in ('classical','legacy','bootstrap'):return {m:np.stack([z[m+'__stats__'+str(float(a))] for a in ALPHAS],1) for m in methods(bundle)}
        key='arms' if bundle['kind']=='teacher' else 'arm_names';np.testing.assert_array_equal(z[key],np.asarray(methods(bundle)));np.testing.assert_array_equal(z['alphas'],np.asarray(ALPHAS))
        st=z['stats'];return {m:st[:,j] for j,m in enumerate(methods(bundle))}
def method_id(bundle,name):
    prefix=bundle['kind']
    if prefix=='teacher':prefix+=':'+bundle['mode']
    return prefix+':'+name
def aggregate_metrics(seed_stats,truth,alpha):
    """Repeated fitted seeds, shared N tables; no 3N independence interval."""
    st=np.asarray(seed_stats);s,n=st.shape[:2];flat=st.reshape(s*n,6);m=metrics(flat,np.tile(truth,s),alpha)
    for key in ('coverage_low','coverage_high'):m.pop(key,None)
    values=np.stack([score_vectors(rows,truth,alpha)['coverage'] for rows in st]);per_table=values.mean(0)
    m.update(attempted_observation_tables=n,fitted_model_seeds=s,evaluation_attempts=s*n,point_valid_attempts=m['valid_attempts'],coverage_mc_se=float(np.std(per_table,ddof=1)/math.sqrt(n)) if n>1 else None,coverage_by_model_seed=values.mean(1).tolist(),dependence_note='Model-seed replicates share tables; MCSE computed after within-table averaging. Training-seed variation descriptive only. Bias/RMSE/mean-SE condition on valid attempts; coverage and loss retain all attempts.')
    return m
def paired_metrics(left,right,truth,alpha):
    l,r=np.asarray(left),np.asarray(right)
    if l.shape[1:]!=r.shape[1:]:raise ValueError('Paired common-table shape mismatch')
    s=max(len(l),len(r))
    if len(l) not in (1,s) or len(r) not in (1,s):raise ValueError('Paired seed roster mismatch')
    if len(l)==1:l=np.repeat(l,s,axis=0)
    if len(r)==1:r=np.repeat(r,s,axis=0)
    result={}
    for key in ('coverage','lower_miss','upper_miss','length','interval_score'):
        ll=np.stack([score_vectors(v,truth,alpha)[key] for v in l]);rr=np.stack([score_vectors(v,truth,alpha)[key] for v in r])
        if not np.isfinite(ll).all() or not np.isfinite(rr).all():result[key]=dict(available=False,reason='Infinite attempted-table loss; paired difference/MCSE undefined');continue
        d=(ll-rr).mean(0);se=float(np.std(d,ddof=1)/math.sqrt(len(d))) if len(d)>1 else None
        result[key]=dict(available=True,difference=float(d.mean()),mc_se=se,mc95_low=float(d.mean()-1.96*se) if se is not None else None,mc95_high=float(d.mean()+1.96*se) if se is not None else None,common_tables=len(d),model_seeds=s)
    return result
def strata(cell):
    result=[('cell',cell['name'])]
    if cell['bank']=='fixed_mechanism':return result
    result += [('all_main','all'),('bank',cell['bank']),('family',cell['family']),('p',str(cell['p'])),('n',str(cell['n'])),('population_kind',cell['population_kind'])]
    if cell['bank']=='synthetic':result.append(('mechanism_exposure','fresh_blind4' if cell['family'] in ('tree','rff','mlp','combinations') else 'design_seen5'))
    return result
def run(args):
    root=Path(args.root);protocol_path=Path(args.protocol);index_path=Path(args.index);p,index=validate_index(root,protocol_path,index_path)
    if args.readiness:
        result=readiness(root,protocol_path,index_path);write(args.output,result);return result
    summary_gate(protocol_path,index_path,args.authorization,args.summary_authorization)
    ready=readiness(root,protocol_path,index_path)
    if not ready['ready']:raise ValueError('Incomplete result suite; use --readiness for missing paths')
    raw_manifest=read(args.raw_manifest)
    if raw_manifest.get('protocol_sha256')!=sha(protocol_path):raise ValueError('Raw manifest protocol drift')
    raw_lookup={g['name']:g for g in raw_manifest['groups']};test_cells=[c for c in p['cells'] if c['split']=='test']
    if set(raw_lookup)!={c['name'] for c in test_cells}:raise ValueError('Final raw manifest must contain exactly all registered test cells')
    collected={};law_reports=[];bindings=[];oracle_diagnostics=[]
    for b in index['bundles']:
        for cell in selected_cells(p,b):
            pp=paths(root,b,cell);g=raw_lookup[cell['name']];verify_raw_descriptor(cell,g,sha(protocol_path));raw=resolve(root,g['raw_path']);expected=expected_count(p,cell,b)
            if sha(raw)!=g['raw_sha256']:raise ValueError('Raw result cohort drift')
            with np.load(raw,allow_pickle=False) as z:truth=z['truth'][:expected];ids=z['mechanism_id'][:expected];obs=z['observation_seed'][:expected];fold=z['fold_seed'][:expected]
            cert=verify_certificate(pp['audit'],protocol_path,args.authorization,pp['prediction'],pp['ledger'],raw,expected);stats=read_stats(pp['prediction'],b)
            if b['kind']=='diagnostics':
                with np.load(pp['prediction'],allow_pickle=False) as z:
                    mse=z['nuisance_mse'];counts=np.isfinite(mse).sum(0);means=[float(mse[np.isfinite(mse[:,j]),j].mean()) if counts[j] else None for j in range(3)]
                    oracle_diagnostics.append(dict(cell=cell['name'],model_seed=b['seed_index'],attempts=expected,nuisance_mse_columns=['g','q0_q1','mu10_mu11'],valid_mse_attempts=counts.tolist(),mean_nuisance_mse=means,base_failed_attempts=int(z['base_failed'].sum()),privileged_oracle_all_recoveries=int(z['oracle_all_recovery'].sum()),maximum_truth_regeneration_delta=float(np.max(np.abs(z['truth_regeneration_delta'])))))
            for name,values in stats.items():
                if values.shape!=(expected,3,6):raise ValueError('Exact method/table/alpha shape mismatch '+name)
                mid=method_id(b,name);seed=int(b['seed_index']);key=(mid,cell['name']);entry=collected.setdefault(key,dict(cell=cell,truth=truth,ids=ids,obs=obs,fold=fold,seeds={}))
                for k,v in (('truth',truth),('ids',ids),('obs',obs),('fold',fold)):np.testing.assert_array_equal(entry[k],v)
                if seed in entry['seeds']:raise ValueError('Duplicated model-seed result '+str(key))
                entry['seeds'][seed]=values
                if cell['bank']=='fixed_mechanism':
                    for law in np.unique(ids):
                        ix=ids==law
                        for j,alpha in enumerate(ALPHAS):law_reports.append(dict(method=mid,model_seed=seed,cell=cell['name'],mechanism_id=str(law),**metrics(values[ix,j],truth[ix],alpha,fixed_mechanism=True)))
            bindings.append(dict(bundle=b['id'],cell=cell['name'],prediction_sha256=sha(pp['prediction']),audit_sha256=sha(pp['audit']),raw_sha256=g['raw_sha256']))
    pooled={};cell_arrays={}
    for (mid,name),entry in collected.items():
        seeds=sorted(entry['seeds']);kind=mid.split(':')[0];expected_seeds=[0,1,2] if kind in ('core','heuristics','teacher','diagnostics') else [-1]
        if seeds!=expected_seeds:raise ValueError('Incomplete fitted-seed roster '+mid+' '+name)
        arr=np.stack([entry['seeds'][s] for s in seeds]);cell_arrays[(mid,name)]=(arr,entry['truth'],entry['cell'])
        if entry['cell']['bank']!='fixed_mechanism':
            for dimension,value in strata(entry['cell']):pooled.setdefault((mid,dimension,value),[]).append((name,arr,entry['truth']))
    rows=[]
    for (mid,dimension,value),parts in sorted(pooled.items()):
        parts.sort(key=lambda x:x[0]);arr=np.concatenate([x[1] for x in parts],axis=1);truth=np.concatenate([x[2] for x in parts])
        for j,alpha in enumerate(ALPHAS):rows.append(dict(method=mid,stratum=dimension,value=value,**aggregate_metrics(arr[:,:,j],truth,alpha)))
    contrasts=[]
    for comparison in index.get('comparisons',[]):
        left,right=comparison['left'],comparison['right'];common=sorted(n for n in set(n for m,n in cell_arrays if m==left)&set(n for m,n in cell_arrays if m==right) if cell_arrays[(left,n)][2]['bank']!='fixed_mechanism')
        if not common:raise ValueError('No matched comparison cohort')
        parts={}
        for name in common:
            l,truth,cell=cell_arrays[(left,name)];r,rt,_=cell_arrays[(right,name)];count=min(len(truth),len(rt));np.testing.assert_array_equal(truth[:count],rt[:count]);l=l[:,:count];r=r[:,:count];truth=truth[:count]
            for dimension,value in strata(cell):parts.setdefault((dimension,value),[]).append((l,r,truth))
        for (dimension,value),pieces in parts.items():
            ll=np.concatenate([x[0] for x in pieces],1);rr=np.concatenate([x[1] for x in pieces],1);tt=np.concatenate([x[2] for x in pieces])
            for j,alpha in enumerate(ALPHAS):contrasts.append(dict(left=left,right=right,stratum=dimension,value=value,alpha=alpha,metrics=paired_metrics(ll[:,:,j],rr[:,:,j],tt,alpha)))
    result=dict(complete=True,protocol_sha256=sha(protocol_path),result_index_sha256=sha(index_path),authorization_sha256=sha(args.authorization),summary_authorization_sha256=sha(args.summary_authorization),source_hashes=source_hashes(),primary_selection=index['primary_selection'],registered_test_tables=sum(int(c['mechanisms'])*int(c['repeats']) for c in test_cells),rows=rows,paired_contrasts=contrasts,fixed_law_diagnostics=law_reports,oracle_diagnostics=oracle_diagnostics,bindings=bindings,candidate_claim='Candidate methods remain candidates; no manual test-based winner is selected.',uncertainty='Paired MCSE averages fitted-model-seed differences within common observation table first; fixed-law repeats reported separately.')
    write(args.output,result);csvpath=Path(args.output).with_suffix('.csv');flatkeys=('method','stratum','value','alpha','attempted_observation_tables','fitted_model_seeds','evaluation_attempts','coverage','coverage_mc_se','mean_length','interval_score','bias','rmse','mean_se','failures','failure_rate','lower_miss','upper_miss','length_p99','length_max','length_top_one_percent_share','IS_p99','IS_max','IS_top_one_percent_share')
    csvrows=[]
    for r in rows:
        rr=dict(r)
        for prefix,key in (('length','length_tails'),('IS','IS_tails')):
            rr[prefix+'_p99']=r[key]['p99'];rr[prefix+'_max']=r[key]['maximum'];rr[prefix+'_top_one_percent_share']=r[key]['top_one_percent_contribution']
        csvrows.append(rr)
    with csvpath.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=flatkeys);writer.writeheader();writer.writerows({k:clean(r.get(k)) for k in flatkeys} for r in csvrows)
    lawkeys=('method','model_seed','cell','mechanism_id','alpha','attempts','valid_attempts','failures','coverage','mean_length','interval_score','bias','rmse','mean_se','empirical_sd','mean_se_over_empirical_sd','rms_se_over_empirical_sd')
    with Path(args.output).with_suffix('.fixed_law.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=lawkeys);writer.writeheader();writer.writerows({k:clean(r.get(k)) for k in lawkeys} for r in law_reports)
    return result
def main():
    a=argparse.ArgumentParser(__doc__)
    for k in ('root','protocol','index','output'):a.add_argument('--'+k,required=True)
    a.add_argument('--raw-manifest');a.add_argument('--authorization');a.add_argument('--summary-authorization');a.add_argument('--readiness',action='store_true');args=a.parse_args();result=run(args);print(json.dumps(dict(complete=result.get('complete',False),missing_count=result.get('missing_count',0))))
if __name__=='__main__':main()
