"""Direct-teacher evaluation: full-table native output, generic error calibration.

This module never labels native outputs EIF estimates. Duplicated compact
features are compatibility fields, not mediation features or independent views.
"""
from pathlib import Path
import argparse,json,math,time,os
import numpy as np
import torch
from teacher_model import load_teacher
from inference import generic_features
import posttrain as pt
from evaluate import sha,clean,EvaluationGuard,EmpiricalErrorCalibration,interval_stats,failed_stats,metrics
from build_cache import raw_group

ALPHAS=(.1,.05,.01)
ARMS=('native_wald','empirical_error','generic_error','generic_mean_corrected')
SAFE={'train','training','head','dev','validation','calibration','calibrate'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),indent=2),encoding='utf-8')
def resolve(p,manifest):
    p=Path(p)
    if p.is_absolute():return p
    candidates=(Path(manifest).parent/p,Path.cwd()/p)
    return next((q for q in candidates if q.exists()),candidates[0])
def sources():
    return {n:sha(Path(__file__).with_name(n)) for n in ('run_teacher_inference.py','teacher_model.py','inference.py','posttrain.py','evaluate.py','build_cache.py')}
def registration(args):
    r=read(args.registration)
    if r.get('frozen') is not True or r.get('protocol_sha256')!=sha(args.protocol):raise ValueError('Frozen teacher evaluation registration required')
    if r.get('source_hashes')!=sources():raise ValueError('Teacher evaluation source registration mismatch')
    return r
def gate(args,split,checkpoint,head=None):
    EvaluationGuard(args.protocol,args.authorization).check(split if split in SAFE else 'test')
    if split not in SAFE:
        a=read(args.authorization)
        if a.get('checkpoint_sha256')!=checkpoint:raise ValueError('Scalar teacher checkpoint test binding required')
        if a.get('teacher_evaluation_registration_sha256')!=sha(args.registration):raise ValueError('Teacher evaluation registration test binding required')
        if head is not None and a.get('teacher_head_artifacts_sha256')!=head:raise ValueError('Teacher error head test binding required')

@torch.inference_mode()
def predict_batch(model,rows,device):
    """Only observed rows enter predictor; one full-table encoding per table."""
    tensor=torch.as_tensor(np.asarray(rows),dtype=torch.float32,device=device)
    context=model.encoder.encode_support(tensor);pool=context['pooled'];scale=context['y_scale'].squeeze(-1)
    est=(model.effect_head(pool).squeeze(-1)*scale).double().cpu().numpy();lv=(model.variance_head(pool).squeeze(-1)+2*scale.log()).double().cpu().numpy();pooled=pool.double().cpu().numpy();results=[]
    for i,r in enumerate(rows):
        try:
            e,v=float(est[i]),float(lv[i]);se=math.exp(v/2)/math.sqrt(len(r))
            if not np.isfinite([e,v,se]).all() or se<=0:raise ValueError('Nonfinite/overflow native teacher output')
            x=generic_features(r,dict(estimate=e,se=se),pooled[i:i+1])
            if not np.isfinite(x).all():raise ValueError('Nonfinite generic feature')
            results.append((e,v,se,x))
        except Exception as exc:results.append(exc)
    return results
def predict(model,rows,device):
    result=predict_batch(model,np.asarray(rows)[None],device)[0]
    if isinstance(result,Exception):raise result
    return result

def cache(args):
    registration(args);manifest=read(args.raw_manifest);ph=sha(args.protocol);ck=sha(args.checkpoint)
    if manifest.get('protocol_sha256')!=ph:raise ValueError('Raw protocol drift')
    selected=[g for g in manifest['groups'] if int(g.get('seed_index',-1)) in (args.seed_index,-1) and (not args.split or g['split'] in args.split)]
    # Authorize before loading checkpoint or any target arrays.
    for g in selected:gate(args,g['split'],ck)
    if args.finalize_manifest:
        out=Path(args.output);groups=[]
        for i in range(args.task_count):
            part=read(out/f'manifest.part-{i}-of-{args.task_count}.json')
            if part['protocol_sha256']!=ph or part['checkpoint_sha256']!=ck or part['source_hashes']!=sources():raise ValueError('Teacher cache shard binding drift')
            groups.extend(part['groups'])
        names=[g['name'] for g in groups]
        if len(set(names))!=len(names) or set(names)!={g['name'] for g in selected}:raise ValueError('Missing/duplicated teacher cache shards')
        for g in groups:
            if sha(g['path'])!=g['sha256'] or sha(g['metadata_path'])!=g['metadata_sha256']:raise ValueError('Teacher cache shard hash drift')
        write(out/'manifest.json',dict(protocol_sha256=ph,checkpoint_sha256=ck,model='direct_teacher',source_hashes=sources(),groups=groups));return
    model=load_teacher(args.checkpoint,args.device);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);groups=[]
    for g in selected[args.task_index::args.task_count]:
        path=out/(g['name']+'.npz');meta=path.with_suffix('.json')
        binding=dict(protocol_sha256=ph,checkpoint_sha256=ck,registration_sha256=sha(args.registration),raw_manifest_sha256=sha(args.raw_manifest),group=g,source_hashes=sources())
        if path.exists() or meta.exists():
            saved=read(meta)
            if saved['binding']!=binding or sha(path)!=saved['sha256']:raise ValueError('Immutable teacher cache drift')
        else:
            raw,provenance=raw_group(g,args.raw_manifest,args.protocol,ph);n=len(raw['truth']);d=22+2*model.encoder.config.width
            estimates=np.full((n,2),np.nan);ses=estimates.copy();lv=np.full(n,np.nan);features=np.full((n,d),np.nan,np.float32);errors=[];seconds=[]
            for start in range(0,n,args.batch_size):
                stop=min(n,start+args.batch_size);rows=np.concatenate((raw['x'][start:stop],raw['a'][start:stop,:,None],raw['m'][start:stop,:,None],raw['y'][start:stop,:,None]),axis=2).astype(np.float64);t=time.perf_counter()
                try:results=predict_batch(model,rows,args.device)
                except Exception:
                    # Isolate a failed table without discarding its batch peers.
                    results=[]
                    for r in rows:
                        try:results.append(predict(model,r,args.device))
                        except Exception as exc:results.append(exc)
                elapsed=time.perf_counter()-t
                for i,result in zip(range(start,stop),results):
                    if isinstance(result,Exception):errors.append(repr(result))
                    else:
                        e,v,s,x=result;estimates[i]=e;ses[i]=s;lv[i]=v;features[i]=x;errors.append('')
                    seconds.append(elapsed/(stop-start))
            fields=dict(x_single=features,x_multi=features.copy(),estimates=estimates,ses=ses,native_logV=lv,truth=raw['truth'],mechanism_id=raw['mechanism_id'],seeds=raw['observation_seed'],fold_seeds=raw['fold_seed'],fold_ids=raw['fold_ids'],failed=np.asarray(errors),success=np.asarray([not e for e in errors]),seconds=np.asarray(seconds),protocol_sha256=np.array(ph),checkpoint_sha256=np.array(ck))
            for k in ('split','bank','family','p','n','seed_index'):fields[k]=np.repeat(g[k],n)
            np.savez_compressed(path,**fields);write(meta,dict(binding=binding,sha256=sha(path),raw=provenance,attempts=n,failed=sum(bool(e) for e in errors),feature_semantics='Full-table generic pooled mean and zero pooled SD; duplicated compatibility fields, no mediation features',score_kind='direct_native'))
        groups.append(dict(name=g['name'],path=str(path.resolve()),sha256=sha(path),metadata_path=str(meta.resolve()),metadata_sha256=sha(meta),**{k:g[k] for k in ('split','bank','seed_index','family','p','n','count')}))
    name='manifest.json' if args.task_count==1 else f'manifest.part-{args.task_index}-of-{args.task_count}.json'
    write(out/name,dict(protocol_sha256=ph,checkpoint_sha256=ck,model='direct_teacher',source_hashes=sources(),groups=groups))

def load_arrays(manifest,allowed):
    # Pre-test compact-merge revision. Old full-table cache producers remain
    # immutable; only this exact old self hash is compatible, with every other
    # dependency identical. Fold matrices remain in their hashed group files.
    h=read(manifest);parts=[];current=sources();legacy=dict(current,**{'run_teacher_inference.py':'c920a7736f5dda7de2386401ec67bc3a612f032b122683b79f3997203e1ac0bc'})
    if h.get('source_hashes') not in (current,legacy):raise ValueError('Unregistered teacher cache producer/dependency source')
    compact=('x_single','x_multi','estimates','ses','native_logV','truth','mechanism_id','seeds','fold_seeds','failed','success','seconds','split','bank','family','p','n','seed_index')
    width=None
    for g in h['groups']:
        if g['split'] not in allowed:raise ValueError('Disallowed split before cache read: '+g['split'])
        p=resolve(g['path'],manifest);metadata=resolve(g['metadata_path'],manifest)
        if sha(p)!=g['sha256'] or sha(metadata)!=g['metadata_sha256']:raise ValueError('Teacher cache hash mismatch')
        binding=read(metadata)['binding']
        if binding.get('source_hashes')!=h['source_hashes'] or any(binding.get(k)!=h[k] for k in ('protocol_sha256','checkpoint_sha256')):raise ValueError('Teacher group producer/scalar metadata mismatch')
        for k in ('name','split','bank','seed_index','p','n','count'):
            if binding['group'].get(k)!=g[k]:raise ValueError('Teacher group descriptor mismatch '+k)
        with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        for k in ('protocol_sha256','checkpoint_sha256'):
            if str(a[k].item())!=h[k]:raise ValueError('Cache scalar binding drift')
        if not np.all(a['split']==g['split']):raise ValueError('Split identity drift')
        count,n=int(g['count']),int(g['n']);folds=a['fold_ids'];features=a['x_single']
        if folds.shape!=(count,n) or not np.isin(folds,(0,1)).all() or np.any((folds==0).sum(1)==0) or np.any((folds==1).sum(1)==0):raise ValueError('Teacher per-group fold array shape/labels mismatch')
        if features.ndim!=2 or features.shape[0]!=count or features.shape[1] not in (150,278,534):raise ValueError('Teacher complete generic feature width mismatch')
        if width is None:width=features.shape[1]
        if features.shape[1]!=width or a['x_multi'].shape!=features.shape or not np.array_equal(a['x_multi'],features,equal_nan=True):raise ValueError('Teacher group feature width/duplicate generic schema mismatch')
        if any(k not in a for k in compact) or a['estimates'].shape!=(count,2) or a['ses'].shape!=(count,2):raise ValueError('Teacher compact schema mismatch')
        if any(a[k].shape!=(count,) for k in compact if k not in ('x_single','x_multi','estimates','ses')):raise ValueError('Teacher compact per-table identity shape mismatch')
        for k in ('bank','family','p','n','seed_index'):
            if not np.all(a[k]==g[k]):raise ValueError('Teacher compact group identity drift '+k)
        parts.append({k:a[k] for k in compact})
    if not parts:raise ValueError('Empty teacher cache bank')
    merged={k:np.concatenate([a[k] for a in parts]) for k in compact}
    for k in ('protocol_sha256','checkpoint_sha256'):merged[k]=np.array(h[k])
    return h,merged
def good(a):return (a['failed'].astype(str)=='')&a['success'].astype(bool)&np.isfinite(a['x_single']).all(1)&np.isfinite(a['estimates'][:,0])&np.isfinite(a['ses'][:,0])&(a['ses'][:,0]>0)&np.isfinite(a['truth'])
def identities(a):return dict(mechanism_id=sorted(set(a['mechanism_id'].astype(str))),seeds=sorted(set(map(int,a['seeds']))),fold_seeds=sorted(set(map(int,a['fold_seeds']))))
def disjoint(a,b):
    for k in ('mechanism_id','seeds','fold_seeds'):
        if set(a[k])&set(b[k]):raise ValueError('Overlapping teacher role identities '+k)
    if (set(a['seeds'])&set(b['fold_seeds'])) or (set(a['fold_seeds'])&set(b['seeds'])):raise ValueError('Cross-role observation/fold identity overlap')

def train_head(args):
    registration(args);h,a=load_arrays(args.cache_manifest,{'train','dev','validation'})
    if h['protocol_sha256']!=sha(args.protocol):raise ValueError('Head protocol mismatch')
    gate(args,'train',h['checkpoint_sha256']);out=Path(args.output)
    if (out/'artifacts.json').exists():raise FileExistsError('Teacher head immutable')
    train=(a['split']=='train')&(a['bank']=='head');dev=np.isin(a['split'],('dev','validation'));valid=good(a)
    if not (train&valid).any() or not (dev&valid).any():raise ValueError('Independent head train/dev required')
    disjoint(identities({k:v[train] for k,v in a.items() if np.ndim(v)>0}),identities({k:v[dev] for k,v in a.items() if np.ndim(v)>0}))
    torch.manual_seed(args.seed);x=a['x_single'];center=x[train&valid].mean(0);scale=np.maximum(x[train&valid].std(0),.01)
    xx=torch.tensor((x[valid]-center)/scale,dtype=torch.float32,device=args.device);z=torch.tensor(((a['estimates'][:,0]-a['truth'])/a['ses'][:,0])[valid],dtype=torch.float32,device=args.device)
    ti=np.flatnonzero(train[valid]);di=np.flatnonzero(dev[valid]);head=pt.ErrorDensity(x.shape[1],args.width,args.components).to(args.device);opt=torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=1e-4);best=float('inf');logs=[];state=None;exposures=0
    devgroups=pt.grouped_indices(a['mechanism_id'],dev,args.min_group,args.max_group);failed_groups=sum(not valid[ix].all() for ix in devgroups)
    epoch_steps=math.ceil(len(ti)/args.batch_size);steps=args.steps if args.steps is not None else args.head_epochs*epoch_steps;validate_every=args.validate_every or epoch_steps;order=np.array([],dtype=int)
    for step in range(steps+1):
        if step:
            if not len(order):order=ti[torch.randperm(len(ti),device=args.device).cpu().numpy()]
            idx=order[:args.batch_size];order=order[args.batch_size:];head.train();loss=pt.density_loss(head(xx[idx]),z[idx]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step();exposures+=len(idx)
        if step%validate_every==0 or step==steps:
            head.eval()
            with torch.no_grad():
                predictions=head(xx[di]);proper_loss=float(pt.density_loss(predictions,z[di]).cpu());qlo=pt.quantile(predictions,.025).cpu().numpy();qhi=pt.quantile(predictions,.975).cpu().numpy()
            lo=np.full(len(x),np.nan);hi=lo.copy();lo[dev&valid]=a['estimates'][dev&valid,0]-a['ses'][dev&valid,0]*qhi;hi[dev&valid]=a['estimates'][dev&valid,0]-a['ses'][dev&valid,0]*qlo
            risk=float(np.mean([pt.group_risk(lo[ix],hi[ix],a['truth'][ix],a['ses'][ix,0]) for ix in devgroups]))
            logs.append(dict(step=step,epoch=step/epoch_steps,exposures=exposures,dev_proper_loss=proper_loss,dev_group_risk=risk,failed_dev_groups=failed_groups,attempted_dev_groups=len(devgroups),attempted_train=int(train.sum()),valid_train=int((train&valid).sum())))
            if risk<best:best=risk;state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};best_step=step
    if state is None:
        write(out/'training_log.json',logs);write(out/'failed_fit.json',dict(failed=True,reason='No finite DEV group risk; failed groups retained',failed_dev_groups=failed_groups,source_hashes=sources()));raise ValueError('Teacher error head has no finite DEV group risk')
    out.mkdir(parents=True,exist_ok=True);np.savez(out/'normalization.npz',center=center,scale=scale);torch.save(dict(state_dict=state,dim=x.shape[1],width=args.width,components=args.components),out/'generic.pt');write(out/'training_log.json',logs)
    write(out/'artifacts.json',dict(complete=True,protocol_sha256=h['protocol_sha256'],checkpoint_sha256=h['checkpoint_sha256'],source_hashes=sources(),registration_sha256=sha(args.registration),cache_manifest_sha256=sha(args.cache_manifest),files={n:sha(out/n) for n in ('generic.pt','normalization.npz','training_log.json')},identities=identities(a),best_step=best_step,optimizer_steps=steps,training_exposures=exposures,args=vars(args),loss='NLL+.2 CRPS',selection='Same core DEV mean mechanism group_risk at alpha .05; epoch zero eligible',failed_train=int((train&~valid).sum()),failed_dev=int((dev&~valid).sum()),failed_dev_groups=failed_groups,feature_semantics='generic only; duplicate multi is not used'))

def load_head(folder,device):
    folder=Path(folder);art=read(folder/'artifacts.json')
    if art.get('complete') is not True or art['source_hashes']!=sources():raise ValueError('Teacher head source drift')
    for n,h in art['files'].items():
        if sha(folder/n)!=h:raise ValueError('Teacher head artifact drift')
    c=torch.load(folder/'generic.pt',map_location='cpu',weights_only=False);head=pt.ErrorDensity(c['dim'],c['width'],c['components']);head.load_state_dict(c['state_dict']);head.to(device).eval()
    with np.load(folder/'normalization.npz') as z:norm=(z['center'],z['scale'])
    return head,norm,art

@torch.inference_mode()
def calibrated_predictions(a,head,norm,cal,device):
    n=len(a['truth']);stats=np.tile(failed_stats(),(n,4,3,1));quantiles=np.full((n,4,3,2),np.nan);weights=np.full((n,head.components),np.nan);means=weights.copy();scales=weights.copy();valid=good(a);errors=np.full(n,'',dtype=object)
    for i in np.flatnonzero(valid):
        e,s=map(float,(a['estimates'][i,0],a['ses'][i,0]));summary=dict(estimate=e,se=s,clipping_rate=0.,failed=False)
        for j,alpha in enumerate(ALPHAS):
            z=__import__('statistics').NormalDist().inv_cdf(1-alpha/2)
            for k,q in enumerate(((-z,z),cal.quantiles(alpha))):quantiles[i,k,j]=q;stats[i,k,j]=interval_stats(summary,alpha,quantiles=q)
        try:
            p=head(torch.tensor((a['x_single'][i:i+1]-norm[0])/norm[1],dtype=torch.float32,device=device));w,m,sd=[v[0].double().cpu().numpy() for v in p];weights[i]=w;means[i]=m;scales[i]=sd;corrected=e-s*float(np.sum(w*m))
            for j,alpha in enumerate(ALPHAS):
                gq=tuple(float(pt.quantile(p,q)[0].cpu()) for q in (alpha/2,1-alpha/2))
                for k in (2,3):
                    q=gq
                    quantiles[i,k,j]=q;summary=dict(estimate=e,se=s,clipping_rate=0.,failed=False);stats[i,k,j]=interval_stats(summary,alpha,quantiles=q)
                    if k==3 and stats[i,k,j,5]==0:stats[i,k,j,0]=corrected
        except Exception as exc:
            errors[i]=repr(exc);stats[i,2:]=failed_stats();quantiles[i,2:]=np.nan;weights[i]=np.nan;means[i]=np.nan;scales[i]=np.nan
    return dict(stats=stats,quantiles=quantiles,cdf_weights=weights,cdf_means=means,cdf_scales=scales,head_errors=np.asarray(errors,dtype=str))

def apply(args):
    registration(args);header=read(args.cache_manifest);head,norm,art=load_head(args.head_dir,args.device);ah=sha(Path(args.head_dir)/'artifacts.json');ck=header['checkpoint_sha256']
    if art['protocol_sha256']!=sha(args.protocol) or art['checkpoint_sha256']!=ck:raise ValueError('Teacher head/model mismatch')
    for g in header['groups']:gate(args,g['split'],ck,ah)
    ch,cal=load_arrays(args.calibration_cache,{'calibration','calibrate'})
    if ch['protocol_sha256']!=sha(args.protocol) or ch['checkpoint_sha256']!=ck:raise ValueError('Teacher calibration mismatch')
    ci=identities(cal);disjoint(ci,art['identities']);valid=good(cal);records=[dict(split='calibration',mechanism_id=str(cal['mechanism_id'][i]),truth=float(cal['truth'][i]),summary=dict(estimate=float(cal['estimates'][i,0]),se=float(cal['ses'][i,0]),failed=not valid[i])) for i in range(len(valid))];calibrator=EmpiricalErrorCalibration().fit(records,art['identities']['mechanism_id'])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);groups=[]
    for g in header['groups'][args.task_index::args.task_count]:
        _,a=load_arrays_from_group(args.cache_manifest,header,g);ids=identities(a);disjoint(ids,ci);disjoint(ids,art['identities']);path=out/(g['name']+'.teacher.npz');meta=path.with_suffix('.json');binding=dict(protocol_sha256=sha(args.protocol),checkpoint_sha256=ck,head_artifacts_sha256=ah,cache_sha256=g['sha256'],calibration_manifest_sha256=sha(args.calibration_cache),source_hashes=sources())
        if path.exists() or meta.exists():
            old=read(meta)
            if old['binding']!=binding or old['sha256']!=sha(path):raise ValueError('Immutable teacher evaluation drift')
        else:
            result=calibrated_predictions(a,head,norm,calibrator,args.device);np.savez_compressed(path,**result,truth=a['truth'],mechanism_id=a['mechanism_id'],seeds=a['seeds'],fold_seeds=a['fold_seeds'],native_estimate=a['estimates'][:,0],native_logV=a['native_logV'],base_score_se=a['ses'][:,0],failed=a['failed'],success=a['success'],arms=np.asarray(ARMS),alphas=np.asarray(ALPHAS))
            write(meta,dict(binding=binding,sha256=sha(path),calibration_state=calibrator.state(),arms=ARMS,score_kind='direct_native',corrected_point_SE_limit='Native population coefficient SE retained; not established SE for corrected point',metrics={arm:{str(alpha):metrics(result['stats'][:,k,j],a['truth'],alpha) for j,alpha in enumerate(ALPHAS)} for k,arm in enumerate(ARMS)}))
        groups.append(dict(name=g['name'],path=str(path.resolve()),sha256=sha(path),metadata_path=str(meta.resolve()),metadata_sha256=sha(meta),split=g['split']))
    write(out/('manifest.json' if args.task_count==1 else f'manifest.part-{args.task_index}-of-{args.task_count}.json'),dict(protocol_sha256=sha(args.protocol),checkpoint_sha256=ck,head_artifacts_sha256=ah,groups=groups))

def load_arrays_from_group(manifest,header,g):
    p=resolve(g['path'],manifest)
    if sha(p)!=g['sha256'] or sha(resolve(g['metadata_path'],manifest))!=g['metadata_sha256']:raise ValueError('Target cache hash drift')
    with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    if any(str(a[k].item())!=header[k] for k in ('protocol_sha256','checkpoint_sha256')) or not np.all(a['split']==g['split']):raise ValueError('Target cache scalar/split drift')
    return header,a

def timing(args):
    registration(args);header=read(args.raw_manifest);ck=sha(args.checkpoint);model=load_teacher(args.checkpoint,args.device);head,norm,art=load_head(args.head_dir,args.device);records=[]
    if art['checkpoint_sha256']!=ck or art['protocol_sha256']!=sha(args.protocol):raise ValueError('Timing model/head mismatch')
    ch,ca=load_arrays(args.calibration_cache,{'calibration','calibrate'})
    if ch['checkpoint_sha256']!=ck or ch['protocol_sha256']!=sha(args.protocol):raise ValueError('Timing calibration mismatch')
    disjoint(identities(ca),art['identities']);cv=good(ca);cal=EmpiricalErrorCalibration().fit([dict(split='calibration',mechanism_id=str(ca['mechanism_id'][i]),truth=float(ca['truth'][i]),summary=dict(estimate=float(ca['estimates'][i,0]),se=float(ca['ses'][i,0]),failed=not cv[i])) for i in range(len(cv))],art['identities']['mechanism_id'])
    for g in header['groups']:
        if args.split and g['split'] not in args.split:continue
        gate(args,g['split'],ck,sha(Path(args.head_dir)/'artifacts.json'));raw,_=raw_group(g,args.raw_manifest,args.protocol,sha(args.protocol))
        rows=np.column_stack([raw[k][0] for k in ('x','a','m','y')]).astype(np.float64)
        for rep in range(args.warmups+args.repeats):
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            t=time.perf_counter();error='';native_seconds=np.nan;empirical_seconds=np.nan
            try:
                e,lv,se,x=predict(model,rows,args.device)
                if args.device.startswith('cuda'):torch.cuda.synchronize()
                native_seconds=time.perf_counter()-t;summary=dict(estimate=e,se=se,clipping_rate=0.,failed=False)
                for alpha in ALPHAS:interval_stats(summary,alpha)
                tt=time.perf_counter()
                for alpha in ALPHAS:interval_stats(summary,alpha,quantiles=cal.quantiles(alpha))
                empirical_seconds=time.perf_counter()-tt
                with torch.inference_mode():p=head(torch.tensor((x[None]-norm[0])/norm[1],dtype=torch.float32,device=args.device));q=[pt.quantile(p,v) for alpha in ALPHAS for v in (alpha/2,1-alpha/2)]
                for j,alpha in enumerate(ALPHAS):interval_stats(summary,alpha,quantiles=(float(q[2*j][0]),float(q[2*j+1][0])))
            except Exception as exc:error=repr(exc)
            if args.device.startswith('cuda'):torch.cuda.synchronize()
            elapsed=time.perf_counter()-t
            if rep>=args.warmups:records.append(dict(group=g['name'],repeat=rep-args.warmups,seconds=elapsed,native_forward_features_seconds=native_seconds,empirical_lookup_seconds=empirical_seconds,error=error))
    write(Path(args.output)/'timing.json',dict(checkpoint_sha256=ck,head_artifacts_sha256=sha(Path(args.head_dir)/'artifacts.json'),source_hashes=sources(),records=records,scope='Loaded model, fresh full-table encoding, native output, generic features, empirical lookup, generic head six quantiles and CI assembly; no saved prediction cache',device=args.device))

def parser():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--stage',choices=('cache','train-head','apply','timing'),required=True)
    for n in ('protocol','registration','checkpoint','raw-manifest','cache-manifest','calibration-cache','head-dir','authorization','output'):p.add_argument('--'+n,required=n in ('protocol','registration','output'))
    p.add_argument('--device',default='cpu');p.add_argument('--seed-index',type=int,default=0);p.add_argument('--task-index',type=int,default=0);p.add_argument('--task-count',type=int,default=1);p.add_argument('--finalize-manifest',action='store_true');p.add_argument('--split',nargs='+');p.add_argument('--seed',type=int,default=20261026);p.add_argument('--steps',type=int);p.add_argument('--head-epochs',type=int,default=40);p.add_argument('--min-group',type=int,default=16);p.add_argument('--max-group',type=int,default=32);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--width',type=int,default=128);p.add_argument('--components',type=int,default=5);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--validate-every',type=int);p.add_argument('--warmups',type=int,default=3);p.add_argument('--repeats',type=int,default=10);p.add_argument('--cpu-threads',type=int)
    return p
def main():
    args=parser().parse_args()
    if args.task_count<1 or not 0<=args.task_index<args.task_count or (args.steps is not None and args.steps<0) or (args.validate_every is not None and args.validate_every<1) or args.head_epochs<0 or args.batch_size<1:raise ValueError('Invalid budget/shard')
    threads=args.cpu_threads or int(os.environ.get('SLURM_CPUS_PER_TASK',os.environ.get('NSLOTS',torch.get_num_threads())));torch.set_num_threads(threads)
    {'cache':cache,'train-head':train_head,'apply':apply,'timing':timing}[args.stage](args)
if __name__=='__main__':main()


