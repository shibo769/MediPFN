"""Candidate post-training objectives on frozen v9 feature caches.

Core thirteen methods are untouched. Truth is used only by offline objectives;
the exported heads and policies take observable original-table features only.
"""
from __future__ import annotations
import argparse,copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import posttrain as pt
from inference import metric_arrays

BIAS_ARMS=tuple(prefix+'_'+mode for prefix in ('anchor','moment') for mode in ('bias','scale','both'))
CONSISTENCY_GRID=(0.,.1,1.)
EXPANDED_ARMS=('expanded_sft','expanded_dpo','expanded_extra_sft_dpo','expanded_rlvr_uniform','expanded_rlvr_replay','expanded_repair_uniform','expanded_repair_replay','expanded_extra_sft_uniform','expanded_extra_sft_replay')
HEAD_ARMS=BIAS_ARMS+tuple('consistency_'+str(v) for v in CONSISTENCY_GRID)
CONTROL_ARMS=('generic_error_mean_corrected','mediation_error_mean_corrected')
APPLICATION_ARMS=CONTROL_ARMS+HEAD_ARMS+EXPANDED_ARMS

def valid_feature_rows(a):
    valid=np.isfinite(a['x_multi']).all(1)&np.isfinite(a['estimates']).all(1)&np.isfinite(a['ses']).all(1)&(a['ses']>0).all(1)
    if 'failed' in a:
        values=np.asarray(a['failed']);failed=values!='' if values.dtype.kind in 'US' else values.astype(bool)
        if failed.ndim>1:failed=failed.any(tuple(range(1,failed.ndim)))
        valid=valid&~failed
    if 'success' in a:
        values=np.asarray(a['success'],bool)
        if values.ndim>1:values=values.all(tuple(range(1,values.ndim)))
        valid=valid&values
    return valid

def cross_half_targets(estimates,se,truth,mechanism_id,seeds,mask,min_tables=16):
    est,s,y=np.asarray(estimates,float),np.asarray(se,float),np.asarray(truth,float);mid=np.asarray(mechanism_id).astype(str);seeds=np.asarray(seeds);mask=np.asarray(mask,bool)
    if est.ndim!=1 or s.shape!=est.shape or y.shape!=est.shape:raise ValueError('One point/SE/truth per table')
    mean=np.full(len(est),np.nan);log_sd=mean.copy();eligible=np.zeros(len(est),bool);pilots=[[] for _ in est];groups=[];failures=[]
    for key in np.unique(mid[mask]):
        ids=np.flatnonzero(mask&(mid==key));ids=ids[np.argsort(seeds[ids],kind='stable')]
        if len(ids)<min_tables:raise ValueError('Same-law pilot group below minimum')
        if np.ptp(y[ids])>1e-12:raise ValueError('One mechanism must have one NIE truth')
        halves=(ids[::2],ids[1::2]);groups.append(ids.tolist())
        for targets,pilot in (halves,halves[::-1]):
            valid=np.isfinite(est[pilot])&np.isfinite(s[pilot])&(s[pilot]>0)&np.isfinite(y[pilot])
            if not valid.all() or len(pilot)<2:
                failures.append(dict(mechanism=key,targets=targets.tolist(),pilot=pilot.tolist(),reason='Invalid independent pilot; targets retained but no moment supervision'));continue
            z=(est[pilot]-y[pilot])/s[pilot];mu=float(z.mean());ls=float(np.log(max(z.std(ddof=1),.1)))
            for i in targets:
                if i in pilot:raise AssertionError('Own error leaked into pilot')
                mean[i]=mu;log_sd[i]=ls;eligible[i]=True;pilots[i]=pilot.tolist()
    return dict(mean=mean,log_sd=log_sd,eligible=eligible,pilot_indices=pilots,groups=groups,failures=failures)

def implied_theta_moments(distribution,estimate,se):
    w,mu,sd=distribution;estimate=estimate.detach();se=se.detach()
    mz=(w*mu).sum(-1);variance=(w*(sd.square()+mu.square())).sum(-1)-mz.square();variance=variance.clamp_min(1e-8)
    return estimate-se*mz,torch.log(se.clamp_min(1e-12))+.5*torch.log(variance)

def equivalence_consistency(original,paired,estimate_original,se_original,estimate_paired,se_paired):
    m1,l1=implied_theta_moments(original,estimate_original,se_original);m2,l2=implied_theta_moments(paired,estimate_paired,se_paired)
    scale=((se_original.detach()+se_paired.detach())/2).clamp_min(1e-8)
    return (((m1-m2)/scale).square()+(l1-l2).square()).mean()

def repair_rewards(actions,intervals,truth,base_se,baseline_ci,cost=None):
    actions=np.asarray(actions,int);ci=np.asarray(intervals,float);truth=np.asarray(truth,float);se=np.asarray(base_se,float);baseline=np.asarray(baseline_ci,float)
    base=metric_arrays(baseline[:,0],baseline[:,1],truth);rewards=[];details=[];idx=np.arange(len(truth))
    for action in actions:
        chosen=ci[idx,action];now=metric_arrays(chosen[:,0],chosen[:,1],truth)
        valid=base['valid'].all() and now['valid'].all() and np.isfinite(se).all() and (se>0).all()
        if not valid:rewards.append(-np.inf);details.append(dict(failed=True,attempts=len(truth)));continue
        relative_is=float(np.mean((base['IS']-now['IS'])/(base['IS']+2*se)))
        rescued=float(np.mean((~base['covered'])&now['covered']));lost=float(np.mean(base['covered']&(~now['covered'])))
        extra=float(np.mean(np.maximum(now['length']-base['length'],0)/np.maximum(base['length'],2*se)))
        tail=float(max(0,now['lower'].mean()-base['lower'].mean())+max(0,now['upper'].mean()-base['upper'].mean()))
        actioncost=0. if cost is None else float(np.asarray(cost)[idx,action].mean())
        reward=relative_is+.5*rescued-2*lost-.1*extra-.5*tail-.01*actioncost
        rewards.append(reward);details.append(dict(failed=False,attempts=len(truth),relative_is=relative_is,rescued=rescued,lost=lost,extra_width=extra,tail_regression=tail,cost=actioncost))
    return np.asarray(rewards,float),details

class MomentHead(nn.Module):
    def __init__(self,dim,width=128,mode='both'):
        super().__init__();self.dim=dim;self.width=width;self.mode=mode;self.components=1
        self.net=nn.Sequential(nn.Linear(dim,width),nn.GELU(),nn.LayerNorm(width),nn.Linear(width,2));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        raw=self.net(x);mu=raw[:,0] if self.mode!='scale' else raw[:,0]*0;logscale=3*torch.tanh(raw[:,1]/3) if self.mode!='bias' else raw[:,1]*0
        return torch.ones_like(mu[:,None]),mu[:,None],logscale.exp()[:,None]

class ExpandedPolicy(nn.Module):
    def __init__(self,dim,experts=6,width=128):
        super().__init__();self.dim=dim;self.width=width;self.experts=experts;self.net=nn.Sequential(nn.Linear(dim,width),nn.GELU(),nn.LayerNorm(width),nn.Linear(width,experts));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)

def interval_from_density(distribution,estimate,se,alpha=.05):
    qlo=pt.quantile(distribution,alpha/2);qhi=pt.quantile(distribution,1-alpha/2)
    return torch.stack((estimate.detach()-se.detach()*qhi,estimate.detach()-se.detach()*qlo),-1)

@torch.no_grad()
def predict_error_head(head,norm,x_multi,estimate,se,alpha=.05,device='cpu'):
    """Original-table-only deployable error head, including replayable density."""
    values=np.asarray(x_multi);est=np.asarray(estimate,float);scale=np.asarray(se,float);valid=np.isfinite(values).all(1)&np.isfinite(est)&np.isfinite(scale)&(scale>0)
    ci=np.full((len(est),2),np.nan);quantiles=ci.copy();parameters={name:np.full((len(est),head.components),np.nan) for name in ('weights','means','scales')}
    if valid.any():
        x=pt.normalized(dict(x_multi=values),'x_multi',norm,device)[valid];p=head(x);qlo=pt.quantile(p,alpha/2).cpu().numpy();qhi=pt.quantile(p,1-alpha/2).cpu().numpy();quantiles[valid]=np.column_stack((qlo,qhi));ci[valid]=np.column_stack((est[valid]-scale[valid]*qhi,est[valid]-scale[valid]*qlo))
        for key,tensor in zip(parameters,p):parameters[key][valid]=tensor.cpu().numpy()
    w,m,s=(parameters[key] for key in ('weights','means','scales'));ez=np.sum(w*m,1);var=np.sum(w*(s*s+m*m),1)-ez*ez
    return dict(intervals=ci,quantiles=quantiles,mixture_evidence=parameters,valid=valid,base_estimate=est,base_score_se=scale,predicted_bias=scale*ez,estimate=est-scale*ez,implied_interval_scale=scale*np.sqrt(np.maximum(var,0)))

def load_expanded_policy(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=False);model=ExpandedPolicy(ck['dim'],ck['experts'],ck['width']);model.load_state_dict(ck['state_dict']);return model.to(device).eval()

@torch.no_grad()
def predict_expanded(policy,core_heads,additional_heads,norm,x_single,x_multi,estimates,ses,alpha=.05,device='cpu'):
    """No paired table, truth, law identifier or teacher enters deployment."""
    core=pt.expert_intervals(core_heads,norm,x_single,x_multi,estimates,ses,alpha,device);ci=[core['intervals']];qq=[core['quantiles']];evidence=dict(core['mixture_evidence'])
    for j,head in enumerate(additional_heads):
        result=predict_error_head(head,norm,x_multi,np.asarray(estimates)[:,1],np.asarray(ses)[:,1],alpha,device);ci.append(result['intervals'][:,None]);qq.append(result['quantiles'][:,None]);evidence['added_'+str(j)]=result['mixture_evidence']
    pool=np.concatenate(ci,1);qpool=np.concatenate(qq,1);x=pt.normalized(dict(x_multi=np.asarray(x_multi)),'x_multi',norm,device);prob=policy(x).softmax(-1).cpu().numpy();action=prob.argmax(-1);indices=np.arange(len(action));pointmap=np.array([0,1,0,1]+[1]*len(additional_heads));base=np.asarray(estimates)[indices,pointmap[action]];base_se=np.asarray(ses)[indices,pointmap[action]]
    meanz=np.zeros((len(action),len(pointmap)));sdz=np.ones_like(meanz)
    for column,key in [(2,'generic_error'),(3,'mediation_error')]+[(4+j,'added_'+str(j)) for j in range(len(additional_heads))]:
        d={k:np.asarray(v,float) for k,v in evidence[key].items()};mean=np.sum(d['weights']*d['means'],1);variance=np.sum(d['weights']*(d['scales']**2+d['means']**2),1)-mean**2;meanz[:,column]=mean;sdz[:,column]=np.sqrt(np.maximum(variance,0))
    predicted_bias=base_se*meanz[indices,action]
    return dict(intervals=pool[indices,action],quantiles=qpool[indices,action],actions=action,probabilities=prob,point_index=pointmap[action],raw_expert_intervals=pool,raw_expert_quantiles=qpool,mixture_evidence=evidence,base_estimate=base,base_score_se=base_se,predicted_bias=predicted_bias,estimate=base-predicted_bias,implied_interval_scale=base_se*sdz[indices,action])
def normal_interval_score(distribution,z):
    _,mu,sd=distribution;lo=mu[:,0]-1.959963984540054*sd[:,0];hi=mu[:,0]+1.959963984540054*sd[:,0]
    return (hi-lo+40*(lo-z).clamp_min(0)+40*(z-hi).clamp_min(0)).mean()
def group_validation(ci,a,groups):
    return float(np.mean([pt.group_risk(ci[g,0],ci[g,1],a['truth'][g],a['ses'][g,0]) for g in groups]))
def save_head(path,head,**extra):torch.save(dict(state_dict=head.state_dict(),dim=head.dim,width=head.width,kind='moment' if isinstance(head,MomentHead) else 'mixture',mode=getattr(head,'mode',None),components=head.components,**extra),path)
def load_head(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=False);head=MomentHead(ck['dim'],ck['width'],ck['mode']) if ck['kind']=='moment' else pt.ErrorDensity(ck['dim'],ck['width'],ck['components']);head.load_state_dict(ck['state_dict']);return head.to(device).eval()
def _raw_stats(a):return dict(estimate=torch.as_tensor(a['estimates'][:,1],dtype=torch.float32),se=torch.as_tensor(a['ses'][:,1],dtype=torch.float32))

def aligned_pair(original,path):
    with np.load(path,allow_pickle=False) as z:
        if 'split' not in z:raise ValueError('Paired training cache requires explicit split identity')
        splits=z['split'].astype(str)
        if not set(splits).issubset({'train','dev'}):raise ValueError('Paired final-test or other labels forbidden in training')
        for key in ('protocol_sha256','checkpoint_sha256'):
            if key not in original or key not in z:raise ValueError('Paired production cache requires '+key)
            if str(np.asarray(original[key]).item())!=str(z[key].item()):raise ValueError('Paired backbone/protocol identity mismatch '+key)
        paired={k:z[k] for k in z.files}
    for key in ('x_multi','estimates','ses','mechanism_id','seeds','truth'): 
        if key not in paired:raise ValueError('Paired cache missing '+key)
    lookup={}
    for i,(mid,seed) in enumerate(zip(paired['mechanism_id'],paired['seeds'])):
        key=(str(mid),int(seed))
        if key in lookup:raise ValueError('Paired transforms require one-to-one row alignment')
        lookup[key]=i
    order=[]
    for mid,seed in zip(original['mechanism_id'],original['seeds']):
        key=(str(mid),int(seed))
        if key not in lookup:raise ValueError('Missing paired row '+str(key))
        order.append(lookup[key])
    result={k:v[order] if np.ndim(v)>0 and len(v)==len(paired['truth']) else v for k,v in paired.items()}
    np.testing.assert_allclose(result['truth'],original['truth'],rtol=0,atol=1e-12)
    for key in ('split','bank'):
        if key not in result or key not in original:raise ValueError('Paired role identity missing '+key)
        np.testing.assert_array_equal(result[key].astype(str),np.asarray(original[key]).astype(str))
    for aliases in (('fold_seeds','fold_seed'),):
        left=next((original[k] for k in aliases if k in original),None);right=next((result[k] for k in aliases if k in result),None)
        if left is None or right is None:raise ValueError('Paired fold identity required')
        np.testing.assert_array_equal(left,right)
    if result['x_multi'].shape!=original['x_multi'].shape:raise ValueError('Equivalence feature dimension changed')
    return result

def train_bias(cache,norm,out,args,logs):
    a=cache.a;train=cache.mask('head');dev=cache.mask('head','dev');groups=pt.grouped_indices(a['mechanism_id'],dev,args.min_group,args.max_group)
    targets=cross_half_targets(a['estimates'][:,1],a['ses'][:,1],a['truth'],a['mechanism_id'],a['seeds'],train,args.min_group)
    pt.save(out/'pilot_targets.json',targets);x=pt.normalized(a,'x_multi',norm,args.device);z=torch.as_tensor((a['estimates'][:,1]-a['truth'])/a['ses'][:,1],dtype=torch.float32,device=args.device)
    est=torch.as_tensor(a['estimates'][:,1],dtype=torch.float64,device=args.device);se=torch.as_tensor(a['ses'][:,1],dtype=torch.float64,device=args.device);target_mean=torch.as_tensor(np.nan_to_num(targets['mean']),dtype=torch.float32,device=args.device);target_logsd=torch.as_tensor(np.nan_to_num(targets['log_sd']),dtype=torch.float32,device=args.device)
    ids=torch.as_tensor(np.flatnonzero(train&cache.good),device=args.device);eligible=torch.as_tensor(targets['eligible'],device=args.device)
    for arm in BIAS_ARMS:
        mode=arm.split('_',1)[1];torch.manual_seed(args.seed);head=MomentHead(x.shape[1],args.width,mode).to(args.device);opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4);best=math.inf;state=None;chosen=0;steps=0
        generator=torch.Generator(device=args.device).manual_seed(args.seed)
        for epoch in range(args.head_epochs+1):
            trainloss=[]
            if epoch:
                for ii in ids[torch.randperm(len(ids),generator=generator,device=args.device)].split(args.batch):
                    p=head(x[ii]);value=-pt.logpdf(p,z[ii]).mean()+.025*normal_interval_score(p,z[ii])
                    if arm.startswith('moment'):
                        good=eligible[ii]
                        if good.any():
                            auxiliary=p[1][:,0]*0
                            if mode!='scale':auxiliary=auxiliary+(p[1][:,0]-target_mean[ii]).square()
                            if mode!='bias':auxiliary=auxiliary+(p[2][:,0].log()-target_logsd[ii]).square()
                            value=value+.2*auxiliary[good].mean()
                    opt.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step();steps+=1;trainloss.append(float(value.detach()))
            with torch.no_grad():
                ci=np.full((len(x),2),np.nan);valid=dev&cache.good;ci[valid]=interval_from_density(head(x[valid]),est[valid],se[valid]).cpu().numpy();score=group_validation(ci,a,groups)
            logs.append(dict(branch='bias',arm=arm,epoch=epoch,dev_group_risk=score,optimizer_steps=steps,train_objective=None if not trainloss else float(np.mean(trainloss)),valid_train=len(ids),attempted_train=int(train.sum()),moment_supervision_tables=int((train&cache.good&targets['eligible']).sum())))
            if score<best:best=score;state=copy.deepcopy(head.state_dict());chosen=epoch
        if state is None:raise ValueError('No finite DEV candidate '+arm)
        head.load_state_dict(state);save_head(out/(arm+'.pt'),head,epoch=chosen,dev_group_risk=best,parameters=sum(p.numel() for p in head.parameters()));print('BIAS',arm,chosen,best,flush=True)

def train_consistency(cache,pair,norm,out,args,logs):
    a=cache.a;train=cache.mask('head')&cache.good;dev=cache.mask('head','dev');groups=pt.grouped_indices(a['mechanism_id'],dev,args.min_group,args.max_group)
    valid_pair=valid_feature_rows(pair)
    train=train&valid_pair
    x=pt.normalized(a,'x_multi',norm,args.device);xp=pt.normalized(pair,'x_multi',norm,args.device);est=torch.as_tensor(a['estimates'][:,1],dtype=torch.float64,device=args.device);se=torch.as_tensor(a['ses'][:,1],dtype=torch.float64,device=args.device);estp=torch.as_tensor(pair['estimates'][:,1],dtype=torch.float64,device=args.device);sep=torch.as_tensor(pair['ses'][:,1],dtype=torch.float64,device=args.device);truth=torch.as_tensor(a['truth'],dtype=torch.float64,device=args.device);z=(est-truth)/se;zp=(estp-truth)/sep;ids=torch.as_tensor(np.flatnonzero(train),device=args.device)
    base=pt.load_heads(args.posttrain_dir,args.device)[1]
    for weight in CONSISTENCY_GRID:
        arm='consistency_'+str(weight);head=copy.deepcopy(base)
        for p in head.parameters():p.requires_grad_(True)
        opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-4);best=math.inf;state=None;chosen=0;steps=0;generator=torch.Generator(device=args.device).manual_seed(args.seed)
        for epoch in range(args.head_epochs+1):
            trainloss=[];consistency_losses=[]
            if epoch:
                for ii in ids[torch.randperm(len(ids),generator=generator,device=args.device)].split(args.batch):
                    p1,p2=head(x[ii]),head(xp[ii]);proper=.5*(pt.density_loss(p1,z[ii])+pt.density_loss(p2,zp[ii]));regularizer=equivalence_consistency(p1,p2,est[ii],se[ii],estp[ii],sep[ii]);value=proper+weight*regularizer;opt.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step();steps+=1;trainloss.append(float(proper.detach()));consistency_losses.append(float(regularizer.detach()))
            with torch.no_grad():
                ci=np.full((len(x),2),np.nan);valid=dev&cache.good;ci[valid]=interval_from_density(head(x[valid]),est[valid],se[valid]).cpu().numpy();score=group_validation(ci,a,groups)
            logs.append(dict(branch='consistency',arm=arm,epoch=epoch,dev_group_risk=score,optimizer_steps=steps,training_forward_calls=2*steps,train_proper_loss=None if not trainloss else float(np.mean(trainloss)),train_consistency_loss=None if not consistency_losses else float(np.mean(consistency_losses)),valid_paired_train=len(ids),invalid_paired_train=int((cache.mask('head')&~valid_pair).sum()),attempted_train=int(cache.mask('head').sum()),coefficient=weight))
            if score<best:best=score;state=copy.deepcopy(head.state_dict());chosen=epoch
        if state is None:raise ValueError('No finite consistency DEV candidate')
        head.load_state_dict(state);save_head(out/(arm+'.pt'),head,epoch=chosen,dev_group_risk=best,coefficient=weight);print('CONSISTENCY',weight,chosen,best,flush=True)

@torch.no_grad()
def expanded_intervals(cache,norm,out,posttrain_dir,device):
    a=cache.a;heads=pt.load_heads(posttrain_dir,device);raw=pt.expert_intervals(heads,norm,a['x_single'],a['x_multi'],a['estimates'],a['ses'],device=device)['intervals'];selection={}
    for branch,names in (('moment',tuple('moment_'+m for m in ('bias','scale','both'))),('consistency',tuple('consistency_'+str(v) for v in CONSISTENCY_GRID))):
        candidates=[(name,torch.load(out/(name+'.pt'),map_location='cpu',weights_only=False)['dev_group_risk']) for name in names];selection[branch]=min(candidates,key=lambda v:(v[1],v[0]))[0]
    x=pt.normalized(a,'x_multi',norm,device);est=torch.as_tensor(a['estimates'][:,1],dtype=torch.float64,device=device);se=torch.as_tensor(a['ses'][:,1],dtype=torch.float64,device=device);added=[]
    for name in selection.values():
        h=load_head(out/(name+'.pt'),device);ci=np.full((len(x),2),np.nan);ci[cache.good]=interval_from_density(h(x[cache.good]),est[cache.good],se[cache.good]).cpu().numpy();added.append(ci)
    pool=np.concatenate((raw,np.stack(added,1)),1);pt.save(out/'expanded_pool.json',dict(experts=list(pt.EXPERTS)+list(selection.values()),point_indices=[0,1,0,1,1,1],selected_by='Original DEV group risk only; identical six-expert pool for ALL expanded policies',selection=selection,expert_hashes={name:pt.sha(out/(name+'.pt')) for name in selection.values()}));return pool

def train_policy(cache,norm,pool,out,args,logs):
    a=cache.a;x=pt.normalized(a,'x_multi',norm,args.device);devgroups=pt.grouped_indices(a['mechanism_id'],cache.mask('sft','dev'),args.min_group,args.max_group);train_sft=pt.grouped_indices(a['mechanism_id'],cache.mask('sft'),args.min_group,args.max_group);rlgroups=pt.grouped_indices(a['mechanism_id'],cache.mask('rlvr'),args.min_group,args.max_group);dpogroups=pt.grouped_indices(a['mechanism_id'],cache.mask('dpo'),args.min_group,args.max_group);width=args.width;experts=pool.shape[1]
    def risk_for_actions(actions,ids):return pt.action_rewards(actions,pool[ids],a['truth'][ids],a['ses'][ids,0])
    def valid_score(model):
        with torch.no_grad():acts=model(x).argmax(-1).cpu().numpy()
        return float(np.mean([pt.group_risk(pool[g,acts[g],0],pool[g,acts[g],1],a['truth'][g],a['ses'][g,0]) for g in devgroups]))
    def fit(name,initial,reference,mode,groups,epochs,hard=None,replay=False):
        model=copy.deepcopy(initial);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);best=math.inf;state=None;chosen=0;steps=0;forwards=0;exposures=0;invalid=0
        with torch.no_grad():ref=reference(x).detach();base_action=ref.argmax(-1).cpu().numpy();baseline_ci=pool[np.arange(len(pool)),base_action]
        pref={}
        for gi,g in enumerate(groups):
            constant=np.tile(np.arange(experts)[:,None],(1,len(g)));reward=risk_for_actions(constant,g);pref[gi]=dict(label=int(np.argmax(reward)),usable=bool(np.isfinite(reward).any()))
            if mode=='dpo':
                with torch.no_grad():actions=torch.distributions.Categorical(logits=ref[g]).sample((args.draws,)).detach();rr=risk_for_actions(actions.cpu().numpy(),g);valid=np.flatnonzero(np.isfinite(rr));order=valid[np.argsort(rr[valid],kind='stable')]
                pref[gi]['usable']=len(order)>1 and rr[order[-1]]-rr[order[0]]>1e-10
                if pref[gi]['usable']:pref[gi].update(winner=actions[order[-1]],loser=actions[order[0]])
        preference_record={str(gi):dict(mechanism_id=str(a['mechanism_id'][groups[gi][0]]),table_indices=groups[gi].tolist(),usable=p['usable'],label=p['label'],winner=p['winner'].cpu().tolist() if 'winner' in p else None,loser=p['loser'].cpu().tolist() if 'loser' in p else None) for gi,p in pref.items()};pt.save(out/(name+'.preferences.json'),preference_record)
        matched_dpo=None
        if name=='expanded_extra_sft_dpo':matched_dpo=json.loads((out/'expanded_dpo.preferences.json').read_text())
        for epoch in range(epochs+1):
            if epoch:
                rng=np.random.default_rng(args.seed+epoch);order=rng.permutation(len(groups)).tolist()
                if hard is not None:
                    extra=list(hard) if replay else rng.choice(len(groups),len(hard),replace=False).tolist();order+=extra;rng.shuffle(order)
                for gi in order:
                    g=groups[gi];logits=model(x[g]);forwards+=1
                    if matched_dpo is not None and not matched_dpo[str(gi)]['usable']:invalid+=1;continue
                    if mode in ('sft','dpo') and not pref[gi]['usable']:invalid+=1;continue
                    if mode=='sft':loss=F.cross_entropy(logits,torch.full((len(g),),pref[gi]['label'],device=args.device,dtype=torch.long))
                    elif mode=='dpo':loss=pt.dpo_loss(logits,ref[g],pref[gi]['winner'],pref[gi]['loser'],args.beta)
                    else:
                        with torch.no_grad():
                            actions=torch.distributions.Categorical(logits=logits.detach()).sample((args.draws,)).detach()
                            reward,details=repair_rewards(actions.cpu().numpy(),pool[g],a['truth'][g],a['ses'][g,0],baseline_ci[g]) if mode=='repair' else (risk_for_actions(actions.cpu().numpy(),g),None)
                            reward=torch.as_tensor(reward,dtype=torch.float32,device=args.device)
                        logs.append(dict(branch='expanded_reward',arm=name,epoch=epoch,mechanism_id=str(a['mechanism_id'][g[0]]),table_indices=g.tolist(),actions=actions.cpu().tolist(),rewards=reward.cpu().tolist(),repair_components=details,failed_reward=not bool(torch.isfinite(reward).all())))
                        if not torch.isfinite(reward).all():invalid+=1;continue
                        loss=pt.rloo_loss(logits,ref[g],actions,reward)
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1;exposures+=len(g)
            score=valid_score(model);logs.append(dict(branch='expanded_policy',arm=name,epoch=epoch,dev_group_risk=score,optimizer_steps=steps,training_forward_calls=forwards,table_exposures=exposures,invalid_updates=invalid,pool_experts=experts,replay_hard_group_indices=None if hard is None else list(map(int,hard)),replay=replay))
            if score<best:best=score;state=copy.deepcopy(model.state_dict());chosen=epoch
        if state is None:raise ValueError('No finite expanded policy DEV checkpoint')
        model.load_state_dict(state);torch.save(dict(state_dict=state,dim=model.dim,width=model.width,experts=experts,epoch=chosen,dev_group_risk=best,pool_hash=pt.sha(out/'expanded_pool.json')),out/(name+'.pt'));print('EXPANDED',name,chosen,best,flush=True);return model
    initial=ExpandedPolicy(x.shape[1],experts,width).to(args.device);sft=fit('expanded_sft',initial,initial,'sft',train_sft,args.sft_epochs)
    fit('expanded_dpo',sft,sft,'dpo',dpogroups,args.post_epochs)
    fit('expanded_extra_sft_dpo',sft,sft,'sft',dpogroups,args.post_epochs)
    with torch.no_grad():baseline=sft(x).argmax(-1).cpu().numpy()
    risks=np.array([pt.group_risk(pool[g,baseline[g],0],pool[g,baseline[g],1],a['truth'][g],a['ses'][g,0]) for g in rlgroups]);hard=np.argsort(-risks,kind='stable')[:max(1,len(rlgroups)//3)]
    pt.save(out/'replay_partition.json',dict(group_mechanism_ids=[str(a['mechanism_id'][g[0]]) for g in rlgroups],baseline_group_risk=risks,hard_indices=hard,rule='Worst third fixed SFT training-group risk. Every epoch visits ALL groups once; adds H hard groups for replay or H uniform groups for control. Easy groups never dropped.'))
    for mode in ('rlvr','repair'):
        for replay in (False,True):fit('expanded_'+mode+('_replay' if replay else '_uniform'),sft,sft,mode,rlgroups,args.post_epochs,hard,replay)
    fit('expanded_extra_sft_uniform',sft,sft,'sft',rlgroups,args.post_epochs,hard,False)
    fit('expanded_extra_sft_replay',sft,sft,'sft',rlgroups,args.post_epochs,hard,True)

def preregister(args,cache):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);path=out/'preregistration.json';files=[Path(__file__),Path(__file__).with_name('posttrain.py'),Path(__file__).with_name('inference.py')]
    core=Path(args.posttrain_dir);artifacts=json.loads((core/'artifacts.json').read_text())
    if artifacts.get('complete') is not True:raise ValueError('Frozen core post-training bundle incomplete')
    for name,digest in artifacts['head_hashes'].items():
        if pt.sha(core/name)!=digest:raise ValueError('Frozen core weights changed '+name)
    if pt.sha(core/'normalization.npz')!=artifacts['normalization_hash']:raise ValueError('Core feature normalization changed')
    registered_args={k:v for k,v in vars(args).items() if k not in ('stage','pair_cache')}
    registration=dict(candidate_only=True,source_hashes={p.name:pt.sha(p) for p in files},inputs=dict(cache=pt.sha(args.cache),core_artifacts=pt.sha(Path(args.posttrain_dir)/'artifacts.json')),args=registered_args,application_arms=APPLICATION_ARMS,primary_candidate='expanded_repair_replay',point_definition='All error-density arms report base canonical point-SE*E[Z] as corrected primary; SE column remains canonical scoreSE, not proven corrected-estimator SE. Core13 unchanged. Generic/mediation mean-corrected controls have identical CIs to core heads.',bias_arms=BIAS_ARMS,consistency_coefficients=CONSISTENCY_GRID,pilot=dict(rule='Opposite sorted-seed-RANK-parity half SAME-mechanism Zmoments; no own error in target; min16tables',standard_deviation_floor=.1,variance_ddof=1),bias_loss='Gaussian NLL+.025 standardized interval score+.2 independent pilot moment MSE; matched anchor controls auxiliary coefficient0',consistency='Half orig+paired NLL+.2CRPS +lambda scaled implied-theta-mean difference squared+logthetaSDdifference squared; bothpoint/SE detached; original-only deployment; paired Mflip/Xperm bank separately hash-bound before any consistency fit',repair='Relative per-table IS +.5 rescued miss−2 lost cover−.1 extra relative width−.5 tail regression−.01 cost; RLOO fixed expanded-SFT reference',cost='No expanded action costs are fabricated: current frozen caches supply no measured N-by6 costs, so cost component is zero; all views computed, no early-exit latency claim.',pool='Same six frozen experts for ALL expanded SFT/DPO/RL/repair controls; new experts chosen by originalDEV groupRisk before finaltest',replay='All groups including easy once each epoch plus H hard or H uniform groups, matched updates; rank fixed training-SFT risk',selection='Same originalDEV groupRisk for every head/policy;epoch0eligible; no TEST labels admitted',no_core_modification=True)
    if path.exists():
        if json.loads(path.read_text())!=pt.clean(registration):raise ValueError('Heuristic preregistration drift')
    else:
        if list(out.glob('*.pt')):raise ValueError('Register all candidates before fitting')
        pt.save(path,registration)
    return out

def train(args):
    with np.load(args.cache,allow_pickle=False) as z:
        if 'split' not in z or not set(z['split'].astype(str)).issubset({'train','dev'}):raise ValueError('Heuristic training rejects test/calibration before loading labels')
    torch.manual_seed(args.seed);cache=pt.Cache(args.cache);cache.good=cache.good&valid_feature_rows(cache.a);out=preregister(args,cache);logs=[]
    with np.load(Path(args.posttrain_dir)/'normalization.npz') as z:norm={k:(z[k+'_center'],z[k+'_scale']) for k in ('x_single','x_multi')}
    if args.stage in ('bias','full'):train_bias(cache,norm,out,args,logs)
    if args.stage in ('consistency','full'):
        if not args.pair_cache:raise ValueError('Causal-equivalence paired cache required')
        binding=dict(paired_cache_sha256=pt.sha(args.pair_cache),original_cache_sha256=pt.sha(args.cache),alignment='Same mechanism_id/observationseed; causal-equivalent transformed task, never extra independent replicate')
        path=out/'paired_binding.json'
        if path.exists() and json.loads(path.read_text())!=binding:raise ValueError('Paired input changed after registration')
        if not path.exists():pt.save(path,binding)
        train_consistency(cache,aligned_pair(cache.a,args.pair_cache),norm,out,args,logs)
    if args.stage in ('repair','full'):pool=expanded_intervals(cache,norm,out,args.posttrain_dir,args.device);train_policy(cache,norm,pool,out,args,logs)
    pt.save(out/('training_'+args.stage+'.json'),logs);weights={p.name:pt.sha(p) for p in out.glob('*.pt')};complete=all((name+'.pt') in weights for name in HEAD_ARMS+EXPANDED_ARMS)
    records={}
    for name in HEAD_ARMS+EXPANDED_ARMS:
        if name+'.pt' in weights:
            ck=torch.load(out/(name+'.pt'),map_location='cpu',weights_only=False);records[name]=dict(dev_group_risk=ck['dev_group_risk'],epoch=ck['epoch'],sha256=weights[name+'.pt'])
    chosen=min(records,key=lambda name:(records[name]['dev_group_risk'],name)) if records else None
    selection=dict(primary='expanded_repair_replay',dev_chosen_candidate=chosen,all_candidate_records=records,all_candidates_complete=complete,criterion='Same original DEV groupRisk, epoch0eligible; all candidates reported, no final-test choice',point_definition='Density-head methods and error-expert actions report base_estimate-base_score_SE*E[Z] as PRIMARY point. Scale-only has E[Z]=0. CI remains base_estimate-base_score_SE*[qhi,qlo]. Reported statsSE is canonical scoreSE, not proven SE of corrected point.',no_inherited_wald_guarantee=True,core_artifacts_sha256=pt.sha(Path(args.posttrain_dir)/'artifacts.json'),preregistration_sha256=pt.sha(out/'preregistration.json'),source_hashes={Path(__file__).name:pt.sha(Path(__file__))})
    pt.save(out/'selection.json',selection);pt.save(out/'artifacts.json',dict(candidate_only=True,stage=args.stage,preregistration_sha256=pt.sha(out/'preregistration.json'),selection_sha256=pt.sha(out/'selection.json'),weights=weights,complete=complete,stage_complete=True,source_hashes={Path(__file__).name:pt.sha(Path(__file__))}))

def verify_candidate(folder,core_folder):
    folder=Path(folder);art=json.loads((folder/'artifacts.json').read_text());registration=json.loads((folder/'preregistration.json').read_text())
    if art.get('complete') is not True:raise ValueError('All candidate branches must finish before final test')
    for name,digest in art['weights'].items():
        if pt.sha(folder/name)!=digest:raise ValueError('Candidate weights changed '+name)
    for name,digest in registration['source_hashes'].items():
        if pt.sha(Path(__file__).with_name(name))!=digest:raise ValueError('Candidate source changed '+name)
    if pt.sha(folder/'preregistration.json')!=art['preregistration_sha256'] or pt.sha(folder/'selection.json')!=art['selection_sha256']:raise ValueError('Candidate selection lock changed')
    if registration['inputs']['core_artifacts']!=pt.sha(Path(core_folder)/'artifacts.json'):raise ValueError('Candidate core reference changed')
    cache=Path(registration['args']['cache'])
    choices=[cache] if cache.is_absolute() else [Path.cwd()/cache,folder/cache,folder.parent/cache];cache=next((p for p in choices if p.exists() and pt.sha(p)==registration['inputs']['cache']),None)
    if cache is None:raise ValueError('Candidate training identity bank missing')
    with np.load(cache,allow_pickle=False) as z:
        mids=set(z['mechanism_id'].astype(str));seeds=set(map(int,z['seeds']))
        if 'fold_seeds' in z:seeds.update(map(int,z['fold_seeds']))
    return art,mids,seeds

def _pad_density(parameters,n,components=5):
    result={key:np.full((n,components),np.nan) for key in ('weights','means','scales')};k=parameters['weights'].shape[1]
    for key,value in parameters.items():
        result[key][:,:k]=value
        if k<components:result[key][:,k:]=0. if key!='scales' else 1.
    return result

@torch.no_grad()
def apply_candidate_group(a,core_heads,heads,policies,norm,pool_selection):
    import apply_posttrain as core_apply
    n=len(a['truth']);alphas=(.10,.05,.01);names=APPLICATION_ARMS;count=len(names);stats=np.tile(core_apply.failed_stats(),(n,count,3,1));base=np.full((n,count),np.nan);base_se=base.copy();bias=base.copy();scale=base.copy();point_index=np.full((n,count),-1,np.int8);actions=np.full((n,count),-1,np.int16);quantiles=np.full((n,count,3,2),np.nan);densities={key:np.full((n,count,5),np.nan) for key in ('weights','means','scales')};raw_pool=np.full((n,6,3,2),np.nan);raw_pool_q=raw_pool.copy();probabilities={};ss=(core_apply.summaries(a,0),core_apply.summaries(a,1));extra=[heads[pool_selection[key]] for key in ('moment','consistency')]
    for ai,alpha in enumerate(alphas):
        ex=pt.expert_intervals(core_heads,norm,a['x_single'],a['x_multi'],a['estimates'],a['ses'],alpha)
        direct={}
        for control,key,point,j in ((CONTROL_ARMS[0],'generic_error',0,2),(CONTROL_ARMS[1],'mediation_error',1,3)):
            d={k:np.asarray(v,float) for k,v in ex['mixture_evidence'][key].items()};ez=np.sum(d['weights']*d['means'],1);var=np.sum(d['weights']*(d['means']**2+d['scales']**2),1)-ez**2;canonical=a['estimates'][:,point];canonical_se=a['ses'][:,point]
            direct[control]=dict(intervals=ex['intervals'][:,j],quantiles=ex['quantiles'][:,j],base_estimate=canonical,base_score_se=canonical_se,predicted_bias=canonical_se*ez,estimate=canonical-canonical_se*ez,implied_interval_scale=canonical_se*np.sqrt(np.maximum(var,0)),mixture_evidence=d,point=point)
        for name,head in heads.items():direct[name]=dict(predict_error_head(head,norm,a['x_multi'],a['estimates'][:,1],a['ses'][:,1],alpha),point=1)
        for name,result in direct.items():
            j=names.index(name);point=result['point'];point_index[:,j]=point;quantiles[:,j,ai]=result['quantiles']
            for key in ('weights','means','scales'):densities[key][:,j]=_pad_density(result['mixture_evidence'],n)[key]
            for i in range(n):
                row=core_apply.interval_stats(ss[point][i],alpha,endpoints=result['intervals'][i]);
                if not row[5]:row[0]=result['estimate'][i]
                stats[i,j,ai]=row
            base[:,j]=result['base_estimate'];base_se[:,j]=result['base_score_se'];bias[:,j]=result['predicted_bias'];scale[:,j]=result['implied_interval_scale']
        for name,policy in policies.items():
            j=names.index(name);result=predict_expanded(policy,core_heads,extra,norm,a['x_single'],a['x_multi'],a['estimates'],a['ses'],alpha);act=result['actions'];pi=result['point_index'];actions[:,j]=act;point_index[:,j]=pi;quantiles[:,j,ai]=result['quantiles'];probabilities[name]=result['probabilities'];raw_pool[:,:,ai]=result['raw_expert_intervals'];raw_pool_q[:,:,ai]=result['raw_expert_quantiles']
            for i in range(n):
                row=core_apply.interval_stats(ss[int(pi[i])][i],alpha,endpoints=result['intervals'][i])
                if not row[5]:row[0]=result['estimate'][i]
                stats[i,j,ai]=row
            base[:,j]=result['base_estimate'];base_se[:,j]=result['base_score_se'];bias[:,j]=result['predicted_bias'];scale[:,j]=result['implied_interval_scale']
            pool_evidence=[]
            for column,key in enumerate(('wald0','wald1','generic_error','mediation_error','added_0','added_1')):
                d=dict(weights=np.ones((n,1)),means=np.zeros((n,1)),scales=np.ones((n,1))) if column<2 else result['mixture_evidence'][key];pool_evidence.append(_pad_density(d,n))
            for key in densities:densities[key][:,j]=np.stack([d[key] for d in pool_evidence],1)[np.arange(n),act]
    result=dict(stats=stats,arm_names=np.array(names),alphas=np.array(alphas),actions=actions,point_index=point_index,base_estimate=base,base_score_se=base_se,predicted_bias=bias,implied_interval_scale=scale,corrected_estimate=base-bias,quantiles=quantiles,raw_pool_intervals=raw_pool,raw_pool_quantiles=raw_pool_q,truth=a['truth'],mechanism_id=a['mechanism_id'],seeds=a['seeds'],fold_seeds=a['fold_seeds'],failed=stats[:,:,:,5].astype(bool),**{'cdf_'+k:v for k,v in densities.items()})
    result.update({name+'_probabilities':prob for name,prob in probabilities.items()});return result

def apply(args):
    from merge_cache import entries,load_entry,read,sha
    from evaluate import EvaluationGuard
    import apply_posttrain as core_apply
    guard=EvaluationGuard(args.protocol,args.authorization);guard.check('dev');header,groups=entries(args.cache_manifest);protocol_hash=sha(args.protocol)
    if header.get('protocol_sha256')!=protocol_hash:raise ValueError('Candidate cache protocol mismatch')
    _,core_ids,core_seeds,identity=core_apply.verify_posttrain(args.posttrain_dir)
    if identity!={'checkpoint_sha256':header['checkpoint_sha256'],'protocol_sha256':protocol_hash}:raise ValueError('Candidate core cache identity mismatch')
    _,train_ids,train_seeds=verify_candidate(args.heuristics_dir,args.posttrain_dir);train_ids|=core_ids;train_seeds|=core_seeds;folder=Path(args.heuristics_dir);pool=json.loads((folder/'expanded_pool.json').read_text());heads={name:load_head(folder/(name+'.pt')) for name in HEAD_ARMS};policies={name:load_expanded_policy(folder/(name+'.pt')) for name in EXPANDED_ARMS};core_heads=pt.load_heads(args.posttrain_dir,'cpu')
    with np.load(Path(args.posttrain_dir)/'normalization.npz') as z:norm={key:(z[key+'_center'],z[key+'_scale']) for key in ('x_single','x_multi')}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);records=[]
    for gi,entry in enumerate(groups):
        if gi%args.task_count!=args.task_index:continue
        split=str(entry.get('split',''))
        if split in ('train','calibration','calibrate'):continue
        is_test=not(split.startswith('dev') or split=='validation')
        if is_test:
            guard.check('test');gate=read(args.authorization)
            if gate.get('posttrain_artifacts_sha256',gate.get('core_artifacts_sha256'))!=sha(Path(args.posttrain_dir)/'artifacts.json') or gate.get('checkpoint_sha256')!=header['checkpoint_sha256'] or gate.get('heuristics_artifacts_sha256')!=sha(folder/'artifacts.json'):raise ValueError('Candidate final-test release does not bind all frozen artifacts')
        path,a=load_entry(entry,args.cache_manifest,protocol_hash)
        if is_test and (set(a['mechanism_id'].astype(str))&train_ids or (set(map(int,a['seeds']))|set(map(int,a['fold_seeds'])))&train_seeds):raise ValueError('Candidate test/training identity overlap')
        name=entry.get('name',path.stem);target=output/(name+'.heuristics.npz');ledger=target.with_suffix('.json');locks=dict(protocol_sha256=protocol_hash,cache_sha256=sha(path),core_artifacts_sha256=sha(Path(args.posttrain_dir)/'artifacts.json'),heuristics_artifacts_sha256=sha(folder/'artifacts.json'),selection_sha256=sha(folder/'selection.json'),source_sha256=sha(Path(__file__)))
        if target.exists():
            old=read(ledger)
            if old['locks']!=locks or old['sha256']!=sha(target):raise ValueError('Candidate application artifact drift')
            records.append(old);continue
        started=time.perf_counter();result=apply_candidate_group(a,core_heads,heads,policies,norm,pool['selection']);np.savez_compressed(target,**result);record=dict(name=name,path=str(target),sha256=sha(target),locks=locks,split=split,tables=len(a['truth']),arms=APPLICATION_ARMS,seconds=time.perf_counter()-started,primary_point='Corrected density mean for all learned error-head candidates/actions; generic/mediation mean-corrected controls use identical core CIs',reported_SE='Canonical scoreSE, NOT proven sampling SE of corrected point',CI='Base canonical estimate-SE*error quantiles; CDF evidence in Z coordinates',candidate_only=True,all_failed_attempts_retained=True);pt.save(ledger,record);records.append(record);print('APPLIED_HEURISTICS',name,len(a['truth']),flush=True)
    suffix='' if args.task_count==1 else f'.part-{args.task_index}-of-{args.task_count}';pt.save(output/('manifest'+suffix+'.json'),dict(protocol_sha256=protocol_hash,heuristics_artifacts_sha256=sha(folder/'artifacts.json'),groups=records))

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True);p=sub.add_parser('train');p.add_argument('--stage',choices=('bias','consistency','repair','full'),default='full');p.add_argument('--cache',required=True);p.add_argument('--pair-cache');p.add_argument('--posttrain-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=2026102697);p.add_argument('--width',type=int,default=128);p.add_argument('--head-epochs',type=int,default=20);p.add_argument('--sft-epochs',type=int,default=15);p.add_argument('--post-epochs',type=int,default=10);p.add_argument('--batch',type=int,default=128);p.add_argument('--draws',type=int,default=4);p.add_argument('--beta',type=float,default=.1);p.add_argument('--min-group',type=int,default=16);p.add_argument('--max-group',type=int,default=32)
    p=sub.add_parser('apply');p.add_argument('--protocol',required=True);p.add_argument('--cache-manifest',required=True);p.add_argument('--posttrain-dir',required=True);p.add_argument('--heuristics-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--authorization');p.add_argument('--task-index',type=int,default=0);p.add_argument('--task-count',type=int,default=1);args=parser.parse_args()
    if args.command=='train':train(args)
    else:
        if not 0<=args.task_index<args.task_count:raise ValueError('Invalid candidate application shard')
        apply(args)
if __name__=='__main__':main()
