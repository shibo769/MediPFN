"""Matched-data direct teacher baselines; no final-test access or GPU at import."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,os,time
import numpy as np
import torch
from data import sample_mechanism,pad_table
from train_backbone import make_pool,make_batch
from teacher_truth import exact_teacher_labels,table_teacher_labels
from teacher_model import DirectTeacherPFN,teacher_loss,save_teacher


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def full_batch(args,step,pool,pool_map,label_cache,dev=False,n=None):
    # Reuse exact original generator/plans, then reconstruct ALL n observed
    # rows, not just its query-budget subset. Underlying table seeds match.
    plan=make_batch(args.data_seed,step,args.dev_batch_size if dev else args.batch_size,n=n,dev=dev,query_budget=args.query_budget,mechanism_pool=None if dev else pool)
    rows=[];masks=[];effects=[];logV=[]
    for provenance in plan['provenance']:
        seed=provenance['mechanism_seed']
        mechanism=sample_mechanism(seed,provenance['p'],provenance['family']) if dev else pool_map[seed]
        if seed not in label_cache:label_cache[seed]=exact_teacher_labels(mechanism)
        table=mechanism.draw(provenance['n'],provenance['draw_seed'],include_truth=False);padded=pad_table(table)
        rows.append(np.column_stack([padded[k] for k in ('x','a','m','y')]).astype(np.float32));masks.append(padded['feature_mask'])
        target=table_teacher_labels(mechanism,table,args.mode,label_cache[seed]);effects.append(target['effect_target']);logV.append(target['log_variance_coefficient_target'])
    return dict(rows=torch.from_numpy(np.stack(rows)),mask=torch.from_numpy(np.stack(masks)),effect=torch.tensor(effects,dtype=torch.float32),logV=torch.tensor(logV,dtype=torch.float32),provenance=plan['provenance'])


def loss_on(model,batch,args):
    pred=model(batch['rows'].to(args.device),batch['mask'].to(args.device))
    return teacher_loss(pred,batch['effect'].to(args.device),batch['logV'].to(args.device),args.variance_weight)


def train(args):
    output=Path(args.output)
    if output.suffix!='.pt':output=output/'teacher.pt'
    if output.exists():raise FileExistsError('Preserving completed teacher checkpoint')
    output.parent.mkdir(parents=True,exist_ok=True)
    threads=args.cpu_threads or int(os.environ.get('SLURM_CPUS_PER_TASK',os.environ.get('NSLOTS','0')))
    if threads:torch.set_num_threads(threads)
    torch.manual_seed(args.seed);np.random.seed(args.seed%(2**32));torch.set_float32_matmul_precision('high')
    model=DirectTeacherPFN(size=args.size).to(args.device);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
    pool=make_pool(args.data_seed,args.mechanism_pool_size);pool_map={m.seed:m for m in pool};label_cache={}
    development=[full_batch(args,k,pool,pool_map,label_cache,True,n) for k,n in enumerate((200,600,1000))]
    history=[];best=math.inf;best_state=None;selected=None;start=time.perf_counter()
    for step in range(args.steps+1):
        if step:
            model.train();batch=full_batch(args,step,pool,pool_map,label_cache);optimizer.zero_grad(set_to_none=True);loss,parts=loss_on(model,batch,args)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite direct-teacher loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite direct-teacher gradient')
            optimizer.step()
        if step==0 or step%args.dev_every==0 or step==args.steps:
            model.eval()
            with torch.inference_mode():validation=float(np.mean([float(loss_on(model,b,args)[0]) for b in development]))
            if not math.isfinite(validation):raise FloatingPointError('Nonfinite direct-teacher DEV objective')
            history.append(dict(step=step,dev_teacher_loss=validation,elapsed_seconds=time.perf_counter()-start))
            if validation<best:best=validation;selected=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(json.dumps(dict(stage='matched_teacher',mode=args.mode,size=args.size,step=step,dev_loss=validation)),flush=True)
    model.load_state_dict(best_state);source={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('teacher_model.py'),Path(__file__).with_name('teacher_truth.py'),Path(__file__).with_name('train_backbone.py'),Path(__file__).with_name('model.py'),Path(__file__).with_name('data.py')]}
    metadata=dict(mode=args.mode,created_utc=datetime.now(timezone.utc).isoformat(),arguments=vars(args),source_hashes=source,parameter_counts=model.parameter_counts,
        selected_step=selected,selected_dev_loss=best,history=history,dev_provenance=[r for b in development for r in b['provenance']],
        training_mechanism_pool_seeds=[m.seed for m in pool],teacher_population_labels=label_cache,
        estimand='Same identified finite-population NIE as nuisance-DML',effect_label='Latent true NIE' if args.mode=='latent_theta' else 'Full-table sample mean of UNCLIPPED oracle mediation EIF score',
        variance_label='log(V) from exact four-cell+Ynoise integration over fixed X population, not noisy sample variance and not V/n',
        normalization='Effect difference divided by fulltable observedYscale; logV prediction restored with2logYscale. NoYcenter shift to NIE.',
        budget='Same capacity configuration, seed,data_seed,SCMpool,n,steps,batch andvalidationplans as newPFN. Fulltable directencoder usesalln, versus PFN half-support+sampledqueries; report actual compute/token exposure, not equal-FLOP claim.',
        selection='Minimum independent train-seen DEV teacher objective; step0eligible; no OOD/test access',variance_head_encoder_detached=True)
    log=output.with_suffix('.training.json');log.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8');save_teacher(output,model,mode=args.mode,training_log_sha256=sha(log),training_metadata=metadata)
    return output


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('latent_theta','FSP_oracle_score'),required=True);p.add_argument('--size',choices=('small','medium','large'),default='medium')
    p.add_argument('--seed',type=int,required=True);p.add_argument('--data-seed',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--dev-batch-size',type=int,default=4);p.add_argument('--dev-every',type=int,default=100);p.add_argument('--query-budget',type=int,default=64)
    p.add_argument('--mechanism-pool-size',type=int,default=256);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--variance-weight',type=float,default=1.);p.add_argument('--cpu-threads',type=int)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.steps<0 or args.dev_every<1 or min(args.batch_size,args.dev_batch_size,args.mechanism_pool_size,args.query_budget)<1 or not args.variance_weight>0:raise ValueError('Invalid teacher budget')
    train(args)
