"""Explicit paired-data overlay around frozen batched cache inference.

This writes NEW paired caches only. Original raw/caches/checkpoints and the
frozen build_cache implementation remain unchanged. Same seeds identify a
pair, not an additional iid draw; never concatenate pairs into standard Cache.
"""
from __future__ import annotations
from pathlib import Path
import copy,json,sys
import numpy as np
import torch
import build_cache as base
from evaluate import EvaluationGuard
from model import load_checkpoint
from equivariance import TRANSFORMS,fixed_permutation

SCHEMA='v9-causal-equivalence-paired-cache-1'

def own_hashes():
    folder=Path(__file__).resolve().parent
    return {name:base.sha(folder/name) for name in ('equivariance.py','build_equivariant_cache.py')}

def registration(args):
    value=base.read(args.registration)
    if value.get('frozen') is not True or value.get('protocol_sha256')!=base.sha(args.protocol):raise ValueError('Frozen innovation registration bound to base protocol required')
    if value.get('source_hashes')!=own_hashes():raise ValueError('Innovation registration source hashes changed')
    if args.transform not in value.get('transforms',[]):raise ValueError('Unregistered paired transform')
    return value

def selected_groups(args,manifest,registered):
    requested=[];test_total=0;all_test_total=sum(int(g['count']) for g in manifest['groups'] if g['split'] not in base.SAFE_SPLITS)
    diagnostics=registered.get('diagnostic_groups',{}).get(args.transform,[])
    if args.transform=='x_permutation' and not diagnostics:raise ValueError('X-permutation requires a predeclared cheap diagnostic group subset')
    test_names=registered.get('test_groups',[])
    if isinstance(test_names,dict):test_names=test_names.get(args.transform,[])
    for original in manifest['groups']:
        if int(original.get('seed_index',0)) not in (args.model_seed_index,-1):continue
        split=original['split'];test=split not in base.SAFE_SPLITS
        if test and not args.include_test:continue
        if not test and split not in ('train','dev'):continue
        if not test and original['bank'] not in ('head','sft','dpo','rlvr','dev'):continue
        if args.split and split not in args.split:continue
        if args.group_name and original['name'] not in args.group_name:continue
        if diagnostics and original['name'] not in diagnostics:continue
        if test and original['name'] not in test_names:continue
        count=int(original['count']);limit=count if args.max_tables is None else min(count,args.max_tables)
        if test:
            registered_limit=int(registered.get('test_tables_per_group',0))
            if registered_limit<1:raise ValueError('Explicit positive test_tables_per_group required')
            limit=min(limit,registered_limit)
        indices=np.arange(limit,dtype=np.int64)
        if test and original['name'] in registered.get('test_indices',{}):indices=np.asarray(registered['test_indices'][original['name']],np.int64)
        if len(indices)<1 or len(np.unique(indices))!=len(indices) or np.any((indices<0)|(indices>=count)):raise ValueError('Invalid paired table subset')
        if test and len(indices)>registered_limit:raise ValueError('Test subset exceeds registered group limit')
        test_total+=len(indices) if test else 0
        permutation=fixed_permutation(int(original['p']),int(registered['permutation_seed']),original['name']) if args.transform=='x_permutation' else np.arange(int(original['p']))
        derived=dict(original);derived.update(name=original['name']+'__'+args.transform,count=len(indices),equivariance=dict(original_name=original['name'],
            transform=args.transform,origindex=indices.tolist(),permutation=permutation.tolist(),registration_sha256=base.sha(args.registration),
            target='Same exact NIE1; no A recoding; original table IDs/seeds/folds retained'))
        requested.append((original,derived,indices,permutation))
    if args.include_test:
        maximum=int(registered.get('max_test_tables',0))
        if not test_names or maximum<1 or test_total>maximum:raise ValueError('Explicit bounded preregistered test transform subset required')
        if test_total>=all_test_total and all_test_total>0:raise ValueError('Full test transform pass is forbidden; original final evaluation remains on original caches')
    if not requested:raise ValueError('No registered paired groups selected')
    return requested

def paired_group(original,derived,indices,permutation,args,model,guard,manifest_hash,checkpoint_hash,source):
    original_reader=base.raw_group
    original_encode,original_predict=model.encode_support,model.predict_nuisance
    budget=dict(encode_forward_calls=0,support_table_encodings=0,query_forward_calls=0,query_patient_evaluations=0)
    def counted_encode(*a,**kw):
        support=a[0] if a else kw.get('support_x')
        budget['encode_forward_calls']+=1;budget['support_table_encodings']+=int(support.shape[0])
        return original_encode(*a,**kw)
    def counted_predict(*a,**kw):
        query=a[1] if len(a)>1 else kw.get('query_x')
        budget['query_forward_calls']+=1;budget['query_patient_evaluations']+=int(query.shape[0]*query.shape[1])
        return original_predict(*a,**kw)
    def transformed_reader(group,manifest_path,protocol_path,protocol_hash):
        # First validate the unchanged raw group with the frozen reader. Only
        # then create a separate in-memory transformed copy for model/scoring.
        raw,binding=original_reader(original,manifest_path,protocol_path,protocol_hash)
        transformed={key:array[indices].copy() for key,array in raw.items()}
        if args.transform=='m_flip':transformed['m']=1-transformed['m']
        else:transformed['x']=transformed['x'][:,:,permutation]
        binding.update(causal_equivalence_overlay=derived['equivariance'],original_raw_unchanged=True)
        return transformed,binding
    try:
        base.raw_group=transformed_reader
        model.encode_support,model.predict_nuisance=counted_encode,counted_predict
        entry=base.process_group(derived,args,model,guard,manifest_hash,checkpoint_hash,source)
    finally:
        base.raw_group=original_reader
        model.encode_support,model.predict_nuisance=original_encode,original_predict
    path=Path(entry['path']);ledger_path=Path(entry['metadata_path']);ledger=base.read(ledger_path)
    with np.load(path,allow_pickle=False) as z:arrays={key:z[key] for key in z.files}
    n=len(indices);permutation_padded=np.full((n,30),-1,np.int64);permutation_padded[:,:len(permutation)]=permutation
    fields=dict(origindex=indices,original_group=np.full(n,original['name']),transform=np.full(n,args.transform),x_permutation=permutation_padded,
        base_table_id=np.asarray([original['name']+':'+str(int(seed)) for seed in arrays['seeds']]),
        pair_id=np.asarray([original['name']+':'+str(int(seed))+'|'+args.transform for seed in arrays['seeds']]))
    if all(key in arrays for key in fields):
        for key,value in fields.items():np.testing.assert_array_equal(arrays[key],value)
    else:
        arrays.update(fields);temporary=path.with_name(path.name+'.paired-fields.tmp')
        if temporary.exists():raise FileExistsError('Interrupted pairing-field commit needs explicit recovery')
        with temporary.open('xb') as f:np.savez_compressed(f,**arrays)
        temporary.replace(path)
        ledger['sha256']=base.sha(path)
    if budget['encode_forward_calls'] or 'additional_inference_budget' not in ledger:
        ledger['additional_inference_budget']=dict(tables=n,views=3,outer_folds=2,**budget,
            scope='Actual additional model calls for this paired cache generation, including failed-batch fallback; original-cache inference is separate; resume adds zero calls')
    ledger.update(pairing=derived['equivariance'],pairing_fields=list(fields),paired_schema=SCHEMA,
        target_note='Original exact NIE retained by causal-equivalence identity; paired model estimates/SEs are allowed to differ')
    base.atomic_json(ledger_path,ledger)
    entry.update(sha256=base.sha(path),metadata_sha256=base.sha(ledger_path),original_group=original['name'],transform=args.transform,pairing_registration_sha256=base.sha(args.registration))
    return entry

def run(args):
    if args.workers is None:args.workers=int(base.os.environ.get('NSLOTS',base.os.environ.get('SLURM_CPUS_PER_TASK',base.os.cpu_count() or 1)))
    if min(args.batch_size,args.workers,args.task_count)<1 or not 0<=args.task_index<args.task_count:raise ValueError('Invalid batch/workers/shard')
    registered=registration(args);manifest=base.read(args.manifest);protocol_hash=base.sha(args.protocol)
    if manifest.get('protocol_sha256')!=protocol_hash:raise ValueError('Raw manifest protocol mismatch')
    selections=selected_groups(args,manifest,registered);guard=EvaluationGuard(args.protocol,args.authorization)
    for original,_,_,_ in selections:base.ensure_guard(guard,original['split'])
    checkpoint_hash=base.sha(args.checkpoint)
    if any(original['split'] not in base.SAFE_SPLITS for original,_,_,_ in selections):base.check_checkpoint_release(base.read(args.authorization),checkpoint_hash)
    source={**base.source_hashes(),**own_hashes()};manifest_hash=base.sha(args.manifest);registration_hash=base.sha(args.registration)
    # This is an independent paired source registration, never a modification
    # to the source hashes in the frozen main experiment protocol.
    if args.finalize_manifest:
        return base.finalize(args,[d for _,d,_,_ in selections],protocol_hash,manifest_hash,checkpoint_hash,source)
    if args.cpu_threads is not None:torch.set_num_threads(args.cpu_threads)
    elif base.os.environ.get('NSLOTS'):torch.set_num_threads(int(base.os.environ['NSLOTS']))
    torch.set_float32_matmul_precision('high');model=load_checkpoint(args.checkpoint,args.device)
    for parameter in model.parameters():parameter.requires_grad_(False)
    entries=[]
    for original,derived,indices,permutation in selections[args.task_index::args.task_count]:
        entry=paired_group(original,derived,indices,permutation,args,model,guard,manifest_hash,checkpoint_hash,source);entries.append(entry)
        print('PAIRED_CACHE',derived['name'],flush=True)
    if source!={**base.source_hashes(),**own_hashes()} or checkpoint_hash!=base.sha(args.checkpoint) or registration_hash!=base.sha(args.registration):raise ValueError('Paired source/checkpoint/registration drift')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    name='manifest.json' if args.task_count==1 else f'manifest.part-{args.task_index}-of-{args.task_count}.json'
    record=dict(schema=SCHEMA,utc=base.now(),protocol_sha256=protocol_hash,checkpoint_sha256=checkpoint_hash,checkpoint_path=str(Path(args.checkpoint).resolve()),
        raw_manifest_sha256=manifest_hash,model=model.config.size,parameter_count=model.parameter_count,source_hashes=source,groups=entries,
        task_index=args.task_index,task_count=args.task_count,model_seed_index=args.model_seed_index,attempts=sum(g['count'] for g in entries),
        registration_sha256=registration_hash,transform=args.transform,
        pairing_contract='Align original_group+origindex or base_table_id/seeds+mechanism_id; pairs intentionally share IDs; never merge as new iid rows')
    base.atomic_json(output/name,record);return output/name

def parser():
    p=base.parser();p.description=__doc__;p.add_argument('--registration',required=True);p.add_argument('--transform',choices=TRANSFORMS,default='m_flip')
    p.add_argument('--split',action='append');p.add_argument('--group-name',action='append');p.add_argument('--max-tables',type=int,default=None);p.add_argument('--include-test',action='store_true')
    return p

if __name__=='__main__':
    args=parser().parse_args()
    if args.model_seed_index is None:raise ValueError('--model-seed-index is required for paired independent-model banks')
    if args.authorization is None:args.authorization=str(Path(args.protocol).parent/'audit/test_authorization.json')
    run(args)
