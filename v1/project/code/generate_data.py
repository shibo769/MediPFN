"""Protocol-indexed CPU raw banks; final data require an explicit frozen release.

Array tasks write independent cell shards. Assemble a manifest only after all
requested cells finish; no concurrent global-manifest mutation.
"""
from pathlib import Path
import argparse,json,os,time
import numpy as np
from data import sample_mechanism,derived_seed,make_folds
from evaluate import sha,clean,EvaluationGuard

TRAIN_SPLITS={'train','dev','cal','calibration','capacity_train','capacity_dev'}


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(clean(value),f,indent=2,allow_nan=False)


def protocol_root(protocol_path):return Path(protocol_path).resolve().parent


def check_release(root,cell,protocol,authorization=None):
    path=Path(protocol.get('_protocol_path',Path(root)/'protocol.json'))
    if sha(path)!=protocol.get('_protocol_sha256',sha(path)):raise ValueError('Protocol changed during generation')
    guard=EvaluationGuard(path,authorization or Path(root)/'audit/test_authorization.json')
    guard.check('dev' if cell['split'] in TRAIN_SPLITS else 'test')


def empirical_population(root,kind):
    """Real covariates only; target labels are never read or retained."""
    kind=str(kind);aliases={'breast-cancer':'breast_cancer','synthetic':'synthetic','none':'synthetic','gaussian':'synthetic'}
    kind=aliases.get(kind,kind)
    if kind=='synthetic':return None,None
    if kind not in ('breast_cancer','diabetes','wine'):raise ValueError('Unregistered empirical X population')
    folder=Path(root)/'raw/populations';folder.mkdir(parents=True,exist_ok=True);path=folder/(kind+'.npy');meta=folder/(kind+'.json');lock=folder/(kind+'.lock')
    if not meta.exists():
        try:lock.mkdir();owner=True
        except FileExistsError:owner=False
        if owner:
            try:
                from sklearn.datasets import load_breast_cancer,load_diabetes,load_wine
                loaders=dict(breast_cancer=load_breast_cancer,diabetes=load_diabetes,wine=load_wine)
                dataset=loaders[kind]();x=np.asarray(dataset.data,np.float32)
                if not np.isfinite(x).all():raise ValueError('Nonfinite empirical X; no silent row deletion')
                with path.open('xb') as f:np.save(f,x,allow_pickle=False)
                write_json(meta,dict(kind=kind,rows=len(x),p=x.shape[1],population_sha256=sha(path),original_labels_discarded=True,
                    source='sklearn.datasets.'+loaders[kind].__name__,population_law='Uniform empirical distribution across these declared real covariate rows'))
            finally:lock.rmdir()
        else:
            deadline=time.monotonic()+60
            while not meta.exists():
                if time.monotonic()>deadline:raise RuntimeError('Empirical-population writer did not finish; preserve and inspect lock')
                time.sleep(.05)
    record=json.loads(meta.read_text(encoding='utf-8'))
    if sha(path)!=record['population_sha256']:raise ValueError('Empirical X population changed')
    return np.load(path,allow_pickle=False),dict(population_path=str(path.resolve()),population_sha256=record['population_sha256'],population_kind=kind)


def generate_cell(root,cell,protocol,authorization=None):
    root=Path(root).resolve();check_release(root,cell,protocol,authorization)
    name=cell['name']
    if Path(name).name!=name or not name or '..' in name:raise ValueError('Unsafe cell name')
    folder=root/'raw';folder.mkdir(parents=True,exist_ok=True);shards=folder/'cells';shards.mkdir(exist_ok=True)
    rawpath=folder/(name+'.npz');metapath=folder/(name+'.metadata.json');shard=shards/(name+'.manifest.json')
    protocol_path=Path(protocol.get('_protocol_path',root/'protocol.json'));digest=sha(protocol_path)
    if shard.exists():
        done=json.loads(shard.read_text(encoding='utf-8'))
        if done['protocol_sha256']!=digest or done['raw_sha256']!=sha(rawpath) or done['metadata_sha256']!=sha(metapath):raise ValueError('Existing cell binding differs')
        return done
    if rawpath.exists() or metapath.exists():raise FileExistsError('Incomplete cell preserved; explicit recovery required: '+name)
    population,popmeta=empirical_population(root,cell.get('population_kind','synthetic'))
    p,n=int(cell['p']),int(cell['n']);laws,repeats=int(cell['mechanisms']),int(cell['repeats'])
    if min(laws,repeats)<1:raise ValueError('Positive mechanism/repeat counts required')
    if population is not None and population.shape[1]!=p:raise ValueError('Empirical X dimension differs from protocol')
    arrays={k:[] for k in ('x','a','m','y','fold_ids','truth','mechanism_id','observation_seed','fold_seed')};tables=[];law_records=[]
    for law in range(laws):
        mechanism_seed=derived_seed(cell['mechanism_seed'],law,202610260301)
        size=8192 if population is None else len(population)
        scm=sample_mechanism(mechanism_seed,p,cell['family'],population_size=size,population=population)
        mechanism=scm.to_dict();law_records.append(mechanism)
        mid=f"v9:{scm.family}:p{p}:s{mechanism_seed}:x{scm.population_hash}"
        for repeat in range(repeats):
            draw_seed=derived_seed(cell['observation_seed'],law,repeat,202610260302);fold_seed=derived_seed(cell['fold_seed'],law,repeat,202610260303)
            table=scm.draw(n,draw_seed,include_truth=True)
            for key in ('x','a','m','y'):arrays[key].append(table[key])
            arrays['fold_ids'].append(make_folds(n,fold_seed));arrays['truth'].append(scm.theta_NIE);arrays['mechanism_id'].append(mid)
            arrays['observation_seed'].append(draw_seed);arrays['fold_seed'].append(fold_seed)
            tables.append(dict(mechanism=mechanism,mechanism_index=law,mechanism_id=mid,observation_seed=draw_seed,draw_seed=draw_seed,
                fold_seed=fold_seed,n=n,p=p,split=cell['split'],bank=cell['bank'],seed_index=cell.get('seed_index',0),**(popmeta or {})))
    payload={k:np.asarray(v,dtype=np.uint64 if k in ('observation_seed','fold_seed') else str if k=='mechanism_id' else None) for k,v in arrays.items()}
    with rawpath.open('xb') as f:np.savez_compressed(f,**payload)
    write_json(metapath,dict(cell=cell,protocol_sha256=digest,source_sha256=sha(__file__),raw_sha256=sha(rawpath),tables=tables,mechanisms=law_records,
        original_real_labels_used=False,exact_target='Declared finite-population mean of (q1-q0)*(mu11-mu10)',all_tables_retained=True))
    group=dict(**cell,count=laws*repeats,protocol_sha256=digest,raw_path=str(rawpath),raw_sha256=sha(rawpath),metadata_path=str(metapath),metadata_sha256=sha(metapath))
    write_json(shard,group);return group


def finalize_manifest(root,protocol,cells,output=None):
    root=Path(root);groups=[];ids={}
    for cell in cells:
        path=root/'raw/cells'/(cell['name']+'.manifest.json');group=json.loads(path.read_text(encoding='utf-8'))
        if group['raw_sha256']!=sha(group['raw_path']) or group['metadata_sha256']!=sha(group['metadata_path']):raise ValueError('Cell changed while finalizing')
        meta=json.loads(Path(group['metadata_path']).read_text(encoding='utf-8'))
        role=(group['split'],group['bank'],group.get('seed_index',0))
        for table in meta['tables']:
            mid=table['mechanism_id'];prior=ids.get(mid)
            if prior is not None and prior!=role:raise ValueError('Mechanism reused across distinct bank/seed roles')
            ids[mid]=role
        groups.append(group)
    path=Path(output) if output else root/'raw/manifest.json'
    write_json(path,dict(protocol_sha256=sha(protocol['_protocol_path']),source_sha256=sha(__file__),groups=groups,
        table_count=sum(g['count'] for g in groups),unique_mechanisms=len(ids),schema='v9-x-a-m-y-raw-1'))
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);parser.add_argument('--root');parser.add_argument('--authorization')
    parser.add_argument('--task-index',type=int,default=int(os.environ.get('SGE_TASK_ID','1'))-1);parser.add_argument('--task-count',type=int,default=1)
    parser.add_argument('--split');parser.add_argument('--bank');parser.add_argument('--seed-index',type=int);parser.add_argument('--finalize-manifest',action='store_true');parser.add_argument('--manifest-output')
    args=parser.parse_args();path=Path(args.protocol).resolve();root=Path(args.root).resolve() if args.root else path.parent
    protocol=json.loads(path.read_text(encoding='utf-8'));protocol.update(_protocol_path=str(path),_protocol_sha256=sha(path));cells=protocol['cells']
    cells=[c for c in cells if (args.split is None or c['split']==args.split) and (args.bank is None or c['bank']==args.bank) and (args.seed_index is None or c.get('seed_index',0)==args.seed_index)]
    if args.finalize_manifest:print(finalize_manifest(root,protocol,cells,args.manifest_output));return
    if not 0<=args.task_index<args.task_count:raise ValueError('Task index is zero-based and below task count')
    for cell in cells[args.task_index::args.task_count]:
        result=generate_cell(root,cell,protocol,args.authorization);print(json.dumps(dict(generated=cell['name'],count=result['count'])),flush=True)


if __name__=='__main__':main()
