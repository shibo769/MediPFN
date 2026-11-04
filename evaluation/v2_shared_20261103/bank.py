"""Create, freeze and verify a single model-free shared ID/OOD table bank.

Only NumPy and the pinned DGP package are used. There is no model import, model
checkpoint access, fitting or evaluation. Every table and its two-fold assignment
is byte-bound before an inference worker may read the bank.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import zipfile
import numpy as np

FORMAT='medipfn-v2-frozen-shared-test-bank-1'
OBSERVED_KEYS=('x','a','m','y','x_types','mediator_types')
_TYPE_DTYPE='<U10'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def hash_object(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def seed(namespace,kind,*parts):
    raw=':'.join(map(str,(namespace,kind)+parts)).encode()
    # Existing factual seeds are <=65 bits, custom error seeds <=164 bits,
    # optional seeds <=32 bits. This new purpose tag is above bit 256.
    return (0x4d50463254455354<<256)|int.from_bytes(hashlib.sha256(raw).digest(),'big')


def fold_seed(namespace,case_id):
    return int.from_bytes(hashlib.sha256((namespace+':folds:'+case_id).encode()).digest()[:4],'big')&0x7fffffff


def outer_folds(a,seed_value):
    r=np.random.default_rng(int(seed_value));fold=np.empty(len(a),np.int8)
    for arm in (0,1):
        ix=np.flatnonzero(a==arm);r.shuffle(ix);fold[ix]=np.arange(len(ix))%2
    return fold


def observed_hash(observed):
    h=hashlib.sha256()
    for key in OBSERVED_KEYS:
        a=np.asarray(observed[key]);header=canonical(dict(key=key,dtype=a.dtype.str,shape=list(a.shape))).encode()
        data=np.ascontiguousarray(a).tobytes()
        h.update(len(header).to_bytes(8,'big'));h.update(header);h.update(len(data).to_bytes(8,'big'));h.update(data)
    return h.hexdigest()


def active_config(source,config):
    """Population definition excludes fields that no longer enter the adapted SEM."""
    c=dict(config)
    for key in ('target','source_target','source_n','quadrature_orders','reference_level','comparison_level'):
        c.pop(key,None)
    if source=='cheng':
        for key in list(c):
            if key.startswith('event_') or key in ('mediator_event_coefficient','outcome_event_coefficient'):c.pop(key)
    if source=='diaz_intermediate':
        for key in list(c):
            if key.startswith('z_') or key in ('m_z','m_az','y_z','y_az'):c.pop(key)
    if source=='miles':
        for key in list(c):
            if key.startswith('confounder_') or key in ('mediator_confounder','mediator_exposure_confounder','outcome_confounder'):c.pop(key)
    if source.startswith('xia_'):
        for key in list(c):
            if key.startswith('c_') or key.startswith('copula_') or key in ('y_c','y_ac'):c.pop(key)
    if source=='shan':
        for key in list(c):
            if key.startswith('missing') or key.startswith('shadow') or key=='incomplete_pattern_count':c.pop(key)
    if source=='zuo':
        for key in list(c):
            if key.startswith('lambda_') or key.startswith('gamma_'):c.pop(key)
    if source=='jones_correlated':c.pop('mediator_error_sd',None)
    return c


def population_hash(source,config):return hash_object(dict(source_id=source,active_sem=active_config(source,config)))


def import_dgp(root):
    root=Path(root).resolve();sys.path.insert(0,str(root))
    dgp=importlib.import_module('dgp')
    if Path(dgp.__file__).resolve().parent!=root/'dgp':raise RuntimeError('A different DGP package is already imported')
    return dgp


def dgp_bindings(root):
    root=Path(root);paths=[root/'dgp/__init__.py',root/'dgp/generators.py',root/'dgp/source_manifest.json']
    paths+=sorted((root/'dgp/vendor').glob('*.py'));paths+=sorted((root/'dgp/vendor').glob('*.json'))
    return {p.relative_to(root).as_posix():file_sha256(p) for p in paths}


def plan(protocol):
    rows=[];sizes=protocol['sample_sizes'];blocks=protocol['mechanism_blocks_per_group_and_n']
    reps=protocol['independent_row_replicates_per_block']
    for ni,n in enumerate(sizes):
        for family,presets in sorted(protocol['id_groups'].items()):
            for j in range(blocks):
                source=presets[(j+ni)%len(presets)]
                rows.append(dict(stratum='id',source_group=family,source_id=source,n=n,block=j))
        for source in protocol['source_ood_presets']:
            family='xia_chan' if source.startswith('xia_') else source
            for j in range(blocks):rows.append(dict(stratum='source_ood',source_group=family,source_id=source,n=n,block=j))
        for source in protocol['composition_ood_presets']:
            for j in range(blocks):rows.append(dict(stratum='composition_ood',source_group='jones',source_id=source,n=n,block=j))
    source_counts=Counter((r['stratum'],r['n'],r['source_group'],r['source_id']) for r in rows)
    for r in rows:
        stratum=r['stratum'];n=r['n'];family=r['source_group'];source=r['source_id']
        families=set(v['source_group'] for v in rows if v['stratum']==stratum and v['n']==n)
        presets=set(v['source_id'] for v in rows if v['stratum']==stratum and v['n']==n and v['source_group']==family)
        r['analysis_weight_within_n']=1/len(families)/len(presets)/source_counts[(stratum,n,family,source)]/reps
        r['analysis_weight']=r['analysis_weight_within_n']/len(sizes)
    return rows


def _write_case(root,dgp,protocol,block,rep,fixture=False,config=None):
    ns=protocol['namespace']+(':fixture' if fixture else '')
    source=block['source_id'];n=block['n'];j=block['block'];stratum=block['stratum']
    block_id=f"{stratum}__{source}__n{n}__m{j:03d}"
    case_id=block_id+f'__r{rep}'
    ms=seed(ns,'mechanism',stratum,source,n,j)
    obs=seed(ns,'observations',stratum,source,n,j,rep)
    task=dgp.sample_task(source,n,obs,mechanism_seed=ms,config=config,compute_truth=True)
    population=population_hash(source,task['metadata']['config'])
    fs=fold_seed(ns,case_id);folds=outer_folds(task['a'],fs)
    observed={k:np.asarray(task[k],dtype=np.float32) for k in ('x','a','m','y')}
    observed.update(x_types=np.asarray(task['x_types'],dtype=_TYPE_DTYPE),mediator_types=np.asarray(task['mediator_types'],dtype=_TYPE_DTYPE))
    metadata=dict(task['metadata'],case_id=case_id,stratum=stratum,n=n,rep=rep,
          mechanism_block_id=block_id,native_mechanism_id=task['mechanism_id'],population_id=population,
          fold_seed=fs,observation_seed=obs,namespace=ns,scientific_test=not fixture,
          active_sem_config=active_config(source,task['metadata']['config']))
    metadata_json=canonical(metadata)
    path=Path('tables')/(case_id+'.npz');destination=root/path
    with destination.open('xb') as f:np.savez_compressed(f,**observed,truth=np.asarray(task['truth'],np.float64),
                  metadata_json=np.asarray(metadata_json),outer_fold_ids=folds)
    return dict(case_id=case_id,mechanism_id=task['mechanism_id'],native_mechanism_id=task['mechanism_id'],
        mechanism_block_id=block_id,cluster_id=population,population_id=population,source_id=source,
        source_group=block['source_group'],stratum=stratum,n=n,rep=rep,fold_seed=fs,
        mechanism_seed=ms,observation_seed=obs,path=path.as_posix(),npz_sha256=file_sha256(destination),
        observed_sha256=observed_hash(observed),metadata_sha256=hashlib.sha256(metadata_json.encode()).hexdigest(),
        truth=float(task['truth']),analysis_weight=block.get('analysis_weight',0.),
        analysis_weight_within_n=block.get('analysis_weight_within_n',0.))


def build_bank(output,dgp_root,protocol_path,fixtures=False):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);(root/'tables').mkdir()
    protocol=json.loads(Path(protocol_path).read_text());dgp=import_dgp(dgp_root)
    if dgp.SOURCE_COMMIT!=protocol['dgp_source_commit']:raise ValueError('Different frozen DGP source')
    cases=[]
    if fixtures:
        specs=[('jones_correlated',200),('jones_correlated',1000),('cheng',200),('diaz_hejazi',1000)]
        for j,(source,n) in enumerate(specs):
            ms=seed(protocol['namespace']+':fixture','mechanism','fixture',source,n,j)
            c=dgp.sample_mechanism(source,ms)
            if source=='jones_correlated':c['p']=100
            b=dict(stratum='fixture',source_group='jones' if source.startswith('jones') else source,source_id=source,n=n,block=j)
            cases.append(_write_case(root,dgp,protocol,b,0,fixture=True,config=c))
    else:
        for block in plan(protocol):
            for rep in range(protocol['independent_row_replicates_per_block']):
                cases.append(_write_case(root,dgp,protocol,block,rep))
            if len(cases)%96==0:print(json.dumps(dict(stage='generate_model_free_bank',tables=len(cases))),flush=True)
    manifest=dict(format=FORMAT,status='FROZEN_ID_ONLY_VALIDATION_FIXTURE' if fixtures else 'FROZEN_SCIENTIFIC_TEST_TABLE_BANK',
       protocol=protocol,protocol_sha256=file_sha256(protocol_path),bank_program_sha256=file_sha256(__file__),
       dgp_source_sha256=dgp_bindings(dgp_root),source_commit=dgp.SOURCE_COMMIT,
       runtime=dict(python=platform.python_version(),numpy=np.__version__),cases=cases,
       counts=dict(Counter(c['stratum'] for c in cases)),tables=len(cases),
       unique_native_configurations=len(set(c['mechanism_id'] for c in cases)),
       unique_structural_populations=len(set(c['cluster_id'] for c in cases)),
       mechanism_blocks=len(set(c['mechanism_block_id'] for c in cases)),
       scientific_model_access='none during generation; tables frozen before inference')
    (root/'manifest.json').write_text(canonical(manifest)+'\n',encoding='utf-8')
    (root/'manifest.sha256').write_text(file_sha256(root/'manifest.json')+'\n')
    (root/'protocol.json').write_bytes(Path(protocol_path).read_bytes())
    return manifest


def load_manifest(bank_root,expected_sha256=None):
    root=Path(bank_root);p=root/'manifest.json'
    expected=expected_sha256 or (root/'manifest.sha256').read_text().strip()
    if file_sha256(p)!=expected:raise ValueError('Frozen bank manifest changed')
    m=json.loads(p.read_text())
    if m.get('format')!=FORMAT:raise ValueError('Wrong bank format')
    return m


def load_case(bank_root,case_doc,verify=True):
    root=Path(bank_root).resolve();p=(root/case_doc['path']).resolve()
    if root not in p.parents:raise ValueError('Case path escapes bank directory')
    if verify and file_sha256(p)!=case_doc['npz_sha256']:raise ValueError('Frozen NPZ changed')
    with np.load(p,allow_pickle=False) as z:
        observed={k:z[k].copy() for k in OBSERVED_KEYS}
        truth=float(z['truth']);raw_meta=str(z['metadata_json'].item());meta=json.loads(raw_meta);folds=z['outer_fold_ids'].copy()
    if verify:
        if observed_hash(observed)!=case_doc['observed_sha256']:raise ValueError('Observed table identity mismatch')
        if hashlib.sha256(raw_meta.encode()).hexdigest()!=case_doc['metadata_sha256']:raise ValueError('Metadata identity mismatch')
        if truth!=case_doc['truth'] or meta['case_id']!=case_doc['case_id']:raise ValueError('Case/target mismatch')
        if not np.array_equal(folds,outer_folds(observed['a'],case_doc['fold_seed'])):raise ValueError('Stored folds differ')
        if population_hash(meta['source_id'],meta['config'])!=case_doc['cluster_id']:raise ValueError('Structural population identity mismatch')
        if len(observed['a'])!=case_doc['n'] or not all(np.isfinite(observed[k]).all() for k in ('x','a','m','y')):raise ValueError('Invalid observed array')
    return dict(observed=observed,truth=truth,outer_fold_ids=folds,metadata=meta)


def verify_case(bank_root,case_doc):load_case(bank_root,case_doc,verify=True);return True
def load_observed_case(bank_root,case_doc,verify=True):return load_case(bank_root,case_doc,verify)['observed']


def archive_bank(bank_root,archive):
    root=Path(bank_root);archive=Path(archive)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(root).as_posix())
    return dict(path=str(archive.resolve()),sha256=file_sha256(archive),bytes=archive.stat().st_size)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--dgp-root',required=True)
    p.add_argument('--protocol',default=str(Path(__file__).with_name('bank_protocol.json')));p.add_argument('--fixtures',action='store_true')
    p.add_argument('--archive');a=p.parse_args();m=build_bank(a.output,a.dgp_root,a.protocol,a.fixtures)
    result=dict(tables=m['tables'],counts=m['counts'],unique_structural_populations=m['unique_structural_populations'],
                manifest_sha256=file_sha256(Path(a.output)/'manifest.json'))
    if a.archive:result['archive']=archive_bank(a.output,a.archive)
    print(json.dumps(result),flush=True)
