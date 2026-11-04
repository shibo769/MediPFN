"""Independent loaded-model CPU timing of exact old-v1 and native-old p3 arms.

Four registered linear cells, first13 observed tables:3warmups+10measured.
Model loading/raw I/O are outside callbacks; each callback includes transfers,
fresh context work and CI construction. CPU timings cannot establish an
architecture-only speed ratio against the new models measured on GPUs.
"""
from pathlib import Path
import argparse,ast,hashlib,json,os,platform,subprocess,sys,time,uuid

ARMS=('old_v1','native_old')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sources():return {name:sha(Path(__file__).with_name(name)) for name in ('run_legacy_timing.py','legacy_worker.py','run_legacy.py')}

def gate(args):
    from evaluate import EvaluationGuard
    EvaluationGuard(args.protocol,args.authorization).check('test')
    from run_legacy import verify_registration
    registration=verify_registration(args.protocol,args.registration,args.legacy_dir)
    for name,digest in sources().items():
        if registration['source_hashes'].get(name)!=digest:raise ValueError('Legacy timing source registration mismatch '+name)
    if read(args.authorization).get('extension_registrations',{}).get('legacy')!=sha(args.registration):raise ValueError('Root release must bind extended legacy registration')
    if args.threads<1 or (os.environ.get('NSLOTS') and args.threads>int(os.environ['NSLOTS'])):raise ValueError('Legacy timer threads must fit allocated CPU slots')
    return registration

def selected_groups(protocol,manifest):
    names={c['name']:c for c in protocol['cells'] if c['split']=='test' and c['bank']=='synthetic' and c['family']=='linear' and c['p']==3 and c['n'] in (200,600,1000,2000)}
    groups=[g for g in manifest['groups'] if g['name'] in names]
    if len(names)!=4 or len(groups)!=4 or len({g['name'] for g in groups})!=4:raise ValueError('Need exactly four registered p3 linear timing cells')
    for group in groups:
        cell=names[group['name']]
        for key in ('name','split','bank','family','p','n','seed_index','population_kind','mechanisms','repeats','mechanism_seed','observation_seed','fold_seed'):
            if group[key]!=cell[key]:raise ValueError('Legacy timing cell descriptor drift '+key)
        if group['count']!=cell['mechanisms']*cell['repeats'] or group['count']<13:raise ValueError('Legacy timing attempt count differs')
    return groups

def frozen_worker_blocks():
    """Compile the actual frozen worker initialization/forward blocks once.

    legacy_worker exposes only its coupled streaming loop. Extracting its two
    unchanged forward bodies permits separate timers without editing/copying
    their numerical implementation. The worker file is source/asset registered.
    """
    path=Path(__file__).with_name('legacy_worker.py');tree=ast.parse(path.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    start=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='model' for t in n.targets))
    end=next(i for i,n in enumerate(function.body[start:],start) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='print')
    outer=next(n for n in function.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='line');inner=next(n for n in outer.body if isinstance(n,ast.For));tries=[n for n in inner.body if isinstance(n,ast.Try)]
    if len(tries)!=2:raise ValueError('Frozen legacy worker forward schema differs')
    def compile_nodes(nodes):return compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec')
    return compile_nodes(function.body[start:end]),compile_nodes(tries[0].body),compile_nodes(tries[1].body)

def callbacks(model,blocks):
    import numpy as np
    import torch
    from scipy.special import expit
    from evaluate import canonical_score,interval_stats
    old_block,native_block=blocks
    def namespace(table,folds):
        if set(table)!={'x','a','m','y'}:raise ValueError('Observed variables only in timed callback')
        rows=np.column_stack((table['x'],table['a'],table['m'],table['y'])).astype(np.float32)
        return dict(np=np,torch=torch,expit=expit,model=model,rows=rows,folds=folds,n=len(rows),i=0)
    def old(table,folds):
        values=namespace(table,folds);values['nu']=np.full((1,values['n'],5),np.nan)
        exec(old_block,values)
        summary=canonical_score(table,values['nu'][0]);return dict(stats=interval_stats(summary,.05),error=summary['error'])
    def native(table,folds):
        values=namespace(table,folds);values.update(point=np.full(1,np.nan),logV=np.full(1,np.nan));exec(native_block,values)
        summary=dict(estimate=float(values['point'][0]),se=float(values['se']),clipping_rate=0.,failed=False)
        return dict(stats=interval_stats(summary,.05),error=None)
    return dict(old_v1=old,native_old=native)

def benchmark(tables,folds,predictors):
    import numpy as np
    from evaluate import failed_stats
    if len(tables)!=13 or len(folds)!=13 or tuple(predictors)!=ARMS:raise ValueError('Fixed13-table/two-arm benchmark required')
    records={name:dict(ms=[],stats=[],errors=[]) for name in ARMS};orders=[]
    for i in range(13):
        order=ARMS[i%2:]+ARMS[:i%2];orders.append(order)
        for name in order:
            tick=time.perf_counter();error=None;row=failed_stats()
            try:
                result=predictors[name](tables[i],folds[i]);row=np.asarray(result['stats'],float);error=result.get('error')
                if row.shape!=(6,) or row[5] not in (0,1) or (not row[5] and (not np.isfinite(row[:5]).all() or row[1]<=0 or row[2]>row[3])):raise ValueError('Invalid successful legacy timing interval')
            except Exception as exc:error=repr(exc);row=failed_stats()
            elapsed=(time.perf_counter()-tick)*1000;records[name]['ms'].append(elapsed);records[name]['stats'].append(row.tolist());records[name]['errors'].append(error)
    output=[]
    for name in ARMS:
        record=records[name];raw=record['ms'][3:];flags=[bool(row[5]) for row in record['stats']]
        output.append(dict(method=name,raw_ms=raw,warmup_ms=record['ms'][:3],median_ms=float(np.median(raw)),mean_ms=float(np.mean(raw)),p10_ms=float(np.quantile(raw,.1)),p90_ms=float(np.quantile(raw,.9)),
                           failure_flags=flags,warmup_failures=sum(flags[:3]),timed_failures=sum(flags[3:]),stats=record['stats'],errors=record['errors']))
    return dict(warmups=3,repetitions=10,rotated_orders=orders,records=output)

def worker(args):
    # Separate interpreter; never import the new model module into this
    # namespace. The source gate runs before any test array is opened.
    gate(args);root=Path(args.legacy_dir).resolve();sys.path.insert(0,str(root/'code'))
    import numpy as np
    import torch
    from nuisance_model import FrozenNuisancePFN
    from evaluate import clean
    torch.set_num_threads(args.threads)
    initializer,old,native=frozen_worker_blocks();namespace=dict(FrozenNuisancePFN=FrozenNuisancePFN,root=root,torch=torch)
    exec(initializer,namespace);predictors=callbacks(namespace['model'],(old,native))
    hardware=dict(device='cpu',threads=torch.get_num_threads(),platform=platform.platform(),processor=platform.processor(),logical_cpu_count=os.cpu_count(),allocated_NSLOTS=os.environ.get('NSLOTS'),HOSTNAME=os.environ.get('HOSTNAME'),JOB_ID=os.environ.get('JOB_ID'),affinity=list(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None)
    print(json.dumps(dict(ready=True,assets=dict(base=sha(root/'base.pt'),head=sha(root/'v1_head.pt')),hardware=hardware)),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('command')=='stop':break
        if request['raw_sha256']!=sha(request['raw_path']):raise ValueError('Raw timing array changed')
        with np.load(request['raw_path'],allow_pickle=False) as z:
            raw={key:z[key][:13] for key in ('x','a','m','y','fold_ids','observation_seed','fold_seed')}
        if raw['x'].shape!=(13,request['n'],3) or raw['fold_ids'].shape!=(13,request['n']):raise ValueError('Old model timing raw schema differs')
        tables=[{key:raw[key][i] for key in ('x','a','m','y')} for i in range(13)]
        result=benchmark(tables,raw['fold_ids'],predictors);result.update(hardware=hardware,seeds=raw['observation_seed'].tolist(),fold_seeds=raw['fold_seed'].tolist(),alpha=.05)
        path=Path(request['output_path'])
        with path.open('x',encoding='utf-8') as f:json.dump(clean(result),f,indent=2,allow_nan=False)
        print(json.dumps(dict(complete=True,sha256=sha(path))),flush=True)

def run(args):
    registration=gate(args);protocol=read(args.protocol);manifest=read(args.raw_manifest);ph=sha(args.protocol)
    if manifest.get('protocol_sha256')!=ph:raise ValueError('Raw manifest protocol mismatch')
    groups=selected_groups(protocol,manifest);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    locks=dict(protocol_sha256=ph,raw_manifest_sha256=sha(args.raw_manifest),registration_sha256=sha(args.registration),authorization_sha256=sha(args.authorization),source_hashes=sources(),assets=registration['assets'],threads=args.threads)
    completed=[];pending=[]
    for group in groups:
        if group.get('protocol_sha256')!=ph or group['raw_sha256']!=sha(group['raw_path']) or group['metadata_sha256']!=sha(group['metadata_path']):raise ValueError('Raw timing group/hash drift')
        path=output/(group['name']+'.legacy_timing.json');ledger=path.with_suffix('.metadata.json');binding=dict(locks,group=group)
        if path.exists() or ledger.exists():
            if not(path.exists() and ledger.exists()):raise ValueError('Interrupted legacy timing output retained')
            record=read(ledger)
            if record['binding']!=binding or record['sha256']!=sha(path):raise ValueError('Existing legacy timing output drift')
            completed.append(record)
        else:pending.append((group,path,ledger,binding))
    if pending:
        stderr=output/('worker-'+uuid.uuid4().hex+'.stderr.log')
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--protocol',args.protocol,'--raw-manifest',args.raw_manifest,'--registration',args.registration,'--legacy-dir',args.legacy_dir,'--authorization',args.authorization,'--output',args.output,'--threads',str(args.threads)]
        with stderr.open('x',encoding='utf-8') as errorlog:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errorlog,text=True)
            try:
                ready=json.loads(process.stdout.readline())
                if ready.get('ready') is not True or ready['assets']!=dict(base=registration['assets']['base.pt'],head=registration['assets']['v1_head.pt']):raise ValueError('Isolated legacy timer initialization failed')
                for group,path,ledger,binding in pending:
                    process.stdin.write(json.dumps(dict(raw_path=group['raw_path'],raw_sha256=group['raw_sha256'],n=group['n'],output_path=str(path)))+'\n');process.stdin.flush();done=json.loads(process.stdout.readline())
                    if done.get('complete') is not True or done['sha256']!=sha(path):raise ValueError('Legacy timing worker output incomplete')
                    record=dict(name=group['name'],path=str(path),sha256=sha(path),binding=binding,hardware=ready['hardware'],worker_stderr=str(stderr),all_attempts_retained=True)
                    with ledger.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
                    completed.append(record);print('LEGACY_TIMED',group['name'],flush=True)
                process.stdin.write(json.dumps(dict(command='stop'))+'\n');process.stdin.flush();process.wait(timeout=60)
                if process.returncode:raise RuntimeError('Isolated legacy timer exited unsuccessfully')
            finally:
                if process.poll() is None:process.terminate();process.wait(timeout=30)
    report=dict(groups=completed,locks=locks,methods=ARMS,groups_count=4,warmups_per_group=3,repetitions_per_group=10,device='cpu',threads=args.threads,
                scope='Loaded exact frozen old model; raw I/O/checkpoint loading excluded; old_v1 fresh full-support2fold nuisance+raw64full-ncanonicalWaldCI; native_old fresh full-table directCI. Transfers/context work included. Both timed independently with rotated order.',
                comparison_limit='CPU hardware/runtime scope only. Never infer architecture-only speed ratios against new-model GPU timings.',truth_input_to_callback=False,all_failed_attempts_retained=True)
    path=output/'manifest.json'
    if path.exists():
        if read(path)!=report:raise ValueError('Legacy timing manifest changed')
    else:
        with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','raw-manifest','registration','legacy-dir','authorization','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);return p

if __name__=='__main__':
    args=parser().parse_args();worker(args) if args.worker else run(args)
