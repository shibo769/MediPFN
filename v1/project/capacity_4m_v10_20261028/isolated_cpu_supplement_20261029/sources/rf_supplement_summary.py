"""Draft pure paired summaries. Caller independently verifies job accounting,
source/protocol hashes and independently reconstructed case truths first.
"""
import math
import numpy as np

def _cluster_se(items):
    """Cluster-robust SE of the per-table mean, even for unequal clusters."""
    clusters={}
    for cluster,value in items:clusters.setdefault(cluster,[]).append(float(value))
    g=len(clusters);n=len(items)
    if not n: return None
    if g<2:return None
    mean=sum(v for _,v in items)/n
    score=[sum(v-mean for v in values) for values in clusters.values()]
    return float(math.sqrt((g/(g-1))*sum(v*v for v in score))/n)

def _hash256(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def _array_identity(value):
    return (isinstance(value,dict) and set(value)=={'sha256','dtype','shape'} and _hash256(value['sha256'])
            and isinstance(value['dtype'],str) and bool(value['dtype']) and isinstance(value['shape'],list)
            and bool(value['shape']) and all(isinstance(x,int) and not isinstance(x,bool) and x>0 for x in value['shape']))

def summarize(cases,pfn,rf,truth,subset,groups):
    n=len(cases)
    if set(pfn)!=set(range(n)) or set(rf)!=set(range(n)) or set(truth)!=set(range(n)):
        raise ValueError('missing registered attempts')
    selected=set(subset); result=[]; paired=[]; identity={}
    for i,c in enumerate(cases):
        a,b=pfn[i],rf[i]
        if a['case']!=c or b['case']!=c or b['bootstrap_planned']!=(i in selected):raise ValueError('case/subset identity')
        if not math.isfinite(float(truth[i])):raise ValueError('invalid independent truth')
        if b.get('common_ok'):
            if not math.isclose(float(b['truth']),float(truth[i]),rel_tol=1e-12,abs_tol=1e-12):raise ValueError('RF truth mismatch')
        fields=((a.get('observation'),b.get('observed')),(a.get('fold'),b.get('fold')),
                (a.get('population_sha256'),b.get('population_sha256')))
        for obs,fold,pop in ((a.get('observation'),a.get('fold'),a.get('population_sha256')),
                             (b.get('observed'),b.get('fold'),b.get('population_sha256'))):
            if obs is not None and not _array_identity(obs):raise ValueError('malformed observation digest')
            if fold is not None and not _array_identity(fold):raise ValueError('malformed fold digest')
            if pop is not None and not _hash256(pop):raise ValueError('malformed population digest')
            if obs is not None and (obs['shape']!=[int(c['n']),int(c['p'])+3] or obs['dtype']!='float32'):raise ValueError('unexpected observation shape/dtype')
            if fold is not None and (fold['shape']!=[int(c['n'])] or fold['dtype']!='int8'):raise ValueError('unexpected fold shape/dtype')
        if any(x is not None and y is not None and x!=y for x,y in fields):
            raise ValueError('PFN/RF byte identity mismatch')
        if all(x is not None and y is not None for x,y in fields):
            identity[i]=True
        else:identity[i]=False
        parity=a.get('cdf95_parity',{})
        if parity.get('status')=='FAIL':raise ValueError('CDF parity failure')
        if a.get('ok'):
            if not math.isclose(float(a['truth']),float(truth[i]),rel_tol=1e-12,abs_tol=1e-12):raise ValueError('PFN truth mismatch')
            if parity.get('status')=='NOT_EVALUABLE':
                if 'original_q95' in a or not parity.get('error') or a.get('methods',{}).get('sft',{}).get('ok',True):
                    raise ValueError('unexplained unavailable CDF parity')
            elif parity.get('status')=='PASS':
                delta=parity.get('max_abs');q=a.get('original_q95')
                if (not isinstance(delta,(float,int)) or isinstance(delta,bool) or not math.isfinite(delta)
                    or delta<0 or delta>1e-6 or not isinstance(q,list) or len(q)!=1 or not isinstance(q[0],list)
                    or len(q[0])!=3 or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in q[0])):
                    raise ValueError('invalid CDF parity evidence')
            else:raise ValueError('missing CDF parity evidence')
    for label,indices,methods,reference in groups:
        if len(set(indices))!=len(indices) or not set(indices).issubset(range(n)):raise ValueError('bad group')
        if any(m.startswith('rf_bootstrap_') for m in methods) and not set(indices).issubset(selected):raise ValueError('bootstrap denominator')
        values={};length_values={}
        for method in methods:
            vals=[];lengths=[];length_by_row=[];infcount=0
            for i in indices:
                row=rf[i] if method.startswith('rf_') else pfn[i]
                common=row.get('common_ok',row.get('ok',False))
                if type(common) is not bool:raise ValueError('malformed common status')
                all_methods=row.get('methods',{})
                if common and (not isinstance(all_methods,dict) or method not in all_methods):
                    raise ValueError('successful common stage missing registered method')
                item=all_methods.get(method,{})
                if common and (not isinstance(item,dict) or type(item.get('ok')) is not bool):
                    raise ValueError('malformed method status')
                ok=common and item['ok']
                if ok:
                    lo,hi=float(item['lo']),float(item['hi'])
                    if not math.isfinite(lo) or not math.isfinite(hi) or lo>hi:raise ValueError('invalid successful endpoint')
                    cover=int(lo<=truth[i]<=hi); lengths.append(hi-lo);length_by_row.append(hi-lo)
                    if item.get('covered') is not None and bool(item['covered'])!=bool(cover):raise ValueError('coverage mismatch')
                else:cover=0;length_by_row.append(None)
                infcount+=bool(item.get('lower_unbounded') or item.get('upper_unbounded'))
                vals.append(cover)
            clusters={}
            for i,v in zip(indices,vals):clusters.setdefault(cases[i]['cluster_id'],[]).append(v)
            if not clusters:raise ValueError('empty group')
            values[method]=vals;length_values[method]=length_by_row
            units=[(cases[i]['cluster_id'],v) for i,v in zip(indices,vals)]
            result.append(dict(group=label,method=method,attempts=len(indices),valid=len(lengths),failures=len(indices)-len(lengths),unbounded=infcount,clusters=len(clusters),coverage=float(np.mean(vals)),cluster_se=_cluster_se(units),mean_length_valid=float(np.mean(lengths)) if lengths else None))
        if reference not in values:raise ValueError('missing reference')
        for method in methods:
            if method==reference:continue
            units=[];length_units=[]
            for i,a,b,la,lb in zip(indices,values[reference],values[method],length_values[reference],length_values[method]):
                # Paired PFN/RF comparisons require actual matching bytes, even
                # when a method failed after successful table reconstruction.
                if method.startswith('rf_') != reference.startswith('rf_'):
                    if not identity[i]:continue
                units.append((cases[i]['cluster_id'],a-b))
                if la is not None and lb is not None:length_units.append((cases[i]['cluster_id'],la-lb))
            paired_count=len(units)
            paired.append(dict(group=label,reference=reference,method=method,attempts=len(indices),
                paired_attempts=paired_count,identity_unavailable=len(indices)-paired_count,clusters=len({c for c,_ in units}),
                coverage_difference=float(np.mean([v for _,v in units])) if units else None,
                cluster_se=_cluster_se(units),length_paired_valid=len(length_units),
                length_excluded=len(indices)-len(length_units),
                length_difference=float(np.mean([v for _,v in length_units])) if length_units else None,
                length_cluster_se=_cluster_se(length_units)))
    return result,paired
