"""Aggregate preregistered CPU cold timing with explicit failures/hardware."""
import hashlib,json,math,statistics,re

def _digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def summarize(records,expected):
 keys=[(x['case_index'],x['method']) for x in expected]
 if len(keys)!=len(set(keys)) or len(records)!=len(keys):raise ValueError('timing denominator or duplicate case')
 wanted={(x['case_index'],x['method']):x for x in expected};groups={};seen=set()
 for r in records:
  k=(r['case_index'],r['method'])
  if k in seen or k not in wanted:raise ValueError('unknown/duplicate timing case')
  seen.add(k);e=wanted[k]
  if r.get('input_table_sha256')!=e['table_sha256'] or r.get('kind')!=e.get('kind') or r.get('scenario')!=e.get('scenario'):raise ValueError('timing case identity')
  seconds=r.get('seconds')
  if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0 or type(r.get('ok')) is not bool:raise ValueError('timing status/time')
  h=r['hardware'];model=h.get('cpu_model');threads=h.get('cpu_threads')
  if h.get('device')!='cpu' or not isinstance(model,str) or not model:raise ValueError('CPU hardware identity')
  if r['ok']:
   if type(threads) is not int or threads<1:raise ValueError('CPU thread identity')
   iv=json.loads(r['interval_json'])
   if not all(math.isfinite(iv[k]) for k in ('lo','hi')) or iv['lo']>iv['hi']:raise ValueError('interval')
   for name,dtypes,shape in (('observed',('float32','float64'),[e['n'],e['p']+3]),('fold',('int8','int16','int32','int64','uint8','uint16','uint32','uint64'),[e['n']])):
    item=r[name];registered=e[name]
    if item!=registered or not re.fullmatch(r'[0-9a-f]{64}',item.get('sha256','')) or item.get('dtype') not in dtypes or item.get('shape')!=shape:raise ValueError('input/fold identity')
  elif not r.get('error_type') or 'error' not in r:raise ValueError('unrecorded failure')
  key=(r['method'],json.dumps(h,sort_keys=True,separators=(',',':')));g=groups.setdefault(key,{'successful':[],'failures':0})
  if r['ok']:g['successful'].append(float(seconds))
  else:g['failures']+=1
 if seen!=set(keys):raise ValueError('missing timing cases')
 output=[]
 for (method,hw),v in sorted(groups.items()):
  values=v['successful'];output.append({'method':method,'hardware':json.loads(hw),'attempts':len(values)+v['failures'],'failures':v['failures'],'successes':len(values),'median_seconds':statistics.median(values) if values else None})
 return {'status':'CPU_COLD_TIMING_SUMMARY_NOT_INDEPENDENTLY_AUDITED','records_sha256':_digest(records),'expected_sha256':_digest(expected),'groups':output,'scope':'read existing table and hash; load/hash method-required models; folds/features; CPU inference; interval JSON; excludes process/module startup and simulation'}
