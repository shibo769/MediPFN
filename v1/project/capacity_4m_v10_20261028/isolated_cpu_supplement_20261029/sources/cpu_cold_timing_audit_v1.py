"""Independent count/median audit for CPU cold timing records."""
import hashlib,json,math,statistics,re

def _digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def audit(records,expected,candidate):
 if candidate.get('records_sha256')!=_digest(records) or candidate.get('expected_sha256')!=_digest(expected):raise ValueError('timing evidence digest')
 wanted={(x['case_index'],x['method']):x for x in expected}
 if len(wanted)!=len(expected) or len(records)!=len(expected):raise ValueError('timing count')
 found=set();groups={}
 for row in records:
  key=(row['case_index'],row['method'])
  if key not in wanted or key in found:raise ValueError('timing identity')
  found.add(key);case=wanted[key]
  if row.get('input_table_sha256')!=case['table_sha256'] or row.get('kind')!=case.get('kind') or row.get('scenario')!=case.get('scenario'):raise ValueError('case changed')
  t=row['seconds'];h=row['hardware']
  if type(t) not in (int,float) or not math.isfinite(t) or t<0 or type(row['ok']) is not bool or h.get('device')!='cpu' or not isinstance(h.get('cpu_model'),str) or not h['cpu_model']:raise ValueError('time/hardware')
  if any(value is not None and (type(value) is not int or value<1) for value in (h.get('allocated_slots'),h.get('affinity_cpu_count'))):raise ValueError('resource count')
  if row['ok']:
   if type(h.get('cpu_threads')) is not int or h['cpu_threads']<1:raise ValueError('thread count')
   iv=json.loads(row['interval_json'])
   if not all(math.isfinite(iv[k]) for k in ('lo','hi')) or iv['lo']>iv['hi']:raise ValueError('interval')
   for name,dtypes,shape in (('observed',('float32','float64'),[case['n'],case['p']+3]),('fold',('int8','int16','int32','int64','uint8','uint16','uint32','uint64'),[case['n']])):
    x=row[name]
    if x!=case[name] or not re.fullmatch(r'[0-9a-f]{64}',x.get('sha256','')) or x.get('dtype') not in dtypes or x.get('shape')!=shape:raise ValueError('observation/fold')
  elif not row.get('error_type') or 'error' not in row:raise ValueError('unrecorded failure')
  g=groups.setdefault((row['method'],json.dumps(h,sort_keys=True,separators=(',',':'))),[]);g.append(row)
 if found!=set(wanted):raise ValueError('missing case')
 out=[]
 for (method,hw),rows in sorted(groups.items()):
  values=[float(x['seconds']) for x in rows if x['ok']];out.append({'method':method,'hardware':json.loads(hw),'attempts':len(rows),'failures':len(rows)-len(values),'successes':len(values),'median_seconds':statistics.median(values) if values else None})
 if out!=candidate['groups']:raise ValueError('timing summary mismatch')
 return {'status':'CPU_COLD_TIMING_RECORDED_IDENTITY_AND_AGGREGATE_PASS','records_sha256':_digest(records),'cases':len(records)}
