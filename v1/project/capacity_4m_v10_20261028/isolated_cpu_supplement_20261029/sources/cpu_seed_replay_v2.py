"""Replay the named SCM/view seed schedule; unsupported schemas fail closed.

This does not enumerate other source implementations or establish historical
independence. It must be bound to the actual case-generating code.
"""
import hashlib
from pathlib import Path
import numpy as np

DATA_SHA="eb24a9c577c24b8cdf99b3ec5cc72b1e660a0930bf016afbf5c93b679707ef10"
LOADER_SHA="36d8c97759e3d688cdd19eca5dc088910189afd8fb b faa29a36bfd08e61980e4".replace(" ","")
_sources_verified=False

def verify_sources(protocol):
 global _sources_verified
 _sources_verified=False
 paths=protocol["project_import_paths"]; specs=protocol["stages"]["evaluation"]["science_imports"]; sources=protocol["source_sha256"]
 loader=[x for x in specs if x["name"]=="cpu_case_loader"]
 if len(loader)!=1:raise ValueError("realX loader binding missing")
 for path,expected in ((paths["data"],DATA_SHA),(loader[0]["path"],LOADER_SHA)):
  p=Path(path).resolve(strict=True)
  if sources.get(str(p))!=expected or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError("RNG implementation source changed")
 _sources_verified=True

def labeled_streams(case):
 if case.get('seed_schema')!='mediation_scm_views_v1':raise ValueError('unsupported RNG schema')
 is_realx=case.get('kind')=='realx'
 if is_realx:
  if not _sources_verified or case.get('population_rng_used') is not False:raise ValueError('realX requires source-bound external population')
 elif case.get('population_rng_used',True) is not True:raise ValueError('only realX uses external population')
 mech=int(case['mechanism_seed']);obs=int(case['observation_seed']);fold=int(case['fold_seed']);view=int(case['view_seed'])
 if min(mech,obs,fold,view)<0:raise ValueError('negative seed')
 streams={'mechanism':mech,'observation':obs,'fold':fold}
 population=int(np.random.SeedSequence([mech,8192,2026102601]).generate_state(1,np.uint64)[0])
 if 'population_seed_derived' in case and int(case['population_seed_derived'])!=population:raise ValueError('population seed mismatch')
 # Even with external X, source computes this number but does not initialize
 # a population RNG. Reserve it conservatively as a derived, unused seed.
 streams['population_derived_unused' if is_realx else 'population']=population
 for v in range(3):
  for f in range(2):streams['view%d_fold%d'%(v,f)]=view+1009*v+37*f
 if len(streams)!=len(set(streams.values())):raise ValueError('within-case RNG collision')
 return streams

def actual_streams(case):return sorted(labeled_streams(case).values())
