"""Strict verification of a frozen RF per-case reservation, before data use.

This independently rebuilds every 32-bit learner state and bootstrap root;
collision against historical/concurrent streams requires separate audit.
"""
import hashlib,json
import numpy as np
def digest(values):return hashlib.sha256('\n'.join(map(str,sorted(values))).encode()).hexdigest()
def task_digest(cases):return hashlib.sha256(json.dumps(cases,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def derive(seed,salt,retry):return int(np.random.SeedSequence([int(seed),int(salt),int(retry)]).generate_state(1,np.uint64)[0])
def validate(mapping,cases,subset):
    if mapping.get('status')!='FROZEN_BEFORE_TEST' or mapping.get('tasks_sha256')!=task_digest(cases):raise ValueError('unfrozen seed reservation or case binding')
    if [c['index'] for c in cases]!=list(range(len(cases))):raise ValueError('case indices')
    chosen=set(subset)
    if len(chosen)!=len(subset) or not chosen.issubset(range(len(cases))):raise ValueError('invalid bootstrap subset')
    groups={}
    for c in cases:groups.setdefault((int(c['observation_seed']),int(c['n'])),[]).append(c)
    assigned=set();all_states=set();all_roots=set();by_index={};keys=set()
    if len(mapping['rows'])!=len(groups):raise ValueError('missing seed groups')
    for row in mapping['rows']:
        key=(int(row['observation_seed']),int(row['n']))
        if key in keys or key not in groups:raise ValueError('duplicate or unknown observation group')
        keys.add(key);members=groups[key];indices=[int(c['index']) for c in members]
        if row['case_indices']!=indices or assigned.intersection(indices):raise ValueError('case group identity')
        if len(members)>1 and (any(c['kind']!='stress' for c in members) or len({(c['cluster_id'],c['rep']) for c in members})!=1 or len({c['scenario'] for c in members})!=len(members)):
            raise ValueError('unapproved observation alias')
        assigned.update(indices);obs,n=key
        retry=int(row['initial_retry'])
        if retry<0:raise ValueError('invalid initial retry')
        initial=derive(obs,mapping['salt_initial'],retry)%(2**31-1)
        if int(row['initial_seed'])!=initial:raise ValueError('initial seed formula')
        states=[initial+10*f+j for f in (0,1) for j in (0,1,2)]
        root=None
        if chosen.intersection(indices):
            if not {'bootstrap_root','bootstrap_retry'}.issubset(row):raise ValueError('missing B100 root')
            retry=int(row['bootstrap_retry'])
            if retry<0:raise ValueError('invalid B100 retry')
            root=derive(obs,mapping['salt_bootstrap'],retry)
            if int(row['bootstrap_root'])!=root or root in all_roots:raise ValueError('B100 root formula/collision')
            all_roots.add(root);rng=np.random.default_rng(root)
            for _ in range(100):
                rng.integers(n,size=n);rng.permutation(n);s=int(rng.integers(2**31-1))
                states.extend(s+10*f+j for f in (0,1) for j in (0,1,2))
        elif 'bootstrap_root' in row or 'bootstrap_retry' in row:raise ValueError('undeclared B100 root')
        if len(states)!=len(set(states)) or all_states.intersection(states):raise ValueError('32-bit RF state collision')
        all_states.update(states)
        for i in indices:by_index[i]=(initial,root)
    if assigned!=set(range(len(cases))) or all_states.intersection(all_roots):raise ValueError('missing cases or cross-domain collisions')
    if (mapping['case_groups']!=len(groups) or mapping['selected_case_references']!=len(chosen)
        or mapping['states_count']!=len(all_states) or mapping['state_digest']!=digest(all_states)
        or mapping['root_count']!=len(all_roots) or mapping['root_digest']!=digest(all_roots)):
        raise ValueError('reservation digest/count mismatch')
    return by_index
