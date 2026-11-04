"""Unregistered, metadata-only proposal for 400 new calibration mechanisms.

Not a registration, collision audit, or permission to generate a table.
Training and DEV use separately frozen existing features. No test is named here.
"""
import hashlib
import json
import numpy as np

NAMESPACE = 202610283300001
LIMIT = 2**63 - 1_000_000
GROUPS = ('id', 'tree_rff', 'tree_mlp', 'rff_mlp')


def derive(seed, *indices):
    return int(np.random.SeedSequence([int(seed), *map(int, indices)]).generate_state(1, np.uint64)[0])


def build():
    tasks = []
    # One entry for each *actual* RNG call. view+0 is not a second stream.
    streams = {}
    for group in GROUPS:
        for offset in range(100):
            index = len(tasks)
            mech, obs, fold, view = [derive(NAMESPACE, index, j) % LIMIT for j in range(4)]
            population = derive(mech, 8192, 2026102601)
            labels = [('mechanism', mech), ('population', population), ('observation', obs), ('fold', fold)]
            labels += [('view%d_fold%d' % (v, f), view + 1009*v + 37*f)
                       for v in range(3) for f in range(2)]
            for label, seed in labels:
                if seed in streams:
                    raise RuntimeError('collision within proposal: ' + repr((seed, streams[seed], (index, label))))
                streams[seed] = (index, label)
            family = ('linear', 'smooth', 'heterosked', 'weak_overlap', 'covariate_shift')[offset % 5] if group == 'id' else 'combinations'
            tasks.append({'index': index, 'split': 'calibration', 'group': group,
                          'n': (200, 600, 2000)[(offset // 3) % 3], 'p': (3, 10, 30)[offset % 3],
                          'family': family, 'mechanism_seed': mech, 'observation_seed': obs,
                          'fold_seed': fold, 'view_seed': view})
    return tasks, set(streams)


def digest(streams):
    return hashlib.sha256('\n'.join(map(str, sorted(streams))).encode()).hexdigest()


if __name__ == '__main__':
    tasks, streams = build()
    print(json.dumps({'status': 'PROPOSAL_ONLY_NOT_REGISTERED', 'calibration_tasks': len(tasks),
                      'actual_streams': len(streams), 'stream_sha256': digest(streams),
                      'namespace': NAMESPACE, 'calibration_order_statistics': {'0.90': 361, '0.99': 397},
                      'random_api': 'default_rng / SeedSequence; no numpy global 32-bit seed for generated tasks'}))
