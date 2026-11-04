"""Build a non-authorizing CPU testbank protocol candidate in memory.

The caller must independently verify all source and input bytes.  This module
does not freeze, write, submit, release or generate observations.  The future
freezer must additionally validate actual seed history, concurrent writers,
permissions and the candidate against the current scheduler before use.
"""
import copy
import hashlib
import json
from collections import Counter
from pathlib import PurePosixPath

AUDIT_PROTOCOL = '1615937ef1a5bade4b4f49875e83a835d1137e79ce6e4059cdfc0cc5f7ffeb43'
AUDIT_REPORT = 'e20a779de53baa4483fdd4023aa77b4990a59adaf1c7ed083a6728fa94c56d40'
AUDIT_RECORD = '807313f59579f3f623df80309801c078a0e8b16e9c9499203b5d1baa3c1c0da0'
LEVELS_FREEZE = 'dbef5feffeaacf52eb347abda108a43b382e673aca3a319d02b46e000f54abc7'
TEST_PLAN = 'a0d5d18bc72af2dd03895f7d72a36f70cf266f84b34203b52e7e1f48fc912980'


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def assemble(base, base_bytes, test_bytes, *, source_paths, source_hashes,
             seed_cases_path, seed_cases_sha, mapping_path, mapping_sha,
             x_provenance_path, x_provenance_sha, x_npz_path, x_npz_sha,
             evidence_paths, bootstrap_indices):
    """Return a candidate; external data and scheduler checks remain necessary."""
    if digest_bytes(base_bytes) != AUDIT_PROTOCOL or json.loads(base_bytes) != base:
        raise ValueError('audit protocol differs from approved bytes')
    if digest_bytes(test_bytes) != TEST_PLAN:
        raise ValueError('test case plan drift')
    cases = json.loads(test_bytes)['cases']
    if len(cases) != 2952 or Counter(x['kind'] for x in cases) != {'realx': 1872, 'stress': 720, 'id': 360}:
        raise ValueError('unexpected case composition')
    ids = [x['case_id'] for x in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate test case identity')
    if (len(bootstrap_indices) != 261 or len(set(bootstrap_indices)) != 261 or
            any(type(i) is not int or i < 0 or i >= 2952 for i in bootstrap_indices)):
        raise ValueError('invalid fixed bootstrap subset')
    names = ('workflow', 'launcher', 'wrapper', 'imports', 'loader', 'rf_core', 'rf_binding', 'testbank_worker')
    if set(source_paths) != set(names) or set(source_hashes) != set(names):
        raise ValueError('exact testbank source bindings required')
    if any(not PurePosixPath(source_paths[k]).is_absolute() or len(source_hashes[k]) != 64 for k in names):
        raise ValueError('invalid source binding')
    root = PurePosixPath(base['output_root'])
    p = copy.deepcopy(base)
    p['status'] = 'INCOMPLETE_TESTBANK_CANDIDATE_NOT_AUTHORIZATION'
    p['authorization'] = False
    for key in names:
        p['source_sha256'][source_paths[key]] = source_hashes[key]
    p['science_imports_source'] = source_paths['imports']
    p['project_import_paths']['cpu_case_loader'] = source_paths['loader']
    p['project_import_paths']['rf_supplement_core'] = source_paths['rf_core']
    p['stages']['evaluation']['science_imports'] = [{'name': 'cpu_case_loader', 'path': source_paths['loader']}]
    p['stages']['calibration_audit']['job_id'] = 8341027
    report = str(root/'outputs/calibration_audit/shard_00001/calibration_audit.json')
    record = str(root/'outputs/calibration_audit/shard_00001/stage_record.json')
    p['input_sha256'].update({report: AUDIT_REPORT, record: AUDIT_RECORD,
                              seed_cases_path: seed_cases_sha, mapping_path: mapping_sha,
                              x_provenance_path: x_provenance_sha, x_npz_path: x_npz_sha})
    if p['case_plans']['test']['sha256'] != TEST_PLAN:
        raise ValueError('base test plan differs')
    freeze = p['model_assets']['levels_freeze']
    if p['artifact_sha256'].get(freeze) != LEVELS_FREEZE:
        raise ValueError('calibration freeze differs')
    if not evidence_paths:
        raise ValueError('review evidence bindings required')
    p['input_sha256'].update(evidence_paths)
    p['rf_seed_cases_path'], p['rf_seed_mapping_path'] = seed_cases_path, mapping_path
    p['covariate_root'] = str(PurePosixPath(x_npz_path).parent)
    p['rf_bootstrap_repeats'] = 100
    p['rf_bootstrap_indices'] = list(bootstrap_indices)
    p['stage_receipts']['testbank'] = str(root/'receipts/testbank.json')
    shards = list(range(1, 297))
    p['stages']['testbank'] = {
        'worker_source': source_paths['testbank_worker'], 'jobname': 'nie4m_cpu_testbank_v1',
        'kind': 'array', 'shards': shards, 'pe_slots': 4,
        'qsub_resources': 'no_gpu=TRUE,mem_per_core=4G,h_rt=24:00:00',
        'expected_qstat_resources': {'no_gpu': 'TRUE', 'mem_per_core': '4G', 'h_rt': '86400'},
        'science_imports': [{'name': 'cpu_case_loader', 'path': source_paths['loader']},
                            {'name': 'rf_supplement_core', 'path': source_paths['rf_core']}],
        'input_paths': [p['case_plans']['test']['path'], seed_cases_path, mapping_path,
                        x_provenance_path, x_npz_path] + sorted(evidence_paths),
        'predecessors': [{'job': 8341027, 'jobname': 'nie4m_cpu_calaudit_v1',
                          'tasks': [0], 'outputs': {report: AUDIT_REPORT, record: AUDIT_RECORD}}],
        'case_ids_by_shard': {str(i): ids[(i-1)*10:i*10] for i in shards},
    }
    if any(not group for group in p['stages']['testbank']['case_ids_by_shard'].values()):
        raise ValueError('empty testbank shard')
    p['required_missing'] = ['Independent final scoped seed/writer review and actual scheduler reconciliation.',
                             'Approved source and license bindings and full import-closure test.',
                             'Outcome-blind 261-index bootstrap schedule and separately verified frozen mapping.',
                             'Independent review and a source-bound freezer; this is not a frozen protocol.']
    return p


if __name__ == '__main__':
    raise RuntimeError('candidate assembly only; no test protocol or registration may be written')
