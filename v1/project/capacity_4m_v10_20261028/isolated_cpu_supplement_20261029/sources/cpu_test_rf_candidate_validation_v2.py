"""Pre-freeze checks for the specified 4M CPU test/RF candidate.

This module does not register or generate test data.  A separate, independently
reviewed freezer must verify scoped historical/parallel streams, the current
writers, licenses, and the actual successful calibration audit before publishing
any protocol.  The main entry deliberately remains disabled.
"""
import copy
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_accounted_calibration(root):
    """Bind the actual independently reviewed CPU-calibration audit."""
    root = Path(root)
    protocol = root / 'protocols/calibration_audit.json'
    directory = root / 'outputs/calibration_audit/shard_00001'
    report, record = directory / 'calibration_audit.json', directory / 'stage_record.json'
    freeze = root / 'outputs/calibration_finalize/shard_00001/alpha_calibration_freeze.json'
    expected = {
        protocol: '1615937ef1a5bade4b4f49875e83a835d1137e79ce6e4059cdfc0cc5f7ffeb43',
        report: 'e20a779de53baa4483fdd4023aa77b4990a59adaf1c7ed083a6728fa94c56d40',
        record: '807313f59579f3f623df80309801c078a0e8b16e9c9499203b5d1baa3c1c0da0',
        freeze: 'dbef5feffeaacf52eb347abda108a43b382e673aca3a319d02b46e000f54abc7',
    }
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError('calibration audit input changed: ' + str(path))
    result = subprocess.run(['qacct', '-j', '8341027'], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
    if result.returncode:
        raise ValueError('calibration audit accounting unavailable')
    fields = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ('jobnumber', 'jobname', 'owner', 'taskid', 'failed', 'exit_status'):
            if parts[0] in fields:
                raise ValueError('ambiguous audit accounting')
            fields[parts[0]] = parts[1]
    if fields != {'jobnumber': '8341027', 'jobname': 'nie4m_cpu_calaudit_v1',
                  'owner': 'shibo', 'taskid': 'undefined', 'failed': '0', 'exit_status': '0'}:
        raise ValueError('calibration audit accounting mismatch')
    doc, rec = json.loads(report.read_text()), json.loads(record.read_text())
    if (doc.get('status') != 'SCOPED_CPU_CALIBRATION_AUDIT_PASS' or doc.get('mechanisms') != 400 or
            doc.get('truth_mismatches') != 0 or doc.get('calibration_job') != 8340506 or
            doc.get('finalizer_job') != 8340929 or doc.get('freeze_sha256') != expected[freeze] or
            rec.get('job_id') != 8341027 or rec.get('manifest_sha256') != expected[protocol] or
            rec.get('result', {}).get('audit_sha256') != expected[report]):
        raise ValueError('calibration audit content mismatch')
    return {str(path): digest for path, digest in expected.items()}


def validate_candidate(test_plan, seed_cases, mapping, subset, binding):
    """Validate case identity and replay RF's preselected random streams.

    ``binding`` is the source-bound rf_seed_binding_v1 module, not a result
    supplied by a caller.  It reconstructs every learner state and root.
    This check does not establish historical or concurrent independence.
    """
    test = test_plan['cases']
    cases = seed_cases['cases']
    if len(test) != 2952 or len(cases) != 2952:
        raise ValueError('wrong number of test cases')
    if Counter(c['kind'] for c in test) != {'realx': 1872, 'stress': 720, 'id': 360}:
        raise ValueError('wrong test case composition')
    if len({c['case_id'] for c in test}) != 2952:
        raise ValueError('duplicate test case')
    if any(any(actual.get(key) != value for key, value in seed.items())
           for seed, actual in zip(cases, test)):
        raise ValueError('RF and test case identity differs')
    if mapping.get('status') != 'PROVISIONAL_MAPPING_NOT_REGISTERED':
        raise ValueError('unexpected original mapping status')
    canonical = binding.task_digest(cases)
    if canonical != mapping.get('canonical_tasks_sha256'):
        raise ValueError('original mapping case digest differs')
    if (len(subset) != 261 or list(subset) != mapping['selected_case_indices'] or
            len(set(subset)) != 261):
        raise ValueError('preselected B100 cases differ')
    # Replay a copy in memory only; no frozen file or authorization is produced.
    proposed = copy.deepcopy(mapping)
    proposed['status'] = 'FROZEN_BEFORE_TEST'
    proposed['tasks_sha256'] = canonical
    by_index = binding.validate(proposed, cases, subset)
    if len(by_index) != 2952 or proposed['states_count'] != 142056 or proposed['root_count'] != 213:
        raise ValueError('unexpected random-stream inventory')
    return {'case_count': len(cases), 'subset_count': len(subset),
            'case_digest': canonical, 'state_count': proposed['states_count'],
            'state_digest': proposed['state_digest'], 'root_count': proposed['root_count'],
            'root_digest': proposed['root_digest'], 'authorization': False}


if __name__ == '__main__':
    raise RuntimeError('candidate validation only; independent scientific review and actual protocol freezer required')
