"""Write-once run identity and allocation-aware CPU settings."""
import hashlib
import json
import os
from pathlib import Path


def allocation_threads():
    if os.environ.get('NSLOTS'):return max(1,int(os.environ['NSLOTS']))
    if hasattr(os,'sched_getaffinity'):return max(1,len(os.sched_getaffinity(0)))
    return max(1,os.cpu_count() or 1)


def initialize_output(path,metadata,program):
    out=Path(path);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Run output is nonempty; choose a new output directory to preserve prior artifacts')
    payload=dict(metadata,program_sha256=hashlib.sha256(Path(program).read_bytes()).hexdigest(),
                 allocation_threads=allocation_threads())
    with (out/'run_manifest.json').open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2);f.write('\n')
    return out
