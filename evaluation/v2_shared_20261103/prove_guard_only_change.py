"""Show that inference math is unchanged from the already submitted fixture."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).parent
OLD=ROOT/'releases/e53991a2f636ba45/routes.py'
NEW=ROOT/'routes.py'


def names(tree):
    result={}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            key=node.name
        elif isinstance(node,ast.Assign):
            key='assign:'+','.join(ast.unparse(t) for t in node.targets)
        else:key=ast.dump(node)
        result[key]=ast.dump(node,include_attributes=False)
    return result


before,after=names(ast.parse(OLD.read_text())),names(ast.parse(NEW.read_text()))
changed=sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))
assert changed==['assign:EXPECTED_RELEASES','verify_assets'],changed
payload=dict(status='PASS_ADMISSION_GUARD_ONLY_CHANGE',old_route_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(),
             hardened_route_sha256=hashlib.sha256(NEW.read_bytes()).hexdigest(),changed_ast_nodes=changed,
             unchanged='All route classes, model loading/prediction helpers, numerical feature/head/calibration logic and route ordering')
output=ROOT/'inference_guard_hardening.json'
output.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
print(json.dumps(payload))
