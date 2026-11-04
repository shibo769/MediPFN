"""Source-bound import finder for a future leased calibration worker.

Every project-local Python import must be in the frozen source map. Explicit
module-to-file mappings prevent Python path shadowing; bytes are checked before
execution. Dependency environment/stdlib modules remain outside this policy.
"""
import hashlib,importlib.abc,importlib.machinery,os,sys
from pathlib import Path

class FrozenImportError(ImportError):pass

def _digest(raw):return hashlib.sha256(raw).hexdigest()

def _inside(path,root):
 try:return path.is_relative_to(root)
 except (ValueError,OSError):return False

class BoundLoader(importlib.abc.Loader):
 def __init__(self,raw,path):self.raw,self.path=raw,path
 def create_module(self,spec):return None
 def exec_module(self,module):exec(compile(self.raw,str(self.path),'exec'),module.__dict__)

class Finder(importlib.abc.MetaPathFinder):
 def __init__(self,source_hashes,import_paths,project_root,environment_root):
  self.sources=source_hashes;self.paths=import_paths;self.project=project_root;self.env=environment_root
 def find_spec(self,fullname,path=None,target=None):
  spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
  if spec is None:
   if fullname in self.paths:raise FrozenImportError('registered project module missing: '+fullname)
   return None
  origin=getattr(spec,'origin',None)
  if not origin or origin in ('built-in','frozen'):
   if fullname in self.paths:raise FrozenImportError('registered module not a file: '+fullname)
   return spec
  file=Path(origin).resolve(strict=True)
  project=_inside(file,self.project) and not _inside(file,self.env)
  expected_path=self.paths.get(fullname)
  if expected_path is not None and file!=Path(expected_path).resolve(strict=True):raise FrozenImportError('module path shadowed: '+fullname)
  if project and expected_path is None:raise FrozenImportError('project module missing name binding: '+fullname)
  if project or expected_path is not None:
   expected=self.sources.get(str(file))
   if not expected or file.suffix!='.py':raise FrozenImportError('unregistered project import: '+fullname)
   raw=file.read_bytes()
   if _digest(raw)!=expected:raise FrozenImportError('project module bytes changed: '+fullname)
   spec.loader=BoundLoader(raw,file)
  return spec

def install(source_hashes,import_paths,project_root,environment_root):
 if not isinstance(source_hashes,dict) or not source_hashes or not isinstance(import_paths,dict):raise FrozenImportError('missing import/source map')
 required={'alpha_artifact_binding_v2','calibration_seed_proposal','calibration','check_cal_only_seed_audit_v3','mixture','data','inference','model'}
 if not required.issubset(import_paths):raise FrozenImportError('missing explicitly bound core imports')
 project=Path(project_root).resolve(strict=True);env=Path(environment_root).resolve(strict=True)
 for name,path in import_paths.items():
  p=Path(path).resolve(strict=True)
  if not _inside(p,project) or _inside(p,env) or str(p) not in source_hashes:raise FrozenImportError('invalid import path: '+name)
  if _digest(p.read_bytes())!=source_hashes[str(p)]:raise FrozenImportError('registered import changed: '+name)
  if name in sys.modules:raise FrozenImportError('project module already loaded before guard: '+name)
 finder=Finder(source_hashes,import_paths,project,env);sys.meta_path.insert(0,finder);return finder
