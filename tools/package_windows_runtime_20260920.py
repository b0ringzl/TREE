"""Physically copy the current runtime with conda-pack relocation support."""
import functools
import json
import re
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BUILD=Path('C:/SST_Pack_20260920_v1')
DEST=BUILD/'SpaceScan-Trees-H120-Windows/runtime'
sys.path.insert(0,str(ROOT/'.codex_build/portable_pack_tools_20260920'))
import conda_pack
from conda_pack.formats import NoArchive

BUILD.mkdir(parents=True,exist_ok=True)
if DEST.exists():
    raise RuntimeError('Never overwrite an earlier runtime build')
DEST.parent.mkdir(parents=True,exist_ok=True)
exception=None
try:
    env=conda_pack.CondaEnv.from_prefix(sys.prefix)
except conda_pack.CondaPackException as exc:
    # Conda's old setuptools 84 record was superseded by installed pip 78.1.
    # This is the only permitted metadata conflict; do not suppress other errors.
    exception=str(exc)
    if re.findall(r'^- (.+):$',exception,re.M) != ['setuptools 84.0.0']:
        raise
    import importlib.metadata
    if importlib.metadata.version('setuptools') != '80.9.0':
        raise RuntimeError('Packaging tool setuptools changed')
    env=conda_pack.CondaEnv.from_prefix(sys.prefix,ignore_missing_files=True)
env=env.exclude('**/__pycache__/*').exclude('**/*.pyc')
original_add=NoArchive._add
def physical_add(self,source,target):
    self.copy_func=functools.partial(shutil.copy2,follow_symlinks=False)
    return original_add(self,source,target)
NoArchive._add=physical_add
def bytes_add(self,source,data,target):
    path=Path(self._absolute_path(target))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)
    shutil.copystat(source,path)
NoArchive._add_bytes=bytes_add
# conda-pack 0.8.1 NoArchive.__exit__ returns self and suppresses I/O errors.
# Make packaging fail closed, and create parents for rewritten text files.
NoArchive.__exit__=lambda self,*args: False
env.pack(str(DEST),format='no-archive',verbose=False)
missing=[f.target for f in env.files if not (DEST/f.target).exists()]
if missing or not (DEST/'python.exe').exists() or not (DEST/'Scripts/conda-unpack-script.py').exists():
    raise RuntimeError('Incomplete runtime: '+str(missing[:10]))
(BUILD/'runtime_build.json').write_text(json.dumps({'files':len(env.files),'source':sys.prefix,
    'physical_copies':True,'metadata_conflict':exception,
    'resolution':'Only obsolete conda setuptools record missing files allowed; actual pip setuptools 78.1.0 files copied; active environment unchanged'},indent=2),encoding='utf-8')
print('RUNTIME_READY',DEST,flush=True)
