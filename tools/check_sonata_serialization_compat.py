"""Bit-exact original vs compatibility encoding, including old container torch."""
import sys,subprocess,importlib.util,json
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'models/vendor/sonata-official-fewshot-v1';OUT=ROOT/'outputs/bauhinia_ptv3_fewshot_20260912_v1'
spec=importlib.util.spec_from_file_location('compat',REPO/'sonata/serialization/hilbert.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
if '--verify' not in sys.argv:
 original=subprocess.check_output(['git','-C',str(REPO),'show','HEAD:sonata/serialization/hilbert.py'],text=True);ns={};exec(compile(original,'original_hilbert.py','exec'),ns);rng=np.random.default_rng(17);arrays={}
 for depth in [1,4,7,12,16]:
  xyz=rng.integers(0,2**depth,size=(2048,3),dtype=np.int64);expected=ns['encode'](torch.from_numpy(xyz),3,depth).numpy();got=mod.encode(torch.from_numpy(xyz),3,depth).numpy();assert np.array_equal(expected,got);arrays['xyz_'+str(depth)]=xyz;arrays['code_'+str(depth)]=expected
 np.savez_compressed(OUT/'serialization_reference.npz',**arrays);print('Modern runtime exact equivalence verified')
else:
 f=np.load(OUT/'serialization_reference.npz');count=0
 for depth in [1,4,7,12,16]:
  got=mod.encode(torch.from_numpy(f['xyz_'+str(depth)]).cuda(),3,depth).cpu().numpy();assert np.array_equal(got,f['code_'+str(depth)]);count+=len(got)
 (OUT/'serialization_compat_check.json').write_text(json.dumps(dict(passed=True,points=count,depths=[1,4,7,12,16],torch=torch.__version__,cuda=torch.version.cuda,check='bit-for-bit equal to unmodified upstream Hilbert encoding under modern torch'),indent=2));print('Old runtime CUDA exact equivalence verified',count)
