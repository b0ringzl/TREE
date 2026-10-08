"""Retain full runtime warnings in log, print compact progress."""
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/bauhinia_ptv3_fewshot_20260912_v1'
with (OUT/'ptv3_feature_runtime.log').open('w',encoding='utf8') as log:
 p=subprocess.Popen(['docker','exec','-w','/workspace','spacescan-ptv3-fewshot-v1','python3','-u','tools/extract_bauhinia_point_features.py','ptv3'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace')
 for line in p.stdout:
  log.write(line);log.flush()
  if line.startswith(('ptv3','Loading','Model params','Traceback','RuntimeError','ValueError')):print(line.strip(),flush=True)
 raise SystemExit(p.wait())
