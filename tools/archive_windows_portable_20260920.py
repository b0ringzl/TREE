"""Versioned ZIP64 delivery and whole-archive integrity verification."""
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=Path('C:/SST_Pack_20260920_v1/SpaceScan-Trees-H120-Windows')
OUT=ROOT/'deliverables/SpaceScan_Trees_Windows_H120_20260920'
AUDIT=ROOT/'outputs/windows_portable_acceptance_20260920_v1'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    verified=json.loads((AUDIT/'VERIFIED.json').read_text(encoding='utf-8'))
    if not verified['passed']:raise RuntimeError('Portable tests not passed')
    if OUT.exists():raise RuntimeError('New delivery directory required')
    OUT.mkdir(parents=True)
    paths=[p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and
           p.suffix!='.pyc' and 'user_data' not in p.relative_to(PACKAGE).parts and not
           (p.parent==PACKAGE/'runtime' and p.name.startswith('tmp') and p.stat().st_size==0)]
    print('HASHING',len(paths),'files',flush=True)
    entries=[]
    for i,p in enumerate(paths):
        entries.append({'path':p.relative_to(PACKAGE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
        if i%5000==0:print('HASHED',i,flush=True)
    manifest=PACKAGE/'package_manifest.json'
    manifest.write_text(json.dumps({'files':entries,'mutable_after_first_launch':['runtime prefixes','user_data','manual reference uploads'],
        'verification':'Hashes represent pristine extracted package before first launch'},indent=2),encoding='utf-8')
    paths.append(manifest)
    archive=OUT/(PACKAGE.name+'.zip')
    print('ZIP',flush=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for i,p in enumerate(paths):
            z.write(p,PACKAGE.name+'/'+p.relative_to(PACKAGE).as_posix())
            if i%5000==0:print('ZIPPED',i,flush=True)
    print('CRC_CHECK',flush=True)
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('Archive corrupted: '+bad)
        if len(z.namelist())!=len(set(z.namelist())):raise RuntimeError('Duplicate archive member')
    result={'file':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
            'unpacked_bytes':sum(p.stat().st_size for p in paths),'files':len(paths),'zip_crc_passed':True,
            'regression_passed':True,'destination_pc_tested':False}
    (OUT/'DELIVERY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'SHA256.txt').write_text(result['sha256']+'  '+archive.name+'\n',encoding='ascii')
    shutil.copy2(PACKAGE/'README.md',OUT/'使用说明.md')
    shutil.copy2(AUDIT/'VERIFIED.json',OUT/'迁移验收记录.json')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
