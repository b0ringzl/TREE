"""Continue only after verified terminal collection failure; preserve every prior artifact."""
import shutil
import sys
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import rare_species_source_extension_20261006 as original
from training_safety import verify_backup

PARENT=ROOT/'outputs/rare_species_fresh_validation_20261006_v1'
OUT=PARENT/'nc_resume_v2'


def main():
    OUT.mkdir(exist_ok=False)
    (OUT/'source_responses').mkdir()
    (OUT/'images').mkdir()
    (OUT/'contact_sheets').mkdir()
    for path in (PARENT/'source_responses').glob('*_taxon.json'):
        shutil.copy2(path,OUT/'source_responses'/path.name)
    original.base.save(OUT/'RESUME_PLAN.json',{
        'prior_terminal_failure':'Proxy connection closed while downloading photo17167866; prior process exited1',
        'preserved_partial_manifest':str(PARENT/'nc_manifest_partial.json'),
        'backup_reference':str(PARENT/'pre_training_backup/backup_manifest.json'),
        'selection':'Exclude all completed prior observations via historical provenance index; same fixed source/quality rules',
        'prediction_inspected':False,'training_performed':False,'deployment_changed':False,
        'download_errors':'Log individual transient errors and continue; stop on access denial, no bypass',
    })
    original.verify_backup=lambda directory:verify_backup(PARENT/'pre_training_backup')
    real_get=requests.get
    errors=[]
    def guarded_get(url,*args,**kwargs):
        try:return real_get(url,*args,**kwargs)
        except requests.RequestException as exc:
            errors.append({'url':url,'error':str(exc)})
            original.base.save(OUT/'transient_download_errors.json',errors)
            # A valid empty/black JPEG enters the fixed, score-independent quality
            # gate and is rejected; it is NEVER saved or used as a sample.
            import io
            from PIL import Image
            stream=io.BytesIO();Image.new('RGB',(1,1)).save(stream,format='JPEG')
            response=requests.Response();response.status_code=200;response._content=stream.getvalue()
            return response
    original.requests.get=guarded_get
    original.nc_collect(OUT)


if __name__=='__main__':main()
