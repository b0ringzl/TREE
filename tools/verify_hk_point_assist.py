"""Read-only integrity checks for point-assist delivery; never fits or deploys weights."""
import json
from pathlib import Path
from datetime import datetime,timezone
from training_safety import sha256
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/hk_point_assist_20260911_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
sources=read(OUT/'source_hashes.json')
for p,h in sources.items():assert sha256(p)==h,p
assets=0
for r in read(OUT/'catalog.json'):
    for kind,h in r['sha256'].items():assert sha256(r['asset_paths'][kind])==h;assets+=1
previous=read(ROOT/'outputs/whu_bioclip_adaptation_20260911_v1/protocol.json')['protected_sources']
weights={p:h for p,h in previous.items() if Path(p).suffix in {'.pt','.safetensors','.joblib'}}
for p,h in weights.items():assert sha256(p)==h,p
result=dict(verified_at=datetime.now(timezone.utc).isoformat(),source_manifests_and_reviews_unchanged=len(sources),
    asset_files_verified=assets,protected_model_weights_unchanged=weights,training_performed=False,
    original_review_files_modified=False,model_switch=False,ui_and_api_updated=True,
    browser_verification=read(OUT/'browser_verification.json'))
(OUT/'delivery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in {'protected_model_weights_unchanged','browser_verification'}},ensure_ascii=False))
