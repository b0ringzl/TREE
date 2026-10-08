"""Run unchanged Demo image preprocessing and teacher on every screened view."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spacescan_trees_demo_v1.server import infer
OUT=ROOT/'outputs/point_assist_case_selection_20260913_v1'
target=OUT/'image_predictions.json'
if target.exists():raise RuntimeError('Refusing to overwrite a previous evaluation')
rows=json.loads((OUT/'manifest.json').read_text(encoding='utf8'))
predictions=[]
for row in rows:
    for view in row['views']:
        result=infer(Path(view['image']).read_bytes(),'street')
        rec=dict(id=row['id'],index=view['index'],image=view['image'],truth=row['species'],
                 correct=result['top3'][0]['species']==row['species'],result=result)
        predictions.append(rec)
        print(json.dumps(dict(id=row['id'],index=view['index'],correct=rec['correct'],top=result['top3'][0]['species']),ensure_ascii=False),flush=True)
        (OUT/'image_predictions.partial.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2),encoding='utf8')
target.write_text(json.dumps(predictions,ensure_ascii=False,indent=2),encoding='utf8')
