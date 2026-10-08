"""Paired, inference-only diagnostic of the reported royal-palm confusion."""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spacescan_trees_demo_v1 import panorama, tree_detection as td, server


def main():
    out=ROOT/'outputs'/('spacescan_panorama_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    old=json.loads((ROOT/'outputs/spacescan_tree_selection_20260910_152011/palms.json').read_text())
    source=Path(old['source']);data=source.read_bytes();im=td.decode(data)
    row=next(r for r in old['candidates'] if r['id']==4)
    box=row['crop_box'];same=[x/s for x,s in zip(box,[im.width,im.height]*2)]
    # Agent-drawn diagnostic ROI, not a replacement user label or training record.
    extended=[.16,.045,.32,.555]
    variants=[]
    for prefix,roi in [('same',same),('full_trunk',extended)]:
        rect=[round(v*s) for v,s in zip(roi,[im.width,im.height]*2)]
        variants.append((prefix+'_raw',im.crop(rect),{'source_region':roi,'rectified':False}))
        result,spec=panorama.perspective_for_region(im,roi)
        variants.append((prefix+'_perspective',result,{'source_region':roi,'rectified':True,'projection_view':spec}))
    summary=[];canvas=Image.new('RGB',(1600,950),'#f2f5ef');draw=ImageDraw.Draw(canvas)
    for i,(name,image,meta) in enumerate(variants):
        encoded=td.jpeg(image);(out/(name+'.jpg')).write_bytes(encoded)
        result=server.infer(encoded,'street');result.pop('model_input',None)
        for r in result['top3']:r.pop('references',None)
        record={'variant':name,'source_sha256':hashlib.sha256(data).hexdigest(),
                'crop_sha256':hashlib.sha256(encoded).hexdigest(),'source':str(source),**meta,**result}
        (out/(name+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
        summary.append(record)
        tile=ImageOps.contain(image,(380,780));x=i*400+(400-tile.width)//2
        canvas.paste(tile,(x,65));draw.text((i*400+10,20),name,fill='black')
        for j,r in enumerate(result['top3']):draw.text((i*400+10,850+j*25),f"{r['species']} {r['score']:.4f}",fill='black')
        print(name,[(r['species'],round(r['score'],4)) for r in result['top3']],flush=True)
    canvas.save(out/'comparison.jpg',quality=95)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(out,flush=True)


if __name__=='__main__':main()
