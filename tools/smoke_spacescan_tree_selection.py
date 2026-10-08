"""Real local inference smoke, not an accuracy benchmark or a training run."""
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spacescan_trees_demo_v1 import tree_detection as td


def main():
    out = ROOT/'outputs'/('spacescan_tree_selection_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    manifest = json.loads((ROOT/'outputs/tree_detector_benchmark_20260909_v1/manifest.json').read_text())
    bridge = json.loads((ROOT/'outputs/vmms20m_teacher_20260909_v1/frames/hewentian_pano__002357/result.json').read_text())
    cases = [('palms', Path(manifest[0]['image'])), ('bridge', Path(bridge['source'])),
             ('roof_crop',ROOT/'outputs/vmms20m_teacher_20260909_v1/frames/hewentian_pano__002357/000_original.jpg'),
             ('botanical', ROOT/'tree_species_demo/test_data/hongkong_025/image.jpg')]
    summary = []
    for name, path in cases:
        data=path.read_bytes();result=td.detect(data)
        original=td.decode(data);scale=min(1400/original.width,700/original.height)
        canvas=original.resize((round(original.width*scale),round(original.height*scale)))
        draw=ImageDraw.Draw(canvas)
        for row in result['candidates']:
            box=[v*s*scale for v,s in zip(row['box'],[original.width,original.height]*2)]
            draw.rectangle(box,outline='#ffaa00',width=3)
            draw.text((box[0],box[1]),f"{row['id']} {row['score']:.3f}",fill='red')
            crop,_,_=td.selected(data,row['token']);(out/f"{name}_{row['id']}.jpg").write_bytes(crop)
            row.pop('thumbnail');row.pop('token')
        canvas.save(out/f'{name}_overlay.jpg')
        (out/f'{name}.json').write_text(json.dumps({'source':str(path),**result},indent=2),encoding='utf8')
        rec={'case':name,'action':result['action'],'candidates':len(result['candidates']),
             'suppressed':result['suppressed'],'seconds':result['seconds']}
        summary.append(rec);print(rec,flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(out,flush=True)


if __name__=='__main__':main()
