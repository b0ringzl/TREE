"""Allowlisted offline application export; no training or production changes."""
import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BUILD=Path('C:/SST_Pack_20260920_v1')
DEST=BUILD/'SpaceScan-Trees-H120-Windows'
APP=DEST/'app'
SOURCES=[]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def cp(source,target=None):
    source=Path(source)
    if not source.is_absolute():source=ROOT/source
    target=target or APP/source.relative_to(ROOT)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    digest=sha(source)
    if sha(target)!=digest or target.samefile(source):raise RuntimeError('Bad copy '+str(source))
    SOURCES.append({'source':str(source),'target':str(target.relative_to(DEST)),'sha256':digest})

def tree(source,target=None):
    source=ROOT/source
    target=target or APP/source.relative_to(ROOT)
    for p in source.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.log'):
            cp(p,target/p.relative_to(source))

def replace(rel,before,after):
    p=APP/rel
    text=p.read_text(encoding='utf-8')
    if before not in text:raise RuntimeError('Missing exact patch '+rel+': '+before[:60])
    write(p,text.replace(before,after))

def extract(rel,names,header):
    text=(ROOT/rel).read_text(encoding='utf-8')
    nodes={n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    write(APP/rel,header+'\n\n'+'\n\n'.join(nodes[n] for n in names)+'\n')

def main():
    if APP.exists():raise RuntimeError('New app export required')
    APP.mkdir(parents=True)
    for name in ('server.py','official_release.py','panorama.py','tree_detection.py','species_names.py','species_names.json',
                 'point_recognition.py','point_assist.py','manual_references.py','reference_library.py','release.json'):
        cp('spacescan_trees_demo_v1/'+name)
    for folder in ('spacescan_trees_demo_v1/static','spacescan_trees_demo_v1/templates','tree_species_teacher_demo/static'):
        tree(folder)
    for p in ('tree_species_teacher_demo/engine.py','tree_species_teacher_demo/server.py',
              'tree_species_demo/demo_locale.py','tree_species_demo/hk_three_species.py',
              'Lidar/tscmdl_whu_sandbox/src/tscmdl_whu/ptv2.py'):
        cp(p)
    write(APP/'Lidar/tscmdl_whu_sandbox/src/tscmdl_whu/__init__.py','from .ptv2 import PointTransformerV2Classifier\n')
    # Exact inference definitions extracted without unrelated dataset/training imports.
    extract('tools/bioclip_probe.py',['ResidualHead'],"import torch\nfrom pathlib import Path\nROOT=Path(__file__).resolve().parents[1]\nSOURCE=ROOT/'outputs/distillation/20260908_0045'")
    extract('tools/distillation_core.py',['read_json','collapse_logits'],'import json\nfrom pathlib import Path\nimport torch')
    extract('tools/training_safety.py',['sha256'],'import hashlib\nfrom pathlib import Path')
    extract('tools/rebuild_banyan_coverage_preview.py',['sample_ids'],'import numpy as np')
    exp=Path('outputs/distillation/bioclip_probe_20260908_0227')
    for name in ('classes.json','teacher_choice.json','cache_metadata.json'):
        cp(exp/name)
    protocol=read(ROOT/exp/'protocol.json')
    encoder=Path(protocol['teacher_weight'])
    write(APP/exp/'protocol.json',{'teacher_weight':encoder.relative_to(ROOT).as_posix(),'teacher_sha256':protocol['teacher_sha256']})
    selection=read(ROOT/exp/'selection_lock.json')
    portable={'selected_runs':[],'weights_sha256':{}}
    for i,row in enumerate(selection['selected_runs'],1):
        rel=f'models/original_adapted/member_{i}'
        original=Path(row['run'])/'best.pt'
        if sha(original)!=selection['weights_sha256'][row['run']]:raise RuntimeError('Original head changed')
        cp(original,APP/rel/'best.pt')
        portable['selected_runs'].append({'run':rel})
        portable['weights_sha256'][rel]=sha(original)
    write(APP/exp/'selection_lock.json',portable)
    cp('outputs/distillation/20260908_0045/aliases.json')
    tree('external_datasets/distillation_models/bioclip25')
    tree('external_datasets/distillation_runtime',APP/'vendor/bioclip')
    tree('outputs/tree_detector_benchmark_20260909_v1/runtime',APP/'vendor/detection')
    replace('tree_species_teacher_demo/engine.py','external_datasets/distillation_runtime','vendor/bioclip')
    replace('spacescan_trees_demo_v1/tree_detection.py','outputs/tree_detector_benchmark_20260909_v1/runtime','vendor/detection')
    tree('outputs/tree_detector_benchmark_20260909_v1/assets/grounding_dino_base')
    cp('outputs/tree_detector_benchmark_20260909_v1/assets/clip/ViT-B-32.pt')
    cp('hk-tree-vmms/derived/training_runs/combined_tree_segmentation/yolo11m_seg_1024_20260901_104757/weights/best.pt')
    head_dir=Path('tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120')
    dep=read(ROOT/head_dir/'deployment.json')
    for m in dep['members']:
        if sha(ROOT/head_dir/m['file'])!=m['sha256']:raise RuntimeError('H120 hash mismatch')
        cp(head_dir/m['file'])
    cp(head_dir/'labels.json')
    write(APP/head_dir/'deployment.json',{k:dep[k] for k in ('model','version','display_version','release_date','output_count',
         'encoder_sha256','temperature','ensemble','labels_sha256','validation_gate_passed','known_regressions')}
         | {'members':[{k:m[k] for k in ('file','sha256','step')} for m in dep['members']]})
    point_path=Path('spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/deployment.json')
    point=read(ROOT/point_path)
    cp(point['model']);cp(point['encoder'])
    write(APP/point_path,{k:point[k] for k in ('version','model','model_sha256','encoder','encoder_sha256','fusion_point_weight')})
    replace('tree_species_teacher_demo/engine.py',"weight = Path(protocol['teacher_weight'])","weight = ROOT / protocol['teacher_weight']")
    replace('tree_species_teacher_demo/engine.py',"path = Path(run['run']) / 'best.pt'","path = ROOT / run['run'] / 'best.pt'")
    replace('spacescan_trees_demo_v1/server.py','import io\n','import io\nimport os\n')
    replace('spacescan_trees_demo_v1/server.py',"UPSTREAM = 'http://127.0.0.1:8039'","UPSTREAM = os.environ.get('SPACESCAN_TEACHER_URL', 'http://127.0.0.1:18339')")
    write(APP/'tree_species_demo/samples.json',[])
    replace('spacescan_trees_demo_v1/templates/index.html','id="testPanorama"','id="testPanorama" hidden style="display:none"')
    replace('spacescan_trees_demo_v1/static/point-upload.js',"const stressLink=el('a','','Low-quality experiment / 低品質補償實驗');stressLink.href='/point-stress';$('cloudPanel').append(stressLink);",'// Research-only pages are not included in the portable delivery.')
    # Remove archive/sample endpoints, not real upload APIs.
    replace('spacescan_trees_demo_v1/server.py',"app.mount('/static'", "app.router.routes = [r for r in app.router.routes if getattr(r,'path','') not in {'/point-assist','/point-stress','/api/test-panorama'}]\napp.mount('/static'")
    refs_dir=Path('external_datasets/spacescan_whole_tree_references_20260910_v2')
    candidates=read(ROOT/refs_dir/'mixed_v3/accepted_manifest.json')
    refs=[]
    for r in candidates:
        if (r.get('license') not in {'cc0','cc-by','cc-by-sa'} or r.get('visual_review')!='accepted_for_reference'
            or not r.get('attribution') or not r.get('author') or not r.get('license_url') or r.get('training_eligible')
            or r.get('framing') not in {'whole_tree_or_crown','whole_tree_or_flower_leaf_detail'}):continue
        path=ROOT/refs_dir/r['file']
        if sha(path)!=r['sha256']:raise RuntimeError('Reference bytes differ '+str(path))
        cp(path);refs.append(r)
    write(APP/refs_dir/'mixed_v3/accepted_manifest.json',refs)
    write(APP/refs_dir/'mixed_v3/pool_manifest.json',refs)
    cp(ROOT/refs_dir/refs[0]['file'],APP/'tree_species_demo/test_data/hongkong_025/image.jpg')
    write(DEST/'licenses/REFERENCE_ATTRIBUTIONS.json',refs)
    for name in ('AGPL-3.0.txt','APACHE-2.0.txt','OPENCLIP-LICENSE.txt','PILLOW-LICENSE.txt'):
        cp(ROOT/'spacescan_android_sstv1/app/src/main/assets'/name,DEST/'licenses'/name)
    cp(ROOT/'external_datasets/distillation_models/bioclip25/README.md',DEST/'licenses/BIOCLIP25_MODEL_CARD.md')
    cp(ROOT/'outputs/tree_detector_benchmark_20260909_v1/assets/grounding_dino_base/README.md',DEST/'licenses/GROUNDING_DINO_MODEL_CARD.md')
    for source in (ROOT/'tools/windows_portable').iterdir():
        if source.is_file():cp(source,DEST/source.name)
    write(DEST/'THIRD_PARTY_NOTICES.md', '''# Third-party notices / 第三方说明

Private local evaluation transfer for the user's Windows computers. Not public redistribution of experimental weights; review applicable training-data rights before broader distribution.

- BioCLIP 2.5 model card declares MIT: https://huggingface.co/imageomics/bioclip-2.5-vith14 . Model card retained.
- OpenCLIP and OpenAI CLIP: MIT. Source, installed package metadata and licence notices retained in app runtime directories.
- Ultralytics detector/software: AGPL-3.0; the installed corresponding Python source and full licence text are bundled. See runtime/Lib/site-packages/ultralytics and its dist-info.
- Grounding DINO: Apache-2.0 model/software notices retained; transformers package sources and licenses retained.
- Python, PyTorch, torchvision, CUDA runtime libraries, NumPy, SciPy, scikit-learn, FastAPI, Jinja2 and other dependencies retain their original licences under runtime, Library and site-packages. NVIDIA driver is NOT bundled. CUDA libraries retain upstream redistribution restrictions.
- Reference images: only recorded CC0, CC BY or CC BY-SA entries, original bytes unchanged; author, photo page and exact licence URL are in licenses/REFERENCE_ATTRIBUTIONS.json and shown beside each picture. Preserve these notices and applicable share-alike terms. Source labels are not a botanical guarantee.
- Training photos, review databases, Google Street View images, private uploads and research scene archives are excluded. No rights in training photographs are granted by this package. H120 model limitations remain in README.md.

Full application source and inference helpers are included in app. No training is performed by launchers. Source export changes only local-path resolution, service-port configuration and omitted research/mobile links; model tensors and inference preprocessing are unchanged.
''')
    write(APP/'spacescan_trees_demo_v1/static/apps.html','''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SpaceScan Trees · Windows</title><link rel="stylesheet" href="/static/style.css"><main><h1>SpaceScan Trees</h1><p>SSTv1-H120 · 2026-09-19 · Windows offline / Windows 離線便攜版</p><p><a href="/image-only">Image-only / 單源影像 →</a></p><p><a href="/multimodal">Image + cloud / 影像＋點雲 →</a></p><p>此包不包含安卓安裝包或研究存檔。模型、硬件及许可说明请查看包内 README.md。</p><p>H120 開發集土沉香正確數 16/22→18/22，誤報 7→10；並非所有指標均提升。</p></main></html>''')
    for kind in ('apk','model','guide'):
        replace('spacescan_trees_demo_v1/official_release.py',f"'/api/official-download/{kind}'",'None')
    write(BUILD/'source_snapshot.json',SOURCES)
    manifest=[]
    for p in APP.rglob('*'):
        if p.is_file() and p.suffix in {'.pt','.safetensors','.joblib','.json','.py','.html','.js','.css'}:
            manifest.append({'file':p.relative_to(DEST).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    write(DEST/'model_manifest.json',manifest)
    write(DEST/'BUILD_INFO.json',{'date':'2026-09-20','version':dep['version'],'references':len(refs),
        'reference_species':len({r['species'] for r in refs}),'training_performed':False,'production_changed':False,
        'packaging_changes':'Relative model paths; packaged environment; port allocation; omitted research archives/Android distribution',
        'model_tensors_unchanged':True,'validation':'See external packaging acceptance report; destination PC not yet tested'})
    print(json.dumps(read(DEST/'BUILD_INFO.json'),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
