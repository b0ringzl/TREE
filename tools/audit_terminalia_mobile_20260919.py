"""Read-only inference comparison. No fitting or deployed-file changes."""
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'outputs/android_toolchain_20260918_v1/python_deps'))
import numpy as np
import onnxruntime as ort
import requests
from PIL import Image, ImageOps

PHOTO = Path('C:/Users/25041725G/Desktop/验证/微信图片_20260916084639_130_2.jpg')
ASSETS = ROOT / 'spacescan_android/app/src/main/assets'
DELIVERY = ROOT / 'deliverables/SpaceScan_Trees_Android_Offline_v0.1_20260918'
OUT = ROOT / 'outputs' / ('terminalia_mobile_diagnosis_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
TARGET = 'Terminalia mantaly'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(im, fmt='PNG', **kwargs):
    b = io.BytesIO()
    im.save(b, format=fmt, **kwargs)
    return b.getvalue()


def center224(im):
    w, h = im.size
    rw, rh = (224, 224*h//w) if w <= h else (224*w//h, 224)
    scaled = im.resize((rw, rh), Image.Resampling.BICUBIC)
    left, top = round((rw-224)/2), round((rh-224)/2)
    return scaled.crop((left, top, left+224, top+224))


def ranked(p, names):
    order = np.argsort(-np.asarray(p))
    idx = names.index(TARGET)
    return {'top5': [{'species': names[i], 'score': float(p[i])} for i in order[:5]],
            'target_rank': int(np.where(order == idx)[0][0])+1, 'target_score': float(p[idx])}


def main():
    OUT.mkdir(exist_ok=False)
    manifest = json.loads((ASSETS/'model_manifest.json').read_text(encoding='utf-8'))
    species = json.loads((ASSETS/'species.json').read_text(encoding='utf-8'))
    labels = [r['scientific_name'] for r in species]
    onnx = (ASSETS/'tree_model.onnx').read_bytes()
    assert digest(onnx) == manifest['model_sha256']
    with zipfile.ZipFile(DELIVERY/'SpaceScan-Trees-Offline-v0.1.apk') as apk:
        assert digest(apk.read('assets/tree_model.onnx')) == digest(onnx)
        assert apk.read('assets/species.json') == (ASSETS/'species.json').read_bytes()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(onnx, sess_options=opts, providers=['CPUExecutionProvider'])
    raw = PHOTO.read_bytes()
    with Image.open(io.BytesIO(raw)) as source:
        orientation = source.getexif().get(274, 1)
        im = ImageOps.exif_transpose(source).convert('RGB')
    padded = ImageOps.pad(im, (min(1024,max(im.size)),)*2,
                          method=Image.Resampling.BICUBIC, color=(128,128,128))
    # Desktop prepare() encodes JPEG before the teacher decodes/resizes it.
    desktop224 = center224(Image.open(io.BytesIO(encoded(padded,'JPEG',quality=95))).convert('RGB'))
    mobile224 = center224(im)
    with Image.open(io.BytesIO(raw)) as source:
        source.draft('RGB', (1536,2048))
        reduced = ImageOps.exif_transpose(source).convert('RGB')
    inputs = {'mobile_center_native_decode': mobile224,
              'mobile_center_half_decode_proxy': center224(reduced),
              'desktop_padded_pixels': desktop224}
    results = {'source_path':str(PHOTO),'source_sha256':digest(raw),'source_size':list(im.size),
               'exif_orientation':orientation,'mobile_manifest':manifest,'apk_assets_verified':True,
               'notes':['Same APK ONNX/labels tested on CPU; not a capture of the user phone.',
                        'Pillow half-size JPEG decoding approximates Android BitmapFactory; exact phone pixels may differ.',
                        'This diagnostic photo is not an independent accuracy benchmark.',
                        'No training, model switching or deployed file edits.'], 'tests':{}}
    for name, pixels in inputs.items():
        pixels.save(OUT/(name+'.png'))
        x=np.asarray(pixels,dtype=np.float32).transpose(2,0,1)[None]/255
        probs=session.run(None,{'image':np.ascontiguousarray(x)})[0][0]
        results['tests']['mobile_'+name]=ranked(probs,labels)
    response=requests.post('http://127.0.0.1:8043/api/predict',
        files={'image':('photo.jpg',raw,'image/jpeg')},
        data={'scenario':'street','source':'image','projection':'photo'},timeout=180)
    response.raise_for_status()
    desktop=response.json()
    distribution=desktop['image_distribution']
    results['desktop_version']=desktop['image_model_version']
    results['tests']['desktop_normal_full_photo']=ranked(list(distribution.values()),list(distribution))
    results['tests']['desktop_normal_full_photo']['api_top3']=desktop['top3']
    # Hold pixels fixed to separate model behavior from center cropping.
    for name in ['mobile_center_native_decode','desktop_padded_pixels']:
        response=requests.post('http://127.0.0.1:8039/api/predict',
            files={'image':('input.png',encoded(inputs[name]),'image/png')},data={'mode':'sstv1'},timeout=180)
        response.raise_for_status()
        data=response.json()['sstv1_distribution']
        results['tests']['teacher_same_pixels_'+name]=ranked(list(data.values()),list(data))
    (OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    concise={k:{key:value for key,value in v.items() if key!='api_top3'} for k,v in results['tests'].items()}
    print(json.dumps({'output':str(OUT),'photo_size':im.size,'desktop_version':results['desktop_version'],
                      'apk_verified':True,'tests':concise},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
