"""Fixed-cache baseline / real BoT-SORT + CLIP regression; no training.

No unbiased identity ground truth exists for these jobs. Structural statistics
are diagnostics, NOT IDF1, HOTA or species accuracy. Never auto-promote based
only on fewer tracks or greater detection retention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spacescan_trees_demo_v1 import corridor_botsort, panorama, tree_detection, vmms_corridor


def signature(row):
    return (int(row["sample_index"]),tuple(round(float(v),6) for v in row["box"]))


def metrics(tracks, frames):
    observed=set()
    selected=[]
    duplicates=0
    for track in tracks:
        ids=[row["sample_index"] for row in track["observations"]]
        duplicates+=len(ids)-len(set(ids))
        observed.update(signature(row) for row in track["observations"])
        selected.append([row["sample_index"]+1 for row in vmms_corridor._select_views(track)])
    return {"stable_candidates":len(tracks),"stable_observations":len(observed),
            "detections":sum(len(f["candidates"]) for f in frames),
            "duplicate_frame_assignments":duplicates,"selected_frame_numbers":selected,
            "source_fragments_reconnected":sum(t.get("merged_fragment_count",0) for t in tracks)}


def reviewed_negatives(tracks):
    old=ROOT/'spacescan_trees_demo_v1/runtime/corridor_jobs/5e5b5e4622d341e3afa091eead3006e0'
    diagnostic=json.loads((old/'tracking_diagnostics.json').read_text(encoding='utf-8'))
    result=json.loads((old/'result.json').read_text(encoding='utf-8'))
    cases=[]
    memberships=[{signature(r) for r in t['observations']} for t in tracks]
    for rank,pairs in ((1,((0,1),(1,2),(0,2))),(2,((0,1),(1,2))),(4,((0,2),(1,2)))):
        observations=diagnostic['stable_tracks'][rank-1]['observations']
        refs=[signature(next(row for row in observations if row['sample_index']==int(v['chainage_m'])))
              for v in result['tracks'][rank-1]['views']]
        for a,b in pairs:
            cases.append({'old_tree':rank,'views':[a+1,b+1],
                          'frames':[refs[a][0]+1,refs[b][0]+1],
                          'wrongly_linked':any({refs[a],refs[b]}<=rows for rows in memberships)})
    return cases


class Images:
    def __init__(self,result):
        route=result['route']
        self.inspection=vmms_corridor.inspect_panorama_folder(route['folder'])
        self.current=None
        self.image=None
    def source(self,frame):
        path=self.inspection['_images'][round(frame['chainage_m'])]['path']
        if frame['frame_id']!=vmms_corridor._frame_id(path,round(frame['chainage_m'])+1):
            raise RuntimeError('Source frame mapping mismatch')
        return path
    def load(self,frame):
        if self.current!=frame['sample_index']:
            with Image.open(self.source(frame)) as source:
                self.image=ImageOps.exif_transpose(source).convert('RGB')
            self.current=frame['sample_index']
        return self.image
    def __call__(self,frame,spec):
        rendered=panorama.render(self.load(frame),spec,max_edge=640)
        return np.asarray(rendered)[:,:,::-1].copy() # BGR for upstream GMC


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    dest=args.output.resolve()
    dest.mkdir(parents=True,exist_ok=False)
    cases=[('campus_reviewed','5e5b5e4622d341e3afa091eead3006e0',
            ROOT/'outputs/corridor_identity_regression_20261003_v2/tracking_diagnostics.json'),
           ('tsim_sha_tsui','f50f31df7c3041709df947539bae69a3',
            ROOT/'spacescan_trees_demo_v1/runtime/corridor_jobs/f50f31df7c3041709df947539bae69a3/tracking_diagnostics.json')]
    report={'purpose':'Fixed-detection regression, not an unbiased accuracy benchmark',
            'baseline_revision':vmms_corridor.TRACKING_REVISION,'candidate_revision':corridor_botsort.REVISION,
            'candidate_config':corridor_botsort.CONFIG,'training_performed':False,'weights_changed':False,
            'seed':0,'cases':[],'production_tracker_changed':False}
    for name,job,cache in cases:
        print(f'Preparing {name}',flush=True)
        result=json.loads((ROOT/'spacescan_trees_demo_v1/runtime/corridor_jobs'/job/'result.json').read_text(encoding='utf-8'))
        frames=json.loads(cache.read_text(encoding='utf-8'))['frames']
        for frame in frames:
            for row in frame['candidates']:
                row['embedding']=np.asarray(row['embedding'],dtype=np.float32)
                row['crop']=Image.new('RGB',(16,16),'green') # placeholder; no classification in this comparison
        provider=Images(result)
        audit={}
        t=time.perf_counter();baseline=vmms_corridor._track_candidates(frames,audit)
        baseline_time=round(time.perf_counter()-t,3)
        print(f'Baseline: {len(baseline)} candidates; starting BoT-SORT / GMC',flush=True)
        counter=0
        def images(frame,spec):
            nonlocal counter
            if frame['sample_index']!=counter:
                counter=frame['sample_index']
                if counter%20==0:print(f'{name}: BoT-SORT frame {counter+1}/{len(frames)}',flush=True)
            return provider(frame,spec)
        candidate_audit={}
        t=time.perf_counter();candidate=corridor_botsort.track_candidates(frames,candidate_audit,images)
        candidate_time=round(time.perf_counter()-t,3)
        case={'name':name,'job':job,'frames':len(frames),'cache':str(cache),
              'cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),
              'baseline':metrics(baseline,frames)|{'elapsed_seconds':baseline_time,'audit':audit},
              'candidate':metrics(candidate,frames)|{'elapsed_seconds':candidate_time,'audit':candidate_audit}}
        if name=='campus_reviewed':
            case['baseline']['reviewed_negative_pairs']=reviewed_negatives(baseline)
            case['candidate']['reviewed_negative_pairs']=reviewed_negatives(candidate)
        for mode,tracks in (('baseline',baseline),('candidate',candidate)):
            folder=dest/name/mode
            folder.mkdir(parents=True)
            serial=[]
            for i,track in enumerate(tracks):
                chosen=vmms_corridor._select_views(track)
                serial.append({'rank':i+1,'observations':[signature(row) for row in track['observations']],
                               'selected_frames':[row['sample_index']+1 for row in chosen]})
                # QA crops are actual source pixels, not the green placeholders.
                if i<6:
                    for j,row in enumerate(chosen):
                        frame=frames[row['sample_index']]
                        view=panorama.render(provider.load(frame),row['view_spec'],max_edge=1024)
                        crop=view.crop(tree_detection.crop_rect(row['view_box'],view.size))
                        vmms_corridor._jpeg(crop,folder/f'track_{i+1:02d}_{j+1}.jpg',edge=480)
            (folder/'tracks.json').write_text(json.dumps(serial,indent=2),encoding='utf-8')
        report['cases'].append(case)
        print(json.dumps({k:v for k,v in case.items() if k in ('name','frames')}|{
              'baseline_tracks':len(baseline),'candidate_tracks':len(candidate),
              'baseline_observations':case['baseline']['stable_observations'],
              'candidate_observations':case['candidate']['stable_observations']},ensure_ascii=False),flush=True)
    report['decision']='Keep the production geometry-gallery tracker. No fully labelled instance validation demonstrates a BoT-SORT improvement; counts alone cannot justify promotion.'
    (dest/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(dest/'report.json'),flush=True)


if __name__=='__main__':main()
