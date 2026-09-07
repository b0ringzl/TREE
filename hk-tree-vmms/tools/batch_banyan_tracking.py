"""Apply the reviewed pilot method to remaining cached banyan candidates.

Never overwrite the pilot or its accepted assets. This is a new, unaccepted
version. In particular, user acceptance is never propagated between frames.
"""
import hashlib
from pathlib import Path
import pilot_banyan_tracking as pilot
from audit_hk_metric_height import DATA, read, write

OUT=DATA/'banyan_tracking_batch_v2'

def main():
    if (OUT/'results.json').exists():
        raise SystemExit('Completed batch exists; refusing to overwrite reviewable geometry.')
    pilot_dir=DATA/'banyan_tracking_pilot_v1'
    original_selection=read(pilot_dir/'selection.json')
    excluded={r['id'] for r in original_selection}
    review_path=pilot_dir/'reviews.json'
    original_hash=hashlib.sha256(review_path.read_bytes()).hexdigest() if review_path.exists() else None
    records=read(DATA/'manifest.json')
    # Older records may omit dense_cache even though the source-keyed cache
    # exists. Resolve it from source metadata without mutating the manifest.
    for r in records:
        source=r.get('dense_source','')
        path=DATA/'dense_cache'/(hashlib.sha256(source.encode()).hexdigest()[:12]+'.npz')
        if source!='frame_color_clouds' and path.exists() and path.with_suffix('.json').exists():
            if read(path.with_suffix('.json')).get('source')==source:
                r['dense_cache']=str(path)
    remaining=[r for r in records if r['species']=='榕树' and r['status']=='geometry_candidate_pending_review' and r['id'] not in excluded]
    selected=[r for r in remaining if r.get('dense_cache') and Path(r['dense_cache']).exists()]
    OUT.mkdir(exist_ok=True)
    write(OUT/'scope.json',{'pilot_ids_excluded':sorted(excluded),'remaining_candidates':len(remaining),
                           'processable_cached_pairs':len(selected),
                           'deferred_no_dense_cache':[r['id'] for r in remaining if r not in selected],
                           'pilot_review_hash':original_hash,
                           'algorithm_sha256':hashlib.sha256(Path(pilot.__file__).read_bytes()).hexdigest()})
    pilot.OUT=OUT
    pilot.select=lambda _:selected
    pilot.read=lambda path:records if Path(path)==DATA/'manifest.json' else read(path)
    pilot.main()
    if original_hash is not None:
        assert hashlib.sha256(review_path.read_bytes()).hexdigest()==original_hash, 'Pilot review changed during processing; inspect concurrent edit.'
    print('Batch complete; original pilot reviews and assets preserved.',flush=True)

if __name__=='__main__':main()
