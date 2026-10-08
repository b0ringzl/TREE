"""Freeze score-independent source-labelled rare-tree evaluation; no model writes."""
import hashlib
import io
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import fresh_bioclip146_benchmark_20261006 as base
from training_safety import backup_before_training, verify_backup, sha256
OUT = ROOT / 'outputs/rare_species_fresh_validation_20261006_v1'
TARGETS = ('Aquilaria sinensis', 'Artocarpus hypargyreus')


def main():
    if (OUT / 'manifest.json').exists() or (OUT / 'predictions.jsonl').exists():
        raise FileExistsError('Existing frozen evaluation cannot be changed')
    verify_backup(OUT / 'pre_training_backup')
    old_obs, old_photos, old_hashes, old_visual = base.prior_index(OUT)
    rows = base.load(OUT / 'nc_manifest_partial.json')
    for row in base.load(OUT / 'nc_resume_v2/nc_manifest.json'):
        rows.append(dict(row, file='nc_resume_v2/' + row['file']))
    exclusions = []
    cloud = {r['pmcid']: r for r in base.load(OUT / 'paper_cloud_sources.json')}
    native = {r['pmcid']: r for r in base.load(OUT / 'paper_native_extraction.json')}
    specs = [('PMC7050300', 'a', [10, 10, 1002, 1712], 'mature_tree'),
             ('PMC10935121', 'A', [24, 16, 562, 1174], 'seedling'),
             ('PMC10935121', 'D', [1170, 15, 1750, 570], 'husk'),
             ('PMC10935121', 'E', [1170, 580, 1750, 1170], 'seed')]
    crops = OUT / 'paper_panels'; crops.mkdir(exist_ok=False)
    for pmcid, panel, rect, view in specs:
        source = native[pmcid]
        with Image.open(OUT / source['file']) as im:
            cropped = im.convert('RGB').crop(rect)
        if min(cropped.size) < 224: raise RuntimeError('No native-resolution fabrication')
        path = crops / f'{pmcid}_figure1_panel_{panel}.png'
        cropped.save(path)
        md = cloud[pmcid]
        rows.append({'id': path.stem, 'label': TARGETS[0], 'source_species': TARGETS[0],
                     'source_kind': 'paper_photo_challenge', 'source_group': 'paper_' + pmcid,
                     'observer_id': None, 'observation_id': pmcid,
                     'source_page': 'https://doi.org/' + md['doi'][0], 'doi': md['doi'][0],
                     'license': 'cc-by-4.0', 'attribution': md['title'] + '; original authors/photographers; see source caption',
                     'caption': md['caption'], 'panel': panel, 'view_type': view, 'crop_rectangle': rect,
                     'native_figure_file': source['file'], 'native_figure_sha256': source['sha256'],
                     'pdf_file': source['pdf_file'], 'pdf_sha256': source['pdf_sha256'],
                     'file': path.relative_to(OUT).as_posix(), 'sha256': sha256(path),
                     'phash': f'{base.phash(cropped):016x}', 'dimensions': list(cropped.size), 'bytes': path.stat().st_size,
                     'training_eligible': False, 'human_review_status': 'author_labelled_not_independent_expert_reviewed',
                     'split': 'paper_domain_shift_challenge', 'hong_kong': False, 'noncommercial_only': False,
                     'recent_after_base_project': False, 'observed_on': None, 'region_query': 'paper', 'resampled': False})
    exclusions.extend([
        {'source': 'PMC7050300', 'panels': 'b-e,f,g', 'reason': 'Microscopy and agarwood products, not natural target photographs'},
        {'source': 'PMC10935121', 'panels': 'B,C', 'reason': 'Tissue culture preparation/callus outside natural botanical photograph scope'},
        {'source': 'PMC8048387', 'panel': 'I', 'reason': 'Native PDF photograph side144x136 and web panel about103x100 are below224; no upscaling'},
        {'source': 'PMC8048387', 'panels': 'all_except_I', 'reason': 'Author captions identify different Artocarpus species'},
        {'source': 'PMC11945267', 'panel': 'A', 'reason': 'Mixed ordinary/Qinan cultivar montage; individual target identity not unambiguous at all ages'},
        {'source': 'PMC11945267', 'panels': 'B-E', 'reason': 'Charts, not plant photographs'},
    ])
    accepted = []; hashes = set(); visuals = {}; seen_ids = set()
    for row in rows:
        why = []
        if row.get('photo_id') == 36887547:
            why.append('Sign-only picture with species text/illustrations, no actual plant photograph')
        if row['source_kind'] == 'community_observation':
            if str(row['observation_id']) in old_obs or str(row['photo_id']) in old_photos:
                why.append('Historical source identity')
            if row['source_evidence']['tier'] != 'multiple_independent_exact_ids':
                why.append('Insufficient source species evidence')
        raw = (OUT / row['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256']: raise RuntimeError('Changed image')
        with Image.open(io.BytesIO(raw)) as src:
            im = ImageOps.exif_transpose(src).convert('RGB')
        v = base.phash(im); gray = np.asarray(im.convert('L'))
        if row['id'] in seen_ids or row['sha256'] in hashes or row['sha256'] in old_hashes:
            why.append('Exact duplicate')
        near_old = base.nearest_hash(v, old_visual); near_new = base.nearest_hash(v, visuals)
        if near_old: why.append('Near historical image')
        if near_new: why.append('Near collected image')
        if min(im.size) < 224 or len(raw) > 20_000_000 or im.width * im.height > 40_000_000:
            why.append('Native size/bytes quality gate')
        if (gray < 15).mean() > .85 or (gray > 245).mean() > .85:
            why.append('Extreme lighting gate')
        if why:
            exclusions.append({'id': row['id'], 'source_page': row['source_page'], 'reason': why,
                               'near_old': near_old, 'near_new': near_new}); continue
        row['cluster_id'] = ('observer_' + str(row['observer_id'])) if row['source_kind'] == 'community_observation' else row['source_group']
        row['phash'] = f'{v:016x}'
        accepted.append(row); hashes.add(row['sha256']); visuals[v] = row['id']; seen_ids.add(row['id'])
    for label in TARGETS:
        observations = {r['observation_id'] for r in accepted if r['label'] == label and r['source_kind'] == 'community_observation'}
        if len(observations) > 100: raise RuntimeError('Predeclared observation cap exceeded')
    base.save(OUT / 'final_source_exclusions.json', exclusions)
    base.save(OUT / 'manifest.json', accepted)
    base.save(OUT / 'quality_review.json', {
        'reviewed_before_predictions': True, 'model_outputs_accessed': False,
        'contact_sheets_reviewed': ['contact_sheets/batch_000.jpg', 'contact_sheets/batch_001.jpg', 'contact_sheets/batch_002.jpg'] +
            ['nc_resume_v2/contact_sheets/batch_%03d.jpg' % i for i in range(7)],
        'paper_originals_reviewed': True, 'review_scope': 'Real photograph/domain/image quality, NOT botanical expert adjudication',
        'one_sign_only_sample_excluded': 'inat_36887547',
        'hard_views_kept': 'Trunks, distant trees, naturally dark photographs, detached seeds and fruit; no score filtering',
    })
    plan_paths = [OUT / p for p in ('PLAN.json', 'catalog146.json', 'NC_EXTENSION_PLAN.json',
        'nc_resume_v2/RESUME_PLAN.json', 'PMC_CLOUD_PLAN_20261007.json', 'quality_review.json')]
    lock = {'locked_at_utc': datetime.now(timezone.utc).isoformat(),
            'manifest_sha256': sha256(OUT / 'manifest.json'), 'rows': len(accepted),
            'plans_sha256': {p.relative_to(OUT).as_posix(): sha256(p) for p in plan_paths},
            'locked_before_inference': True, 'training_eligible': False, 'model_score_selection': False,
            'primary': 'Community natural photographs; paper photos reported separately',
            'cluster': 'Observer bootstrap conservatively accounts for repeated visits/photos; observation equal weights also reported',
            'source_counts': dict(Counter(r['source_kind'] for r in accepted))}
    base.save(OUT / 'collection_lock.json', lock)
    scripts = [ROOT / 'tools' / name for name in ('rare_species_freeze_20261007.py',
        'rare_species_source_extension_20261006.py', 'rare_species_nc_resume_v2_20261006.py',
        'rare_species_pmc_cloud_20261007.py', 'rare_species_extract_paper_native_20261007.py')]
    backup_before_training(OUT / 'evaluation_protocol_snapshot', plan_paths + scripts +
        [OUT / p for p in ('manifest.json', 'collection_lock.json', 'final_source_exclusions.json',
                          'paper_cloud_sources.json', 'paper_native_extraction.json')],
        {'purpose': 'Fixed evaluation protocol/data provenance snapshot, no training',
         'physical_weight_backup': str(OUT / 'pre_training_backup/backup_manifest.json')})
    print('FROZEN', len(accepted), Counter((r['label'], r['source_kind']) for r in accepted), flush=True)


if __name__ == '__main__': main()
