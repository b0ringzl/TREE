"""Bounded licensed raw collection; no label approval, splitting, or training.

Species/negative choices are preregistered from training-only diagnostics.
Freshness and human decisions are audited separately before any fitting.
"""
import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps

from collect_conservation_sources_20260917 import identifiers, query
from collect_sstv1_incremental import LICENSES, ROOT, download, old_index, save

TARGETS = [
    ('Artocarpus hypargyreus', 10),
    ('Artocarpus heterophyllus', 6),
    ('Litsea glutinosa', 6),
    ('Ficus hispida', 6),
    ('Sterculia lanceolata', 6),
]


def main(out):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ('images', 'source_responses', 'contact_sheets'):
        (out / name).mkdir()
    save(out / 'PLAN.json', {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'targets_observation_caps': TARGETS,
        'negative_choice': 'Training-only confusion pool in outputs/conservation_optimization_diagnostics_20260917_v1/diagnostics.json; NOT test mining.',
        'api_policy': 'Serial public API; >=1.1 s wait; stop entire source on HTTP 401/403/429/503.',
        'licenses': sorted(LICENSES), 'max_pages_per_taxon': 2,
        'source_gate': 'Exact active species; >=2 independent current exact identifiers including a non-observer; no current species conflict.',
        'sampling': 'At most one new observation per observer per species; up to three uniformly spaced gallery photographs, not visually selected yet.',
        'status': 'Quarantined raw collection, not human-reviewed/approved. Historical identity and organ/quality audit still required.',
        'deployment_changed': False, 'training_performed': False,
    })
    prior_obs, prior_sha, prior_photos, _ = old_index()
    rows, events, errors = [], [], []
    seen_photos, seen_hashes = set(), set()
    stopped = False
    for ti, (species, cap) in enumerate(TARGETS):
        try:
            taxa = query('taxa/autocomplete', {'q': species, 'per_page': 10})
            save(out / 'source_responses' / f'{ti:02d}_taxon.json', taxa)
            hits = [r for r in taxa.get('results', []) if r.get('is_active') and r.get('name') == species]
            if len(hits) != 1:
                raise ValueError('Exact active name not uniquely resolved; no synonym substitution')
            taxon_id = hits[0]['id']
            selected, observers, exclusions = [], set(), Counter()
            for page in (1, 2):
                data = query('observations', {'taxon_id': taxon_id, 'photos': 'true',
                    'photo_license': ','.join(sorted(LICENSES)), 'per_page': 200,
                    'page': page, 'order_by': 'id', 'order': 'desc'})
                save(out / 'source_responses' / f'{ti:02d}_observations_{page}.json', data)
                for obs in data.get('results', []):
                    ev = identifiers(obs, taxon_id)
                    reasons = []
                    if (obs.get('taxon') or {}).get('id') != taxon_id:
                        reasons.append('not_exact_species')
                    if str(obs['id']) in prior_obs:
                        reasons.append('known_historical_observation')
                    if obs.get('flags') or ev['tier'] != 'multiple_independent_exact_ids':
                        reasons.append('insufficient_species_source_evidence')
                    if obs['user']['id'] in observers:
                        reasons.append('observer_already_selected_for_species')
                    photos = [p for p in obs.get('photos', []) if p.get('license_code') in LICENSES
                              and not p.get('flags') and not p.get('hidden')
                              and p['id'] not in prior_photos and p['id'] not in seen_photos]
                    if not photos:
                        reasons.append('no_new_licensed_photo')
                    events.append({'species': species, 'observation_id': obs['id'],
                                   'source_evidence': ev, 'exclusions': reasons})
                    exclusions.update(reasons)
                    if reasons or len(selected) >= cap:
                        continue
                    observers.add(obs['user']['id'])
                    picks = sorted({0, len(photos) // 2, len(photos) - 1})
                    selected.append((obs, ev, [photos[i] for i in picks]))
                    seen_photos.update(photos[i]['id'] for i in picks)
                if len(selected) >= cap or page * 200 >= data.get('total_results', 0):
                    break
            print('SELECTED', species, len(selected), 'observations', dict(exclusions), flush=True)
            for obs, ev, photos in selected:
                for photo in photos:
                    row = {'id': 'artofresh_inat_' + str(photo['id']), 'source': 'iNaturalist',
                        'source_species': species, 'source_page': obs.get('uri'), 'source_url': obs.get('uri'),
                        'source_evidence': ev, 'source_taxon_id': taxon_id,
                        'observation_id': obs['id'], 'inat_observation_id': obs['id'],
                        'observer_id': obs['user']['id'], 'observed_on': obs.get('observed_on'),
                        'observer_group': 'inat_observer_' + str(obs['user']['id']),
                        'group_id': 'inat_observation_' + str(obs['id']),
                        'photo_id': photo['id'], 'photo_url': photo['url'].replace('/square.', '/large.'),
                        'license': photo['license_code'], 'photo_license_code': photo['license_code'],
                        'attribution': photo.get('attribution'), 'photo_attribution': photo.get('attribution'),
                        'human_review_status': 'not_requested', 'agent_content_review': 'pending',
                        'organ_review': 'not_yet_inspected', 'training_eligible': False, 'split': 'quarantine',
                        'label_authority': 'Source community identification, NOT human expert verification',
                        'identity_warning': 'Distinct observations/observers may still depict the same individual tree',
                    }
                    time.sleep(1.1)
                    row = download((row, out))
                    flags = []
                    if row['status'] == 'downloaded':
                        row['source_image'] = str((out / row['file']).resolve())
                        if row['sha256'] in prior_sha: flags.append('historical_exact_sha')
                        if row['sha256'] in seen_hashes: flags.append('within_batch_exact_sha')
                        if min(row['dimensions']) < 224: flags.append('small_dimensions')
                        if row['quality']['dark_fraction'] > .65: flags.append('mostly_dark')
                        seen_hashes.add(row['sha256'])
                    else:
                        flags.append('download_failed')
                    row['quality_flags'] = flags
                    rows.append(row)
                    save(out / 'manifest.json', rows)
                    print(row['id'], species, row['status'], flags, flush=True)
                    if row['status'] == 'failed' and any(str(c) in row.get('error', '') for c in (401, 403, 429, 503)):
                        stopped = True
                        raise RuntimeError('Source denied/rate-limited/unavailable; no bypass or retry')
        except Exception as exc:
            errors.append({'species': species, 'type': type(exc).__name__, 'error': str(exc)})
            print('ERROR', errors[-1], flush=True)
            if isinstance(exc, requests.HTTPError) and exc.response.status_code in (401, 403, 429, 503):
                stopped = True
            if stopped:
                break
        finally:
            save(out / 'source_audit.json', events)
            save(out / 'collection_errors.json', errors)
    save(out / 'manifest.json', rows)
    good = [r for r in rows if r['status'] == 'downloaded']
    for start in range(0, len(good), 12):
        sheet = Image.new('RGB', (1440, 1200), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(good[start:start + 12]):
            with Image.open(row['source_image']) as src:
                image = ImageOps.exif_transpose(src).convert('RGB')
            image.thumbnail((350, 320))
            x, y = (j % 4) * 360, (j // 4) * 400
            sheet.paste(image, (x + (350 - image.width) // 2, y))
            draw.text((x + 4, y + 325), f"{start+j+1:03d} {row['photo_id']}", fill='black')
            draw.text((x + 4, y + 343), row['source_species'], fill='black')
            draw.text((x + 4, y + 361), f"obs {row['observation_id']} / by {row['observer_id']}", fill='black')
            draw.text((x + 4, y + 379), 'RAW CANDIDATE - NOT TRAINING', fill='black')
        sheet.save(out / 'contact_sheets' / f'raw_{start//12:03d}.jpg', quality=94)
    summary = {'downloaded': len(good), 'species_counts': dict(Counter(r['source_species'] for r in good)),
               'observations': len({r['observation_id'] for r in good}),
               'observers': len({r['observer_id'] for r in good}), 'errors': errors,
               'stopped_source': stopped, 'training_performed': False, 'deployment_changed': False,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(out / 'SUMMARY.json', summary)
    print('COMPLETE', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
