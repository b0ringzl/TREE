"""Bounded source-labelled pending white-kwai-muk collection; never train/approve.

Reads the root agent's already-recorded public API response. No fresh API search,
no model labels, no source-page screenshot masquerading as a photograph.
"""
import hashlib
import io
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps

from import_conservation_browser_images_20260917 import digest, phash, save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'external_datasets/artocarpus_whole_source_20260917_v1'
PREVIOUS = ROOT / 'external_datasets/artocarpus_fresh_inat_20260917_v1'
OBS = {353356916, 265280952, 100271692, 76933959, 7456890}
ALLOWED = {'cc0', 'cc-by', 'cc-by-sa'}


def main():
    # Source documents were acquired for this same run beforehand. Never replace
    # them or any manifest. New image folders make a repeated invocation fail.
    (OUT / 'images').mkdir(exist_ok=False)
    (OUT / 'source_responses').mkdir(exist_ok=False)
    (OUT / 'contact_sheets').mkdir(exist_ok=False)
    response = PREVIOUS / 'source_responses/00_observations_1.json'
    shutil.copy2(response, OUT / 'source_responses/00_observations_1.json')
    observations = json.loads(response.read_text(encoding='utf-8'))['results']
    audit = json.loads((PREVIOUS / 'source_audit.json').read_text(encoding='utf-8'))
    evidence = {r['observation_id']: r['source_evidence'] for r in audit
                if r.get('species') == 'Artocarpus hypargyreus'}
    rows, events = [], []
    for obs in observations:
        if obs['id'] not in OBS:
            continue
        ev = evidence[obs['id']]
        if (obs.get('taxon') or {}).get('name') != 'Artocarpus hypargyreus':
            raise ValueError('Source species changed; stop rather than relabel')
        if ev.get('conflicting_current_identifications') or obs.get('flags'):
            events.append({'observation_id': obs['id'], 'status': 'conflict_quarantine_no_download'})
            continue
        photos = [p for p in obs.get('photos', []) if p.get('license_code') in ALLOWED
                  and not p.get('flags') and not p.get('hidden')][:3]
        for photo in photos:
            url = photo['url'].replace('/square.', '/original.')
            row = {
                'id': 'artowhole_inat_' + str(photo['id']),
                'source_species': 'Artocarpus hypargyreus', 'source_species_zh_hant': '白桂木',
                'source': 'iNaturalist', 'source_site': 'https://www.inaturalist.org',
                'source_page': obs['uri'], 'page_url': obs['uri'], 'source_url': obs['uri'],
                'photo_url': url, 'image_url': url, 'api_photo_url': photo['url'],
                'photo_size_requested': 'original', 'photo_id': photo['id'],
                'observation_id': obs['id'], 'inat_observation_id': obs['id'],
                'source_taxon_id': obs['taxon']['id'],
                'observer_id': obs['user']['id'], 'observer_login': obs['user']['login'],
                'observer_group': 'inat_observer_' + str(obs['user']['id']),
                'source_group': 'inat_observation_' + str(obs['id']),
                'group_id': 'inat_observation_' + str(obs['id']),
                'observed_on': obs.get('observed_on'), 'location': obs.get('location'),
                'place_guess': obs.get('place_guess'), 'positional_accuracy': obs.get('positional_accuracy'),
                'geoprivacy': obs.get('geoprivacy'),
                'source_evidence': ev, 'label_authority': 'source_label_only',
                'license': photo['license_code'], 'photo_license_code': photo['license_code'],
                'rights_status': 'source_explicit_cc_license', 'attribution': photo.get('attribution'),
                'human_review_status': 'pending_human_confirmation',
                'training_eligible': False, 'split': 'quarantine_unsplit',
                'history_dedup': 'pending_full_history_audit',
                'visual_qa': 'pending', 'acquisition_method': 'public_photo_download',
                'identity_warning': 'Different observations and observers can photograph the same individual tree.',
                'source_response': str((OUT / 'source_responses/00_observations_1.json').resolve()),
                'source_response_sha256': digest(response),
            }
            time.sleep(1.2)
            try:
                r = requests.get(url, timeout=35,
                    headers={'User-Agent': 'SpaceScanTrees/1.0 limited public botanical photo review'})
                r.raise_for_status()
                raw = r.content
                if len(raw) > 30_000_000:
                    raise ValueError('Image above 30 MB collection cap')
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
                    native_size, native_format = im.size, im.format
                    rgb = ImageOps.exif_transpose(im).convert('RGB')
                    visual_hash = format(phash(rgb), '016x')
                extension = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}.get(native_format)
                if not extension:
                    raise ValueError('Unexpected original image format')
                path = OUT / 'images' / (row['id'] + extension)
                with path.open('xb') as stream:
                    stream.write(raw)
                row.update(status='downloaded', file=path.relative_to(OUT).as_posix(),
                    source_image=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                    phash=visual_hash, dimensions=list(native_size), bytes=len(raw),
                    native_format=native_format, downloaded_at=datetime.now(timezone.utc).isoformat())
                rows.append(row)
                print('DOWNLOADED', row['id'], native_size, len(raw), flush=True)
            except Exception as exc:
                events.append({'observation_id': obs['id'], 'photo_id': photo['id'],
                               'url': url, 'error': str(exc)})
                print('ERROR', str(exc), flush=True)
                if isinstance(exc, requests.HTTPError) and exc.response.status_code in (401, 403, 429, 503):
                    break
        if events and events[-1].get('error'):
            break
    save(OUT / 'manifest.json', rows)
    save(OUT / 'collection_errors.json', events)
    sheet = Image.new('RGB', (1600, ((len(rows) + 3) // 4) * 450), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        with Image.open(row['source_image']) as src:
            image = ImageOps.exif_transpose(src).convert('RGB')
        image.thumbnail((388, 365))
        x, y = (i % 4) * 400, (i // 4) * 450
        sheet.paste(image, (x + (388 - image.width) // 2, y))
        draw.text((x + 5, y + 370), f"{i+1:02d} photo {row['photo_id']}", fill='black')
        draw.text((x + 5, y + 390), f"obs {row['observation_id']} / {row['observer_login']}", fill='black')
        draw.text((x + 5, y + 410), 'SOURCE LABEL ONLY / NO TRAINING', fill='black')
    if rows:
        sheet.save(OUT / 'contact_sheets/weak_source_all.jpg', quality=95)
    save(OUT / 'RAW_SUMMARY.json', {
        'downloaded': len(rows), 'observations': len({r['observation_id'] for r in rows}),
        'observers': len({r['observer_id'] for r in rows}),
        'training_eligible': 0, 'human_accepted': 0, 'whole_crown_count': 'not_yet_visually_checked',
        'full_history_dedup': 'pending_separate_audit', 'errors': events,
        'training_performed': False, 'deployment_changed': False,
        'script_sha256': digest(Path(__file__)),
    })


if __name__ == '__main__':
    main()
