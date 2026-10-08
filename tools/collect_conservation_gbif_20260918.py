"""Small, serial GBIF institution-image provenance collection; never training.

Uses the documented occurrence-image API as its PRIMARY image delivery endpoint,
not as a retry around a publisher refusal. Media license is independently checked.
Preserved specimens are isolated from any street/tree-photo training candidates.
"""
import argparse
import hashlib
import io
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/conservation_authoritative_collection_20260918_v1/gbif'
SPECIES = ['Rhodoleia championii', 'Keteleeria fortunei', 'Aquilaria sinensis', 'Artocarpus hypargyreus']
LIMIT = {'Rhodoleia championii': 6, 'Keteleeria fortunei': 6, 'Aquilaria sinensis': 2, 'Artocarpus hypargyreus': 2}
API = 'https://api.gbif.org/v1'
UA = 'SpaceScanTreesResearch/1.0 (local noncommercial provenance audit; serial requests)'
SOURCES = [
 'https://techdocs.gbif.org/en/openapi/v1/occurrence',
 'https://techdocs.gbif.org/en/openapi/images',
 'https://techdocs.gbif.org/en/data-publishing/multimedia-publishing',
 'https://www.gbif.org/terms/data-user']


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


class Client:
    def __init__(self, out):
        self.out = out; self.session = requests.Session()
        self.session.headers.update({'User-Agent': UA, 'Accept': 'application/json'})
        self.calls = []; self.last = 0.; self.datasets = {}
        self.robot = None

    def request(self, url, params=None, image=False):
        # One connection and >=1 second between request starts; no retry loop.
        time.sleep(max(0., 1.0-(time.monotonic()-self.last)))
        self.last = time.monotonic()
        response = self.session.get(url, params=params, timeout=(15, 40),
                                    headers={'Accept': 'image/*' if image else 'application/json'})
        self.calls.append({'url': response.url, 'status': response.status_code,
                           'content_type': response.headers.get('content-type'),
                           'date': datetime.now(timezone.utc).isoformat()})
        write(self.out/'request_log.json', self.calls)
        if response.status_code in (401, 403, 429):
            raise RuntimeError('Access/rate-limit refusal; no bypass or automated retry: '+response.url)
        response.raise_for_status()
        return response

    def json(self, url, params=None):
        return self.request(url, params=params).json()

    def dataset(self, key):
        if key not in self.datasets:
            self.datasets[key] = self.json(API+'/dataset/'+key)
            write(self.out/'api'/('dataset_'+key+'.json'), self.datasets[key])
        return self.datasets[key]


def canonical_media_license(value):
    text = str(value or '').strip().lower()
    if re.search(r'^https?://creativecommons\.org/publicdomain/zero/1\.0/?(?:legalcode)?$', text): return 'CC0-1.0'
    match = re.search(r'^https?://creativecommons\.org/licenses/(by|by-sa|by-nc|by-nc-sa)/(\d\.\d)/?(?:legalcode)?$', text)
    return 'CC-'+match[1].upper()+'-'+match[2] if match else None


def collect(out):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/'api').mkdir(); (out/'specimen_domain').mkdir(); (out/'living_candidate_unreviewed').mkdir()
    client = Client(out)
    write(out/'PROTOCOL.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'species': SPECIES, 'per_species_success_limit': LIMIT, 'requests': 'Serial one connection; >=1second gap; no refusal retry',
        'primary_image_delivery': 'Documented GBIF image cache API; original publisher URL retained; maximum1200px cache derivative, not claimed original full resolution.',
        'license_gate': 'Explicit media.license CC0/BY/BY-SA/BY-NC/BY-NC-SA only. Occurrence license is NEVER substituted.',
        'excluded_sources': 'All iNaturalist occurrence/media/dataset evidence; unresolved source name or infraspecific/synonym assignment; missing media license',
        'domains': 'PRESERVED_SPECIMEN/MATERIAL_SAMPLE go to specimen_domain only; no automatic street training eligibility',
        'scope': 'Local non-commercial research data collection only, no training, no deployment/publication; preserve image content/watermarks and attribution.',
        'official_documentation': SOURCES})
    # The official API documents programmatic use; retain its robots response.
    robots = client.request('https://api.gbif.org/robots.txt')
    (out/'api'/'robots.txt').write_text(robots.text, encoding='utf-8')
    if robots.text.lstrip().startswith('<'):
        raise RuntimeError('Unexpected robots response; stop instead of guessing')
    parser = RobotFileParser(); parser.parse(robots.text.splitlines())
    if not parser.can_fetch(UA, API+'/occurrence/search'):
        raise RuntimeError('API robots disallows query')
    candidates = []; exclusions = []; species_summaries = {}; downloaded = []; duplicate_hashes = set()
    for name in SPECIES:
        slug = name.replace(' ', '_')
        match = client.json(API+'/species/match', {'name': name, 'strict': 'true'})
        write(out/'api'/(slug+'_taxon.json'), match)
        if match.get('canonicalName') != name or match.get('matchType') != 'EXACT' or match.get('rank') != 'SPECIES':
            raise RuntimeError('Species match not exact: '+name)
        key = match['usageKey']; records = []
        for offset in (0, 300, 600):
            page = client.json(API+'/occurrence/search', {'taxonKey': key, 'mediaType': 'StillImage', 'limit': 300, 'offset': offset})
            write(out/'api'/(slug+'_occurrences_'+str(offset)+'.json'), page)
            records.extend(page.get('results', []))
            if page.get('endOfRecords') or offset+300 >= page.get('count', 0): break
        local = []; reasons = Counter(); inst = Counter()
        for record in records:
            rid = record['key']; rawtext = json.dumps(record).lower()
            if 'inaturalist' in rawtext:
                reasons['community_iNaturalist_excluded'] += 1; continue
            scientific = record.get('scientificName', '')
            if record.get('speciesKey') != key or ' '.join(scientific.split()[:2]) != name or re.search(r'\b(var\.|subsp\.|cf\.|aff\.)', scientific):
                reasons['non_exact_original_species_or_infraspecific'] += 1; continue
            basis = record.get('basisOfRecord', '')
            domain = 'specimen_domain' if basis in ('PRESERVED_SPECIMEN', 'MATERIAL_SAMPLE') else 'living_candidate_unreviewed'
            for mi, media in enumerate(record.get('media', [])):
                if media.get('type') != 'StillImage': continue
                lic = canonical_media_license(media.get('license')); raw = media.get('identifier')
                if not lic:
                    reasons['missing_or_not_allowlisted_media_license'] += 1
                    exclusions.append({'species': name, 'record_id': rid, 'reason': 'media_license_not_approved', 'media': media,
                                       'occurrence_license_not_substituted': record.get('license')}); continue
                if not raw or urlparse(raw).scheme not in ('https', 'http'):
                    reasons['missing_raw_image_identifier'] += 1; continue
                institution = record.get('institutionCode') or media.get('rightsHolder') or media.get('publisher')
                if not institution:
                    reasons['no_institution_evidence'] += 1; continue
                row = {'id': f'gbif_{rid}_{mi}', 'source_species': name, 'species_key': key,
                    'source_scientific_name': scientific, 'source_page': f'https://www.gbif.org/occurrence/{rid}',
                    'record_id': rid, 'occurrence_identifier': record.get('occurrenceID'),
                    'catalog_number': record.get('catalogNumber'), 'institution': institution,
                    'dataset_key': record.get('datasetKey'), 'publisher_org_key': record.get('publishingOrgKey'),
                    'identified_by': record.get('identifiedBy'), 'date_identified': record.get('dateIdentified'),
                    'event_date': record.get('eventDate'), 'recorded_by': record.get('recordedBy'),
                    'country': record.get('country'), 'locality': record.get('locality'),
                    'basis_of_record': basis, 'domain': domain, 'raw_url': raw,
                    'source_image_page': media.get('references'), 'media_license_raw': media.get('license'),
                    'media_license': lic, 'occurrence_license': record.get('license'),
                    'creator': media.get('creator'), 'rights_holder': media.get('rightsHolder'),
                    'media_title': media.get('title'), 'media_description': media.get('description'),
                    'label_authority': 'Institutional occurrence label; identifiedBy field retained when provided; not independently taxonomically certified.',
                    'group_id': 'gbif_record:'+str(rid), 'training_eligible': False,
                    'review_status': 'not_reviewed_for_street_tree_training', 'downloaded': False}
                local.append(row); inst[institution] += 1
        # Prefer living photographs if the institutional inventory contains any;
        # otherwise obtain a modest source-diverse specimen reference batch.
        local.sort(key=lambda r: (r['domain'] == 'specimen_domain', not bool(r['identified_by']), r['record_id']))
        counts = Counter(); success = 0
        for row in local:
            if success >= LIMIT[name]: break
            if counts[row['institution']] >= 2: continue
            dataset = client.dataset(row['dataset_key'])
            if 'inaturalist' in json.dumps(dataset).lower():
                reasons['community_dataset_excluded'] += 1; continue
            row['dataset_title'] = dataset.get('title'); row['dataset_citation'] = dataset.get('citation')
            row['dataset_publisher'] = dataset.get('publishingOrganizationKey')
            identifier_md5 = hashlib.md5(row['raw_url'].encode()).hexdigest()
            image_url = f"{API}/image/cache/occurrence/{row['record_id']}/media/{identifier_md5}"
            if not parser.can_fetch(UA, image_url):
                reasons['robots_image_disallowed'] += 1; continue
            row['retrieval_url'] = image_url
            try:
                response = client.request(image_url, image=True)
                blob = response.content; h = hashlib.sha256(blob).hexdigest()
                with Image.open(io.BytesIO(blob)) as img:
                    img.load(); width, height = img.size; fmt = img.format
                if min(width, height) < 224: raise ValueError('Image too small')
                if h in duplicate_hashes:
                    row['duplicate_download_sha256'] = h; reasons['duplicate_bytes'] += 1; continue
                extension = '.png' if fmt == 'PNG' else '.jpg' if fmt == 'JPEG' else '.'+fmt.lower()
                path = out/row['domain']/(row['id']+extension)
                path.write_bytes(blob); duplicate_hashes.add(h)
                row.update(downloaded=True, source_image=str(path), sha256=h, dimensions=[width, height],
                           bytes=len(blob), format=fmt, retrieval_date=datetime.now(timezone.utc).isoformat(),
                           rights_scope='Attribution and original media terms retained; NC restricted to non-commercial use; no Demo/public distribution authorized by this collection.')
                downloaded.append(dict(row)); success += 1; counts[row['institution']] += 1
                print('DOWNLOADED', name, row['record_id'], row['domain'], width, height, row['media_license'], flush=True)
            except RuntimeError:
                raise
            except Exception as error:
                row['download_error'] = type(error).__name__+': '+str(error)
                reasons['delivery_failed_no_retry'] += 1
            write(out/'downloaded_manifest.json', downloaded)
        candidates.extend(local)
        species_summaries[name] = {'taxon_key': key, 'records_seen': len(records),
            'candidate_media_explicit_license': len(local), 'institution_media_counts': dict(inst),
            'downloaded': success, 'downloaded_domains': dict(Counter(r['domain'] for r in downloaded if r['source_species'] == name)),
            'excluded_reasons': dict(reasons)}
        write(out/'candidate_inventory.json', candidates); write(out/'exclusions.json', exclusions)
        write(out/'SUMMARY.json', {'species': species_summaries, 'downloaded': len(downloaded), 'training_eligible': 0,
            'no_training': True, 'no_production_change': True, 'image_license_independent': True,
            'official_sources': SOURCES, 'fresh_independent_tree_test_claim': False})
    write(out/'COMPLETE.json', {'completed_at': datetime.now(timezone.utc).isoformat(), 'downloaded': len(downloaded),
        'manifest_sha256': hashlib.sha256((out/'downloaded_manifest.json').read_bytes()).hexdigest(),
        'scope': 'Collection only. Institutional specimen-domain references are not street-image training data.'})
    return downloaded


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args(); collect(args.out)
