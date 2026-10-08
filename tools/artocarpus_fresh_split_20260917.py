"""Read-only identity firewall for fresh Artocarpus and lookalike acquisitions.

No model, fitting, prediction access, network, historical raster decoding, or DB writes.
Historical test *identity metadata* is exclusion-only. Output files must be new.
Raw collection manifests do not grant botanical, human, or training approval.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
SPECIES = 'Artocarpus hypargyreus'
DEFAULT_OUT = ROOT / 'outputs/artocarpus_fresh_split_20260917_v1'
CACHE = ROOT / 'external_datasets/conservation_browser_20260917_v3/history_metadata_cache.json'
PRIOR = ROOT / 'outputs/SST_Conserve_Browser_20260917_v3'
NEW_DIRS = ('artocarpus_whole_source_20260917_v1', 'artocarpus_lookalike_negatives_20260917_v1', 'artocarpus_fresh_inat_20260917_v1')
HASH_KEYS = ('sha256', 'source_sha256', 'image_sha256')
PATH_KEYS = ('source_image', 'local_path', 'local_file', 'image_path', 'path', 'file', 'image', 'materialized_path')
PAGE_KEYS = ('page_url', 'source_page', 'source_url', 'inat_uri')
PHOTO_KEYS = ('image_url', 'photo_url', 'original_media_url')
HELD = {'planning', 'selection', 'test', 'field', 'case', 'validation', 'val', 'holdout', 'challenge'}
TOKEN_TYPES = ('sha256', 'path', 'photo', 'sample', 'observation', 'image_url', 'group', 'gallery', 'observer')
BLOCKED_REVIEW = {'rejected', 'reject', 'reject_source', 'uncertain', 'unusable', 'pending', 'deleted', 'delete'}
ACCEPTED_REVIEW = {'accepted', 'accept', 'confirm_source', 'confirmed'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_new(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def list_rows(value):
    if isinstance(value, list):
        return [r for r in value if isinstance(r, dict)]
    if isinstance(value, dict):
        for key in ('items', 'rows', 'records', 'images', 'manifest'):
            if isinstance(value.get(key), list):
                return list_rows(value[key])
    return []


def canon_url(value):
    if not isinstance(value, str):
        return None
    try:
        u = urlsplit(value)
        if u.scheme not in {'http', 'https'} or not u.netloc:
            return None
        return urlunsplit(('https', u.netloc.lower().removeprefix('www.'), unquote(u.path), u.query, ''))
    except ValueError:
        return None


def text_value(value):
    return str(value).strip() if isinstance(value, (str, int)) and not isinstance(value, bool) else ''


def values(row, keys):
    return [text_value(row[k]) for k in keys if row.get(k) is not None and text_value(row[k])]


def record_identity(row, origin):
    """Metadata normalization only. Never interpret a model prediction as a label."""
    origin = Path(origin)
    nested = row.get('source_record') if isinstance(row.get('source_record'), dict) else {}
    merged = dict(nested, **row)
    evidence = merged.get('source_evidence') or nested.get('evidence') or {}
    species = next(iter(values(merged, ('source_species', 'target_species', 'species', 'scientific_name', 'label'))), '')
    tokens = {k: set() for k in TOKEN_TYPES}
    for h in values(merged, HASH_KEYS):
        if re.fullmatch('[a-fA-F0-9]{64}', h):
            tokens['sha256'].add(h.lower())
    paths = []
    for s in values(merged, PATH_KEYS):
        if '://' in s or Path(s).suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.bmp'}:
            continue
        p = Path(s)
        if not p.is_absolute():
            choices = [origin.parent / p, origin.parent.parent / p, ROOT / p]
            p = next((p for p in choices if p.is_file()), choices[0])
        p = p.resolve()
        paths.append(str(p)); tokens['path'].add(str(p).replace('\\', '/').casefold())
    pages = {x for s in values(merged, PAGE_KEYS) if (x := canon_url(s))}
    image_urls = {x for s in values(merged, PHOTO_KEYS) if (x := canon_url(s))}
    tokens['image_url'] |= image_urls
    for s in values(merged, ('inat_observation_id', 'observation_id')):
        if s.isdigit():
            tokens['observation'].add('inat:' + s)
    for s in values(merged, ('photo_id', 'inat_photo_id')):
        if s.isdigit():
            tokens['photo'].add('inat:' + s)
    blobs = values(merged, ('id', 'sample_id', 'group_id', 'source_group', 'split_group')) + list(pages | image_urls) + paths
    for s in blobs:
        for x in re.findall(r'(?:/observations/|(?:inat_)?observation[_:])(\d+)', s):
            tokens['observation'].add('inat:' + x)
        for x in re.findall(r'(?:/photos/|inat_photo_|(?:conserve|sstv1)_inat_)(\d+)', s):
            tokens['photo'].add('inat:' + x)
    for value in values(merged, ('sample_id',)) + [Path(p).stem for p in paths]:
        if re.fullmatch('[a-f0-9]{20}', value):
            tokens['sample'].add('legacy_sample:' + value)
    if merged.get('commons_page_id'):
        tokens['photo'].add('commons:' + str(merged['commons_page_id']))
    for s in values(merged, ('group_id', 'source_group', 'split_group')):
        tokens['group'].add(s)
    tokens['group'] |= {'observation:' + x for x in tokens['observation']}
    site = text_value(evidence.get('source_site')) if isinstance(evidence, dict) else ''
    site = site or text_value(merged.get('source_site'))
    domain = next((urlsplit(u).netloc for u in sorted(pages | image_urls)), '')
    if tokens['observation'] or any('inaturalist' in u for u in pages | image_urls):
        site = 'iNaturalist'
    elif 'commons.wikimedia' in domain or merged.get('commons_page_id'):
        site = 'Wikimedia Commons'
    elif not site:
        site = {'herbarium.gov.hk': 'HKHerbarium', 'greening.gov.hk': 'Greening', 'indiaflora-ces.iisc.ac.in': 'IISc', 'treesandshrubsonline.org': 'TreesAndShrubsOnline'}.get(domain, domain)
    # Non-iNat gallery pages bind related source images conservatively.
    if site != 'iNaturalist':
        tokens['gallery'] |= pages
        tokens['gallery'] |= {s for s in tokens['group'] if s.startswith(('source_', 'browser_', 'gallery_', 'commons_author_date_'))}
    observer_id = text_value(merged.get('observer_id') or merged.get('inat_user_id'))
    observer_login = text_value(merged.get('observer_login') or merged.get('inat_user_login'))
    user = merged.get('observer') if isinstance(merged.get('observer'), dict) else {}
    observer_id = observer_id or text_value(user.get('id'))
    observer_login = observer_login or text_value(user.get('login'))
    if observer_id:
        tokens['observer'].add(('inat:id:' if site == 'iNaturalist' and observer_id.isdigit() else 'source:author:') + observer_id.casefold())
    if observer_login:
        tokens['observer'].add('inat:login:' + observer_login.casefold())
    if merged.get('observer_group'):
        tokens['observer'].add('source:group:' + str(merged['observer_group']))
    attr = next(iter(values(merged, ('photo_attribution', 'attribution'))), '')
    # Display-name proxy is only a conservative collision check, not a proven user ID.
    author = re.sub(r'^\s*(?:\(c\)|©)\s*', '', attr, flags=re.I).split(', some rights reserved')[0].strip()
    if author and author.casefold() not in {'no rights reserved', 'all rights reserved', 'unknown'} and '<' not in author and len(author) < 180:
        tokens['observer'].add('display:' + author.casefold())
    phashes = [p.lower() for p in values(merged, ('phash', 'p_hash')) if re.fullmatch('[a-fA-F0-9]{16}', p)]
    phase = text_value(merged.get('phase') or merged.get('split'))
    if not species:
        species = next((SPECIES for p in paths if SPECIES.lower() in p.lower().replace('_', ' ')), '')
    return {'id': text_value(merged.get('id') or merged.get('sample_id') or merged.get('item_id')),
            'species': species, 'tokens': {k: sorted(v) for k, v in tokens.items()}, 'phashes': phashes,
            'source_site': site or 'unspecified', 'paths': sorted(set(paths)), 'phase': phase,
            'observer_identity_quality': 'stable_id' if any(x.startswith('inat:id:') for x in tokens['observer']) else ('login' if observer_login else ('display_proxy_only' if tokens['observer'] else 'missing')),
            'origin': str(origin.resolve()), 'review_status': None,
            'license': text_value(merged.get('photo_license_code') or merged.get('license')), 'attribution': attr}


def db_decisions(path):
    before = digest(path)
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as con:
        con.execute('PRAGMA query_only=ON'); con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute('SELECT * FROM decisions')]
    assert digest(path) == before, f'Concurrent DB change: {path}'
    return rows, before


def queue_config():
    return [
        ('external_datasets/herbarium_training_candidates_20260907/curated_manifest.json', 'external_datasets/herbarium_training_candidates_20260907/human_review.sqlite3', False),
        ('external_datasets/sstv1_incremental_20260914_v1/curated_manifest.json', 'external_datasets/sstv1_incremental_20260914_v1/human_review.sqlite3', False),
        ('external_datasets/palm_confusion_20260908/curated_manifest.json', 'external_datasets/palm_confusion_20260908/human_review.sqlite3', True),
        ('external_datasets/conservation_sources_20260917_v1/review_manifest.json', 'conservation_review/data/conservation_sources_20260917_v1/human_review.sqlite3', False),
        ('outputs/conservation_expansion_20260917_curation_v1/human_review_required.json', 'conservation_expansion_review/data/expansion_20260917_v1/human_review.sqlite3', False),
    ]


def metadata_inputs(excluded):
    """Explicit metadata directories only; no weights, image glob, or backup traversal."""
    paths = set()
    for d in (ROOT / 'external_datasets').iterdir():
        if not d.is_dir() or d.name in NEW_DIRS or d.name in {'distillation_models', 'distillation_runtime', 'species_scale_trials', 'reference_framing_model'}:
            continue
        paths.update(d.glob('*manifest*.json'))
        if d.name in {'inat_max150', 'inat_max150_test', 'inat_parent_name_supplement', 'inaturalist_dated', 'palm_confusion_20260908'}:
            paths.update(d.glob('**/metadata.jsonl'))
        if d.name == 'herbarium_training_candidates_20260907':
            paths.add(d / 'existing_image_index.jsonl')
    for base in (PRIOR, ROOT / 'outputs/distillation/20260908_0045'):
        paths.update(base.glob('*rows.json'))
    for base in (PRIOR, ROOT / 'outputs/SSTv1_incremental_20260914_v1', ROOT / 'outputs/SSTv1_Aquilaria_User_20260914_v1'):
        paths.update(base.glob('*manifest.json'))
        if (base / 'feature_rows.json').is_file(): paths.add(base / 'feature_rows.json')
    paths.update((ROOT / 'outputs/conservation_browser_delivery_20260917_v1').glob('*candidates.json'))
    for d in (ROOT / 'outputs').glob('SST_Conserve*human_challenge*'):
        paths.update(p for p in (d / 'source_manifest.json', d / 'challenge_rows.json') if p.is_file())
    return sorted(p for p in paths if p.is_file() and not any(p.resolve().is_relative_to(x) for x in excluded))


class UnionFind:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while x != self.p[x]: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def merge_records(records):
    # Merge actual image identities; observations/observers instead form split groups.
    uf = UnionFind(len(records)); first = {}
    for i, r in enumerate(records):
        for kind in ('sha256', 'path', 'photo', 'sample'):
            for value in r['tokens'][kind]:
                if (kind, value) in first: uf.union(i, first[kind, value])
                else: first[kind, value] = i
    buckets = collections.defaultdict(list)
    for i, r in enumerate(records): buckets[uf.find(i)].append(r)
    merged = []
    for members in buckets.values():
        out = {'tokens': {k: sorted({x for r in members for x in r['tokens'][k]}) for k in TOKEN_TYPES},
               'ids': sorted({r['id'] for r in members if r['id']}),
               'species': sorted({r['species'] for r in members if r['species']}),
               'phashes': sorted({p for r in members for p in r['phashes']}),
               'source_sites': sorted({r['source_site'] for r in members if r['source_site'] != 'unspecified'}),
               'phases': sorted({r['phase'] for r in members if r['phase']}),
               'origins': sorted({r['origin'] for r in members}),
               'paths': sorted({p for r in members for p in r['paths']}),
               'review_statuses': sorted({r['review_status'] for r in members if r['review_status']}),
               'observer_identity_quality': sorted({r['observer_identity_quality'] for r in members}),
               'licenses': sorted({r['license'] for r in members if r['license']}),
               'attributions': sorted({r['attribution'] for r in members if r['attribution']})}
        merged.append(out)
    return merged


def build_history(out, excludes=()):
    out.mkdir(parents=True, exist_ok=False)
    excluded = [Path(x).resolve() for x in excludes] + [ROOT / 'external_datasets' / d for d in NEW_DIRS]
    cache = load(CACHE)
    assert not cache['parse_errors']
    records = []; files = []
    for path in metadata_inputs(excluded):
        raw = path.read_text(encoding='utf-8-sig')
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()] if path.suffix == '.jsonl' else list_rows(json.loads(raw))
        for row in rows:
            rec = record_identity(row, path)
            if any(rec['tokens'][k] for k in ('sha256', 'photo', 'path', 'observation')):
                if path.parent.name.endswith('human_challenge1') or path.parent.name.endswith('human_challenge2'):
                    rec['phase'] = 'challenge'
                if path.name.startswith('old_') and path.name.endswith('_rows.json'):
                    rec['phase'] = path.name.removeprefix('old_').removesuffix('_rows.json')
                elif path.parent.name == '20260908_0045' and path.name.endswith('_rows.json'):
                    rec['phase'] = path.name.removesuffix('_rows.json')
                records.append(rec)
        files.append({'path': str(path), 'sha256': digest(path), 'rows': len(rows), 'identity_only': True})
    # Recover actual observer IDs from saved public-API responses, without new
    # requests. Only enrich observations/photos already present in this registry;
    # fetched-but-unused observations do not become imaginary historical images.
    known_obs = {x for r in records for x in r['tokens']['observation']}
    known_photo = {x for r in records for x in r['tokens']['photo']}
    enriched = 0
    for path in sorted((ROOT / 'external_datasets').glob('*/source_responses/*.json')):
        if any(path.resolve().is_relative_to(x) for x in excluded): continue
        data = load(path)
        if not isinstance(data, dict): continue
        observations = data.get('results', [])
        if not isinstance(observations, list): continue
        n = 0
        for o in observations:
            if not isinstance(o, dict) or not isinstance(o.get('user'), dict) or not isinstance(o.get('photos'), list): continue
            if 'inat:' + str(o.get('id')) not in known_obs: continue
            for photo in o['photos']:
                if 'inat:' + str(photo.get('id')) not in known_photo: continue
                row = {'source_species': (o.get('taxon') or {}).get('name'), 'inat_observation_id': o['id'],
                    'photo_id': photo['id'], 'source_page': o.get('uri'), 'photo_url': photo.get('url'),
                    'observer_id': o['user'].get('id'), 'observer_login': o['user'].get('login'),
                    'photo_attribution': photo.get('attribution'), 'photo_license_code': photo.get('license_code')}
                records.append(record_identity(row, path)); n += 1
        if n:
            enriched += n
            files.append({'path': str(path), 'sha256': digest(path), 'rows': n, 'identity_only': True, 'use': 'Historical observer-ID enrichment only'})
    review_audit = []
    for manifest, database, palm_filter in queue_config():
        mp, dp = ROOT / manifest, ROOT / database
        rows = list_rows(load(mp)); decisions, dbhash = db_decisions(dp)
        by_id = {text_value(d.get('item_id') or d.get('id')): d for d in decisions}
        statuses = collections.Counter(); excluded_duplicates = 0
        for row in rows:
            if palm_filter and row.get('exact_duplicate_matches'):
                excluded_duplicates += 1; continue
            rec = record_identity(row, mp)
            d = by_id.get(rec['id'])
            valid = d and (not d.get('sha256') or d['sha256'] in rec['tokens']['sha256']) and (not d.get('source_species') or d['source_species'] == rec['species'])
            status = str(d.get('decision') or d.get('status')) if valid else 'pending'
            rec['review_status'] = status; statuses[status] += 1; records.append(rec)
        review_audit.append({'manifest': str(mp), 'database': str(dp), 'database_sha256': dbhash,
                             'actual_queue_statuses': dict(statuses), 'excluded_collection_duplicates': excluded_duplicates})
    merged = merge_records(records)
    registry = {'schema_version': 1, 'purpose': 'Exclusion identity registry, NEVER an accepted-training manifest',
        'scope': 'Existing cached all-history identity union plus refreshed canonical split/source metadata and live review queues. No historical images or prediction files accessed.',
        'cache_path': str(CACHE), 'cache_sha256': digest(CACHE),
        'cache_limit': 'Old cached identities are kept conservatively even when an origin has changed; refreshed metadata is additive. Cache is not approval evidence.',
        'excluded_current_acquisition_roots': list(map(str, excluded)), 'metadata_files': files, 'review_queues': review_audit,
        'observer_enrichment_metadata_rows': enriched,
        'global_cached_identity_maps': cache['identity_maps'], 'records': merged,
        'limitations': ['Photographer display names are collision proxies, not verified user IDs.', 'pHash can miss severe crops, mirrors and different views; it is not proof of same-tree identity.', 'A source gallery or observer group is not a verified individual tree.', 'Freshness is relative to recorded project history, not unknown BioCLIP pretraining images.', 'No historical raster was opened; missing old perceptual hashes remain visible in inventory.']}
    save_new(out / 'history_registry.json', registry)
    white = [r for r in merged if SPECIES in r['species']]
    sources = {}
    for source in sorted({s for r in white for s in (r['source_sites'] or ['unspecified'])}):
        subset = [r for r in white if source in (r['source_sites'] or ['unspecified'])]
        sources[source] = {'images': len(subset), 'observations': len({x for r in subset for x in r['tokens']['observation']}),
            'observer_tokens': sorted({x for r in subset for x in r['tokens']['observer']}),
            'stable_observer_ids': sorted({x for r in subset for x in r['tokens']['observer'] if x.startswith('inat:id:')}),
            'galleries': sorted({x for r in subset for x in r['tokens']['gallery']}),
            'missing_observer_images': sum(not r['tokens']['observer'] for r in subset),
            'phase_counts': dict(collections.Counter(p for r in subset for p in r['phases'])),
            'actual_review_status_counts': dict(collections.Counter(s for r in subset for s in r['review_statuses']))}
    summary = {'species': SPECIES, 'unique_image_identity_groups': len(white), 'images_with_sha': sum(bool(r['tokens']['sha256']) for r in white),
        'images_with_phash': sum(bool(r['phashes']) for r in white), 'observations': len({x for r in white for x in r['tokens']['observation']}),
        'stable_observer_ids': sorted({x for r in white for x in r['tokens']['observer'] if x.startswith('inat:id:')}),
        'display_or_group_observer_tokens': sorted({x for r in white for x in r['tokens']['observer'] if not x.startswith(('inat:id:', 'inat:login:'))}),
        'source_coverage': sources, 'all_registry_images': len(merged), 'cached_identity_counts': {k: len(v) for k, v in cache['identity_maps'].items()},
        'actual_review_status_counts': dict(collections.Counter(s for r in white for s in r['review_statuses'])),
        'phase_counts': dict(collections.Counter(p for r in white for p in r['phases'])),
        'independent_source_options_not_yet_verified': ['New iNaturalist observer IDs + non-overlapping observations; audit across positive and negative classes together.', 'A different documented botanical institution or independently attributed photo source not in source_coverage; verify species and rights first.'],
        'limitations': registry['limitations']}
    save_new(out / 'historical_artocarpus_inventory.json', white)
    save_new(out / 'historical_summary.json', summary)
    write_protocol(out, summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('display_or_group_observer_tokens', 'source_coverage')}, ensure_ascii=False, indent=2))


def lookup_maps(registry):
    maps = {k: collections.defaultdict(set) for k in TOKEN_TYPES}; visual = collections.defaultdict(set)
    for i, r in enumerate(registry['records']):
        for k in TOKEN_TYPES:
            for value in r['tokens'][k]: maps[k][value].add(i)
        for p in r['phashes']: visual[int(p, 16)].add(i)
    return maps, visual


def verify_new_phash(path, supplied):
    """Decode only an incoming acquisition, never a historical/test raster."""
    import numpy as np
    from PIL import Image, ImageOps
    from scipy.fft import dctn
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image)
        coeff = dctn(np.asarray(rgb.convert('L').resize((32, 32)), dtype=float), norm='ortho')[:8, :8].flatten()
        value = int(''.join('1' if x else '0' for x in coeff > np.median(coeff[1:])), 2)
    computed = f'{value:016x}'
    return computed in supplied, computed


def historical_matches(row, registry, maps, visual, distance):
    matches = {}; indices = set()
    for kind in TOKEN_TYPES:
        hit = {i for value in row['tokens'][kind] for i in maps[kind].get(value, ())}
        if hit: matches[kind] = sorted(hit); indices |= hit
    close = {i for p in row['phashes'] for h, values_ in visual.items() if (int(p, 16) ^ h).bit_count() <= distance for i in values_}
    if close: matches['near_phash'] = sorted(close); indices |= close
    cached = registry['global_cached_identity_maps']; cmatches = {}
    cache_types = {'sha256': 'hashes', 'path': 'paths', 'photo': 'photos', 'observation': 'observations', 'image_url': 'urls'}
    for kind, ck in cache_types.items():
        hits = [v for v in row['tokens'][kind] if v in cached[ck] or (kind == 'path' and v.replace('/', '\\') in cached[ck])]
        if hits: cmatches[kind] = hits
    phits = [f'{int(h):016x}' for p in row['phashes'] for h in cached['visual'] if (int(p, 16) ^ int(h)).bit_count() <= distance]
    if phits: cmatches['near_phash'] = phits[:50]
    return matches, indices, cmatches


def audit_new(registry_path, manifests, out, reserve_observers=(), reserve_sources=(), near=6):
    """Callable audit for positive and negative raw manifests in the SAME batch.

    Reserve groups before any fitting/score inspection. No random split. All outputs
    retain training_eligible=False until a separate explicit label/rights review.
    """
    out.mkdir(parents=True, exist_ok=False)
    reg = load(registry_path); maps, visual = lookup_maps(reg)
    normalized = []; raw_rows = []; input_snapshots = []
    for path in map(Path, manifests):
        raw = path.read_bytes()
        input_snapshots.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()})
        for r in list_rows(json.loads(raw.decode('utf-8-sig'))):
            rec = record_identity(r, path)
            if rec['tokens']['sha256'] or rec['paths']:
                normalized.append(rec); raw_rows.append(r)
    audited = []
    for r, original in zip(normalized, raw_rows):
        hits, ids, cached = historical_matches(r, reg, maps, visual, near)
        # Observer-only reuse excludes fresh observer-heldout status but is not
        # by itself a train-vs-historical-test image leakage claim.
        content_keys = {'sha256', 'path', 'photo', 'sample', 'observation', 'image_url', 'group', 'gallery', 'near_phash'}
        content_ids = {i for k, vs in hits.items() if k in content_keys for i in vs}
        hold_content = sorted(i for i in content_ids if HELD & set(reg['records'][i]['phases']))
        hold_observer = sorted(i for i in set(hits.get('observer', [])) if HELD & set(reg['records'][i]['phases']))
        hold = sorted(set(hold_content) | set(hold_observer))
        bad = sorted(i for i in content_ids if BLOCKED_REVIEW & set(reg['records'][i]['review_statuses']))
        old = bool(content_ids or cached)
        stable = [x for x in r['tokens']['observer'] if x.startswith(('inat:id:', 'inat:login:'))]
        site_known = any(r['source_site'] in x['source_sites'] for x in reg['records'])
        observer_novel = bool(stable) and not hits.get('observer')
        new_source = r['source_site'] != 'unspecified' and not site_known
        integrity = False; phash_verified = False; computed_phash = None; error = None
        if r['paths'] and r['tokens']['sha256']:
            try:
                integrity = digest(r['paths'][0]) in r['tokens']['sha256']
                if integrity: phash_verified, computed_phash = verify_new_phash(r['paths'][0], r['phashes'])
            except (OSError, ValueError) as e: error = str(e)
        reasons = []
        if not integrity: reasons.append('new_image_integrity_missing_or_failed')
        if not phash_verified: reasons.append('new_phash_missing_or_failed')
        if bad: reasons.append('actual_historical_reject_pending_uncertain_or_unusable')
        if hold: reasons.append('historical_holdout_identity_or_near_duplicate_or_gallery')
        if old: reasons.append('historically_seen_image_observation_or_gallery')
        holdout_reasons = list(reasons)
        if not observer_novel and not new_source: holdout_reasons.append('no_proven_new_observer_or_source')
        # These are identity-screening passes ONLY, not scientific-label approval.
        train_pass = integrity and phash_verified and not bad and not hold and not old
        fresh_pass = train_pass and (observer_novel or new_source)
        audited.append(dict(r, raw_manifest_id=original.get('id'), integrity_verified=integrity, integrity_error=error,
            phash_verified=phash_verified, recomputed_new_phash=computed_phash,
            historical_matches=hits, cached_historical_matches=cached, old_holdout_record_ids=hold, blocked_review_record_ids=bad,
            old_holdout_content_record_ids=hold_content, old_holdout_observer_record_ids=hold_observer,
            submit_as_new_review_item=not bool(bad),
            identity_train_screen_pass=train_pass, fresh_holdout_screen_pass=fresh_pass, train_screen_reasons=reasons,
            fresh_holdout_reasons=holdout_reasons, eligible_holdout_unit=('observer' if observer_novel else 'source' if new_source else None),
            freshness_scope='Only recorded project identities. Missing historical observer IDs and cross-site re-use still need review; not proof of independent trees or absence from pretrained data.',
            fresh_holdout_claim_ready=False,
            training_eligible=False, label_decision='not_made_by_this_identity_tool', split='unassigned_pending_label_rights_and_visual_review'))
    # Bind multi-photo observations, observer series and source galleries across
    # positive+negative acquisitions. Near duplicates are bound, never discarded.
    uf = UnionFind(len(audited)); first = {}; edges = []
    for i, r in enumerate(audited):
        for k in ('sha256', 'photo', 'sample', 'observation', 'group', 'gallery', 'observer'):
            for value in r['tokens'][k]:
                key = k, value
                if key in first: uf.union(i, first[key]); edges.append({'a': first[key], 'b': i, 'reason': k, 'value': value})
                else: first[key] = i
        for j in range(i):
            if any((int(a, 16) ^ int(b, 16)).bit_count() <= near for a in r['phashes'] for b in audited[j]['phashes']):
                uf.union(i, j); edges.append({'a': j, 'b': i, 'reason': 'near_phash'})
    groups = collections.defaultdict(list)
    for i in range(len(audited)): groups[uf.find(i)].append(i)
    group_rows = []
    for indices in groups.values():
        members = [audited[i] for i in indices]
        observers = {x for r in members for x in r['tokens']['observer']}
        sites = {r['source_site'] for r in members}
        reserve = bool(set(reserve_observers) & observers or set(reserve_sources) & sites)
        permitted = all(r['fresh_holdout_screen_pass'] if reserve else r['identity_train_screen_pass'] for r in members)
        proposal = ('reserved_fresh_holdout_pending_review' if reserve else 'unassigned_train_or_development_pending_review') if permitted else 'quarantine_group'
        key = 'identity_component_' + hashlib.sha256('|'.join(sorted(r['id'] or str(i) for i, r in zip(indices, members))).encode()).hexdigest()[:16]
        for r in members: r['split_group'] = key; r['proposed_split'] = proposal
        group_rows.append({'split_group': key, 'row_indices': indices, 'ids': [r['id'] for r in members], 'observer_tokens': sorted(observers), 'source_sites': sorted(sites), 'proposed_split': proposal})
    for item in input_snapshots:
        assert digest(item['path']) == item['sha256'], f"Source manifest changed during audit: {item['path']}"
    result = {'schema_version': 1, 'tool_path': str(Path(__file__).resolve()), 'tool_sha256': digest(__file__),
        'registry_path': str(Path(registry_path).resolve()), 'registry_sha256': digest(registry_path),
        'manifests': input_snapshots,
        'near_phash_hamming64': near, 'reserved_observer_tokens': list(reserve_observers), 'reserved_sources': list(reserve_sources),
        'rows': audited, 'groups': group_rows, 'group_edges': edges,
        'summary': {'images': len(audited), 'identity_train_screen_pass': sum(r['identity_train_screen_pass'] for r in audited),
            'fresh_holdout_screen_pass': sum(r['fresh_holdout_screen_pass'] for r in audited), 'training_eligible': 0,
            'proposed_splits': dict(collections.Counter(r['proposed_split'] for r in audited))},
        'required_next_gates': ['Source species corroboration and visual whole-tree/crown quality review, never model-derived labels.', 'Source-specific rights check and recorded experiment authority; accepted quality is not a licence.', 'Freeze all observer/gallery components into train/development/holdout before fitting or any score inspection.', 'Check near-hash ambiguous pairs manually; never relax an old human rejection automatically.', 'Verify every-run weight backups and fixed fair baseline before any fit.'],
        'limitations': reg['limitations']}
    save_new(out / 'new_batch_identity_audit.json', result)
    exclusions = []
    for r in audited:
        if r['blocked_review_record_ids']:
            exclusions.append({'id': r['id'], 'sha256': r['tokens']['sha256'],
                'historical_decisions': [{'ids': reg['records'][i]['ids'], 'statuses': reg['records'][i]['review_statuses'], 'origins': reg['records'][i]['origins']} for i in r['blocked_review_record_ids']],
                'submit_as_new_review_item': False, 'training_eligible': False,
                'reason': 'Do not erase or reopen an actual historical human decision by redownloading with a new ID.'})
    save_new(out / 'historical_review_exclusions.json', exclusions)
    print(json.dumps(result['summary'], ensure_ascii=False))
    return result


def write_protocol(out, summary):
    lines = ['# 白桂木新数据隔离与复核协议', '',
        '本工具只审计身份；不训练、不看历史测试图片或预测、不生成模型标签，不写任何旧数据库。历史测试集只读取身份元数据以排除泄漏。', '',
        f"历史白桂木：{summary['unique_image_identity_groups']} 个图像身份组；{summary['observations']} 个记录的 iNaturalist 观察；{len(summary['stable_observer_ids'])} 个已记录稳定观察者 ID。多张图或多个观察并不等于独立树。", '',
        '## 强制规则', '',
        '1. Fresh holdout 排除历史任意 train / development / test / pending / reject 的同图、同照片 ID、同观察、多角度系列、近重复和来源图库；需要新观察者或新独立来源。',
        '2. 新训练图排除真实人审 pending / reject / uncertain / unusable，以及历史 holdout 的同 SHA、pHash 近重复、观察、图库组；重抓旧训练图也不算新数据。',
        '3. SHA / 原照片 ID / 观察 ID 是硬身份；pHash ≤ 6 是保守隔离，不直接宣称同树。不同裁切、镜像及跨时相仍需人工排查。',
        '4. 正例与相似负例必须一起审计；同观察、摄影者及图库组成连通分组。绝不逐图随机拆分，同树多角度全留一侧。',
        '5. 先按来源与观察者锁定 holdout，再看任何模型分数；少量新图不足以宣称独立树泛化。已多次用过的 15 张仍为开发集。',
        '6. 人审以 live human_review.sqlite3 为准；Palm 的 exact_duplicate_matches 记录不属于其实际审核队列。导入器占位 pending 不冒充真实人工待审，也不冒充确认。',
        '7. 工具所有输出 training_eligible=false。物种来源＋必要人工复核、图像质量、版权、实验授权与训练备份是后续独立条件。', '',
        '## 来源覆盖', '', '| 来源 | 图像身份组 | 观察 | 缺少摄影者信息 |', '|---|---:|---:|---:|']
    for s, d in summary['source_coverage'].items(): lines.append(f"| {s} | {d['images']} | {d['observations']} | {d['missing_observer_images']} |")
    lines += ['', '同图出现在不同来源时来源计数可重叠；总数不是来源数相加。完整 SHA、pHash、身份别名、来源文件及审核状态见 JSON。', '',
        '## 使用', '',
        '`python tools/artocarpus_fresh_split_20260917.py audit --registry outputs/artocarpus_fresh_split_20260917_v1/history_registry.json --manifest <positive manifest> --manifest <negative manifest> --out <new versioned audit directory>`', '',
        '需要保留某观察者时增加 `--reserve-observer inat:id:123`；保留新机构来源时增加 `--reserve-source SiteName`。未明确保留时不随机分配，也不自动将图加入训练。', '',
        '## 新独立来源方向', '',
        '优先未出现在历史清单的 iNaturalist 稳定观察者及其新观察；另可找不同植物机构或有独立摄影者、明确版权与种名依据的来源。这里只提出检索方向，未声称已取得新图片。', '',
        '观察者显示名可能重名、改名；历史信息缺失时不能证明观察者独立。Fresh 仅指本项目已记录历史，不保证未出现在 BioCLIP 预训练中。', '']
    with (out / 'PROTOCOL.md').open('x', encoding='utf-8') as f: f.write('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('--out', type=Path, default=DEFAULT_OUT); b.add_argument('--exclude', type=Path, action='append', default=[])
    a = sub.add_parser('audit'); a.add_argument('--registry', type=Path, default=DEFAULT_OUT / 'history_registry.json')
    a.add_argument('--manifest', type=Path, action='append', required=True); a.add_argument('--out', type=Path, required=True)
    a.add_argument('--reserve-observer', action='append', default=[]); a.add_argument('--reserve-source', action='append', default=[])
    a.add_argument('--near', type=int, default=6, choices=range(0, 13))
    args = p.parse_args()
    if args.command == 'build': build_history(args.out.resolve(), args.exclude)
    else: audit_new(args.registry.resolve(), args.manifest, args.out.resolve(), args.reserve_observer, args.reserve_source, args.near)


if __name__ == '__main__': main()
