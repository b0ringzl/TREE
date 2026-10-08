"""Collect a small licensed reference library; never touches training data/weights.

Public iNaturalist API, globally rate-limited to <= 1 request/second.
Only exact scientific-name/synonym matches and research-grade observations.
Every photo has its own reusable license; observation license is insufficient.
"""
import argparse
import hashlib
import io
import json
import re
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
APP = Path(__file__).resolve().parent
LICENSES = {'cc0': 'https://creativecommons.org/publicdomain/zero/1.0/',
            'cc-by': 'https://creativecommons.org/licenses/by/4.0/',
            'cc-by-sa': 'https://creativecommons.org/licenses/by-sa/4.0/'}
API = 'https://api.inaturalist.org/v1/'
GROUPS = {
    'Ficus microcarpa / Ficus benjamina (榕樹)': ['Ficus microcarpa', 'Ficus benjamina'],
    'Araucaria columnaris(Araucaria heterophylla)': ['Araucaria columnaris', 'Araucaria heterophylla'],
}


def norm(s):
    return re.sub(r'\s+', ' ', s.replace('×', 'x')).strip().lower()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class Client:
    def __init__(self, root):
        self.root = root
        self.cache = root / 'api_cache'
        self.cache.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'SpaceScanTreesReferenceGallery/1.0 (small licensed botanical reference collection)'
        self.last = 0

    def get(self, url, params=None):
        # One serial downloader; bounded responses and no rate-limit bypass.
        if url.startswith(API):
            time.sleep(max(0, 1.1 - (time.monotonic() - self.last)))
            self.last = time.monotonic()
        r = self.session.get(url, params=params, timeout=(10, 30), stream=True)
        if r.status_code in (401, 403, 429):
            r.close()
            raise RuntimeError('Access restriction or rate limit: ' + str(r.status_code) + ' ' + url)
        r.raise_for_status()
        data = bytearray()
        for chunk in r.iter_content(65536):
            data.extend(chunk)
            if len(data) > 20_000_000:
                r.close()
                raise ValueError('Response exceeds 20 MB')
        r.close()
        return bytes(data)

    def json(self, endpoint, params):
        key = hashlib.sha256(json.dumps([endpoint, params], sort_keys=True).encode()).hexdigest()
        dest = self.cache / (key + '.json')
        if dest.exists():
            return json.loads(dest.read_text(encoding='utf-8'))['response']
        d = json.loads(self.get(API + endpoint, params))
        write_json(dest, {'url': API + endpoint, 'params': params, 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'response': d})
        return d

    def resolve(self, name):
        d = self.json('taxa', {'q': name, 'per_page': 10, 'all_names': 'true'})
        matches = []
        for t in d['results']:
            names = [t['name']] + [n['name'] for n in t.get('names', []) if n.get('locale') == 'sci']
            if t.get('is_active') and t.get('iconic_taxon_id') == 47126 and t.get('rank_level', 100) <= 10 and norm(name) in {norm(n) for n in names}:
                matches.append(t)
        exact = [t for t in matches if norm(t['name']) == norm(name)]
        if len(exact) == 1:
            return exact[0]
        if len(matches) != 1:
            raise ValueError('No unique exact scientific-name/synonym match: ' + name)
        return matches[0]


def quality(raw):
    with Image.open(io.BytesIO(raw)) as original:
        if original.width * original.height > 40_000_000:
            raise ValueError('Too many pixels')
        im = ImageOps.exif_transpose(original).convert('RGB')
    if min(im.size) < 320:
        raise ValueError('Short edge below 320 px')
    gray = np.asarray(im.convert('L').resize((160, 160)), dtype=float)
    dark = float(np.mean(gray < 25))
    bright = float(np.mean(gray > 248))
    lap = float(np.var(gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1] - 4 * gray[1:-1, 1:-1]))
    if dark > .65 or bright > .65 or lap < 35:
        raise ValueError('Poor exposure or low texture')
    small = np.asarray(im.convert('L').resize((9, 8)))
    bits = small[:, 1:] > small[:, :-1]
    dhash = int(''.join('1' if b else '0' for b in bits.flat), 2)
    return im, {'width': im.width, 'height': im.height, 'dark_fraction': dark, 'bright_fraction': bright, 'laplacian': lap, 'dhash': format(dhash, '016x')}


def collect(client, label, queries, entries, seen, errors, per_class=2):
    current = [r for r in entries if r['species'] == label and r.get('visual_review') != 'rejected_for_reference']
    for query in queries:
        quota = per_class // len(queries)
        if sum(r['query'] == query for r in current) >= quota:
            continue
        taxon = client.resolve(query)
        observations = client.json('observations', {'taxon_id': taxon['id'], 'quality_grade': 'research',
            'photos': 'true', 'photo_license': ','.join(LICENSES), 'per_page': 40, 'order_by': 'votes', 'order': 'desc'})['results']
        # Prefer different observations; only reuse one when no other eligible image exists.
        for observation_index, obs in enumerate(observations + observations):
            if sum(r['query'] == query for r in current) >= quota:
                break
            ot = obs.get('taxon') or {}
            if obs.get('quality_grade') != 'research' or obs.get('flags') or obs.get('spam'):
                continue
            if taxon['id'] != ot.get('id') and taxon['id'] not in ot.get('ancestor_ids', []):
                continue
            if observation_index < len(observations) and any(r['observation_id'] == obs['id'] for r in current):
                continue
            for photo in obs.get('photos', []):
                license_code = (photo.get('license_code') or '').lower()
                if license_code not in LICENSES or photo.get('flags') or str(photo['id']) in seen:
                    continue
                dims = photo.get('original_dimensions') or {}
                if min(dims.get('width', 0), dims.get('height', 0)) < 600:
                    continue
                url = re.sub(r'/(square|small|medium)\.', '/large.', photo['url'])
                if urlparse(url).scheme != 'https' or urlparse(url).hostname not in {'inaturalist-open-data.s3.amazonaws.com', 'static.inaturalist.org'}:
                    continue
                try:
                    raw = client.get(url)
                    _, q = quality(raw)
                    sha = hashlib.sha256(raw).hexdigest()
                    if any(r['sha256'] == sha for r in entries):
                        continue
                    if any((int(r['quality']['dhash'], 16) ^ int(q['dhash'], 16)).bit_count() <= 5 for r in current):
                        continue
                    ident = 'inat_' + str(photo['id'])
                    dest = client.root / 'images' / (ident + '.jpg')
                    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != sha:
                        raise RuntimeError('Existing image bytes changed: ' + ident)
                    if not dest.exists():
                        dest.write_bytes(raw)  # Downloaded original large variant, no watermark removal/edit.
                    row = {'id': ident, 'species': label, 'query': query, 'resolved_taxon': taxon['name'],
                        'observed_taxon': ot.get('name'), 'taxon_id': taxon['id'], 'observation_id': obs['id'],
                        'photo_id': photo['id'], 'source_url': obs['uri'], 'photo_page': 'https://www.inaturalist.org/photos/' + str(photo['id']),
                        'image_url': url, 'file': 'images/' + dest.name, 'sha256': sha, 'bytes': len(raw), 'quality': q,
                        'license': license_code, 'license_url': LICENSES[license_code], 'attribution': photo.get('attribution'),
                        'author': photo.get('attribution_name') or obs['user'].get('name') or obs['user']['login'],
                        'date': obs.get('observed_on'), 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                        'identification_basis': 'iNaturalist research grade; not independent botanical verification',
                        'visual_review': 'pending', 'training_eligible': False}
                    entries.append(row)
                    current.append(row)
                    seen.add(str(photo['id']))
                    write_json(client.root / 'manifest.json', entries)
                    break
                except (requests.RequestException, ValueError, OSError) as e:
                    errors.append({'species': label, 'photo_id': photo['id'], 'error': str(e)})


def sheets(root, entries):
    for start in range(0, len(entries), 24):
        canvas = Image.new('RGB', (1600, 6 * 260), 'white')
        for j, row in enumerate(entries[start:start + 24]):
            x, y = (j % 4) * 400, (j // 4) * 260
            with Image.open(root / row['file']) as src:
                im = ImageOps.exif_transpose(src).convert('RGB')
                im.thumbnail((390, 210))
                canvas.paste(im, (x + (400-im.width)//2, y))
            draw = ImageDraw.Draw(canvas)
            draw.text((x+5,y+213), str(start+j+1) + ' ' + row['query'], fill='black')
            draw.text((x+5,y+231), row['id'] + ' ' + row['license'], fill='black')
        canvas.save(root / ('contact_%02d.jpg' % (start//24+1)), quality=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int, default=148)
    parser.add_argument('--per-class', type=int, default=2)
    parser.add_argument('--seed-library', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and not args.resume:
        raise RuntimeError('Use a new versioned directory or explicit --resume for collection (not training).')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(exist_ok=True)
    names = json.loads((APP / 'species_names.json').read_text(encoding='utf-8'))
    labels = json.loads((ROOT / 'outputs/distillation/bioclip_probe_20260908_0227/classes.json').read_text(encoding='utf-8'))['labels'] + ['Wodyetia bifurcata']
    entries = json.loads((out/'manifest.json').read_text(encoding='utf-8')) if (out/'manifest.json').exists() else []
    if args.seed_library and not entries:
        seed = args.seed_library.resolve()
        entries = json.loads((seed/'manifest.json').read_text(encoding='utf-8'))
        for row in entries:
            shutil.copy2(seed/row['file'], out/row['file'])
            row['visual_review'] = 'pending_whole_tree_review'
        write_json(out/'manifest.json', entries)
    decisions = json.loads((out/'visual_review.json').read_text(encoding='utf-8')) if (out/'visual_review.json').exists() else {}
    for row in entries:
        if row['id'] in decisions:
            row['visual_review'] = decisions[row['id']]['status']
            row['review_note'] = decisions[row['id']]['note']
    seen = {str(r['photo_id']) for r in entries}
    errors = []
    client = Client(out)
    if args.seed_library:
        for p in (args.seed_library/'api_cache').glob('*.json'):
            dest = client.cache/p.name
            if not dest.exists():
                shutil.copy2(p, dest)
    for i, label in enumerate(labels[:args.limit]):
        queries = GROUPS.get(label, [label.split('(')[0].strip()])
        try:
            collect(client, label, queries, entries, seen, errors, args.per_class)
        except (requests.RequestException, ValueError, OSError) as e:
            errors.append({'species': label, 'error': str(e)})
        counts = {s: sum(r['species'] == s and r.get('visual_review') != 'rejected_for_reference' for r in entries) for s in labels}
        write_json(out/'summary.json', {'classes': len(labels), 'images': len(entries), 'complete_classes': sum(n == 2 for n in counts.values()),
            'counts': counts, 'missing': {s:n for s,n in counts.items() if n < 2}, 'training_eligible': False})
        write_json(out/'errors.json', errors)
        print(f'{i+1}/{len(labels)} {label}: {counts[label]}/{args.per_class}; total={len(entries)}', flush=True)
    sheets(out, entries)
    write_json(out/'manifest.json', entries)
    write_json(out/'collection_complete.json', {'images':len(entries), 'per_class_requested':args.per_class})


if __name__ == '__main__':
    main()
