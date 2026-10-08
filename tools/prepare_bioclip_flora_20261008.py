"""Pin and verify official BioCLIP 2.5 text embeddings; no training or image upload."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests

from training_safety import sha256, verify_backup

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/bioclip_flora_integration_20261008_v1'
DEST = ROOT / 'external_datasets/bioclip25_flora_20261008_v1'
REPO = 'imageomics/TreeOfLife-200M'
FILES = ['embeddings/txt_emb_bioclip-2.5-vith14.json',
         'embeddings/txt_emb_bioclip-2.5-vith14.npy']
ENCODER_SHA = 'ac2e37c2f89ef8e6b889176a9a3f418970ad9db15a218bd29e3321e95c46ae97'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def download(url, target, size, digest):
    if target.exists():
        if target.stat().st_size != size or sha256(target) != digest:
            raise RuntimeError('Existing source failed checksum; never overwrite: ' + str(target))
        return
    partial = target.with_suffix(target.suffix + '.part')
    last_log = 0
    for attempt in range(8):
        start = partial.stat().st_size if partial.exists() else 0
        if start == size:
            break
        try:
            headers = {'Range': f'bytes={start}-'} if start else {}
            with requests.get(url, stream=True, headers=headers, timeout=(30, 45)) as response:
                response.raise_for_status()
                if start and response.status_code != 206:
                    raise RuntimeError('Server did not honor download resume')
                with partial.open('ab' if start else 'wb') as stream:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        stream.write(chunk)
                        start += len(chunk)
                        if time.monotonic() - last_log > 12:
                            print(f'{target.name}: {start / size:.1%}', flush=True)
                            last_log = time.monotonic()
        except requests.RequestException as exc:
            print(f'download retry {attempt + 1}: {type(exc).__name__}', flush=True)
            continue
    if partial.stat().st_size != size or sha256(partial) != digest:
        raise RuntimeError('Downloaded official artifact failed checksum')
    partial.rename(target)


def main():
    verify_backup(RUN / 'pre_training_backup')
    DEST.mkdir(exist_ok=True)
    lock = DEST / 'source_lock.json'
    if lock.exists():
        source = json.loads(lock.read_text(encoding='utf-8'))
    else:
        response = requests.get(f'https://huggingface.co/api/datasets/{REPO}', timeout=30)
        response.raise_for_status()
        revision = response.json()['sha']
        response = requests.get(f'https://huggingface.co/api/datasets/{REPO}/tree/{revision}/embeddings', timeout=30)
        response.raise_for_status()
        items = {r['path']: r for r in response.json()}
        source = {'repo': REPO, 'revision': revision, 'license': 'CC0-1.0',
                  'documentation': 'https://huggingface.co/datasets/imageomics/TreeOfLife-200M/blob/main/embeddings/README.md',
                  'files': [{'path': path, 'bytes': items[path]['size'], 'sha256': items[path]['lfs']['oid']}
                            for path in FILES]}
        write(lock, source)
    for row in source['files']:
        download(f"https://huggingface.co/datasets/{REPO}/resolve/{source['revision']}/{row['path']}",
                 DEST / Path(row['path']).name, row['bytes'], row['sha256'])
    manifest = DEST / 'manifest.json'
    if manifest.exists():
        print('Already prepared; no overwrite.', flush=True)
        return
    labels = json.loads((DEST / Path(FILES[0]).name).read_text(encoding='utf-8'))
    embeddings = np.load(DEST / Path(FILES[1]).name, mmap_mode='r')
    print('Official embeddings:', embeddings.shape, embeddings.dtype, flush=True)
    if embeddings.ndim != 2 or 1024 not in embeddings.shape or len(labels) not in embeddings.shape:
        raise RuntimeError('Official labels/encoder embedding dimensions mismatch')
    rows_first = embeddings.shape[0] == len(labels)
    records, indices, seen = [], [], set()
    for index, (taxa, common) in enumerate(labels):
        if len(taxa) != 7 or taxa[0] != 'Plantae':
            continue
        genus, epithet = taxa[5], taxa[6]
        name = epithet if epithet.startswith(genus + ' ') else genus + ' ' + epithet
        name = name.strip()
        # Collapse exact species-name duplicates; retain the first official taxonomic path.
        if name in seen:
            continue
        seen.add(name)
        records.append({'species': name, 'common_name': common, 'family': taxa[4],
                        'genus': genus, 'taxonomy': taxa, 'official_index': index})
        indices.append(index)
    output = np.lib.format.open_memmap(DEST / 'plant_embeddings.npy', mode='w+',
                                      dtype=np.float32, shape=(len(indices), 1024))
    for start in range(0, len(indices), 4096):
        block_indices = indices[start:start + 4096]
        block = np.asarray(embeddings[block_indices, :] if rows_first else embeddings[:, block_indices].T,
                           dtype=np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        if not np.isfinite(block).all() or (norms < .1).any():
            raise RuntimeError('Invalid official text vectors')
        output[start:start + len(block_indices)] = block / norms
    output.flush()
    del output
    write(DEST / 'plants.json', records)
    result = {'version': 'bioclip25-official-plantae-20261008-v1', 'model': 'imageomics/bioclip-2.5-vith14',
              'encoder_sha256': ENCODER_SHA, 'embedding_dimension': 1024,
              'plant_species_count': len(records), 'official_taxon_count': len(labels),
              'source': source, 'training': False, 'scope': 'Official TreeOfLife Plantae catalogue; not all plants on Earth',
              'score_policy': 'Softmax over this fixed plant catalogue at original model logit scale, temperature 1; uncalibrated',
              'derived_files': [{'file': p, 'sha256': sha256(DEST / p), 'bytes': (DEST / p).stat().st_size}
                                for p in ['plant_embeddings.npy', 'plants.json']]}
    write(manifest, result)
    print('Verified plant species:', len(records), flush=True)


if __name__ == '__main__':
    main()
