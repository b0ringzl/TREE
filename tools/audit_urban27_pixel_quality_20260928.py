"""Non-destructive per-image quality and near-duplicate features for the frozen snapshot."""

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/urban27_image_quality_20260928_v1'
BACKUP = RUN / 'pre_curation_backup'
OUTPUT = RUN / 'pixel_quality.jsonl'


def entropy(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = hist[hist > 0] / hist.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def dhash(gray):
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (resized[:, 1:] > resized[:, :-1]).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f'{value:016x}'


def main():
    verification = json.loads((RUN / 'pre_curation_backup_verification.json').read_text(encoding='utf-8'))
    if not verification['sha256_verified']:
        raise RuntimeError('Backup was not verified')
    images = sorted(path for folder in BACKUP.iterdir() if folder.is_dir() for path in folder.iterdir() if path.is_file())
    if len(images) != verification['files']:
        raise RuntimeError('Image count mismatch')
    with OUTPUT.open('w', encoding='utf-8') as handle:
        for index, path in enumerate(images, 1):
            record = {'relative_path': str(path.relative_to(BACKUP)), 'bytes': path.stat().st_size}
            try:
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert('RGB')
                record['width'], record['height'] = image.size
                image.thumbnail((256, 256), Image.Resampling.BICUBIC)
                rgb = np.asarray(image)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                record.update(
                    status='ok',
                    laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    dark_fraction=float((gray < 15).mean()),
                    bright_fraction=float((gray > 245).mean()),
                    entropy_bits=entropy(gray),
                    dhash64=dhash(gray),
                    aspect_ratio=float(max(record['width'], record['height']) / min(record['width'], record['height'])),
                )
            except Exception as exc:
                record.update(status='error', error=str(exc))
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            if index % 500 == 0:
                print(f'Pixel audit {index}/{len(images)}', flush=True)
    print(f'Done: {len(images)}', flush=True)


if __name__ == '__main__':
    main()
