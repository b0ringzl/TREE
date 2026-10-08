"""Pinned, local-only official distribution; never choose a model by file mtime."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RELEASE = json.loads((ROOT / 'release.json').read_text(encoding='utf-8'))


def version_state(actual):
    return 'unavailable' if not actual else ('aligned' if actual == RELEASE['classifier_version'] else 'mismatch')


def public_release(actual=None):
    # Do not expose absolute paths, training manifests or source images.
    return {
        'release_id': RELEASE['release_id'],
        'classifier_version': RELEASE['classifier_version'],
        'android_classifier_version': RELEASE['android_classifier_version'],
        'android_classifier_alignment': version_state(RELEASE['android_classifier_version']),
        'actual_web_classifier_version': actual,
        'web_version_state': version_state(actual),
        'class_count': RELEASE['class_count'],
        'web_pipeline': RELEASE['web_pipeline'],
        'portable_windows_pipeline': RELEASE.get('portable_windows_pipeline'),
        'portable_windows_current': RELEASE.get('portable_windows_current', False),
        'android_pipeline': RELEASE['android_pipeline'],
        'full_android_pipeline_parity': False,
        'windows_url': '/',
        'demo_editions': {'image_only': '/image-only', 'image_and_cloud': '/multimodal'},
        'android_apk_url': '/api/official-download/apk',
        'android_model_url': '/api/official-download/model',
        'android_guide_url': '/api/official-download/guide',
        'android_apk_sha256': RELEASE['android_apk_sha256'],
        'android_model_sha256': RELEASE['android_model_sha256'],
        'android_no_internet_permission': True,
        'android_physical_phone_tested': False,
        'regression': {'n': 2, 'same_top1': 2, 'max_abs_fixed_rgb': 0.0093725,
                       'classifier_version': RELEASE['android_classifier_version'],
                       'applies_to_current_web': RELEASE['android_classifier_version'] == RELEASE['classifier_version'],
                       'kind': 'Inspected fixed-photo H120 conversion agreement; not independent species accuracy or physical-phone acceptance'},
        'known_regressions': RELEASE['known_regressions'],
        'research_archives_preserved': True,
        'point_fusion_changed': False,
    }


def result_provenance(mode, actual, data, prepared, pipeline=None):
    return {
        'release_id': RELEASE['release_id'],
        'classifier_alignment': version_state(actual) if mode == 'sstv1' else 'separate-evaluated-fusion-image-branch',
        'expected_single_image_classifier': RELEASE['classifier_version'],
        'actual_image_classifier': actual,
        'target_bytes_sha256': hashlib.sha256(data).hexdigest(),
        'padded_jpeg_sha256': hashlib.sha256(prepared).hexdigest(),
        'pipeline': pipeline or (RELEASE['web_pipeline'] if mode == 'sstv1' else 'bioclip25-original-image-for-ptv2-fusion'),
        'android_full_pipeline_equivalent': False,
    }


def download_path(key):
    # Only three pinned private deliverables can be downloaded, no arbitrary path.
    names = {'apk': RELEASE['android_apk'], 'model': RELEASE['android_model'], 'guide': '安装与验收说明.md'}
    if key not in names:
        raise KeyError(key)
    directory = (PROJECT / RELEASE['android_delivery']).resolve()
    path = (directory / names[key]).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise KeyError(key)
    return path
