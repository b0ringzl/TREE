"""Offline identity-firewall regressions; synthetic metadata only."""
import unittest
from pathlib import Path
from artocarpus_fresh_split_20260917 import record_identity, merge_records, lookup_maps, historical_matches

ORIGIN = Path(__file__).resolve().parent / 'synthetic_manifest.json'


def normalize(**kw):
    return record_identity(kw, ORIGIN)


def registry(rows):
    return {'records': merge_records(rows), 'global_cached_identity_maps': {k: {} for k in ('hashes', 'paths', 'photos', 'observations', 'urls', 'visual')}}


class IdentityTests(unittest.TestCase):
    def test_photo_url_resolutions_alias(self):
        a = normalize(photo_url='https://inaturalist-open-data.s3.amazonaws.com/photos/123/medium.jpg')
        b = normalize(photo_url='https://inaturalist-open-data.s3.amazonaws.com/photos/123/original.jpg')
        self.assertEqual(a['tokens']['photo'], b['tokens']['photo'])
        self.assertEqual(len(merge_records([a, b])), 1)

    def test_multiple_photos_one_observation_not_multiple_trees(self):
        a = normalize(photo_id=1, inat_observation_id=123)
        b = normalize(photo_id=2, group_id='A species|observation:123')
        reg = registry([a]); maps, visual = lookup_maps(reg)
        hits, _, _ = historical_matches(b, reg, maps, visual, 6)
        self.assertIn('observation', hits)

    def test_legacy_materialized_crop_alias(self):
        a = normalize(sample_id='1234567890abcdefabcd', sha256='1' * 64)
        b = normalize(path='some/crop/1234567890abcdefabcd.jpg', sha256='2' * 64)
        merged = merge_records([a, b])
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]['tokens']['sha256']), 2)

    def test_generic_cc0_notice_not_photographer(self):
        a = normalize(photo_id=1, photo_attribution='no rights reserved')
        self.assertEqual(a['tokens']['observer'], [])

    def test_observer_and_display_aliases_preserved(self):
        a = normalize(inat_observation_id=1, observer_id=7, observer_login='Somebody', photo_attribution='(c) Writer, some rights reserved (CC BY)')
        self.assertEqual(set(a['tokens']['observer']), {'inat:id:7', 'inat:login:somebody', 'display:writer'})

    def test_actual_review_decision_survives_identity_merge(self):
        a = normalize(sha256='a' * 64); a['review_status'] = 'rejected'
        b = normalize(sha256='a' * 64, phase='train')
        row = merge_records([a, b])[0]
        self.assertEqual(row['review_statuses'], ['rejected'])
        self.assertEqual(row['phases'], ['train'])

    def test_near_hash_exclusion(self):
        a = normalize(sha256='1' * 64, phash='aaaaaaaaaaaaaaaa')
        b = normalize(sha256='2' * 64, phash='aaaaaaaaaaaaaaab')
        reg = registry([a]); maps, visual = lookup_maps(reg)
        hits, _, _ = historical_matches(b, reg, maps, visual, 6)
        self.assertEqual(hits['near_phash'], [0])

    def test_no_prediction_field_becomes_source_label(self):
        a = normalize(sha256='a' * 64, prediction='Artocarpus hypargyreus', candidate='Artocarpus hypargyreus')
        self.assertEqual(a['species'], '')


if __name__ == '__main__': unittest.main()
