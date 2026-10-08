"""Unit tests and real-output regression checks; not detection accuracy measurement."""
import json
from pathlib import Path
import re
import unittest
from finalize_vmms_object_gate import decide
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/vmms20m_teacher_20260909_v1'

class GateTests(unittest.TestCase):
    def row(self,margin,support=.0):
        return {'views':[{'background_minus_tree':margin,'background_similarity':.28}]*2,
            'grounding_support':[{'score':support}]}
    def test_background_veto_not_overridden_by_detector(self):
        self.assertEqual(decide(self.row(.04,.42))[0],'suppress_non_tree')
    def test_ambiguous_without_support(self):
        self.assertEqual(decide(self.row(.015,.23))[0],'suppress_non_tree')
    def test_ambiguous_with_support(self):
        self.assertEqual(decide(self.row(.015,.40))[0],'retain_unverified')
    def test_tree_evidence_retained(self):
        self.assertEqual(decide(self.row(-.01))[0],'retain_unverified')
    def test_real_report_regressions(self):
        active=json.loads((OUT/'active_object_gate.json').read_text())['directory']
        rows=json.loads((OUT/active/'decisions.json').read_text(encoding='utf8'))
        lookup={r['id']:r for r in rows}
        for i in range(3):
            self.assertEqual(lookup[f'hewentian_pano__002357__tree_{i:03d}']['decision'],'suppress_non_tree')
        positives=[r for r in rows if r['group']=='historical_positive_regression']
        self.assertEqual(len(positives),166)
        self.assertTrue(all(r['decision']=='retain_unverified' for r in positives))
        for key in ['hewentian_pano__003683__tree_001','hewentian_pano__003684__tree_001','jianshazui_pano_2__002972__tree_000']:
            self.assertEqual(lookup[key]['decision'],'retain_unverified')
        html=(OUT/'index.html').read_text(encoding='utf8')
        payload=json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>',html,re.S).group(1))
        frame=next(f for f in payload['frames'] if f['frame_key']=='hewentian_pano__002357')
        self.assertEqual(frame['instances'],[])
        self.assertEqual(frame['suppressed_non_tree_count'],3)
        self.assertTrue((OUT/frame['overview']).exists())
        raw=json.loads((OUT/'anchors/hewentian_pano__002357.json').read_text(encoding='utf8'))
        self.assertEqual(len(raw['instances']),3)  # reversible; originals untouched
        self.assertEqual(payload['summary']['candidate_count'],794)

if __name__=='__main__':unittest.main()
