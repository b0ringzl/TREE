import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_aquilaria_sstv1 import read_json,ROOT,AQ,aqmetrics,labels
import numpy as np
P=ROOT/'outputs/SSTv1_Aquilaria_20260914_v1'
class SafetyTests(unittest.TestCase):
    def test_observer_splits(self):
        rows=read_json(P/'accepted_new_manifest.json')
        for r in rows:
            self.assertEqual(r['target'],AQ);self.assertEqual(r['review_authority'],'agent_source_and_visual_review')
            self.assertFalse(r['quality_flags']);self.assertGreaterEqual(len(r['source_evidence']['exact_identifiers']),2)
            self.assertEqual(r['source_evidence']['conflicting_current_ids'],[])
            self.assertTrue(all(o['phase']==r['phase'] for o in rows if o['observer_id']==r['observer_id']))
    def test_no_augmented_holdout(self):
        rows=read_json(P/'feature_rows.json')
        for r in rows:
            if r['phase']!='train':self.assertEqual(r['brightness'],1.)
        for r in rows:
            self.assertTrue(all(o['phase']==r['phase'] for o in rows if o['group_id']==r['group_id']))
    def test_prior_replay_only_accepted_train(self):
        for r in read_json(P/'prior_replay_manifest.json'):
            self.assertEqual(r['phase'],'train')
            d=read_json(P/'previous_incremental_human_review.json')[r['id']]
            self.assertEqual(d['status'],'accepted');self.assertEqual(d['sha256'],r['sha256'])
    def test_no_old_holdout_leakage(self):
        held=[r for phase in ('planning','selection','test','field','case') for r in read_json(P/f'old_{phase}_rows.json')]
        heldgroups={r['group_id'] for r in held};heldhash={r.get('source_sha256') for r in held}
        for r in read_json(P/'accepted_replay_manifest.json'):
            self.assertNotIn(r['group_id'],heldgroups);self.assertNotIn(r.get('source_sha256'),heldhash)
    def test_metric_counts_false_positives(self):
        self.assertEqual(labels(([{'target':AQ}],None,None)).dtype,np.int64)
        p=np.zeros((4,148));p[0,AQ]=1;p[1,0]=1;p[2,AQ]=1;p[3,0]=1
        m=aqmetrics(([{'target':t} for t in [AQ,AQ,0,0]],None,None),p)
        self.assertEqual(m['recall'],.5);self.assertEqual(m['precision'],.5);self.assertEqual(m['false_positive'],1)
    def test_initialized_from_live_sstv1(self):
        for h in read_json(P/'base_contract.json')['initial_heads']:
            self.assertIn('models\\SSTv1_20260914_v1',h['path'])
        self.assertEqual(read_json(P/'config.json')['accepted_compatible'],26)
if __name__=='__main__':unittest.main()
