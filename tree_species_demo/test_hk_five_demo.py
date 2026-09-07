"""Policy and preprocessing tests; fake heads, no model training/review writes."""
import io,unittest
import numpy as np
from engine import point_input
from hk_five_species import decision,choose,NAMES

class Head:
    classes_=np.arange(5)
    def __init__(self,p):self.p=np.array(p)
    def predict_proba(self,x):return self.p[None]

class FiveDemoTests(unittest.TestCase):
    def result(self,index,mode='image',severe=False):
        p=np.full(5,.025);p[index]=.9
        return decision({'heads':{mode:Head(p)}},mode,np.array([1.,0.]),severe)
    def test_five_class_order_and_uncalibrated_score(self):
        for i in range(5):
            r=self.result(i);self.assertEqual(r['label'],NAMES[i]);self.assertEqual(r['candidate_label'],NAMES[i]);self.assertEqual(len(r['top3']),3)
        self.assertEqual(self.result(0,'point')['reliability'],'no_independent_class_evaluation')
        self.assertEqual(self.result(2,'point')['label'],NAMES[2]) # Not disabled by lack of reliability evaluation.
    def test_low_quality_and_low_margin_keep_candidates(self):
        self.assertEqual(self.result(1,severe=True)['label'],'Unknown')
        r=decision({'heads':{'image':Head([.22,.21,.20,.19,.18])}},'image',np.array([1.,0.]))
        self.assertEqual(r['label'],'Unknown');self.assertEqual(len(r['top3']),3)
    def test_disagreement_and_unpaired_policy(self):
        outputs={'image':self.result(1),'point':self.result(3,'point'),'fusion':self.result(4,'fusion')};quality={m:{'severe':False} for m in ['image','point']}
        r,mode,_=choose(outputs,quality,True);self.assertEqual(r['label'],'Unknown');self.assertEqual(mode,'image')
        r,mode,_=choose(outputs,quality,False);self.assertEqual(r['label'],NAMES[1])
        self.assertEqual(outputs['image']['label'],NAMES[1]) # No mutation of branch result.
    def test_quality_fallback_and_single_source(self):
        outputs={'image':self.result(1,severe=True),'point':self.result(3,'point')}
        r,mode,_=choose(outputs,{'image':{'severe':True},'point':{'severe':False}},True)
        self.assertEqual(mode,'point');self.assertEqual(r['label'],NAMES[3])
        r,mode,_=choose({'point':outputs['point']},{'point':{'severe':False}},False);self.assertEqual(mode,'point')
    def test_training_sparse_asset_parity(self):
        p=np.random.default_rng(9).normal(size=(1081,3)).astype(np.float32);p/=np.linalg.norm(p,axis=1).max();b=io.BytesIO();np.savez(b,points_xyz=p,centroid_xyz=np.ones(3),scale=7.)
        actual,q,_=point_input(b.getvalue(),'point.npz',2048,preserve_sparse=True)
        np.testing.assert_array_equal(actual,p);self.assertEqual(q['unique_count'],1081)
        legacy,_,_=point_input(b.getvalue(),'point.npz',2048);self.assertEqual(len(legacy),2048)
    def test_duplicate_points_not_padded_in_new_mode(self):
        p=np.random.default_rng(1).normal(size=(500,3));b=io.BytesIO();np.save(b,np.r_[p,p])
        actual,_,_=point_input(b.getvalue(),'point.npy',2048,preserve_sparse=True)
        self.assertEqual(len(actual),500);self.assertEqual(len(np.unique(actual,axis=0)),500)

if __name__=='__main__':unittest.main()
