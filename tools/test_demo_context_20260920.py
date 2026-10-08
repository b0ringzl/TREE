"""Fast unit tests: equal-probability fusion, unchanged legacy calls, warnings."""
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tree_species_teacher_demo.engine import Engine,CONTEXT_PIPELINE

class Stub(Engine):
    def __init__(self):
        self.labels=['Aquilaria sinensis','Artocarpus hypargyreus']
        self.calls=[]
    def predict(self,data,mode='adapted'):
        self.calls.append((data,mode))
        p=[.8,.2] if data==b'padded' else [.1,.9]
        dist=dict(zip(self.labels,p));top=[dict(species=max(dist,key=dist.get),score=max(p))]
        return dict(deployment='unchanged-h120',comparison={m:top for m in ('sstv1','teacher','adapted')},
            top3=top,model_input=data.decode(),inference_seconds=.05,
            **{m+'_distribution':dict(dist) for m in ('sstv1','teacher','adapted')},
            quality=dict(warnings=['Center crop omits much','Low resolution warning'],crop_retained_fraction=.5))

class ContextTest(unittest.TestCase):
    def test_uses_both_distinct_original_inputs(self):
        engine=Stub();r=engine.predict_context(b'padded',b'original')
        self.assertEqual(engine.calls,[(b'padded','sstv1'),(b'original','sstv1')])
        self.assertEqual(r['inference_pipeline'],CONTEXT_PIPELINE)
        self.assertAlmostEqual(r['sstv1_distribution']['Aquilaria sinensis'],.45)
        self.assertEqual(r['top3'][0]['species'],'Artocarpus hypargyreus')
        self.assertEqual(r['comparison']['sstv1'],r['top3'])
        self.assertEqual(r['deployment'],'unchanged-h120')
        self.assertEqual(r['model_input_detail'],'original')
        self.assertEqual(r['view_evidence']['independent_images'],1)
        self.assertFalse(r['view_evidence']['agreement'])
        self.assertAlmostEqual(r['inference_seconds'],.1)
        self.assertTrue(any('Low resolution' in w for w in r['quality']['warnings']))
        self.assertFalse(any('Center crop omits' in w for w in r['quality']['warnings']))
    def test_equal_views_are_not_artificially_sharpened(self):
        r=Stub().predict_context(b'padded',b'padded')
        self.assertAlmostEqual(r['top3'][0]['score'],.8)
        self.assertTrue(r['view_evidence']['agreement'])

if __name__=='__main__':unittest.main()
