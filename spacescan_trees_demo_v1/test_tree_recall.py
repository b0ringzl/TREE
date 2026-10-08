import io,unittest
from unittest.mock import patch
from PIL import Image
from . import tree_detection as td

def row(box,score=.5):return dict(box=box,score=score,detector='test')
class TreeRecallTests(unittest.TestCase):
    def test_fragment_removed_without_score_transfer(self):
        parent=row([.0,.2,.4,1],.45);fragment=row([.1,.75,.3,1],.8)
        result=td.remove_partial_boxes([fragment,parent])
        self.assertEqual(result,[parent]);self.assertEqual(result[0]['score'],.45)
    def test_separate_trees_retained(self):
        rows=[row([0,.2,.35,1]),row([.28,.15,.72,1]),row([.55,.3,1,1])]
        self.assertEqual(td.remove_partial_boxes(rows),rows)
    def test_group_parent_does_not_erase_children(self):
        rows=[row([0,.1,.9,1]),row([.02,.7,.2,1]),row([.65,.7,.85,1])]
        self.assertEqual(td.remove_partial_boxes(rows),rows)
    def test_single_yolo_hit_triggers_supplement(self):
        class Fake:
            weights={}
            calls=0
            def boxes(self,im):return [row([.1,.75,.3,1],.58)]
            def supplemental_boxes(self,im):
                self.calls+=1
                return [row([0,.2,.4,1],.47),row([.3,.17,.73,1],.41),row([.56,.33,1,1],.45)]
            def background_checks(self,crops):return [dict(keep=True,background_minus_tree=[-.1,-.1]) for _ in crops]
        engine=Fake();buf=io.BytesIO();Image.new('RGB',(865,869)).save(buf,format='PNG')
        with patch.object(td,'ENGINE',engine):d=td.detect(buf.getvalue())
        self.assertEqual(engine.calls,1);self.assertEqual(len(d['candidates']),3)
        self.assertEqual(d['partial_boxes_removed'],1);self.assertEqual(d['action'],'choose')
        self.assertTrue(all(r['score']>=.4 for r in d['candidates']))
        for r in d['candidates']:td.selected(buf.getvalue(),r['token'])
