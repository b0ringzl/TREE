import io,unittest
import numpy as np
from . import point_recognition as p,point_assist

class PointRecognitionTests(unittest.TestCase):
    def test_distribution_and_equal_weights(self):
        image={c:1/148 for c in p.CLASSES}
        image['Ficus microcarpa / Ficus benjamina (榕樹)']=image.pop('Ficus microcarpa')
        point=dict(scores=[0,0,0,0,0,0,1])
        a,b,f,m=p.fuse(image,point)
        np.testing.assert_allclose(a,np.ones(7)/7)
        np.testing.assert_allclose(f,.5*a+.5*b)
        self.assertEqual(np.argmax(f),6);self.assertAlmostEqual(m,7/148)
    def test_missing_teacher_scores_fail_closed(self):
        with self.assertRaises(RuntimeError):p.fuse({},dict(scores=[1/7]*7))
    def test_bad_point_scores_fail_closed(self):
        with self.assertRaises(RuntimeError):p.fuse({c:1/148 for c in p.CLASSES},dict(scores=[float('nan')]*7))
    def test_metric_array_prioritized(self):
        b=io.BytesIO();x=np.random.default_rng(5).normal(size=(300,3));np.savez(b,xyz=x,points_xyz=x*0)
        actual,_=point_assist.decode(b.getvalue(),'input.npz','metres')
        np.testing.assert_array_equal(actual,x)
    def test_axis_gate_does_not_load_model(self):
        b=io.BytesIO();np.save(b,np.random.default_rng(5).normal(size=(300,3)))
        e,model=p.infer_upload(b.getvalue(),'input.npy','unknown',False)
        self.assertIsNone(model);self.assertFalse(e['fusion_used'])
if __name__=='__main__':unittest.main()
