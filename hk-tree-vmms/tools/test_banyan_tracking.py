import unittest
import numpy as np
from pilot_banyan_tracking import track

class TrackingTests(unittest.TestCase):
    def test_bent_stem_uses_real_points(self):
        z,a=np.meshgrid(np.arange(.35,7.8,.08),np.linspace(0,2*np.pi,24,endpoint=False))
        p=np.c_[.15*z.ravel()+.3*np.cos(a.ravel()),.3*np.sin(a.ravel()),z.ravel()]
        out,stats=track(p,np.array([0.,0.]),0,7.)
        self.assertIsNotNone(out)
        self.assertGreater(stats['observed_stem_bin_coverage'],.9)
        self.assertGreater(np.ptp(out[:,0]),1)
        from scipy.spatial import cKDTree
        self.assertEqual(cKDTree(p).query(out)[0].max(),0)
    def test_floating_crown_not_a_trunk(self):
        p=np.random.default_rng(7).uniform([-1,-1,5],[1,1,8],(4000,3))
        out,_=track(p,np.array([0.,0.]),0,7.)
        self.assertIsNone(out)

if __name__=='__main__':unittest.main()
