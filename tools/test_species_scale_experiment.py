import unittest
import numpy as np
from species_scale_experiment import grouped_name,FICUS,aggregate,metrics,collapsed_baseline,partition_validation


class ScaleTests(unittest.TestCase):
    def test_only_requested_ficus_merge(self):
        self.assertEqual(grouped_name('Ficus benjamina'),FICUS)
        self.assertEqual(grouped_name('Ficus microcarpa'),FICUS)
        self.assertEqual(grouped_name('Ficus virens'),'Ficus virens')
        self.assertNotEqual(grouped_name('Leucaena leucocephala'),grouped_name('Albizia lebbeck'))

    def test_baseline_keeps_outside_probability(self):
        p=np.array([[.1,.2,.3,.4]])
        got=collapsed_baseline(p,['Ficus microcarpa','Ficus benjamina','A','B'],[FICUS,'A'])
        np.testing.assert_allclose(got,[[.3,.3,.4]])
        self.assertAlmostEqual(float(got.sum()),1)

    def test_reject_everything_does_not_claim_accuracy(self):
        p=np.array([[.1,.1,.8],[.15,.05,.8],[.1,.1,.8]])
        m=metrics(np.array([0,1,2]),p,2,.9)
        self.assertEqual(m['known_coverage'],0)
        self.assertEqual(m['accepted_precision_all_sources'],0)
        self.assertEqual(m['known_correct_and_accepted_rate'],0)

    def test_unknown_false_accept_is_counted(self):
        p=np.array([[.9,.05,.05],[.05,.9,.05],[.9,.05,.05]])
        m=metrics(np.array([0,1,2]),p,2,.5)
        self.assertEqual(m['known_closed_accuracy'],1)
        self.assertEqual(m['other_false_accept_rate'],1)
        self.assertAlmostEqual(m['accepted_precision_all_sources'],2/3)

    def test_group_partition_not_image_partition(self):
        rows=[{'label':'A','split':'val','group_id':str(i//2)} for i in range(16)]
        partition_validation(rows)
        self.assertEqual({r['phase'] for r in rows},{'planning','selection'})
        for i in range(0,16,2):self.assertEqual(rows[i]['phase'],rows[i+1]['phase'])
        with self.assertRaises(ValueError):
            aggregate([{'group_id':'x'},{'group_id':'x'}],np.eye(2),np.array([0,1]))


if __name__=='__main__':unittest.main()
