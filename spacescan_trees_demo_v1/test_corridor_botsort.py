import unittest

from . import corridor_botsort
from . import test_vmms_corridor


class BoTSortCorridorTests(unittest.TestCase):
    def frames(self,positions):
        frames=[]
        for sample,items in enumerate(positions):
            frame=test_vmms_corridor.VmmsCorridorTests._tracking_frame(sample,items)
            for row in frame['candidates']:
                row['view_box']=row['box'][:]
                row['view_spec']={'yaw':0,'pitch':15,'hfov':90,'vfov':105,'width':786,'height':1024}
            frames.append(frame)
        return frames

    def test_real_botsort_reid_keeps_stationary_identity(self):
        frames=self.frames([[(0,[1,0],[.45,.2,.55,.8])]]*4)
        audit={}
        tracks=corridor_botsort.track_candidates(frames,audit)
        self.assertEqual(len(tracks),1)
        self.assertEqual(len(tracks[0]['observations']),4)
        self.assertEqual(audit['botsort_streams'],1)

    def test_native_ids_cannot_bypass_spherical_geometry(self):
        frames=self.frames([[(0,[1,0],[.48,.35,.54,.55])]]*2+
                           [[(0,[1,0],[.48,.10,.54,.28])]]*2)
        self.assertEqual(len(corridor_botsort.track_candidates(frames)),2)

    def test_close_same_species_trees_are_not_one_identity(self):
        frames=self.frames([[(4,[1,0],[.49,.15,.52,.75]),
                             (12,[.999,.02],[.53,.15,.56,.75])]]*3)
        tracks=corridor_botsort.track_candidates(frames)
        self.assertEqual(len(tracks),2)
        self.assertEqual([len(t['observations']) for t in tracks],[3,3])

    def test_empty_detection_frames_do_not_generate_observations(self):
        frames=self.frames([[(0,[1,0],[.45,.2,.55,.8])],[],
                            [(0,[1,0],[.45,.2,.55,.8])]])
        tracks=corridor_botsort.track_candidates(frames)
        self.assertEqual(len(tracks),1)
        self.assertEqual([r['sample_index'] for r in tracks[0]['observations']],[0,2])

    def test_separate_view_streams_have_no_cross_camera_state(self):
        frames=self.frames([[(0,[1,0],[.45,.2,.55,.8]),
                             (50,[0,1],[.65,.2,.75,.8])]]*3)
        for frame in frames:
            frame['candidates'][1]['view_spec']={**frame['candidates'][1]['view_spec'],'yaw':45}
        audit={};tracks=corridor_botsort.track_candidates(frames,audit)
        self.assertEqual(audit['botsort_streams'],2)
        self.assertEqual(len(tracks),2)


if __name__=='__main__':unittest.main()
