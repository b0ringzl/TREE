"""Five-class review-only expansion; preserve older runs and original taxonomy.

HWT density experiment uses the same nearest-three original clouds as before,
but retains real points at 6 cm instead of 15 cm. This does not certify alignment.
Other surveys retain the existing DS source path, explicitly recorded per item.
"""
import re
from pathlib import Path
import numpy as np
import expand_hk_species_pilot as experiment
from build_hk_lidar_pairs import Maps, VMMS, voxel, read_binary_xyzi_rgb


class DenserFrameMaps(Maps):
    def nearby(self, row):
        survey=Path(row['source_image_relpath']).parts[0]
        if survey!='2023-10-27_hewentian':
            self.sampling_m=.15
            return super().nearby(row)
        self.sampling_m=.06
        if self.color_files is None:
            self.color_files=[]
            for path in (VMMS/survey/'pointcloud/ColorCloudPoint').glob('*.pcd'):
                match=re.search(r'_img(\d+)_',path.name)
                if match:self.color_files.append((int(match[1]),path))
        nearest=sorted(self.color_files,key=lambda v:abs(v[0]-int(row['frame_id'])))[:3]
        if not nearest:raise ValueError('no_original_frame_clouds')
        pos=np.array([float(row['local_'+k]) for k in 'xy'])
        arrays=[]
        for _,path in nearest:
            raw=read_binary_xyzi_rgb(path)
            xyz=np.column_stack([raw[k] for k in 'xyz'])
            xyz=xyz[np.isfinite(xyz).all(1)&(np.linalg.norm(xyz[:,:2]-pos,axis=1)<45)]
            arrays.append(xyz)
        xyz=np.concatenate(arrays)
        return xyz[voxel(xyz,self.sampling_m)],[str(p) for _,p in nearest]


if __name__=='__main__':
    experiment.NAMES={'Delonix regia':'凤凰木','Aleurites moluccana':'石栗',
                      'Leucaena leucocephala':'银合欢','Albizia lebbeck':'大叶合欢',
                      'Araucaria columnaris(Araucaria heterophylla)':'南洋杉（原训练合并类）'}
    experiment.LABEL_ALIASES={n:'Araucaria columnaris(Araucaria heterophylla)'
                              for n in ('Araucaria columnaris','Araucaria heterophylla')}
    experiment.OUT=experiment.DATA.parent/'species_expansion_five_v2_20260907'
    if experiment.OUT.exists():raise SystemExit('Output exists; preserve this run and choose a new version.')
    experiment.Maps=DenserFrameMaps
    experiment.USE_FRAME_SOURCES=True
    experiment.LIMIT_PER_CLASS=8
    experiment.main()
