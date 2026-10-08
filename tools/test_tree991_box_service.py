"""Non-browser QA. Synthetic exports never enter the user's export directory."""
import tempfile, unittest, subprocess, json
from pathlib import Path
import numpy as np
import tree991_box_service as svc

def op(mode='replace',right=(1,0,0),up=(0,1,0),bounds=(-1,1,-1,1)):
    return dict(mode=mode,right=list(right),up=list(up),bounds=list(bounds))

def slot(ops=None,confirmed=False):
    return dict(ops=ops or [op()],zmin=-10,zmax=80,confirmed=confirmed,note='QA synthetic data')

class SelectionTests(unittest.TestCase):
    def test_polygon_concavity_boundary_and_js_parity(self):
        polygon=[[0,0],[3,0],[3,1],[1,1],[1,3],[0,3]]
        q=np.vstack([np.random.default_rng(7).uniform(-1,4,(1000,2)),polygon,[[.5,2],[2,2],[1,2],[0,0]]])
        expected=svc.polygon_mask(q[:,0],q[:,1],polygon)
        np.testing.assert_array_equal(expected[-4:],[1,0,1,1])
        script="const {containsPolygon}=require('./tools/tree991_box_ui/polygon.js');let s='';process.stdin.on('data',c=>s+=c);process.stdin.on('end',()=>{const d=JSON.parse(s);console.log(JSON.stringify(d.q.map(p=>containsPolygon(...p,d.p))))});"
        result=subprocess.run(['node','-e',script],input=json.dumps(dict(q=q.tolist(),p=polygon)),text=True,capture_output=True,check=True,cwd=svc.ROOT)
        np.testing.assert_array_equal(expected,json.loads(result.stdout))
        o=op(bounds=(0,3,0,3));o['polygon']=polygon;s=svc.validate_slot(slot([o]))
        xyz=np.column_stack([q,np.zeros(len(q))]);np.testing.assert_array_equal(svc.selection_mask(xyz,s),expected)
        s['ops'].append(dict(o,mode='subtract'));self.assertFalse(svc.selection_mask(xyz,s).any())

    def test_polygon_rejection(self):
        for polygon in [[[0,0],[1,1]],[[0,0],[1,1],[2,2]],[[0,0],[1,0],[1,0],[0,1]],[[0,0],[1,0],[0,float('nan')]]]:
            o=op();o['polygon']=polygon
            with self.assertRaises((AssertionError,ValueError)):svc.validate_slot(slot([o]))

    def test_projection_and_depth_trim(self):
        x=np.array([[0,0,0],[0,0,3],[2,0,0],[0,2,0]],dtype=np.float32)
        s=slot();np.testing.assert_array_equal(svc.selection_mask(x,s),[1,1,0,0])
        s['ops'].append(op('intersect',up=(0,0,1)))
        np.testing.assert_array_equal(svc.selection_mask(x,s),[1,0,0,0])

    def test_subtraction_and_z(self):
        x=np.array([[0,0,0],[.7,.7,0],[.7,.7,2]],dtype=np.float32)
        s=slot([op(),op('subtract',bounds=(-.5,.5,-.5,.5))]);s['zmax']=1
        np.testing.assert_array_equal(svc.selection_mask(x,s),[0,1,0])

    def test_rotated_projection(self):
        r=np.sqrt(.5);s=slot([op(right=(r,r,0),up=(0,0,1))])
        np.testing.assert_array_equal(svc.selection_mask(np.array([[.5,.5,0],[1,1,0]]),s),[1,0])

    def test_invalid(self):
        for o in [op(right=(2,0,0)),op(up=(1,0,0)),op(bounds=(0,0,0,1)),op(bounds=(0,float('nan'),0,1)),op('intersect')]:
            with self.assertRaises((AssertionError,ValueError)):svc.validate_slot(slot([o]))

    def test_original_export_and_label_provenance(self):
        with tempfile.TemporaryDirectory(prefix='tree991_qa_') as directory:
            root=Path(directory);source=root/'synthetic.pcd'
            xyz=np.random.default_rng(42).uniform(-2,2,(10000,3)).astype('<f4')
            h=f'VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(xyz)}\nHEIGHT 1\nPOINTS {len(xyz)}\nDATA binary\n'
            source.write_bytes(h.encode()+xyz.tobytes());digest=svc.sha256(source)
            s=slot();s['ops'][0]['polygon']=[[-1,-1],[1,-1],[1,1],[-1,1]]
            s=svc.validate_slot(s);expected=np.flatnonzero(svc.selection_mask(xyz,s))
            for confirmed in [False,True]:
                s['confirmed']=confirmed;out=root/str(confirmed)
                rec=svc.export_cloud('A',s,out,source)
                with np.load(out/'full.npz') as full:
                    np.testing.assert_array_equal(full['xyz'],xyz[expected])
                    np.testing.assert_array_equal(full['source_indices'],expected)
                with np.load(out/'sample2048.npz') as sample:
                    self.assertEqual(len(sample['metric_xyz']),2048)
                    self.assertEqual(len(np.unique(sample['source_indices'])),2048)
                    np.testing.assert_array_equal(sample['metric_xyz'],xyz[sample['source_indices']])
                payload=(out/'full.ply').read_bytes().split(b'end_header\n',1)[1]
                np.testing.assert_array_equal(np.frombuffer(payload,dtype='<f4').reshape(-1,3),xyz[expected])
                self.assertEqual(rec['species'],'Aquilaria sinensis' if confirmed else None)
                self.assertFalse(rec['training_eligible']);self.assertFalse(rec['image_projection_verified'])
                self.assertEqual(svc.sha256(source),digest)
                for name,info in rec['files'].items():self.assertEqual(svc.sha256(out/name),info['sha256'])
                with self.assertRaises(FileExistsError):svc.export_cloud('A',s,out,source)

if __name__=='__main__':unittest.main(verbosity=2)
