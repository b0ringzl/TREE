import io
import hashlib
import time
import unittest
import numpy as np
from unittest.mock import patch
from PIL import Image
from spacescan_trees_demo_v1 import panorama as p, tree_detection as td


class PanoramaTests(unittest.TestCase):
    def test_projection_detection_and_manual_override(self):
        self.assertEqual(p.resolve_projection((8192,4096))['kind'],'equirectangular')
        self.assertEqual(p.resolve_projection((2000,1000),'photo')['kind'],'photo')
        self.assertEqual(p.resolve_projection((2000,1200))['kind'],'photo')
        with self.assertRaises(ValueError):p.resolve_projection((1000,1000),'equirectangular')
        from fastapi.testclient import TestClient
        from spacescan_trees_demo_v1 import server
        b=io.BytesIO();Image.new('RGB',(400,200),'green').save(b,'PNG')
        with patch.object(server,'infer') as infer:
            response=TestClient(server.app).post('/api/predict',files={'image':('pano.png',b.getvalue())})
        self.assertEqual(response.status_code,409);infer.assert_not_called()

    def test_camera_center_orientation(self):
        for yaw,pitch in [(0,0),(90,30),(-180,-30)]:
            s=p.view_spec((1000,500),yaw,pitch)
            u,v=p.rays_uv(p.uv_rays(.5,.5,s),yaw)
            self.assertAlmostEqual(float(u),yaw/360+.5)
            self.assertAlmostEqual(float(v),.5-pitch/180)

    def test_rectilinear_plane_stays_straight(self):
        # A 3D vertical line x/z=.3 in a level pinhole view projects to constant x.
        f,r,up=p.basis(35,0)
        points=np.array([f+.3*r+y*up for y in np.linspace(-1,1,20)])
        x=(points@r)/(points@f)
        self.assertTrue(np.allclose(x,.3))

    def test_seam_wrap(self):
        a=[-.02,.3,.03,.7];b=[.98,.3,1.03,.7]
        self.assertAlmostEqual(p.circular_iou(a,b),1)
        # Green seam neighbors; red middle. A rear-facing view must sample green.
        im=np.zeros((100,200,3),dtype=np.uint8);im[:,:,0]=255
        im[:,:20]=[0,255,0];im[:,-20:]=[0,255,0]
        result=p.render(Image.fromarray(im),p.view_spec((200,100),180,0,40,40))
        color=np.array(result)[result.height//2,result.width//2]
        self.assertGreater(color[1],250)

    def test_panorama_detection_sees_only_rectified_views(self):
        class FakeEngine:
            weights={}
            def boxes(self,im):
                self.calls.append(im.size);return []
            def supplemental_boxes(self,im):return []
            def background_checks(self,crops):return []
        engine=FakeEngine();engine.calls=[]
        b=io.BytesIO();Image.new('RGB',(400,200),'green').save(b,'PNG')
        with patch.object(td,'ENGINE',engine):r=td.detect(b.getvalue())
        self.assertEqual(len(engine.calls),16)
        self.assertNotIn((400,200),engine.calls)
        self.assertEqual(r['projection']['method'],'spherical_to_pinhole')

    def test_signed_crop_reprojects_source(self):
        b=io.BytesIO();Image.new('RGB',(400,200),'green').save(b,'PNG');data=b.getvalue()
        spec=p.view_spec((400,200),0,30)
        token=td.sign({'sha256':hashlib.sha256(data).hexdigest(),'expires':time.time()+30,
                       'crop_box':[5,6,50,80],'projection_view':spec})
        crop,_,_=td.selected(data,token)
        self.assertEqual(Image.open(io.BytesIO(crop)).size,(45,74))


if __name__=='__main__':unittest.main()
