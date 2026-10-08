import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/SpaceScan_Android_SSTv1_20260919_v1'
paths=json.loads((ROOT/'outputs/android_toolchain_20260918_v1/paths.json').read_text())
jdk=Path(paths['jdk'])/'bin';project=ROOT/'spacescan_android_sstv1'
subprocess.run([str(jdk/'javac.exe'),'-d',str(OUT/'java'),str(project/'app/src/main/java/org/spacescan/trees/offline/Preprocessor.java'),str(project/'tests/PadCheck.java')],check=True)
checks=[]
for w,h in [(173,291),(300,201),(224,224),(1024,2048),(11,19),(865,869),(3072,4096)]:
    path=OUT/f'pad_{w}_{h}.bin';edge=min(1024,max(w,h))
    subprocess.run([str(jdk/'java.exe'),'-cp',str(OUT/'java'),'PadCheck',str(w),str(h),str(path)],check=True)
    yy,xx=np.mgrid[:h,:w];a=np.stack(((xx*7+yy*3)%256,(xx*2+yy*5)%256,(xx+yy*11)%256),-1).astype(np.uint8)
    expected=np.asarray(ImageOps.pad(Image.fromarray(a),(edge,edge),method=Image.Resampling.BICUBIC,color=(128,128,128)))
    packed=np.fromfile(path,dtype='>u4').reshape(edge,edge)
    actual=np.stack([(packed>>shift)&255 for shift in [16,8,0]],-1).astype(np.uint8)
    maxdiff=int(abs(actual.astype(int)-expected.astype(int)).max())
    checks.append(dict(width=w,height=h,max_pixel_delta=maxdiff));assert maxdiff==0,checks[-1]
(OUT/'java_padding_parity.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks))
