"""Export all original mapped points in eight frozen 45 m scene windows.

No DS inputs, voxels, random sampling, deduplication, denoising or segmentation.
Four worker processes parse original source files once and fan out exact point
records to all intersecting scenes. Resumable source checkpoints are kept outside
the delivery folder. Original map XYZ, intensity/RGB and point provenance persist.
"""
from __future__ import annotations
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw

import build_selected_raw_multimodal_scenes as shared

BASE = shared.TREE_ROOT / '交付/人工标注_原始LiDAR配准场景_20260907_v3_指定树种'
OUT = shared.TREE_ROOT / '交付/人工标注_原始LiDAR配准场景_20260907_v4_不降采样'
WORK = shared.PROJECT / 'derived/full_density_scene_package_20260907'
DTYPE = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('intensity','<f4'),('rgb','<u4'),('source_file_id','<u4'),('source_point_id','<u4')])
W,H = 2048,1024


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)


def header(path):
    result={}
    with path.open('rb') as f:
        while True:
            line=f.readline()
            if not line:raise ValueError('Incomplete header '+str(path))
            parts=line.decode('ascii').strip().split()
            if parts and not parts[0].startswith('#'):result[parts[0]]=parts[1:]
            if line.startswith(b'DATA '):
                result['offset']=f.tell();break
    assert result['FIELDS'] in (['x','y','z','intensity'],['x','y','z','rgb'])
    assert result['SIZE']==['4']*4 and result.get('COUNT',['1']*4)==['1']*4
    return result


def chunks(path,meta):
    n=int(meta['POINTS'][0])
    if meta['DATA']==['binary']:
        types=[(name,'<u4' if kind=='U' else '<f4') for name,kind in zip(meta['FIELDS'],meta['TYPE'])]
        dtype=np.dtype(types)
        assert path.stat().st_size==meta['offset']+n*dtype.itemsize
        data=np.memmap(path,mode='r',offset=meta['offset'],dtype=dtype,shape=(n,))
        for start in range(0,n,1_000_000):
            part=data[start:start+1_000_000]
            yield np.column_stack([part[k] for k in 'xyz']),part[meta['FIELDS'][3]],start
    elif meta['DATA']==['ascii']:
        with path.open('rb') as f:
            f.seek(meta['offset']);start=0
            while True:
                text=f.read(16*1024*1024)
                if not text:break
                text+=f.readline()
                values=np.fromstring(text.decode('ascii'),sep=' ',dtype=np.float32)
                if len(values)%4:raise ValueError('Incomplete ASCII record '+str(path))
                a=values.reshape(-1,4)
                yield a[:,:3],a[:,3],start
                start+=len(a)
            assert start==n,(str(path),start,n)
    else:raise ValueError('Unsupported PCD DATA '+str(meta['DATA']))


def source_worker(job):
    path=Path(job['path']);work=WORK/f"source_{job['id']:02d}"
    work.mkdir(parents=True,exist_ok=True)
    marker=work/'complete.json'
    signature={'source_size':path.stat().st_size,'source_mtime_ns':path.stat().st_mtime_ns,'scenes':job['scenes'],'revision':1}
    if marker.exists():
        old=shared.load_json(marker)
        if old.get('signature')==signature:
            print('RESUME',job['id'],path.name,flush=True);return old
    meta=header(path);total=int(meta['POINTS'][0]);scenes=job['scenes']
    streams={s['number']:(work/f"scene_{s['number']:02d}.bin").open('wb') for s in scenes}
    counts={s['number']:0 for s in scenes}
    depths={s['number']:np.full(W*H,np.inf,dtype=np.float32) for s in scenes}
    invalid=0;read_count=0;last=0.;start_time=time.time()
    try:
        for xyz,attribute,start in chunks(path,meta):
            valid=np.isfinite(xyz).all(1);invalid+=int((~valid).sum());read_count+=len(xyz)
            for s in scenes:
                position=np.array(s['position'],dtype=np.float64)
                dx=xyz[:,0].astype(np.float64)-position[0];dy=xyz[:,1].astype(np.float64)-position[1]
                keep=valid & (dx*dx+dy*dy<=45.0**2)
                ids=np.flatnonzero(keep)
                if not len(ids):continue
                points=xyz[ids]
                rec=np.empty(len(ids),dtype=DTYPE)
                for k,axis in enumerate('xyz'):rec[axis]=points[:,k]
                rec['intensity']=attribute[ids] if meta['FIELDS'][3]=='intensity' else np.nan
                rec['rgb']=attribute[ids].astype(np.uint32) if meta['FIELDS'][3]=='rgb' else 0
                rec['source_file_id']=job['id'];rec['source_point_id']=(ids+start).astype(np.uint32)
                streams[s['number']].write(rec.tobytes());counts[s['number']]+=len(ids)
                cam=(points.astype(np.float64)-position)@np.array(s['world_from_camera'])
                distance=np.linalg.norm(cam,axis=1);good=distance>1e-6
                cam=cam[good];distance=distance[good]
                u=(.5-np.arctan2(cam[:,1],cam[:,0])/(2*np.pi))%1
                v=.5-np.arcsin(np.clip(cam[:,2]/distance,-1,1))/np.pi
                px=np.minimum((u*W).astype(np.int32),W-1);py=np.clip((v*H).astype(np.int32),0,H-1)
                np.minimum.at(depths[s['number']],py*W+px,distance.astype(np.float32))
            if time.time()-last>8:
                save(work/'progress.json',{'source_id':job['id'],'file':path.name,'rows_read':read_count,'total':total,'scene_points':counts,'elapsed_s':round(time.time()-start_time,1)})
                last=time.time()
    finally:
        for f in streams.values():f.close()
    assert read_count==total,(path,read_count,total)
    for n,depth in depths.items():np.save(work/f'depth_{n:02d}.npy',depth,allow_pickle=False)
    report={'id':job['id'],'path':str(path),'header':meta,'source_points':total,'rows_read':read_count,'nonfinite_xyz_rows':invalid,'retained_by_scene':counts,'signature':signature,'elapsed_s':time.time()-start_time}
    save(marker,report)
    print('COMPLETE',job['id'],path.name,total,counts,flush=True)
    return report


def render(depth,annotation_path,path,count):
    base=Image.open(annotation_path).convert('RGBA');assert base.size==(W,H)
    d=depth.reshape(H,W);valid=np.isfinite(d)
    rgba=np.zeros((H,W,4),dtype=np.uint8)
    # Pixel visibility is rasterization of ALL exported points, not point sampling.
    t=np.clip((d[valid]-2)/60,0,1)
    rgba[valid,0]=(30+225*t).astype(np.uint8);rgba[valid,1]=(235-80*t).astype(np.uint8)
    rgba[valid,2]=(255-220*t).astype(np.uint8);rgba[valid,3]=125
    image=Image.alpha_composite(base,Image.fromarray(rgba)).convert('RGB')
    draw=ImageDraw.Draw(image);draw.rectangle((0,0,W,54),fill='#152835')
    draw.text((15,10),f'原密度 {count:,} 点 | 全部点参与投影 | 同像素显示最近点',font=shared.font(28),fill='white')
    image.save(path,quality=95)
    return int(valid.sum())


def main():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(OUT).free
    if free<100*1024**3:raise RuntimeError('At least 100 GiB free disk space required for full-density export')
    rows=shared.read_coordinates();scenes=[]
    for folder in sorted(BASE.glob('[0-9][0-9]_*')):
        old=shared.load_json(folder/'场景说明.json');row=rows[old['frame_key']]
        attitude=[float(row[k+'_rad']) for k in ('roll','pitch','heading')]
        rotation=shared.world_from_flu(*attitude)@np.diag([-1.,-1.,1.])
        scenes.append({'number':old['scene_number'],'folder':folder.name,'row':row,'old':old,'survey':Path(row['source_image_relpath']).parts[0],'position':old['camera_local_xyz'],'world_from_camera':rotation.tolist()})
    assert len(scenes)==8
    jobs=[]
    for survey in sorted({s['survey'] for s in scenes}):
        sources=sorted(p for p in (shared.VMMS/survey/'pointcloud/mapping').glob('pointcloud_*.pcd') if re.fullmatch(r'pointcloud_\d+\.pcd',p.name))
        for p in sources:
            jobs.append({'id':len(jobs),'path':str(p),'survey':survey,'scenes':[{'number':s['number'],'position':s['position'],'world_from_camera':s['world_from_camera']} for s in scenes if s['survey']==survey]})
    assert len(jobs)==16
    save(WORK/'jobs.json',jobs)
    print('READ ORIGINAL SOURCES',len(jobs),'using 4 workers',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(source_worker,job) for job in jobs]
        reports=[future.result() for future in concurrent.futures.as_completed(futures)]
    reports.sort(key=lambda r:r['id'])
    save(OUT/'原始点云文件清单.json',[{k:v for k,v in r.items() if k!='signature'} for r in reports])
    summary=[];contact=Image.new('RGB',(1600,1800),'#08131f')
    for i,s in enumerate(scenes):
        directory=OUT/s['folder'];pcdir=directory/'点云数据';imdir=directory/'影像数据';prdir=directory/'配准图片'
        for d in (pcdir,imdir,prdir):d.mkdir(parents=True,exist_ok=True)
        for file in (BASE/s['folder']/'影像数据').iterdir():shutil.copy2(file,imdir/file.name)
        selected=[r for r in reports if str(s['number']) in {str(k) for k in r['retained_by_scene']}]
        count=sum(int(r['retained_by_scene'].get(s['number'],r['retained_by_scene'].get(str(s['number']),0))) for r in selected)
        assert count>s['old']['final_point_count'],(s['number'],count)
        pcd=pcdir/'lidar_scene_full_density.pcd'
        head=('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity rgb source_file_id source_point_id\n'
              'SIZE 4 4 4 4 4 4 4\nTYPE F F F F U U U\nCOUNT 1 1 1 1 1 1 1\n'
              f'WIDTH {count}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {count}\nDATA binary\n').encode('ascii')
        digest=hashlib.sha256();depth=np.full(W*H,np.inf,dtype=np.float32);source_counts=[]
        with pcd.open('wb') as target:
            target.write(head);digest.update(head)
            for r in selected:
                n=int(r['retained_by_scene'].get(s['number'],r['retained_by_scene'].get(str(s['number']),0)))
                work=WORK/f"source_{r['id']:02d}";part=work/f"scene_{s['number']:02d}.bin"
                assert part.stat().st_size==n*DTYPE.itemsize
                with part.open('rb') as inp:
                    for buf in iter(lambda:inp.read(8*1024*1024),b''):
                        target.write(buf);digest.update(buf)
                np.minimum(depth,np.load(work/f"depth_{s['number']:02d}.npy"),out=depth)
                source_counts.append({'source_file_id':r['id'],'source_file':r['path'],'points':n})
        assert pcd.stat().st_size==len(head)+count*DTYPE.itemsize
        pixels=render(depth,imdir/'panorama_annotated.jpg',prdir/'registration_overlay.jpg',count)
        info={'scene_number':s['number'],'frame_key':s['old']['frame_key'],'route':s['old']['route'],'species':s['old']['species'],'human_label_count':s['old']['human_label_count'],
              'full_density_points':count,'previous_sparse_points':s['old']['final_point_count'],'density_ratio':count/s['old']['final_point_count'],
              'point_coordinates':'Original vendor mapping local XYZ (metres), unchanged; not camera coordinates',
              'crop':'horizontal distance from original camera local XY <=45 m; no vertical cut; nonfinite XYZ cannot be spatially assigned',
              'not_used':['DS point clouds','voxel cache','voxel downsampling','random sampling','point count cap','deduplication','CEDD','SegmentAnyTree','denoising','ground removal'],
              'source_counts':source_counts,'field_notes':'Source intensity and RGB retained where present. Intensity NaN for RGB-only files; RGB=0 for intensity-only files. source_file_id references top-level source manifest; source_point_id is original zero-based row. Overlapping source points retained.',
              'camera_position_map_xyz':s['position'],'world_from_camera':s['world_from_camera'],'projection':'camera = (map_xyz - camera_position) @ world_from_camera; u=(.5-atan2(y,x)/(2pi))%1; v=.5-asin(z/r)/pi',
              'matching':'45 m spatial matching from original mapping clouds, timestamps unavailable; no claim of per-frame +/-3 s',
              'registration_status':'Uses corrected INS axis convention and existing empirical 180-degree camera yaw; no new precision extrinsic calibration',
              'rasterized_occupied_pixels':pixels,'pcd_sha256':digest.hexdigest(),'original_image_sha256':shared.sha256(imdir/'panorama_original.jpg')}
        assert info['original_image_sha256']==s['old']['source_panorama_sha256']
        save(directory/'场景说明.json',info)
        summary.append({'序号':s['number'],'目录':s['folder'],'路段':s['old']['route'],'帧':s['old']['frame_key'],'标注数':s['old']['human_label_count'],'原密度点数':count,'上一版点数':s['old']['final_point_count'],'点数倍数':round(info['density_ratio'],2),'PCD字节数':pcd.stat().st_size})
        preview=Image.open(prdir/'registration_overlay.jpg').resize((780,390),Image.Resampling.LANCZOS)
        x=(i%2)*800+10;y=(i//2)*450+48;contact.paste(preview,(x,y))
        ImageDraw.Draw(contact).text((x,y-38),f"{s['number']:02d} {s['old']['route']} {count:,} 点",font=shared.font(25),fill='white')
        print('EXPORTED',s['number'],count,'points',flush=True)
    with (OUT/'总清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    contact.save(OUT/'8个场景配准总览.jpg',quality=94)
    readme=f'''# v4 不降采样场景包

保留 v3 的全部 8 个场景、原编号与人工标注，其中 06、07 为狐尾椰子。其余为尖沙咀、司徒拔道的红花羊蹄甲、榕树、石栗场景。每帧多个标注，共 31 个跨帧实例。

## 原密度点云

从两个采集目录共 16 个 pointcloud_数字.pcd 未降采样建图文件逐点读取，并保留相机周围 45 m 水平半径内的全部有限坐标点。未使用 DS、体素缓存、随机采样、去重、CEDD、SegmentAnyTree、去噪或地面剔除。不切高度，不设置最大点数。来源建图数据本身经过供应方建图，本包不声称恢复为原始传感器数据包。

每个 `点云数据/lidar_scene_full_density.pcd` 为 binary PCD，保留原始建图局部 XYZ（米），以及原有 intensity/RGB。与 v3 的相机坐标 PCD 不同，本版点坐标不作变换，配准图片计算时才按场景说明中的矩阵变换。source_file_id 和 source_point_id 可以逐点追溯来源。跨文件重复观测不合并。不存在的 intensity 为 NaN，RGB 为0；对应源文件字段可在清单核对。

## 影像及配准

影像数据原样复制 v3，包含原始 JPG、人工标注图、标注 JSON。全部导出点参与投影；二维图片同一像素显示距离最近的点，是可见性处理，不是对导出点云降采样。配准仍基于原项目 INS 与180度偏航关系；局部遮挡/运动目标重影可能存在。建图文件缺少逐点时间戳，本版采用45米空间匹配，不声称逐帧±3秒时间匹配。

## 数量及验证

总计导出 {sum(r['原密度点数'] for r in summary):,} 点（相邻场景可重复包含同一原始点）。每个源文件实际读取点数须等于PCD头声明点数；每场景二进制字节数按点数核验；原始JPG通过SHA256与v3原图比较。PCD SHA256保存在各场景说明中。可用 CloudCompare 打开点云，注意原密度版本的加载时间和内存需求高于稀疏版。
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    save(OUT/'核验结果.json',{'all_source_declared_counts_match':True,'source_files':len(reports),'source_points_scanned':sum(r['rows_read'] for r in reports),'exported_scene_points':sum(r['原密度点数'] for r in summary),'all_pcd_binary_sizes_match':True,'original_image_hashes_match':True,'scenes':summary})
    save(WORK/'status.json',{'stage':'creating_zip','output':str(OUT),'summary':summary})
    archive=OUT.with_suffix('.zip')
    print('CREATING ZIP',archive,flush=True)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,str(Path(OUT.name)/p.relative_to(OUT)))
    print('CHECKING ZIP CRC',flush=True)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    save(WORK/'status.json',{'stage':'complete','output':str(OUT),'zip':str(archive),'zip_bytes':archive.stat().st_size,'summary':summary})
    print('DONE',str(archive),archive.stat().st_size,flush=True)


if __name__=='__main__':
    main()
