"""Independent invariants, export round-trip checks and same-points comparisons."""
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import build_selected_raw_multimodal_scenes as build


def read_export(path):
    with path.open(encoding='ascii') as stream:
        while stream.readline().strip() != 'DATA ascii':
            if stream.tell() >= path.stat().st_size:
                raise ValueError('Invalid PCD')
        return np.loadtxt(stream)


def old_rotation(roll, pitch, heading):
    cr, sr, cp, sp, ch, sh = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(heading), math.sin(heading)
    return np.array([[ch,-sh,0],[sh,ch,0],[0,0,1]]) @ np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]) @ np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])


def main():
    # Independent cardinal-direction checks: heading zero means north in ENU.
    np.testing.assert_allclose(build.world_from_flu(0,0,0) @ [1,0,0], [0,1,0], atol=1e-12)
    np.testing.assert_allclose(build.world_from_flu(0,0,np.pi/2) @ [1,0,0], [1,0,0], atol=1e-12)
    np.testing.assert_allclose(build.world_from_flu(0,0,0) @ [0,1,0], [-1,0,0], atol=1e-12)
    uv,_ = build.project_camera(np.array([[1.,0,0],[0,1,0],[0,0,1],[0,0,-1]]))
    np.testing.assert_allclose(uv, [[.5,.5],[.25,.5],[.5,0],[.5,1]], atol=1e-12)
    rows = build.read_coordinates()
    findings=[]
    comparison=Image.new('RGB',(1600,5*455),'#091522')
    comparison_index = 0
    diagnostics=build.OUTPUT/'核验对照'
    diagnostics.mkdir(exist_ok=True)
    for directory in sorted(build.OUTPUT.glob('[0-9][0-9]_*')):
        meta=build.load_json(directory/'场景说明.json')
        assert meta['human_label_count'] >= 2
        key=meta['frame_key']; row=rows[key]
        xyz=read_export(directory/'点云数据/lidar_scene_sparse.pcd')
        assert xyz.shape==(meta['final_point_count'],3)
        assert np.isfinite(xyz).all() and len(xyz)<=150000
        original=directory/'影像数据/panorama_original.jpg'
        assert build.sha256(original)==meta['source_panorama_sha256']
        for name, relative in [('lidar_scene_sparse.pcd','点云数据'),('panorama_original.jpg','影像数据'),('panorama_annotated.jpg','影像数据'),('registration_overlay.jpg','配准图片')]:
            assert build.sha256(directory/relative/name)==meta['generated_sha256'][name]
        if meta['route'] in ('尖沙咀', '司徒拔道'):
            points,_=build.tst_points(row)
        else:
            points,_=build.hwt_points(meta['frame_id'])
        expected=build.sample_exact(points,key).astype(np.float32).astype(np.float64)
        error=float(np.max(np.abs(xyz-expected)))
        assert error<=.000001
        item={'frame_key':key,'point_count':len(xyz),'max_export_roundtrip_error_m':error,'original_image_byte_copy_verified':True,'file_hashes_verified':True}
        if meta['scene_number'] in (6, 7):
            previous=build.TREE_ROOT/'交付/人工标注_原始LiDAR配准场景_20260907_v2_多标注'
            old_folder=next(previous.glob(f"{meta['scene_number']:02d}_*"))
            old_meta=build.load_json(old_folder/'场景说明.json')
            assert old_meta['frame_key']==key
            for relative in ('点云数据/lidar_scene_sparse.pcd','影像数据/panorama_original.jpg','影像数据/annotation.json'):
                assert build.sha256(old_folder/relative)==build.sha256(directory/relative)
            item['preserved_from_v2_byte_identical_pcd_image_annotations']=True
        else:
            assert any(any(t in n for t in ('Bauhinia purpurea','Ficus','Aleurites moluccana')) for n in meta['species'])
        item['human_label_count']=meta['human_label_count']
        item['species']=meta['species']
        if meta['route']=='尖沙咀':
            attitude=[float(row[k+'_rad']) for k in ('roll','pitch','heading')]
            offset=np.diag([-1.,-1.,1.])
            corrected=build.world_from_flu(*attitude)@offset
            wrong=old_rotation(*attitude)@offset
            # Recover world-relative points, then apply the faulty old transform
            # to exactly the same samples. Density and image do not change.
            old_xyz=xyz@corrected.T@wrong
            delta_deg=math.degrees(math.acos(np.clip((np.trace(corrected.T@wrong)-1)/2,-1,1)))
            item['old_vs_correct_rotation_deg']=delta_deg
            image=Image.open(directory/'影像数据/panorama_annotated.jpg').convert('RGB')
            p=diagnostics/f"{meta['scene_number']:02d}_old_wrong_same_points.jpg"
            build.registration_image(image,old_xyz,p)
            # Replace renderer's v2 banner; the old view is diagnostic only.
            old=Image.open(p).convert('RGB'); d=ImageDraw.Draw(old)
            d.rectangle((0,0,old.width,54),fill='#6b1824')
            d.text((18,10),'旧变换（错误）— 与右图使用相同点云、相同点数',font=build.font(28),fill='white')
            old.save(p,quality=94)
            index=comparison_index
            comparison_index+=1
            for col,path in enumerate([p,directory/'配准图片/registration_overlay.jpg']):
                thumb=Image.open(path).convert('RGB').resize((780,390),Image.Resampling.LANCZOS)
                comparison.paste(thumb,(10+col*800,index*455+50))
                label=f"{meta['scene_number']:02d} {meta['frame_id']} " + ('旧版错误变换' if col==0 else '修正INS变换')
                ImageDraw.Draw(comparison).text((16+col*800,index*455+10),label,font=build.font(25),fill='white')
        findings.append(item)
        print(key, len(xyz), 'verified',flush=True)
    assert len(findings)==8
    comparison.save(diagnostics/'尖沙咀_同点数修正前后对照.jpg',quality=94)
    build.write_json(build.OUTPUT/'核验结果.json',{'invariants':'north/east INS headings and panorama cardinal projection passed','scenes':findings,'interpretation':'Numerical consistency tests are not a measured calibration accuracy. Visual static-structure alignment improved; occlusion and moving objects remain.'})
    report='''# v3 指定树种场景核验说明

保留 06（尖沙咀 P1 002749）和 07（尖沙咀 P2 001485）的狐尾椰子场景。与 v2 多标注版逐字节比较，两场景的 PCD、原始影像、人工标注 JSON 一致；配准图仅统一了标题。

替换其余六场景：01 尖沙咀红花羊蹄甲与榕树（4 标注）；02 尖沙咀红花羊蹄甲（3）；03 尖沙咀榕树（6）；04 司徒拔道石栗（2）；05 司徒拔道榕树（3）；08 司徒拔道榕树（2）。加上 06 的 7 个和 07 的 4 个，本包共 31 个跨帧标注实例。01、02 相邻帧可能包含重复实体树，不将该数量解释为独立树木总数。

## 已修正的问题

原导出脚本错误采用普通平面偏航角及 Rz @ Ry @ Rx。此版继续调用修正后的项目共享 INS 函数：北向零、顺时针，level @ roll @ pitch。两个路段均使用对应采集目录的建图点云和该影像的 INS 位置姿态。

## 检查结果

8 个场景均通过点数/有限坐标检查、原始 JPG 逐字节哈希校验、输出哈希校验及从来源点云重新计算后的坐标往返校验（最大误差小于 0.000001 m，仅为导出精度）。独立的北/东航向与全景正交方向公式检查通过。

五个尖沙咀场景额外生成使用同一批点的错误变换/修正变换对照，因此图像差异不来自采样密度变化。查看“核验对照/尖沙咀_同点数修正前后对照.jpg”。旧变换图仅为诊断对照，不作为可用配准结果。

## 视觉核验范围

已检查八场景总览和新增司徒拔道场景的完整分辨率叠加图。主要树冠和道路结构位于对应方向。第04场景墙体、树冠，第05场景挡土墙与树木，第08场景道路两侧结构可供进一步人工核对。相机与地图观测位置不同，后台建筑点可能投到前景树冠、车辆上，稀疏点云逐像素深度测试不能完全处理此类遮挡。移动巴士等存在累积重影，不适合作为静态配准基准。

本次完成的是导出坐标变换错误修复及可视核验，不代表精细外参重新标定或达到某个厘米/像素误差。未提供实测同名控制点，因此不报告虚构的配准精度。原有局部点云缺失也不会由此补齐。

## 使用版本

请使用名称末尾带 _v3_指定树种 的场景包，每帧均含至少两个人工标注。建图点云只经过空间裁剪、坐标变换和降采样，未进行 CEDD 掩膜、SegmentAnyTree、地面剔除或去噪。由于源建图点云无逐点时间戳，本版为 45 米空间匹配，不宣称逐影像 ±3 秒匹配。
'''
    (build.OUTPUT/'配准修正与核验说明.md').write_text(report,encoding='utf-8')
    archive=shutil.make_archive(str(build.OUTPUT),'zip',root_dir=build.OUTPUT.parent,base_dir=build.OUTPUT.name)
    print('ZIP',archive,flush=True)


if __name__=='__main__':
    main()
