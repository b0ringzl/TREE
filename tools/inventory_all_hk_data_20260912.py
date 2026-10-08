"""Read-only source inventory, header audits and authoritative review counts."""
import os,json,csv,sqlite3,hashlib,struct
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import numpy as np
import laspy,yaml
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/hk_data_inventory_20260912_v1'
NEW=ROOT/'lidar data/hk'
IMAGE={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
SKIP={'.git','node_modules','__pycache__','.venv','venv','envs'}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def walk(root):
    for base,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if d not in SKIP and not Path(base,d).is_symlink()]
        for n in files:
            p=Path(base,n)
            if p.is_symlink():continue
            yield p
def stats(files):
    c=Counter();b=Counter()
    for r in files:c[r['extension']]+=1;b[r['extension']]+=r['bytes']
    return dict(files=len(files),bytes=sum(r['bytes'] for r in files),image_files=sum(c[x] for x in IMAGE),by_extension={x:dict(files=c[x],bytes=b[x]) for x in sorted(c)})
def pcd(p):
    header={}
    with p.open('rb') as f:
        for _ in range(100):
            line=f.readline(10000).decode('ascii').strip()
            if line and not line.startswith('#'):
                k,*v=line.split();header[k]=v
                if k=='DATA':break
        offset=f.tell();kind=header['DATA'][0];n=int(header['POINTS'][0]);out=dict(header=header,points=n,data_offset=offset,encoding=kind)
        if kind=='binary':
            stride=sum(int(s)*int(c) for s,c in zip(header['SIZE'],header.get('COUNT',['1']*len(header['FIELDS']))));out['expected_bytes']=offset+stride*n;out['size_consistent']=p.stat().st_size==out['expected_bytes']
        elif kind=='binary_compressed':
            comp,raw=struct.unpack('<II',f.read(8));out.update(compressed_bytes=comp,uncompressed_bytes=raw,size_consistent=p.stat().st_size==offset+8+comp)
        else:out['size_consistent']=None
    return out
def las(p):
    with laspy.open(p) as f:
        h=f.header;crs=h.parse_crs()
        return dict(points=int(h.point_count),format=int(h.point_format.id),version=str(h.version),bounds_min=h.mins.tolist(),bounds_max=h.maxs.tolist(),scales=h.scales.tolist(),offsets=h.offsets.tolist(),crs=crs.to_string() if crs else None,dimensions=list(h.point_format.dimension_names),creation_date=str(h.creation_date),validation='header only; no complete point-content scan')
def sql_audit(p):
    result={}
    with sqlite3.connect(p.as_uri()+'?mode=ro',uri=True) as c:
        c.execute('BEGIN')
        names=[r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'") if not r[0].startswith('sqlite_')]
        for name in names:
            quote='"'+name.replace('"','""')+'"';cols=[r[1] for r in c.execute('PRAGMA table_info('+quote+')')]
            result[name]=dict(rows=c.execute('SELECT COUNT(*) FROM '+quote).fetchone()[0],columns=cols)
            for k in ['decision','status','review_status']:
                if k in cols:result[name][k]=dict(c.execute('SELECT "'+k+'", COUNT(*) FROM '+quote+' GROUP BY "'+k+'"').fetchall())
    return result
def main():
    OUT.mkdir(parents=True,exist_ok=False);errors=[]
    roots=[('recent_hk',NEW),('VMMS_HoManTin',ROOT.parent/'vmms/2023-10-27_hewentian'),('VMMS_TsimShaTsui',ROOT.parent/'vmms/20250210-jianshazui'),('VMMS_Stubbs',ROOT.parent/'vmms/Stubbs Road_2024-07-25_HyD_collect'),('VMMS_location_unconfirmed',ROOT.parent/'vmms/campus'),('VMMS_floor6_unconfirmed',ROOT.parent/'vmms/floor6'),('VMMS_library_unconfirmed',ROOT.parent/'vmms/library'),('vmms_project',ROOT/'hk-tree-vmms'),('legacy_web_dataset',ROOT/'dataset')]
    roots += [('external_'+p.name,p) for p in (ROOT/'external_datasets').iterdir() if p.is_dir()]
    roots += [('baseline_'+s,ROOT/s) for s in ['image_species_baseline','full_image_species_baseline','yolo11_species_baseline','tree_species_teacher_demo','tree_species_demo','spacescan_trees_demo_v1','hk-tree-labeler','LabelPaw-web-images']]
    roots += [('output_'+p.name,p) for p in (ROOT/'outputs').iterdir() if p.is_dir() and (p.name.startswith(('vmms','focus8','hk_point','plantnet','distillation')))]
    inventories={};allnew=[];by_sub={};manifest=OUT/'file_inventory.jsonl'
    with manifest.open('w',encoding='utf-8') as f:
        for key,root in roots:
            rs=[]
            for p in walk(root):
                try:
                    st=p.stat();r=dict(collection=key,path=str(p.resolve()),relative_path=p.relative_to(root).as_posix(),bytes=st.st_size,extension=p.suffix.lower() or '(none)',modified_at_utc=datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat());rs.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n')
                except OSError as e:errors.append(dict(path=str(p),error=str(e)))
            inventories[key]=dict(root=str(root),**stats(rs));print(key,len(rs),round(inventories[key]['bytes']/1e9,3),'GB',flush=True)
            groups=defaultdict(list)
            for r in rs:groups[Path(r['relative_path']).parts[0]].append(r)
            by_sub[key]={k:stats(v) for k,v in groups.items()}
            if key=='recent_hk':allnew=rs
    headers=[]
    for r in allnew:
        p=Path(r['path'])
        if p.suffix.lower() not in {'.las','.laz','.pcd'}:continue
        try:headers.append(dict(path=str(p),relative_path=r['relative_path'],bytes=r['bytes'],**(pcd(p) if p.suffix.lower()=='.pcd' else las(p))))
        except Exception as e:errors.append(dict(path=str(p),error=str(e)))
    sessions=[]
    for p in sorted(NEW.rglob('test.bag')):
        folder=p.parent;entry=dict(id=folder.name,path=str(folder),assets=[x.relative_to(folder).as_posix() for x in walk(folder)],bag_magic=p.open('rb').read(13).decode('ascii',errors='replace'))
        for name in ['info.yaml','start_info.yaml']:
            q=folder/name
            if q.exists():entry[name]=yaml.safe_load(q.read_text(encoding='utf-8'))
        q=folder/'pose.csv'
        if q.exists():
            with q.open(encoding='utf-8-sig') as f:poses=list(csv.DictReader(f))
            times=np.array([float(r['timestamp']) for r in poses]);coords=np.array([[float(r['pos_'+a]) for a in 'xyz'] for r in poses])
            entry['trajectory']=dict(rows=len(poses),first_timestamp=float(times[0]),last_timestamp=float(times[-1]),duration_s=float(times[-1]-times[0]),nonincreasing_timestamps=int((np.diff(times)<=0).sum()),finite_coordinates=bool(np.isfinite(coords).all()),xyz_extent=np.ptp(coords,axis=0).tolist(),accumulated_step_distance=float(np.linalg.norm(np.diff(coords,axis=0),axis=1).sum()),coordinate_system='local pose, external CRS not established')
        entry['pointclouds']=[r for r in headers if Path(r['path']).parent==folder];entry['bytes']=sum(Path(x).stat().st_size for x in walk(folder));sessions.append(entry)
    blocks=[]
    for p in sorted(NEW.glob('data/data/*/*_summary.json')):
        folder=p.parent;rs=[h for h in headers if Path(h['path']).is_relative_to(folder)];trees=[h for h in rs if 'individual_tree_output' in Path(h['path']).parts]
        blocks.append(dict(id=folder.name,path=str(folder),semantic_summary=read(p),las_files=len(rs),individual_tree_files=len(trees),individual_tree_points=sum(x['points'] for x in trees),tree_point_min=min([x['points'] for x in trees],default=0),tree_point_max=max([x['points'] for x in trees],default=0),primary_header=next((h for h in rs if Path(h['path']).name==folder.name+'.las'),None),tree_ids=[Path(h['path']).stem for h in trees]))
    review_audits={}
    for root in [ROOT/'external_datasets/herbarium_training_candidates_20260907',ROOT/'external_datasets/palm_confusion_20260908']:
        p=root/'human_review.sqlite3'
        if p.exists():review_audits[str(p)]=sql_audit(p)
    important={}
    selected=[ROOT/'hk-tree-vmms/outputs/coordinates/coordinate_summary.json',ROOT/'outputs/hk_point_assist_20260911_v1/summary.json',ROOT/'hk-tree-vmms/derived/reviewed_five_species_pilot_v1_20260907/metrics.json',ROOT/'outputs/vmms20m_teacher_20260909_v1/summary.json',ROOT/'outputs/vmms20m_teacher_20260909_v1/object_gate_20260910_reviewed_v4/summary.json']
    for root in (ROOT/'external_datasets').iterdir():
        if root.is_dir():selected += [root/n for n in ['summary.json','collection_summary.json','curation_summary.json']]
    for p in selected:
        if p.exists():important[str(p)]=dict(sha256=digest(p),data=read(p))
    live_reviews={}
    for p in [ROOT/'hk-tree-vmms/derived/exhaustive_species_frame_labeler/runtime/frame_review_state.json',ROOT/'hk-tree-vmms/derived/exhaustive_species_frame_labeler/runtime/frame_drafts.json',ROOT/'outputs/hk_point_assist_20260911_v1/geometry_reviews.json']:
        if p.exists():
            d=read(p);live_reviews[str(p)]=dict(records=len(d),statuses=dict(Counter(v.get('frame_status',v.get('decision','unspecified')) for v in d.values())),labels=sum(len(v.get('labels',[])) for v in d.values()),sha256=digest(p))
    data=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='Hong Kong project data and associated training/reference assets. Auxiliary web data is not necessarily photographed in Hong Kong. Excludes WHU-only data, dependencies, git, worktrees; physical counts are not unique samples.',inventories=inventories,subdirectories=by_sub,recent_sessions=sessions,processed_blocks=blocks,point_headers=headers,sqlite_reviews=review_audits,important_sources=important,live_review_counts=live_reviews,errors=errors,source_mutations=False,full_raw_hash_scan=False)
    write(OUT/'inventory.json',data)
    print('COMPLETE',len(headers),'point headers;',len(sessions),'sessions;',sum(b['individual_tree_files'] for b in blocks),'tree files;',len(errors),'errors',flush=True)

if __name__=='__main__':main()
