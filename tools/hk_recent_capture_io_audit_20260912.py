"""Isolated I/O dependencies; no imports of the training environment's model stack."""
import sys,json,subprocess,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/hk_fullscene_extract_20260912_v1';sys.path.insert(0,str(OUT/'io_runtime'))
from rosbags.rosbag1 import Reader
import imageio_ffmpeg
def main():
    dest=OUT/'capture_io_audit';dest.mkdir(exist_ok=False);records=[]
    for bag in sorted((ROOT/'lidar data/hk').rglob('*.bag')):
        record=dict(source=str(bag),capture=bag.parent.name)
        try:
            with Reader(bag) as reader:record.update(start_time=reader.start_time,end_time=reader.end_time,message_count=reader.message_count,connections=[dict(topic=c.topic,type=c.msgtype,count=c.msgcount) for c in reader.connections])
        except Exception as e:record['error']=str(e)
        records.append(record);print(bag.parent.name,record,flush=True)
    videos=[];ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for p in sorted((ROOT/'lidar data/hk').rglob('*.insv')):
        capture=next(x for x in ['tree981','tree982'] if x in p.parts);folder=dest/capture;folder.mkdir(exist_ok=True);probe=subprocess.run([ffmpeg,'-hide_banner','-i',str(p)],capture_output=True,text=True,encoding='utf-8',errors='replace');(folder/(p.stem+'_probe.txt')).write_text(probe.stderr,encoding='utf-8');images=[]
        for sec in [0,30,90]:
            target=folder/(p.stem+f'_{sec:03d}.jpg');res=subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-ss',str(sec),'-i',str(p),'-frames:v','1','-vf','scale=960:-1','-n',str(target)],capture_output=True,text=True,encoding='utf-8',errors='replace')
            if target.exists():images.append(str(target))
        videos.append(dict(source=str(p),capture=capture,frames=images,projection='raw video view; fisheye/stitching not yet calibrated'))
    (dest/'manifest.json').write_text(json.dumps(dict(bags=records,videos=videos,ffmpeg=ffmpeg,source_mutations=False),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
