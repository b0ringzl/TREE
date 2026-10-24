"""Offline Windows supervisor. Starts and stops only its own child processes."""
import argparse
import ctypes
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from ctypes import wintypes

ROOT=Path(__file__).resolve().parent
APP=ROOT/'app'
RUNTIME=ROOT/'runtime'
USER=ROOT/'user_data'
USER.mkdir(exist_ok=True)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
    YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(USER/'ultralytics'),
    PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
os.environ.pop('PYTHONPATH',None)
os.environ.pop('PYTHONHOME',None)
os.environ['PATH']=os.pathsep.join([str(RUNTIME),str(RUNTIME/'Library/bin'),str(RUNTIME/'DLLs'),
    str(RUNTIME/'Scripts'),os.environ.get('PATH','')])
os.environ.setdefault('SAM_PYTHON', str(RUNTIME/'python.exe'))
os.environ.setdefault('SPACESCAN_SAM_WEIGHT', str(APP/'LabelPaw-web-images/weights/sam_weights/sam2.1_hiera_tiny.pt'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def prepare_runtime():
    marker=USER/'runtime_location.json'
    if marker.exists():
        if json.loads(marker.read_text())['path']!=str(ROOT):
            raise RuntimeError('This initialized folder was moved. Extract the ORIGINAL ZIP again at the final location. / 請將原始 ZIP 重新解壓到最終位置。')
        return
    script=RUNTIME/'Scripts/conda-unpack-script.py'
    if not script.is_file():raise RuntimeError('Runtime incomplete; extract the full ZIP.')
    print('Preparing bundled runtime (first launch only)...',flush=True)
    subprocess.run([sys.executable,'-s',str(script)],check=True,cwd=ROOT)
    marker.write_text(json.dumps({'path':str(ROOT)}),encoding='utf-8')


def check_files(force=False):
    manifest=ROOT/'model_manifest.json'
    manifest_hash=sha(manifest)
    entries=json.loads(manifest.read_text(encoding='utf-8'))
    cache_path=USER/'verified_files.json'
    if not force and cache_path.is_file():
        try:
            cache=json.loads(cache_path.read_text(encoding='utf-8'))
            if (cache.get('root')==str(ROOT) and cache.get('manifest_sha256')==manifest_hash
                    and len(cache.get('files',[]))==len(entries)):
                for row,stored in zip(entries,cache['files']):
                    path=ROOT/row['file']
                    stat=path.stat()
                    if (stored[0]!=row['file'] or stat.st_size!=row['bytes']
                            or stat.st_mtime_ns!=stored[1]):
                        break
                else:
                    print('Verified files unchanged; using fast startup.',flush=True)
                    return
        except (OSError,ValueError,TypeError,KeyError,IndexError):
            pass
    print('Verifying installed files (full check)...',flush=True)
    verified=[]
    for row in entries:
        path=ROOT/row['file']
        if not path.is_file():
            raise RuntimeError('Missing/corrupted model or source: '+row['file'])
        stat=path.stat()
        if stat.st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise RuntimeError('Missing/corrupted model or source: '+row['file'])
        verified.append([row['file'],stat.st_mtime_ns])
    temporary=cache_path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'root':str(ROOT),'manifest_sha256':manifest_hash,
                                      'files':verified}),encoding='utf-8')
    os.replace(temporary,cache_path)


def hardware():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('NVIDIA CUDA unavailable. Update the NVIDIA driver; this package requires a supported NVIDIA GPU.')
    if torch.cuda.get_device_properties(0).major < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This exact-model build requires native BF16 (RTX 30/40/50 series or compatible Ampere+ NVIDIA GPU). No silent model replacement.')
    p=torch.cuda.get_device_properties(0)
    free,total=torch.cuda.mem_get_info()
    print(f'GPU: {p.name}; total {total/2**30:.1f} GiB; free {free/2**30:.1f} GiB',flush=True)
    if total < 11*2**30:raise RuntimeError('At least 12 GB VRAM is required for this full-model build; 16 GB or more is recommended. Smaller cards are not validated.')
    if free < 9*2**30:raise RuntimeError('Less than 9 GiB VRAM is free. Close other GPU applications and retry.')
    if total < 15*2**30:print('WARNING: below recommended 16 GB VRAM; full panorama/detector workload may run out of memory.',flush=True)
    return {'name':p.name,'total_bytes':total,'free_bytes':free,'torch':torch.__version__}


def ports():
    # Never reuse, terminate or attach to unrelated localhost services.
    for image in range(18339,18439,2):
        sockets=[]
        try:
            for p in (image,image+1):
                s=socket.socket();s.bind(('127.0.0.1',p));sockets.append(s)
            return image,image+1
        except OSError:pass
        finally:
            for s in sockets:s.close()
    raise RuntimeError('No free local port pair')


def child_job():
    # Closing the console also closes this handle, terminating only our children.
    class Limits(ctypes.Structure):
        _fields_=[('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),('flags',wintypes.DWORD),
                  ('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),('active',wintypes.DWORD),
                  ('affinity',ctypes.c_size_t),('priority',wintypes.DWORD),('schedule',wintypes.DWORD)]
    class Extended(ctypes.Structure):
        _fields_=[('basic',Limits),('io',ctypes.c_ulonglong*6),('process_mem',ctypes.c_size_t),
                  ('job_mem',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.restype=wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]
    kernel.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    job=kernel.CreateJobObjectW(None,None)
    if not job:raise ctypes.WinError(ctypes.get_last_error())
    info=Extended();info.basic.flags=0x2000
    if not kernel.SetInformationJobObject(job,9,ctypes.byref(info),ctypes.sizeof(info)):
        kernel.CloseHandle(job);raise ctypes.WinError(ctypes.get_last_error())
    return kernel,job


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--edition',choices=['image-only','multimodal','video','corridor'],default='multimodal')
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    prepare_runtime()
    import requests
    previous=USER/'running.json'
    stop=USER/'stop.request'
    if stop.exists():
        print('Waiting for the previous desktop session to finish...',flush=True)
        for _ in range(40):
            if not previous.exists():break
            time.sleep(0.5)
        if previous.exists():
            raise RuntimeError('The previous desktop session is still shutting down. Retry in a moment.')
        # A stale request from an interrupted shutdown must not stop this run.
        stop.unlink(missing_ok=True)
    if previous.exists() and not args.check_only:
        saved=json.loads(previous.read_text())
        try:
            status=requests.get(saved['base_url']+'/api/health',timeout=2).json()
            if status.get('ready') and status.get('single_image_version')==saved['version']:
                url=saved['base_url']+'/'+args.edition
                print('Already running: '+url,flush=True)
                if not args.no_browser:webbrowser.open(url)
                return
        except requests.RequestException:pass
    print('Checking model files...',flush=True)
    check_files(force=args.check_only)
    gpu=hardware()
    if args.check_only:
        print('CHECK OK',json.dumps(gpu));return
    teacher_port,web_port=ports()
    os.environ['SPACESCAN_TEACHER_URL']=f'http://127.0.0.1:{teacher_port}'
    version=json.loads((APP/'spacescan_trees_demo_v1/release.json').read_text())['classifier_version']
    logs=USER/('logs_'+time.strftime('%Y%m%d_%H%M%S'))
    logs.mkdir()
    children=[];handles=[]
    kernel,job=child_job()
    try:
        for module,port,name in [('tree_species_teacher_demo.server:app',teacher_port,'teacher'),('spacescan_trees_demo_v1.server:app',web_port,'demo')]:
            handle=(logs/(name+'.log')).open('w',encoding='utf-8');handles.append(handle)
            children.append(subprocess.Popen([sys.executable,'-s','-m','uvicorn',module,'--host','127.0.0.1','--port',str(port)],
                cwd=APP,env=os.environ.copy(),stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW))
            if not kernel.AssignProcessToJobObject(job,wintypes.HANDLE(children[-1]._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
        base=f'http://127.0.0.1:{web_port}'
        for _ in range(300):
            if any(p.poll() is not None for p in children):raise RuntimeError('Service exited; see '+str(logs))
            try:
                info=requests.get(base+'/api/health',timeout=2).json()
                if info.get('ready') and info.get('single_image_version')==version:break
            except requests.RequestException:pass
            time.sleep(1)
        else:raise RuntimeError('Model startup timed out; see '+str(logs))
        state={'url':base+'/'+args.edition,'base_url':base,'teacher_url':os.environ['SPACESCAN_TEACHER_URL'],
               'version':version,'pids':[p.pid for p in children],'gpu':gpu,'logs':str(logs),'supervisor_pid':os.getpid()}
        (USER/'running.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        print('READY '+state['url'],flush=True)
        print('Keep this window open. Press Ctrl+C to stop. / 保持此視窗開啟，Ctrl+C 停止。',flush=True)
        if not args.no_browser:webbrowser.open(state['url'])
        while not (USER/'stop.request').exists():
            if any(p.poll() is not None for p in children):raise RuntimeError('Service stopped unexpectedly; see '+str(logs))
            time.sleep(1)
    except KeyboardInterrupt:pass
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill()
        for h in handles:h.close()
        kernel.CloseHandle(job)
        for name in ('running.json','stop.request'):
            (USER/name).unlink(missing_ok=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('ERROR:',exc,flush=True)
        sys.exit(1)
