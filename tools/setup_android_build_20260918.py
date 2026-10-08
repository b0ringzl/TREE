"""Download checksummed, project-local Android build dependencies. No system PATH edits."""
import concurrent.futures
import hashlib
import json
import zipfile
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/android_toolchain_20260918_v1'

def download(url, target, expected):
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Existing download hash mismatch: ' + str(target))
        return
    partial = target.with_suffix(target.suffix + '.part')
    h = hashlib.sha256()
    with requests.get(url, stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        with partial.open('wb') as handle:
            for block in response.iter_content(1024 * 1024):
                if block:
                    handle.write(block); h.update(block)
    if h.hexdigest() != expected:
        raise RuntimeError('Download hash mismatch: ' + url)
    partial.replace(target)
    print('DOWNLOADED', target.name, target.stat().st_size, flush=True)

def extract(archive, destination):
    if destination.exists():
        return
    destination.mkdir()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            resolved = (destination / item.filename).resolve()
            if not resolved.is_relative_to(destination.resolve()):
                raise ValueError('Unsafe archive member')
        z.extractall(destination)

def main():
    OUT.mkdir(exist_ok=True)
    api = 'https://api.adoptium.net/v3/assets/latest/17/hotspot?architecture=x64&image_type=jdk&os=windows&vendor=eclipse'
    r = requests.get(api, timeout=45); r.raise_for_status(); meta = r.json()
    package = meta[0]['binary']['package']
    (OUT/'jdk_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    gradle = 'https://services.gradle.org/distributions/gradle-8.9-bin.zip'
    checksum = requests.get(gradle + '.sha256', timeout=45); checksum.raise_for_status()
    sources = [
        (package['link'], OUT/'jdk.zip', package['checksum']),
        (gradle, OUT/'gradle.zip', checksum.text.strip().split()[0]),
        ('https://dl.google.com/android/repository/commandlinetools-win-15859902_latest.zip', OUT/'cmdline.zip',
         '90ae805d20434428bffcb699c290860f19bb5f66a67e6b330067e3de801fb04a')]
    (OUT/'download_manifest.json').write_text(json.dumps([dict(url=u, file=str(p), sha256=h) for u,p,h in sources], indent=2), encoding='utf-8')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda args: download(*args), sources))
    for name in ('jdk','gradle','cmdline'):
        extract(OUT/(name+'.zip'), OUT/name)
    paths = dict(jdk=str(next((OUT/'jdk').glob('jdk-*'))),
                 gradle=str(OUT/'gradle/gradle-8.9/bin/gradle.bat'),
                 sdkmanager=str(OUT/'cmdline/cmdline-tools/bin/sdkmanager.bat'),
                 sdk=str(OUT/'sdk'))
    (OUT/'paths.json').write_text(json.dumps(paths, indent=2), encoding='utf-8')
    print(json.dumps(paths), flush=True)

if __name__ == '__main__':
    main()
