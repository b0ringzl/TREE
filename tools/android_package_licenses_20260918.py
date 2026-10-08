"""Retrieve upstream licence texts (not images) for the local evaluation package."""
from pathlib import Path
import urllib.request,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'spacescan_android/app/src/main/assets'
URLS={
 'APACHE-2.0.txt':'https://www.apache.org/licenses/LICENSE-2.0.txt',
 'AGPL-3.0.txt':'https://raw.githubusercontent.com/ultralytics/ultralytics/main/LICENSE',
 'ONNXRUNTIME-LICENSE.txt':'https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/LICENSE',
 'ONNXRUNTIME-ThirdPartyNotices.txt':'https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/ThirdPartyNotices.txt',
 'PILLOW-LICENSE.txt':'https://raw.githubusercontent.com/python-pillow/Pillow/11.1.0/LICENSE',
}
def main():
    result=[]
    for name,url in URLS.items():
        data=urllib.request.urlopen(url,timeout=60).read()
        assert len(data)>500 and b'<html' not in data[:100].lower()
        (ASSETS/name).write_bytes(data)
        result.append(dict(file=name,url=url,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    (ASSETS/'license_sources.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()
