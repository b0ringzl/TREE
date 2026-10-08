"""Local test emulator only: inspect UI, capture screenshots, report diagnostics."""
import subprocess,sys,xml.etree.ElementTree as ET,json
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ADB=ROOT/'outputs/android_toolchain_20260918_v1/sdk/platform-tools/adb.exe'
OUT=ROOT/'outputs/SpaceScan_Android_Offline_20260918_v2'
def adb(*args):return subprocess.check_output([str(ADB),'-s','emulator-5580',*args],encoding='utf-8')
def snapshot():
    adb('shell','uiautomator','dump','/sdcard/spacescan-qa-ui.xml')
    raw=adb('shell','cat','/sdcard/spacescan-qa-ui.xml')
    (OUT/'latest_ui.xml').write_text(raw,encoding='utf-8')
    root=ET.fromstring(raw);nodes=[]
    for n in root.iter('node'):
        a=n.attrib
        if a.get('text') or a.get('content-desc'):
            nodes.append({k:a.get(k) for k in ('text','content-desc','bounds','clickable','enabled','class')})
    print(json.dumps(nodes,ensure_ascii=False,indent=2))
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='screen':
        name=sys.argv[2];assert name in ('home','result','polygon','photo','camera','history')
        adb('shell','screencap','-p','/sdcard/spacescan-qa.png');adb('pull','/sdcard/spacescan-qa.png',str(OUT/(name+'.png')))
    else:snapshot()
