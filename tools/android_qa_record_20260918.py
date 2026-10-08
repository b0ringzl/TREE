"""Record observed integration checks from our own offline Android 35 test AVD."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ADB=ROOT/'outputs/android_toolchain_20260918_v1/sdk/platform-tools/adb.exe'
OUT=ROOT/'outputs/SpaceScan_Android_Offline_20260918_v2'
def adb(*args):return subprocess.check_output([str(ADB),'-s','emulator-5580',*args],encoding='utf-8').strip()
def main():
    selftest=json.loads(adb('shell','run-as','org.spacescan.trees.offline','cat','files/selftest_result.json'))
    history=json.loads(adb('shell','run-as','org.spacescan.trees.offline','cat','files/history.json'))
    exported=json.loads(adb('shell','cat','/sdcard/Download/SpaceScan-result-1789746338936.json'))
    assert selftest['pass'] and len(history)>=2 and exported==history[0]
    assert history[0]['manual_polygon'] and not history[1]['manual_polygon']
    assert history[0]['width']==195 and history[0]['height']==341
    assert history[1]['candidates'][0]['scientific_name']=='Aquilaria sinensis'
    assert history[0]['candidates'][0]['scientific_name']=='Ficus elastica'
    flags={key:adb('shell','settings','get','global',key) for key in ['airplane_mode_on','wifi_on','mobile_data']}
    assert flags=={'airplane_mode_on':'1','wifi_on':'0','mobile_data':'0'}
    installed=adb('shell','pm','path','org.spacescan.trees.offline').removeprefix('package:')
    installed_sha=adb('shell','sha256sum',installed).split()[0]
    apk=ROOT/'spacescan_android/app/build/outputs/apk/debug/app-debug.apk'
    assert installed_sha==hashlib.sha256(apk.read_bytes()).hexdigest()
    manifest=adb('shell','dumpsys','package','org.spacescan.trees.offline')
    assert 'android.permission.INTERNET' not in manifest
    crashlog=adb('logcat','-d','-b','crash');assert 'org.spacescan.trees.offline' not in crashlog,crashlog
    report=dict(device='Local x86_64 Android 15/API35 test emulator; not a physical phone',apk_sha256=installed_sha,
        offline_flags=flags,selftest=selftest,installed_and_started=True,photo_picker_and_decode=True,
        polygon_seven_vertices_then_undo_to_six_then_apply=True,camera_capture_returned_1392x1856=True,
        history_and_json_export_verified=True,export_matches_saved_record=True,no_internet_permission=True,
        app_crash_log_empty=True,physical_arm_phone_tested=False,
        observed_photo_results=[history[1],history[0]],
        caveat='These are UI/inference smoke tests, NOT accuracy evidence. The same user-labelled Aquilaria photo was correct Top1 before crop and wrong after polygon crop.')
    (OUT/'android_qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='observed_photo_results'},ensure_ascii=True,indent=2))
if __name__=='__main__':main()
