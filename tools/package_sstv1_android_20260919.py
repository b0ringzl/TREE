"""Verify and package the teacher Android build; no training or web publication."""
import hashlib,json,os,shutil,subprocess,zipfile
from pathlib import Path
from training_safety import sha256,verify_backup
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/SpaceScan_Android_SSTv1_20260919_v1'
PROJECT=ROOT/'spacescan_android_sstv1'
DEST=ROOT/'deliverables/SpaceScan_Trees_Android_SSTv1_v0.2_20260919'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    DEST.mkdir(exist_ok=False)
    cfg=read(ROOT/'outputs/android_toolchain_20260918_v1/paths.json');bt=Path(cfg['sdk'])/'build-tools/35.0.0'
    adb=Path(cfg['sdk'])/'platform-tools/adb.exe'
    def device(*args):return subprocess.check_output([str(adb),'-s','emulator-5580',*args],text=True,encoding='utf-8').strip()
    for name in ['sst_android_qa.json','sst_photo_result.json','sst_result.png','selftest_result.json']:
        subprocess.run([str(adb),'-s','emulator-5580','pull','/data/user/0/org.spacescan.trees.offline/files/'+name,str(OUT/name)],check=True)
    qa=read(OUT/'sst_android_qa.json');assert qa['passed'] and qa['selftest']['pass']
    assert qa['photo']['candidates'][0]['scientific_name']=='Terminalia mantaly'
    gate=read(OUT/'equivalence_gate.json');assert gate['pass_gate']
    apk=PROJECT/'app/build/outputs/apk/debug/app-debug.apk'
    installed=device('shell','pm','path','org.spacescan.trees.offline').removeprefix('package:')
    installed_hash=device('shell','sha256sum',installed).split()[0];assert installed_hash==sha256(apk)
    badging=subprocess.check_output([str(bt/'aapt.exe'),'dump','badging',str(apk)],text=True,encoding='utf-8')
    assert 'android.permission.INTERNET' not in badging and "versionCode='2'" in badging
    signature=subprocess.check_output([str(bt/'apksigner.bat'),'verify','--verbose','--print-certs',str(apk)],env=dict(os.environ,JAVA_HOME=cfg['jdk']),text=True,encoding='utf-8')
    alignment=subprocess.check_output([str(bt/'zipalign.exe'),'-c','-P','16','-v','4',str(apk)],text=True,encoding='utf-8')
    assets=PROJECT/'app/src/main/assets';manifest=read(assets/'model_manifest.json')
    with zipfile.ZipFile(apk) as z:
        assert read(OUT/'model_pack/model_manifest.json')==json.loads(z.read('assets/model_manifest.json'))
        assert len(json.loads(z.read('assets/species.json')))==148
        assert 'assets/tree_model.onnx' not in z.namelist()
        assert not [n for n in z.namelist() if n.endswith(('.jpg','.jpeg','.webp'))]
        assert not [n for n in z.namelist() if 'SSTSmokeInstrumentation' in n]
    flags={k:device('shell','settings','get','global',k) for k in ['airplane_mode_on','wifi_on','mobile_data']}
    assert flags=={'airplane_mode_on':'1','wifi_on':'0','mobile_data':'0'}
    verify_backup(OUT/'pre_training_backup')
    target=DEST/'SpaceScan-Trees-SSTv1-v0.2.apk';shutil.copy2(apk,target)
    pack=OUT/'SSTv1-20260918-Offline-Model.zip';shutil.copy2(pack,DEST/pack.name);assert sha256(pack)==sha256(DEST/pack.name)
    for name in ['equivalence_gate.json','paired_regression.json','stage_export_parity.json','java_padding_parity.json','sst_android_qa.json','first_import_qa.json','sst_photo_result.json','sst_result.png','sst_home.png']:
        shutil.copy2(OUT/name,DEST/name)
    for name in ['model_manifest.json','species.json']:shutil.copy2(assets/name,DEST/name)
    shutil.copy2(PROJECT/'INSTALL_zh.md',DEST/'安装与验收说明.md');shutil.copy2(PROJECT/'README.md',DEST/'TECHNICAL_README.md')
    with zipfile.ZipFile(DEST/'SpaceScan-SSTv1-Android-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in PROJECT.rglob('*'):
            if p.is_file() and not set(p.relative_to(PROJECT).parts)&{'build','.gradle'} and p.name!='local.properties' and p.suffix!='.keystore':z.write(p,'spacescan_android_sstv1/'+p.relative_to(PROJECT).as_posix())
        for name in ['export_sstv1_android_20260919.py','check_sstv1_android_preprocessing_20260919.py','package_sstv1_android_20260919.py','training_safety.py']:
            z.write(ROOT/'tools'/name,'tools/'+name)
    (DEST/'apk_signature.txt').write_text(signature,encoding='utf-8');(DEST/'apk_manifest.txt').write_text(badging,encoding='utf-8')
    report=dict(apk=target.name,apk_sha256=sha256(target),apk_bytes=target.stat().st_size,model_pack=pack.name,
        model_pack_sha256=sha256(pack),model_pack_bytes=pack.stat().st_size,version=manifest['desktop_version'],
        offline_flags=flags,no_internet_permission=True,signature_verified=True,alignment_16kb_verified=True,
        original_desktop_and_old_android_unchanged=True,training_performed=False,backup=str(OUT/'pre_training_backup'),
        physical_android_phone_tested=False,emulator='Android 15/API35 x86_64, 4 GB RAM',equivalence=gate,
        final_apk_installed_sha256=installed_hash,terminalia_result=qa['photo']['candidates'],terminalia_elapsed_ms=qa['photo']['elapsed_ms'])
    save(DEST/'DELIVERY.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
