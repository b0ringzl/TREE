"""Package the offline evaluation app with auditable checks and corresponding source."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,zipfile
from training_safety import verify_backup
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'spacescan_android'
EVAL=ROOT/'outputs/SpaceScan_Android_Offline_20260918_v2'
DEST=ROOT/'deliverables/SpaceScan_Trees_Android_Offline_v0.1_20260918'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    DEST.mkdir(parents=True,exist_ok=False)
    paths=read(ROOT/'outputs/android_toolchain_20260918_v1/paths.json');bt=Path(paths['sdk'])/'build-tools/35.0.0'
    env=dict(os.environ,JAVA_HOME=paths['jdk'])
    apk=PROJECT/'app/build/outputs/apk/debug/app-debug.apk'
    badging=subprocess.check_output([str(bt/'aapt.exe'),'dump','badging',str(apk)],text=True,encoding='utf-8')
    assert 'android.permission.INTERNET' not in badging
    assert "sdkVersion:'26'" in badging
    signature=subprocess.check_output([str(bt/'apksigner.bat'),'verify','--verbose','--print-certs',str(apk)],env=env,text=True,encoding='utf-8')
    alignment=subprocess.check_output([str(bt/'zipalign.exe'),'-c','-P','16','-v','4',str(apk)],text=True,encoding='utf-8')
    manifest=read(PROJECT/'app/src/main/assets/model_manifest.json')
    with zipfile.ZipFile(apk) as archive:
        assert hashlib.sha256(archive.read('assets/tree_model.onnx')).hexdigest()==manifest['model_sha256']
        assert json.loads(archive.read('assets/model_manifest.json'))==manifest
        assert json.loads(archive.read('assets/selftest.json'))==read(EVAL/'selftest.json')
        assert len(json.loads(archive.read('assets/species.json')))==147
        assert not [n for n in archive.namelist() if n.endswith(('.jpg','.jpeg','.webp'))]
        for n in ('AGPL-3.0.txt','APACHE-2.0.txt','ONNXRUNTIME-LICENSE.txt','ONNXRUNTIME-ThirdPartyNotices.txt','PILLOW-LICENSE.txt'):
            assert len(archive.read('assets/'+n))>500
    backup=ROOT/'outputs/SpaceScan_Android_Offline_20260918_v1/pre_training_backup';verify_backup(backup)
    target=DEST/'SpaceScan-Trees-Offline-v0.1.apk';shutil.copy2(apk,target)
    for name in ('model_manifest.json','species.json'):
        shutil.copy2(PROJECT/'app/src/main/assets'/name,DEST/name)
    for name in ('mobile_validation.json','java_preprocessing_parity.json'):
        shutil.copy2(EVAL/name,DEST/name)
    shutil.copy2(PROJECT/'README.md',DEST/'TECHNICAL_README.md')
    if (PROJECT/'INSTALL_zh.md').exists():shutil.copy2(PROJECT/'INSTALL_zh.md',DEST/'安装与验收说明.md')
    if (EVAL/'android_qa.json').exists():shutil.copy2(EVAL/'android_qa.json',DEST/'android_qa.json')
    for name in ('home.png','result.png','polygon.png'):
        if (EVAL/name).exists():shutil.copy2(EVAL/name,DEST/name)
    with zipfile.ZipFile(DEST/'SpaceScan-Android-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in PROJECT.rglob('*'):
            if p.is_file() and not set(p.relative_to(PROJECT).parts)&{'.gradle','build'} and p.name!='local.properties' and p.suffix!='.keystore':
                z.write(p,'spacescan_android/'+p.relative_to(PROJECT).as_posix())
        for name in ('export_spacescan_android_20260918.py','validate_android_preprocessing_20260918.py','android_package_licenses_20260918.py','package_android_delivery_20260918.py','setup_android_build_20260918.py','distillation_core.py','training_safety.py'):
            z.write(ROOT/'tools'/name,'tools/'+name)
    (DEST/'apk_manifest.txt').write_text(badging,encoding='utf-8')
    (DEST/'apk_signature.txt').write_text(signature,encoding='utf-8')
    report=dict(apk=target.name,apk_bytes=target.stat().st_size,apk_sha256=sha(target),model=manifest,
        no_internet_permission=True,apk_signature_verified=True,zip_16kb_alignment_verified=True,
        desktop_originals_unchanged=True,training_performed=False,backup=str(backup),
        physical_android_phone_tested=False,source_zip_sha256=sha(DEST/'SpaceScan-Android-source.zip'))
    (DEST/'DELIVERY.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
