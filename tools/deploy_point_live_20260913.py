"""Versioned pre-deployment snapshot and physical copy; never fits a model."""
import json, shutil, sys
from pathlib import Path
from training_safety import backup_before_training, sha256
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/point_live_deployment_20260913_v1'
MODEL=ROOT/'outputs/aquilaria_ptv2_fewshot_20260913_44056021/runs/final_3shot/model.joblib'
ENCODER=ROOT/'experiments/whu18_yolo11_ptv2_fusion_20260826/ptv2/run/best.pt'
if __name__=='__main__':
    files=[ROOT/p for p in ['spacescan_trees_demo_v1/server.py','spacescan_trees_demo_v1/point_assist.py','spacescan_trees_demo_v1/static/index.html','spacescan_trees_demo_v1/static/app.js','spacescan_trees_demo_v1/static/point-upload.js','spacescan_trees_demo_v1/static/point-evidence.js','tree_species_teacher_demo/engine.py']]
    backup=backup_before_training(RUN,files+[MODEL,ENCODER],{'purpose':'pre-deployment backup; no training','authorized_switch':'PTv2 seven-class upload inference','image_weights_changed':False})
    dest=ROOT/'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1';dest.mkdir(parents=True,exist_ok=False)
    shutil.copy2(MODEL,dest/'model.joblib')
    assert sha256(dest/'model.joblib')==sha256(MODEL)
    config=dict(version='ptv2-seven-20260913-v1',model=str((dest/'model.joblib').relative_to(ROOT)),model_sha256=sha256(MODEL),encoder=str(ENCODER.relative_to(ROOT)),encoder_sha256=sha256(ENCODER),backup=str(backup),source=str(MODEL),fusion_point_weight=.5)
    (dest/'deployment.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(config,ensure_ascii=False))
