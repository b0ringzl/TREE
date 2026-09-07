"""Conservative structural QC after visual spot-check; preserves previous artifacts."""
import shutil
from collections import Counter
from pathlib import Path
from build_hk_lidar_pairs import read,write,TARGETS
ROOT=Path(__file__).resolve().parents[1]/'derived/hk_dualsource_three_species_20260907'

def main():
    manifest=ROOT/'manifest.json';records=read(manifest);backup=ROOT/'manifest.before_structure_audit.json'
    if not backup.exists():shutil.copy2(manifest,backup)
    flagged=[{'id':r['id'],'reason':r['reason']} for r in records if r['status']=='needs_geometry_refinement']
    for r in records:
        if r['status']!='geometry_candidate_pending_review':continue
        reason=None
        if r['id']=='jianshazui_pano_2__000420__cde7cffb18':reason='visual_spotcheck_regular_background_and_incomplete_palm'
        elif r['species']!=TARGETS[0] and r['quality']['trunk_bin_occupancy']<.95:reason='palm_vertical_coverage_below_95pct_bin_threshold'
        if reason:
            r.update(status='needs_geometry_refinement',training_eligible=False,reason=reason,agent_quality_review={'source':'assistant_structural_quality_audit','not_human_acceptance':True,'rule':'explicit mixed-surface example or palm stem-bin occupancy <0.95'})
            write(ROOT/'instances'/r['id']/'record.json',r);flagged.append({'id':r['id'],'reason':reason})
    write(manifest,records);summary=read(ROOT/'summary.json');valid=[r for r in records if r['status']=='geometry_candidate_pending_review']
    summary.update(status=dict(Counter(r['status'] for r in records)),candidate_species=dict(Counter(r['species'] for r in valid)),candidate_tree_groups=len({r['tree_group'] for r in valid}),quality_audit_flagged=len(flagged))
    write(ROOT/'summary.json',summary);write(ROOT/'structure_audit.json',{'flagged':flagged,'policy':'Not used to label species. Excluded from training/recommended tests; raw sources and derived point files retained.'})
    archive=ROOT/'experiment/before_structure_audit';archive.mkdir(exist_ok=True)
    for name in ('model.joblib','report.json','split.json','demo_samples.json','demo_behavior.json','demo_holdout_predictions.json'):
        src=ROOT/'experiment'/name
        if src.exists() and not (archive/name).exists():shutil.copy2(src,archive/name)
    print('flagged',len(flagged),'remaining',len(valid),'groups',summary['candidate_tree_groups'])

if __name__=='__main__':main()
