"""New presentation order + visible-issue notes; never change pending labels/DB."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'external_datasets/conservation_sources_20260917_v1'
NOTES=ROOT/'outputs/conservation_agent_visual_notes_20260917.json'
TEXT={
 'visible_sign_conflicts_with_source_taxon_and_mixed_canopy':
 'The visible sign reads Aquilaria malaccensis, but the source record says Aquilaria sinensis. Check which tree the sign identifies; palm foliage also overlaps this view. / 可見樹牌寫著 Aquilaria malaccensis，但來源紀錄為 Aquilaria sinensis。請核對樹牌對應的目標；畫面亦混有棕櫚葉。',
 'bark_wound_only_without_diagnostic_foliage':
 'Bark and wound only; foliage and whole-tree context are absent. Use the source record if helpful; choose unsure when evidence is insufficient. / 僅見樹皮及傷口，缺少枝葉與全株脈絡。可對照來源紀錄；依據不足請選擇無法判斷。',
 'detached_ground_object_unverified_parent_tree':
 'A detached fruit-like object on the ground; its connection to the recorded tree is unverified. / 地面上的離枝果實狀物，其與來源樹木的關係尚未核實。',
 'detached_dry_fruit_without_parent_context':
 'Detached dry fruit without its parent branch or tree. If identifiable, treat it as a fruit detail, not a whole-tree view. / 離枝乾果，未見所屬枝條或樹木；若能辨認，僅作果實細節，不當作全株影像。',
 'dense_mixed_vegetation_target_ambiguity':
 'Dense overlapping vegetation; confirm which branch or plant is the target before accepting. / 多層植物互相重疊；接受前請先確認目標枝條或植株。',
 'underexposed_target_and_occluding_background':
 'The target leaves remain dark despite passing the coarse brightness check. Inspect detail; choose unsure if it is insufficient. / 雖通過粗略亮度檢查，目標葉片仍偏暗。請檢查細節，不足時選擇無法判斷。',
 'diagnostic_flower_detail_requires_actual_species_review':
 'Clear flower and leaf details may help verification; the source label alone is not an identification guarantee. / 清楚的花葉細節可供核驗，但來源標籤本身不保證鑑定正確。',
 'trunk_partly_obscured_by_decorations':
 'Ribbons and decorations obscure much of the trunk; check whether enough botanical detail remains. / 彩帶及裝飾物遮擋大量樹幹；請確認是否仍有足夠植物細節。',
 'very_small_plant_in_building_dominated_photo':
 'Most of the image shows the building; the plant is small. A target crop may be needed before use. / 畫面主要為建築，植株佔比很小；使用前可能需要裁出目標。',
 'multiple_plants_or_host_tree_in_negative_context':
 'Multiple plants or a possible host tree occur in the view; confirm the particular target rather than labelling the entire scene. / 畫面包含多株植物或可能的宿主樹；請確認具體目標，勿將整個場景視為同一樹種。',
 'soft_macro_detail_and_incomplete_leaf_context':
 'Soft-focus close-up with incomplete branch context; review whether the image itself supports a reliable decision. / 微距影像偏軟焦且枝條脈絡不完整；請判斷圖像本身是否足以支持可靠結論。',
 'detached_fruit_negative_not_whole_tree':
 'A single detached fruit in hand; if accepted, use as a fruit detail rather than whole-tree evidence. / 手持單個離枝果實；若接受，應作果實細節而非全株證據。'
}


def main():
    source=DATA/'review_manifest.json'; original=source.read_bytes()
    rows=json.loads(original); notes=json.loads(NOTES.read_text(encoding='utf8'))
    lookup={i:n for n in notes['agent_notes'] for i in n['ids']}
    ranks={'critical':0,'high':1,'normal':2}
    for index,row in enumerate(rows):
        row['original_review_index']=index
        row['original_source_priority']=row.get('review_priority')
        note=lookup.get(row['id'])
        row['review_priority']='critical' if note and note['priority']=='highest' else (note['priority'] if note else 'normal')
        row['review_note']=TEXT[note['reason_code']] if note else ''
        row['review_note_authority']='agent_visible_issue_only_not_species_label_or_human_decision'
        row['review_note_code']=note['reason_code'] if note else None
    rows.sort(key=lambda r:(ranks[r['review_priority']],r['source_species']!='Aquilaria sinensis',r['original_review_index']))
    output=DATA/'review_priority_manifest.json'
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert source.read_bytes()==original
    assert all(r['training_eligible'] is False and r['human_review_status']=='pending' and r['split']=='quarantine' for r in rows)
    assert len(rows)==108 and len({r['id'] for r in rows})==108
    assert rows[0]['id']=='conserve_inat_368821903'
    report={'rows':108,'notes':len(lookup),'first_id':rows[0]['id'],'original_manifest_unchanged':True,
            'human_decisions_written':False,'original_sha256':hashlib.sha256(original).hexdigest(),
            'priority_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    (DATA/'review_priority_VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report))


if __name__=='__main__':main()
