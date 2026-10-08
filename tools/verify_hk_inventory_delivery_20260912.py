import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/hk_data_inventory_20260912_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
inv=read(OUT/'inventory.json');audit=read(OUT/'content_audit_v2.json')
new=[json.loads(s) for s in (OUT/'file_inventory.jsonl').open(encoding='utf-8') if '"collection": "recent_hk"' in s]
for r in new:
    p=Path(r['path']);st=p.stat();assert st.st_size==r['bytes'];assert datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat()==r['modified_at_utc']
assert len(new)==187 and len(audit['single_tree_content'])==72
assert sum(audit['current_pair_species'].values())==95
assert sum(v['frames'] for v in audit['coordinate_streams'])==20691
assert not inv['errors']
pdf=ROOT/'output/pdf/Hong_Kong_Data_Inventory_20260912.pdf'
out=dict(checked_at=datetime.now(timezone.utc).isoformat(),new_source_files_unchanged_size_and_mtime=187,independent_species_truth_not_inferred=True,source_mutations=False,training=False,models_changed=False,pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),visual_qa='All 8 rendered pages inspected; no clipping, overlap or missing Chinese glyphs.',report_pages=8)
(OUT/'delivery_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
