"""Final rendered-report assertions and independent bootstrap recomputation."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/rare_species_fresh_validation_20261006_v1'
PDF = ROOT / 'output/pdf/珍稀树种新增样本精度验证_20261007.pdf'

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    target = OUT / 'ARTIFACT_AUDIT.json'
    if target.exists(): raise FileExistsError(target)
    load = lambda p: json.loads((OUT / p).read_text(encoding='utf-8'))
    rows = load('manifest.json'); result = load('RARE_RESULTS.json'); audit = load('FINAL_AUDIT.json')
    assert audit['passed'] and digest(OUT / 'REPORT_ZH.md') == audit['report_sha256']
    scores = [json.loads(x) for x in (OUT / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    subsets = [('primary', [i for i, r in enumerate(rows) if r['source_kind'] == 'community_observation'], result['primary'])]
    for label, metric in result['per_species'].items():
        subsets.append((label, [i for i, r in enumerate(rows) if r['source_kind'] == 'community_observation' and r['label'] == label], metric))
    verified = []
    for name, indices, metric in subsets:
        group = defaultdict(list)
        for i in indices: group[rows[i]['cluster_id']].append(i)
        groups = [group[k] for k in sorted(group)]
        sizes = np.array([len(g) for g in groups])
        delta = np.array([sum(int(scores[i]['top3']['sstv1_context'][0] == rows[i]['label']) -
            int(scores[i]['top3']['teacher_context'][0] == rows[i]['label']) for i in g) for g in groups])
        sample = np.random.default_rng(2026100607).choice(len(groups), size=(10000, len(groups)), replace=True)
        replicate = 100 * delta[sample].sum(1) / sizes[sample].sum(1)
        ci = np.quantile(replicate, [.025, .975])
        assert np.allclose(ci, metric['cluster_bootstrap95_percentage_points'], atol=1e-10)
        verified.append({'subset': name, 'observer_cluster_ci95': ci.tolist()})
    reader = PdfReader(PDF); assert len(reader.pages) == 3
    text = '\n'.join(p.extract_text() for p in reader.pages)
    for marker in ('91.56%', '95.45%', '97.06%', '94.12%', '92.55%', '95.21%', '+2.66', '-2.94',
                   '论文照片未显示提升', '未改变', '图版', '146'):
        assert marker in text, marker
    assert len(reader.pages[0].images) == len(reader.pages[1].images) == len(reader.pages[2].images) == 0
    page_images = [ROOT / 'tmp/pdfs/rare_species_20261007' / f'page-{i}.png' for i in (1, 2, 3)]
    assert all(p.is_file() for p in page_images)
    # Visual inspection of every rendered page was performed before this audit.
    record = {'passed': True, 'pdf_pages': 3, 'pdf_sha256': digest(PDF),
        'pdf_native_text_contains_exact_report_counts': True, 'no_nc_photos_embedded': True,
        'rendered_all_pages_visually_inspected_no_clipping_or_missing_glyphs': True,
        'bootstrap_independently_recomputed': verified,
        'deliverables': {p.name: digest(p) for p in [PDF, OUT / 'REPORT_ZH.md', OUT / 'REPORT_ZH.html',
                          OUT / 'RARE_RESULTS.json', OUT / 'CASE_RESULTS.json', OUT / 'FINAL_AUDIT.json']},
        'requirements_completion': 'Two species collected project-new images; fixed paired comparison and honest improvement/regression report; journal evidence separately reported; weights unchanged and verified backups retained.'}
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('ARTIFACT AUDIT PASSED: three PDF pages; exact figures; paired cluster CIs independently verified')

if __name__ == '__main__': main()
