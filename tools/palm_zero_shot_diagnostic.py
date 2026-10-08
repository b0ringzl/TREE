"""No training: compare current and expanded zero-shot candidate lists.

Pending web source labels are diagnostics only, never validation or training GT.
"""
import json
import sys
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tree_species_teacher_demo.engine import Engine, decode
from distillation_core import collapse_logits


def main():
    import open_clip
    data = ROOT / 'external_datasets/palm_confusion_20260908'
    output = data / 'zero_shot_diagnostic.json'
    if output.exists(): raise FileExistsError(output)
    engine = Engine()
    engine.load()
    rows = json.loads((data / 'curated_manifest.json').read_text(encoding='utf-8'))
    labels = engine.classes['labels'] + ['Wodyetia bifurcata']
    results = []
    with torch.inference_mode():
        text = open_clip.get_tokenizer('ViT-H-14')(['a photo of Wodyetia bifurcata']).cuda()
        extra = F.normalize(engine.model.encode_text(text, normalize=True).float(), dim=-1)
        for row in rows:
            pixels, _, _ = decode((data / row['file']).read_bytes())
            feature = engine.model.encode_image(((pixels[None].cuda() - engine.mean) / engine.std).to(engine.dtype), normalize=True).float()
            logits = collapse_logits(feature @ engine.bank.T * engine.model.logit_scale.exp().float(), engine.classes['groups'])
            added = feature @ extra.T * engine.model.logit_scale.exp().float()
            expanded = torch.cat([logits, added], dim=-1) / engine.choice['calibration_temperature']
            top = expanded.softmax(-1)[0].topk(3)
            results.append({'id': row['id'], 'unreviewed_source_label': row['source_species'],
                'old_top1': engine.classes['labels'][logits.argmax(-1).item()],
                'expanded_top3': [{'species': labels[i], 'score': p} for i, p in zip(top.indices.cpu().tolist(), top.values.cpu().tolist())]})
    report = {'training_performed': False, 'production_changed': False, 'labels': labels,
        'notice': 'Unreviewed source-label agreement only, NOT classification accuracy. No fitted head uses these samples.',
        'source_agreement': {s: {'n': sum(r['unreviewed_source_label'] == s for r in results),
            'old_matches': sum(r['unreviewed_source_label'] == s and r['old_top1'] == s for r in results),
            'expanded_matches': sum(r['unreviewed_source_label'] == s and r['expanded_top3'][0]['species'] == s for r in results)}
            for s in sorted({r['source_species'] for r in rows})}, 'rows': results}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['source_agreement']))


if __name__ == '__main__': main()
