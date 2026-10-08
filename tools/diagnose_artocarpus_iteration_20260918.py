"""Read-only train/development diagnosis; never opens test/regression assets."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/SSTv1_Artocarpus_User_20260918_v1'
OUT = ROOT / 'outputs/artocarpus_iteration_diagnostics_20260918_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def source(row):
    p = row['source_image'].replace('\\', '/')
    if '/inat_max150/' in p:
        return 'iNaturalist legacy catalog'
    if '/artocarpus_user_20260918_v1/' in p:
        return 'user supplied (source identity unresolved)'
    for token in ('herbarium', 'greening', 'IISc', 'conservation_browser'):
        if token.lower() in p.lower():
            return token
    return str(row.get('source') or row.get('source_name') or Path(p).parent.parent.name)


def record(row, before, after, labels):
    a, b = before.argsort()[::-1], after.argsort()[::-1]
    return dict(sample_id=row['sample_id'], source_image=row['source_image'],
                group_id=row['group_id'], source=source(row), truth=row['target'],
                baseline_top1=labels[int(a[0])], selected_top1=labels[int(b[0])],
                baseline_white=float(before[16]), selected_white=float(after[16]),
                white_rank=int(np.where(b == 16)[0][0] + 1),
                selected_top1_score=float(after[b[0]]),
                white_minus_top_other=float(after[16] - np.delete(after, 16).max()),
                selected_top3=[dict(species=labels[int(i)], score=float(after[i])) for i in b[:3]])


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    labels = read(RUN / 'labels.json')
    baseline = np.load(RUN / 'fit/baseline_predictions.npz', allow_pickle=False)
    selected = np.load(RUN / 'fit/selected_predictions.npz', allow_pickle=False)
    inspected, dev_summary = [], {}
    for phase in ('planning', 'newdev'):
        rows = read(RUN / f'{phase}_rows.json')
        pp = selected[phase]
        positives = [i for i, r in enumerate(rows) if r['target'] == 16]
        fps = [i for i, r in enumerate(rows) if r['target'] != 16 and pp[i].argmax() == 16]
        dev_summary[phase] = dict(positive_count=len(positives), correct=sum(int(pp[i].argmax() == 16) for i in positives),
                                  false_positives=len(fps))
        for i in positives + fps:
            rec = record(rows[i], baseline[phase][i], pp[i], labels)
            rec['phase'] = phase
            rec['case'] = 'TP' if rows[i]['target'] == 16 and pp[i].argmax() == 16 else ('FN' if rows[i]['target'] == 16 else 'FP')
            with Image.open(rows[i]['source_image']) as raw:
                im = ImageOps.exif_transpose(raw).convert('RGB')
                rec['image_width'], rec['image_height'] = im.size
                lum = np.asarray(im.convert('L'), dtype=float) / 255.
                rec['image_statistics'] = dict(mean_luminance=float(lum.mean()), fraction_dark_below_0_15=float((lum < .15).mean()),
                                                fraction_bright_above_0_95=float((lum > .95).mean()), luminance_std=float(lum.std()))
            inspected.append(rec)
    assert sum(x['truth'] == 16 for x in inspected) == 16
    assert sum(x['case'] == 'FP' for x in inspected) == 4
    assert len(inspected) == 20

    train = read(RUN / 'train_rows.json')
    ti = [i for i,r in enumerate(train) if r['target'] == 16]
    gr = Counter(train[i]['group_id'] for i in ti)
    source_stats = defaultdict(lambda: dict(images=0, selected_correct=0, groups=set(), competing_classes=Counter()))
    training_records = []
    for i in ti:
        rec = record(train[i], baseline['train'][i], selected['train'][i], labels)
        rec['class_internal_weight'] = 1 / (len(gr) * gr[train[i]['group_id']])
        training_records.append(rec)
        s = source_stats[rec['source']]
        s['images'] += 1
        s['selected_correct'] += int(selected['train'][i].argmax() == 16)
        s['groups'].add(rec['group_id'])
        if selected['train'][i].argmax() != 16:
            s['competing_classes'][rec['selected_top1']] += 1
    for s in source_stats.values():
        s['groups'] = len(s['groups'])
        s['competing_classes'] = dict(s['competing_classes'])
    train_summary = dict(images=len(ti), groups=len(gr), group_sizes=dict(sorted(gr.items())),
                         source_summary=dict(source_stats), baseline_correct=sum(int(baseline['train'][i].argmax() == 16) for i in ti),
                         selected_correct=sum(int(selected['train'][i].argmax() == 16) for i in ti),
                         limitation='Training replay; not accuracy validation; no training photos visually reviewed in this script.')
    (OUT / 'development_cases.json').write_text(json.dumps(inspected, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'training_white_metadata.json').write_text(json.dumps(dict(summary=train_summary, rows=training_records), ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'SUMMARY.json').write_text(json.dumps(dict(development=dev_summary, training=train_summary,
        scope='Only planning2640, newdev15, train4198 metadata/predictions. No test/regression files loaded. No label edits or training.'), ensure_ascii=False, indent=2), encoding='utf-8')

    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    for start in range(0, len(inspected), 6):
        sheet = Image.new('RGB', (1800, 1450), '#f3f5f1')
        draw = ImageDraw.Draw(sheet)
        for slot, rec in enumerate(inspected[start:start+6]):
            x, y = (slot % 3)*600, (slot // 3)*725
            with Image.open(rec['source_image']) as raw:
                im = ImageOps.exif_transpose(raw).convert('RGB')
                im.thumbnail((570, 510), Image.Resampling.LANCZOS)
            sheet.paste(im, (x+(600-im.width)//2, y+55+(510-im.height)//2))
            title=f"{start+slot+1:02d} {rec['phase']} {rec['case']}  {rec['image_width']}x{rec['image_height']}"
            draw.text((x+15,y+15),title,fill='#17372b',font=font)
            lines=[rec['sample_id'][:55], f"truth: {labels[rec['truth']]}",
                   f"top: {rec['selected_top1']} {rec['selected_top1_score']:.1%}",
                   f"white: {rec['baseline_white']:.1%} -> {rec['selected_white']:.1%} (rank {rec['white_rank']})",
                   f"white-minus-other: {rec['white_minus_top_other']:+.1%}"]
            for n,line in enumerate(lines): draw.text((x+15,y+585+n*24),line,fill='#172f28',font=small)
        sheet.save(OUT/f'development_contact_{start//6+1:02d}.jpg', quality=94)
    print(json.dumps(dict(development=dev_summary, training_sources=dict(source_stats), white_train_correct=train_summary['selected_correct']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
