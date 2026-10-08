"""Presentation-only names. Never modify classifier labels or scores."""
import json
from pathlib import Path

NAMES = json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))


def bilingual_names(species):
    row = NAMES[species]
    scientific = row['scientific_name']
    chinese = row['chinese_name']
    return {'scientific_name': scientific, 'chinese_name': chinese,
            'name': f'{scientific} / {chinese}'}
