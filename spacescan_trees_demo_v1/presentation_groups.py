"""Coarse presentation groups over the unchanged 148-class image distribution.

These groups are not retrained species classifiers. Keep the raw scores and
labels available to the point-fusion path and exported evidence.
"""

import math

BAUHINIA_GROUP = 'Bauhinia group'
BAUHINIA_MEMBERS = (
    'Bauhinia purpurea',
    'Bauhinia variegata',
    'Bauhinia x blakeana',
)


def grouped_distribution(raw):
    if not isinstance(raw, dict) or len(raw) != 148 or BAUHINIA_GROUP in raw:
        raise ValueError('Expected the complete, ungrouped 148-class distribution')
    if not all(member in raw for member in BAUHINIA_MEMBERS):
        raise ValueError('Bauhinia source classes are missing')
    scores = {species: float(score) for species, score in raw.items()}
    if any(not math.isfinite(score) or score < 0 or score > 1 for score in scores.values()):
        raise ValueError('Invalid class probabilities')
    if abs(sum(scores.values()) - 1.0) > 0.005:
        raise ValueError('Incomplete or unnormalized class probabilities')
    grouped = {species: score for species, score in scores.items() if species not in BAUHINIA_MEMBERS}
    grouped[BAUHINIA_GROUP] = sum(scores[member] for member in BAUHINIA_MEMBERS)
    return grouped


def top_grouped(raw, count=3):
    grouped = grouped_distribution(raw)
    return sorted(grouped.items(), key=lambda row: (-row[1], row[0]))[:count]
