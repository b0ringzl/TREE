"""Auditable classification KD helpers. No production writes or implicit label edits."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF, InterpolationMode


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def class_groups(names):
    labels, groups = [], []
    for i, name in enumerate(names):
        label = 'Ficus microcarpa / Ficus benjamina (榕樹)' if name in ('Ficus microcarpa', 'Ficus benjamina') else name
        if label not in labels:
            labels.append(label)
            groups.append([])
        groups[labels.index(label)].append(i)
    return labels, groups


def collapse_logits(logits, groups):
    return torch.stack([torch.logsumexp(logits[..., ids], dim=-1) for ids in groups], dim=-1)


def canonical_image(path):
    # Identical spatial input for both teachers and student; no stochastic crop-cache mismatch.
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source).convert('RGB')
        im = TF.center_crop(TF.resize(im, 224, interpolation=InterpolationMode.BICUBIC), [224, 224])
        return TF.to_tensor(im)


def probabilities(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=-1, keepdims=True)


def confidence_weight(p):
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
    return (1 - entropy / np.log(p.shape[-1])).clamp(0, 1)


def kd_loss(student_logits, teacher_p, temperature=2.0, sample_weights=None):
    """KL(p_teacher || p_student); calibration already in p, KD T applied once."""
    if student_logits.shape != teacher_p.shape:
        raise ValueError('Class order/dimension mismatch')
    if not torch.isfinite(teacher_p).all() or (teacher_p < 0).any():
        raise ValueError('Invalid teacher distribution')
    if not torch.allclose(teacher_p.sum(-1), torch.ones_like(teacher_p[:, 0]), atol=1e-4):
        raise ValueError('Teacher distribution not normalized')
    target = F.softmax(teacher_p.clamp_min(1e-12).log() / temperature, dim=-1)
    loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=-1), target,
                    reduction='none').sum(-1) * temperature ** 2
    loss = loss * confidence_weight(teacher_p).detach()
    return loss.mean() if sample_weights is None else (loss * sample_weights).sum() / sample_weights.sum()


def fuse(p, method):
    p = np.asarray(p, dtype=np.float64)
    if method == 'mean':
        return p.mean(0)
    if method == 'logprob':
        return probabilities(np.log(np.maximum(p, 1e-12)).mean(0))
    if method == 'weighted':
        w = np.maximum(1e-8, 1 + (p * np.log(np.maximum(p, 1e-12))).sum(1) / np.log(p.shape[1]))
        return np.average(p, axis=0, weights=w)
    raise ValueError(method)


def metrics(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p)
    pred = p.argmax(1)
    c = p.shape[1]
    cm = np.zeros((c, c), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp, support = cm.diagonal(), cm.sum(1)
    denominator = cm.sum(0) + support
    f1 = np.divide(2 * tp, denominator, out=np.zeros(c), where=denominator > 0)
    recall = np.divide(tp, support, out=np.zeros(c), where=support > 0)
    present = support > 0
    top5 = np.argsort(-p, axis=1)[:, :min(5, c)]
    return {'n': len(y), 'classes_with_support': int(present.sum()),
            'top1': float((pred == y).mean()), 'macro_f1': float(f1[present].mean()),
            'balanced_accuracy': float(recall[present].mean()),
            'top5': float((top5 == y[:, None]).any(1).mean()),
            'nll': float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
            'mean_confidence': float(p.max(1).mean()), 'per_class_f1': f1.tolist(),
            'per_class_recall': recall.tolist(), 'support': support.tolist(), 'confusion_matrix': cm.tolist()}


def grouped_metrics(rows, p):
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r['group_id'], []).append(i)
    truths, predictions = [], []
    for indices in groups.values():
        labels = {rows[i]['target'] for i in indices}
        if len(labels) != 1:
            raise ValueError('Conflicting group labels')
        truths.append(labels.pop())
        predictions.append(p[indices].mean(0))
    return metrics(truths, predictions)


def bootstrap_gain(y, baseline_p, candidate_p, group_ids, seed=20260908, repeats=500):
    # Paired cluster bootstrap of accuracy difference, not image-IID resampling.
    indices = {}
    for i, group in enumerate(group_ids):
        indices.setdefault(group, []).append(i)
    clusters = list(indices.values())
    delta = (candidate_p.argmax(1) == y).astype(float) - (baseline_p.argmax(1) == y).astype(float)
    sums = np.array([delta[ids].sum() for ids in clusters])
    sizes = np.array([len(ids) for ids in clusters])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(clusters), len(clusters))
        values.append(sums[chosen].sum() / sizes[chosen].sum())
    return {'metric': 'image_accuracy_difference_cluster_bootstrap', 'point': float(delta.mean()),
            'ci95': np.quantile(values, [.025, .975]).tolist(), 'repeats': repeats,
            'group_unit': 'existing source groups; NOT verified VMMS trees'}
