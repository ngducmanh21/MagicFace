"""Evidence tables and metrics. No classifier predictions are inferred here."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

from .au import AU_NAMES


FER_FIELDS = ('source_true', 'source_pred', 'result_true', 'result_pred')


def validate_fer_annotations(cases):
    """One image must have consistent annotations even if used in many edits."""
    annotations = {}
    for case in cases:
        fer = case.get('fer', {})
        if not isinstance(fer, dict) or set(fer) - set(FER_FIELDS):
            raise ValueError(f'fer must be a mapping with fields: {", ".join(FER_FIELDS)}')
        for field, value in fer.items():
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'fer.{field} must be a non-empty class name or null.')
            role, kind = field.split('_')
            key = (str(Path(case[role]).resolve()), kind)
            label = value.strip()
            if key in annotations and annotations[key] != label:
                raise ValueError(f'Conflicting FER {kind} labels for {case[role]}: '
                                 f'{annotations[key]!r} vs {label!r}')
            annotations[key] = label
    return annotations


def _average(values):
    return mean(values) if values else None


def _stats(values):
    return {'n': len(values), 'mean': _average(values),
            'std': stdev(values) if len(values) > 1 else None}


def classification_metrics(pairs, labels):
    """Rows=true, columns=predicted; zero-support recall is undefined (null)."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, prediction in pairs:
        matrix[index[truth]][index[prediction]] += 1
    per_class = []
    for i, label in enumerate(labels):
        tp, support = matrix[i][i], sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        # F1 = 2TP/(2TP+FP+FN); absent true classes are excluded from macro metrics.
        f1 = 2 * tp / (support + predicted) if support + predicted else None
        per_class.append({'class': label, 'support': support, 'predicted': predicted,
                          'precision': precision, 'recall': recall, 'f1': f1})
    supported = [row for row in per_class if row['support']]
    count = len(pairs)
    return {'n_evaluated': count, 'labels': labels, 'confusion_counts': matrix,
            'confusion_row_normalized': [[v / sum(row) if sum(row) else None for v in row] for row in matrix],
            'accuracy': sum(matrix[i][i] for i in range(len(labels))) / count if count else None,
            'macro_precision': _average([r['precision'] or 0.0 for r in supported]),
            'macro_recall': _average([r['recall'] for r in supported]),
            'macro_f1': _average([r['f1'] or 0.0 for r in supported]),
            'per_class': per_class}


def summarize_evidence(report):
    cases = report['cases']
    annotations = validate_fer_annotations(cases)
    labels = sorted(set(annotations.values()))
    fer = {}
    for role in ('source', 'result'):
        paths = list(dict.fromkeys(case[role] for case in cases))
        pairs = [(annotations[(path, 'true')], annotations[(path, 'pred')]) for path in paths
                 if (path, 'true') in annotations and (path, 'pred') in annotations]
        fer[role] = {**classification_metrics(pairs, labels), 'n_unique_images': len(paths),
                     'n_missing_labels': len(paths) - len(pairs),
                     'status': 'available' if pairs else 'unavailable',
                     'reason': None if pairs else 'No images have both true and predicted FER labels.'}
    au_summary, response = [], []
    for name in AU_NAMES:
        rows = [r for case in cases for r in case['au_rows']
                if r['au'] == name and r['measured_delta'] is not None]
        edited = [r for r in rows if r['requested_delta'] != 0]
        unchanged = [r for r in rows if r['requested_delta'] == 0]
        au_summary.append({'au': name, 'n_scored_pairs': len(rows), 'n_edited_pairs': len(edited),
                           'source_mean': _average([r['source_intensity'] for r in rows]),
                           'result_mean': _average([r['result_intensity'] for r in rows]),
                           'change_mean': _average([r['measured_delta'] for r in rows]),
                           'edited_mae': _average([r['absolute_error'] for r in edited if r['absolute_error'] is not None]),
                           'unchanged_drift': _average([abs(r['measured_delta']) for r in unchanged])})
        # Include explicitly requested zero controls; do not add implicit zeros to sweeps.
        controls = sorted({case['requested_aus'][name] for case in cases if name in case['requested_aus']})
        for control in controls:
            selected = [case for case in cases if case['requested_aus'].get(name) == control]
            measurements = [r['measured_delta'] for case in selected for r in case['au_rows']
                            if r['au'] == name and r['measured_delta'] is not None]
            stats = _stats(measurements)
            response.append({'au': name, 'requested_delta': control, 'n_total': len(selected),
                             'n_scored': stats['n'], 'change_mean': stats['mean'], 'change_std': stats['std'],
                             'expected_delta': control * report['au_delta_scale']
                             if report['au_delta_scale'] is not None else None})
    errors = Counter()
    seen = set()
    for case in cases:
        for role in ('source', 'result'):
            if case[role] in seen:
                continue
            seen.add(case[role])
            score = case[f'{role}_score']
            if score['status'] != 'ok':
                errors[score.get('error', 'AU scoring unavailable')] += 1
    return {'schema_version': 1, 'n_cases': len(cases), 'n_scored_pairs': report['scored_cases'],
            'n_unscored_pairs': len(cases) - report['scored_cases'],
            'au': au_summary, 'au_control_response': response, 'fer': fer,
            'scoring_failures': [{'reason': reason, 'n_unique_images': count} for reason, count in errors.items()],
            'definitions': {
                'au_unit': 'Paired edit case; a source repeated across edits contributes once per edit.',
                'au_control_response': 'Pooled by AU and requested value, including combination edits; descriptive, not an isolated causal effect.',
                'au_error': 'Computed only with explicit au_delta_scale. Missing scores excluded, never imputed as zero.',
                'fer_unit': 'Unique image path within source/result; identical image annotations are deduplicated.',
                'fer_ground_truth': 'Provided source_true/result_true annotations; requested target emotion is not ground truth.',
                'fer_macro': 'Mean over classes with true support > 0. Undefined precision contributes zero to macro precision.',
                'fer_comparison': 'Source/result evaluated separately; their supports may differ. No improvement claim is inferred.',
            }}


def _write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = {key: value for key, value in row.items() if key in fields}
            for key, value in clean.items():
                if isinstance(value, str) and value.startswith(('=', '+', '-', '@')):
                    clean[key] = "'" + value
            writer.writerow(clean)


def write_evidence(report, output_dir, formats=('png', 'svg', 'pdf')):
    """Export auditable tables plus figures; called only when evidence is enabled."""
    from .evidence_figures import render_evidence_figures

    output_dir = Path(output_dir)
    summary = summarize_evidence(report)
    tables = output_dir / 'tables'
    tables.mkdir(exist_ok=True)
    artifacts = []
    for name, rows in (('au_summary', summary['au']), ('au_control_response', summary['au_control_response'])):
        path = tables / f'{name}.csv'
        _write_csv(path, rows, list(rows[0]))
        artifacts.append(str(path.relative_to(output_dir)))
    for role, data in summary['fer'].items():
        if data['status'] != 'available':
            continue
        path = tables / f'fer_{role}_per_class.csv'
        _write_csv(path, data['per_class'], list(data['per_class'][0]))
        artifacts.append(str(path.relative_to(output_dir)))
        for key in ('confusion_counts', 'confusion_row_normalized'):
            path = tables / f'fer_{role}_{key}.csv'
            rows = [{'true_class': truth, 'predicted_class': prediction, 'value': data[key][i][j]}
                    for i, truth in enumerate(data['labels']) for j, prediction in enumerate(data['labels'])]
            _write_csv(path, rows, ['true_class', 'predicted_class', 'value'])
            artifacts.append(str(path.relative_to(output_dir)))
    figures = render_evidence_figures(report, summary, output_dir, formats)
    artifacts.extend(path for figure in figures for path in figure['files'])
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    config = {key: report[key] for key in ('schema_version', 'created_at', 'title', 'estimator',
              'estimator_version', 'estimator_device', 'au_delta_scale', 'metadata')}
    config['figure_formats'] = list(formats)
    (output_dir / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    artifacts.extend(('summary.json', 'run_config.json'))
    # Checksums cover precisely the exported evidence files, not an entire working directory.
    index = [{'path': path, 'sha256': hashlib.sha256((output_dir / path).read_bytes()).hexdigest()}
             for path in artifacts]
    (output_dir / 'artifact_index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    return {'summary': summary, 'figures': figures, 'artifacts': artifacts + ['artifact_index.json']}
