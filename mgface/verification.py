"""Portable PNG/HTML/CSV verification reports, independent of torch/diffusers."""

import csv
from datetime import datetime, timezone
import html
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .au import AU_NAMES, edit_metadata, validate_request
from .evidence import validate_fer_annotations
from .cell_sheet import render_cell_sheets

CASE_METADATA_FIELDS = ('label', 'requested_aus', 'seed', 'inference_steps', 'fer',
                        'sample_id', 'dataset_id', 'input_source', 'dataset_metadata',
                        'background', 'generation_seconds', 'dataset_name', 'dataset_split',
                        'source_emotion', 'source_valence', 'source_arousal', 'edit_type',
                        'active_aus', 'edit_au', 'edit_level', 'cell_id', 'cell_A',
                        'target_au', 'control_aus', 'selection_status', 'acceptance_status')
CASE_METADATA_FIELDS += ('target_threshold_og',)


def add_report_arguments(parser):
    parser.add_argument('--au_backend', choices=('libreface', 'none'), default='libreface',
                        help='Use none for a visual-only report with explicitly unscored images.')
    parser.add_argument('--au_python', default=sys.executable,
                        help='Python executable of the separate LibreFace environment.')
    parser.add_argument('--au_device', default='cpu')
    parser.add_argument('--au_weights_dir', default='weights_libreface')
    parser.add_argument('--au_delta_scale', type=float, default=None,
                        help='Explicit conversion: expected measured delta = requested delta * scale. '
                             'Omit if the model and estimator scales have not been calibrated.')
    parser.add_argument('--report_title', default='MagicFace | AU verification')
    parser.add_argument('--evidence', action='store_true',
                        help='Export AU figures, summary tables, config and optional FER evaluation.')
    parser.add_argument('--figure_formats', nargs='+', choices=('png', 'svg', 'pdf'),
                        default=['png', 'svg', 'pdf'], help='Evidence figure formats; HTML always includes PNG previews.')


def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError('At least one verification case is required.')
    validated = []
    for case in cases:
        item = dict(case)
        item['requested_aus'] = validate_request(case['requested_aus'])
        for key, value in edit_metadata(item['requested_aus']).items():
            item.setdefault(key, value)
        for key in ('source', 'result'):
            path = Path(case[key]).resolve()
            with Image.open(path) as image:
                image.verify()
            item[key] = str(path)
        item['label'] = str(case.get('label', f'Edit {len(validated) + 1}'))
        validated.append(item)
    validate_fer_annotations(validated)
    return validated


def validate_scale(scale):
    if scale is not None and (not math.isfinite(scale) or scale == 0):
        raise ValueError('au_delta_scale must be a finite, non-zero number.')


def score_with_libreface(paths, python=sys.executable, device='cpu', weights_dir='weights_libreface'):
    paths = list(dict.fromkeys(str(Path(path).resolve()) for path in paths))
    with tempfile.TemporaryDirectory(prefix='magicface-score-') as temp:
        request, response = Path(temp) / 'request.json', Path(temp) / 'response.json'
        request.write_text(json.dumps(paths))
        command = [str(python), str(Path(__file__).with_name('au_worker.py')),
                   '--request', str(request), '--response', str(response),
                   '--device', device, '--weights_dir', str(Path(weights_dir).resolve())]
        try:
            subprocess.run(command, check=True, timeout=1800)
            return json.loads(response.read_text())
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            return {'estimator': 'LibreFace', 'version': None, 'device': device,
                    'images': {path: {'status': 'error', 'error': str(exc)} for path in paths}}


def _score(bundle, path):
    item = bundle.get('images', {}).get(path, {'status': 'skipped', 'error': 'AU scoring not run.'})
    if item.get('status') == 'ok':
        try:
            values = {name: float(item['intensities'][name]) for name in AU_NAMES}
            if any(not math.isfinite(v) or not 0 <= v <= 5 for v in values.values()):
                raise ValueError('Expected finite intensities on the 0..5 scale.')
            return {'status': 'ok', 'intensities': values}
        except (KeyError, TypeError, ValueError) as exc:
            return {'status': 'error', 'error': f'Invalid AU scores: {exc}'}
    return {'status': item.get('status', 'error'), 'error': item.get('error', 'No AU scores.')}


def _evaluate(case, source, result, scale):
    measured = source['status'] == result['status'] == 'ok'
    rows = []
    for name in AU_NAMES:
        requested = case['requested_aus'].get(name, 0.0)
        before = source.get('intensities', {}).get(name)
        after = result.get('intensities', {}).get(name)
        delta = after - before if measured else None
        expected = requested * scale if scale is not None else None
        target = before + expected if before is not None and expected is not None else None
        rows.append({'au': name, 'requested_delta': requested, 'source_intensity': before,
                     'result_intensity': after, 'measured_delta': delta, 'expected_delta': expected,
                     'absolute_error': abs(delta - expected) if measured and expected is not None else None,
                     'target_out_of_range': target is not None and not 0 <= target <= 5})
    edited = [row for row in rows if row['requested_delta'] != 0]
    unchanged = [row for row in rows if row['requested_delta'] == 0]
    return {'au_rows': rows, 'status': 'scored' if measured else 'unscored',
            'edited_au_mae': (sum(r['absolute_error'] for r in edited) / len(edited)
                              if measured and scale is not None and edited else None),
            'unchanged_au_drift': (sum(abs(r['measured_delta']) for r in unchanged) / len(unchanged)
                                   if measured and unchanged else None)}


def _number(value, signed=False):
    return 'N/A' if value is None else format(value, '+.2f' if signed else '.2f')


def _font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def _thumbnail(path, size):
    with Image.open(path) as image:
        return ImageOps.pad(ImageOps.exif_transpose(image).convert('RGB'), size,
                            method=Image.Resampling.LANCZOS, color='#e9edf3')


def _draw_sheets(report, output_dir):
    files = []
    for start in range(0, len(report['cases']), 4):
        cases = report['cases'][start:start + 4]
        canvas = Image.new('RGB', (1440, 160 + 390 * len(cases)), '#edf1f7')
        draw = ImageDraw.Draw(canvas)
        draw.text((28, 20), report['title'][:85], font=_font(26), fill='#17243b')
        draw.text((28, 58), 'Intensity: 0-5 | Measured change = result - source | N/A = not scored',
                  font=_font(16), fill='#43516a')
        mapping = ('Error: N/A (request-to-intensity scale not configured)' if report['au_delta_scale'] is None
                   else f"Expected change = requested change x {report['au_delta_scale']:g}; targets are not clipped")
        draw.text((28, 83), mapping, font=_font(16), fill='#43516a')
        for index, case in enumerate(cases):
            y = 120 + index * 390
            draw.rounded_rectangle((20, y, 1420, y + 374), radius=14, fill='white')
            draw.text((36, y + 12), f"{start + index + 1:02d}  {case['label'][:68]}",
                      font=_font(19), fill='#17243b')
            draw.text((36, y + 44), 'SOURCE', font=_font(13), fill='#64748b')
            draw.text((324, y + 44), 'RESULT', font=_font(13), fill='#64748b')
            canvas.paste(_thumbnail(output_dir / case['source_asset'], (272, 272)), (36, y + 66))
            canvas.paste(_thumbnail(output_dir / case['result_asset'], (272, 272)), (324, y + 66))
            xs = (628, 701, 818, 931, 1045, 1160, 1287)
            for x, label in zip(xs, ('AU', 'Request', 'Source', 'Result', 'Change', 'Expected', '|Error|')):
                draw.text((x, y + 45), label, font=_font(14), fill='#64748b')
            for i, row in enumerate(case['au_rows']):
                ry = y + 69 + i * 22
                if row['requested_delta'] != 0:
                    draw.rectangle((618, ry - 1, 1398, ry + 21), fill='#e7f6f2')
                values = (row['au'], _number(row['requested_delta'], True),
                          _number(row['source_intensity']), _number(row['result_intensity']),
                          _number(row['measured_delta'], True), _number(row['expected_delta'], True),
                          _number(row['absolute_error']) + ('*' if row['target_out_of_range'] else ''))
                for x, value in zip(xs, values):
                    draw.text((x, ry), value, font=_font(15), fill='#17243b')
            footer = (f"{case['status'].upper()} | seed: {case.get('seed', 'N/A')} | "
                      f"edited AU MAE: {_number(case['edited_au_mae'])} | "
                      f"unchanged AU drift: {_number(case['unchanged_au_drift'])}")
            draw.text((36, y + 346), footer, font=_font(14), fill='#43516a')
        draw.text((28, canvas.height - 26),
                  '* Expected absolute intensity outside 0-5. Full details and scoring errors: report.html',
                  font=_font(13), fill='#43516a')
        name = f'grid_{len(files) + 1:03d}.png'
        canvas.save(output_dir / name)
        files.append(name)
    return files


def _write_html(report, output_dir):
    esc = lambda value: html.escape(str(value), quote=True)
    cards = []
    for index, case in enumerate(report['cases']):
        rows = []
        for row in case['au_rows']:
            values = (row['au'], _number(row['requested_delta'], True), _number(row['source_intensity']),
                      _number(row['result_intensity']), _number(row['measured_delta'], True),
                      _number(row['expected_delta'], True), _number(row['absolute_error']))
            warning = ' *' if row['target_out_of_range'] else ''
            rows.append(f'<tr class="{"edited" if row["requested_delta"] else ""}">' +
                        ''.join(f'<td>{esc(v)}</td>' for v in values[:-1]) +
                        f'<td>{esc(values[-1] + warning)}</td></tr>')
        errors = ' '.join(f'{label}: {score["error"]}' for label, score in
                          (('Source', case['source_score']), ('Result', case['result_score'])) if 'error' in score)
        cards.append(f'''<article data-status="{case['status']}">
<header><h2>{index + 1:02d} &nbsp; {esc(case['label'])}</h2><span class="badge">{case['status']}</span></header>
<p class="muted">Seed {esc(case.get('seed', 'N/A'))} · Steps {esc(case.get('inference_steps', 'N/A'))}</p>
<div class="comparison"><figure><a href="{case['source_asset']}"><img src="{case['source_asset']}" alt="Source image" loading="lazy"></a><figcaption>Source</figcaption></figure>
<figure><a href="{case['result_asset']}"><img src="{case['result_asset']}" alt="Edited result" loading="lazy"></a><figcaption>Result</figcaption></figure>
<div class="scores"><table><thead><tr><th>AU</th><th>Request</th><th>Source</th><th>Result</th><th>Change</th><th>Expected</th><th>|Error|</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>
<p>Edited AU MAE <b>{_number(case['edited_au_mae'])}</b> &nbsp;·&nbsp; Unchanged AU drift <b>{_number(case['unchanged_au_drift'])}</b></p>
{f'<p class="error">{esc(errors)}</p>' if errors else ''}</article>''')
    scale = report['au_delta_scale']
    calibration = ('Request values are model controls. Their conversion to measured intensity is not configured; expected change and error are N/A.'
                   if scale is None else f'Expected change = request × {scale:g}. Error is absolute measured-minus-expected change. Targets are not clipped.')
    links = ' '.join(f'<a href="{name}">Grid {i + 1}</a>' for i, name in enumerate(report['grids']))
    dataset_html = ''
    dataset_counts = report['metadata'].get('dataset_counts')
    if isinstance(dataset_counts, dict):
        dataset_html = (f'<p><b>Dataset coverage:</b> {esc(dataset_counts.get("selected_images", "N/A"))} selected inputs; '
                        f'{esc(dataset_counts.get("images_with_output", "N/A"))} images with output; '
                        f'{esc(dataset_counts.get("generated_edits", "N/A"))} generated edits; '
                        f'{esc(dataset_counts.get("failed_events", "N/A"))} input/preprocessing/generation failure events.</p>')
    for name in ('dataset_summary.json', 'dataset_inputs.json', 'cell_selection.json', 'scope_audit.json',
                 'samples.csv', 'failures.csv', 'excluded_annotations.csv'):
        if (output_dir / name).is_file():
            links += f' <a href="{name}">{name}</a>'
    evidence_html = ''
    if 'evidence' in report:
        evidence = report['evidence']
        evidence_html = '<section class="evidence"><h2>Experiment evidence</h2><p>'
        evidence_html += ('AU plots describe the supplied edits. Repeated sources count per edit in AU statistics. '
                          'FER uses unique image paths and supplied ground-truth labels; source and result supports may differ.</p>')
        for role, metrics in evidence['summary']['fer'].items():
            if metrics['status'] == 'available':
                evidence_html += (f'<p><b>FER {role}</b>: n={metrics["n_evaluated"]}; '
                                  f'missing labels={metrics["n_missing_labels"]}; '
                                  f'accuracy={_number(metrics["accuracy"])}; '
                                  f'macro precision={_number(metrics["macro_precision"])}; '
                                  f'macro recall={_number(metrics["macro_recall"])}; '
                                  f'macro F1={_number(metrics["macro_f1"])}</p>')
            else:
                evidence_html += f'<p><b>FER {role}: unavailable.</b> {esc(metrics["reason"])}</p>'
        evidence_html += '<details><summary>Download tables, configuration and checksums</summary><nav>'
        evidence_html += ''.join(f'<a href="{esc(path)}">{esc(Path(path).name)}</a>'
                                 for path in evidence['artifacts'] if not path.startswith('figures/'))
        evidence_html += '</nav></details><div class="evidence-figures">'
        for figure in evidence['figures']:
            evidence_html += (f'<figure><figcaption>{esc(figure["title"])}</figcaption>'
                              f'<a href="{figure["preview"]}"><img src="{figure["preview"]}" '
                              f'alt="{esc(figure["title"])}" loading="lazy"></a><nav>')
            evidence_html += ''.join(f'<a href="{path}">{Path(path).suffix[1:].upper()}</a>' for path in figure['files'])
            evidence_html += '</nav></figure>'
        evidence_html += '</div></section>'
    cell_html = ''
    if report.get('cell_grids'):
        cell_html = ('<section class="evidence"><h2>Per-source rare-cell sheets</h2>'
                     '<p>Original, generated +0 control and doses +1…+4. Bars are LibreFace diagnostics; '
                     'official OpenGraphAU/AUCANet/dlib gates remain explicitly unevaluated.</p>'
                     '<div class="cell-sheets">')
        for path in report['cell_grids']:
            cell_html += (f'<figure><a href="{esc(path)}"><img src="{esc(path)}" '
                          f'alt="Rare cell evidence sheet" loading="lazy"></a>'
                          f'<figcaption>{esc(Path(path).stem)}</figcaption></figure>')
        cell_html += '</div></section>'
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AU verification</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf1f7;color:#17243b;font:15px system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:32px}h1{font-size:32px;margin:8px 0}h2{font-size:19px;margin:0}
p{line-height:1.6}.muted,figcaption{color:#59677d}nav{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}a{color:#076954}
article{background:white;border:1px solid #dce3ed;border-radius:16px;padding:22px;margin:20px 0;break-inside:avoid}
header{display:flex;justify-content:space-between;align-items:center;gap:12px}.badge{background:#e7f6f2;padding:5px 12px;border-radius:20px}
.comparison{display:grid;grid-template-columns:minmax(180px,1fr) minmax(180px,1fr) minmax(520px,2fr);gap:18px}
figure{margin:0}img{width:100%;aspect-ratio:1;object-fit:contain;background:#edf1f7;border-radius:8px}figcaption{margin-top:8px}
.scores{overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}td,th{padding:7px;text-align:right;border-bottom:1px solid #e9edf3}td:first-child,th:first-child{text-align:left}.edited{background:#e7f6f2}
.error{color:#a33220;overflow-wrap:anywhere}input[type=search]{padding:10px;border:1px solid #cbd5e1;border-radius:8px;width:300px;max-width:100%}label{margin-left:16px}details{margin:16px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.evidence{background:white;border-radius:16px;padding:22px;margin-bottom:24px}.evidence-figures{display:grid;grid-template-columns:1fr 1fr;gap:24px}.evidence-figures img{aspect-ratio:auto;border:1px solid #dce3ed}.evidence-figures figcaption{font-weight:600;margin:14px 0}.evidence-figures nav{margin:6px 0}
.cell-sheets{display:grid;grid-template-columns:1fr;gap:24px}.cell-sheets img{aspect-ratio:auto;border:1px solid #dce3ed}.cell-sheets figcaption{overflow-wrap:anywhere}
@media(max-width:1000px){.comparison{grid-template-columns:1fr 1fr}.scores{grid-column:1/-1}main{padding:16px}}
@media(max-width:700px){.evidence-figures{grid-template-columns:1fr}}
@media print{nav,.filters{display:none}body{background:white}main{padding:0}article{border-radius:0}}
</style><main>'''
    document += f'''<p class="muted">MAGICFACE / RESULT REVIEW</p><h1>{esc(report['title'])}</h1>
<p>{len(report['cases'])} results · {report['scored_cases']} scored · {len(report['cases']) - report['scored_cases']} unscored<br>
Estimator: {esc(report['estimator'])} {esc(report.get('estimator_version') or '')} · Intensity scale: 0–5</p>
{dataset_html}
<p>Change = result intensity − source intensity. Highlighted rows are requested non-zero edits.<br>{esc(calibration)}<br>
Unchanged AU drift is the mean absolute change of AUs requested at zero. * Expected absolute intensity outside 0–5. AU estimates are not ground truth.</p>
<nav><a href="scores.csv">Download CSV</a><a href="results.json">Results JSON</a><a href="manifest.json">Manifest</a>{links}</nav>
{cell_html}{evidence_html}<h2>Individual results</h2>
<div class="filters"><input type="search" id="search" placeholder="Filter by label or AU" aria-label="Filter results"><label><input type="checkbox" id="unscored">Unscored only</label></div>
{''.join(cards)}<details><summary>Run metadata</summary><pre>{esc(json.dumps(report['metadata'], indent=2, ensure_ascii=False))}</pre></details></main>
<script>function filterCards(){{const q=document.getElementById('search').value.toLowerCase();const missing=document.getElementById('unscored').checked;document.querySelectorAll('article').forEach(c=>{{c.hidden=!c.textContent.toLowerCase().includes(q)||(missing&&c.dataset.status!=='unscored')}})}}document.getElementById('search').addEventListener('input',filterCards);document.getElementById('unscored').addEventListener('change',filterCards);</script></html>'''
    (output_dir / 'report.html').write_text(document, encoding='utf-8')


def write_verification_report(cases, output_dir, scores=None, au_delta_scale=None,
                              title='MagicFace | AU verification', metadata=None,
                              evidence=False, figure_formats=('png', 'svg', 'pdf')):
    """Reusable after inference or a validation step. Missing scores stay missing.

    ``scores`` uses the score_with_libreface return format. All 12 intensities
    are required for each successful image. Paths in scores are absolute.
    """
    validate_scale(au_delta_scale)
    if evidence:
        if not figure_formats or set(figure_formats) - {'png', 'svg', 'pdf'}:
            raise ValueError('figure_formats must contain only png, svg, pdf.')
        # Fail before scoring/export if the optional plotting dependency is missing.
        import matplotlib  # noqa: F401
    cases = validate_cases(cases)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    scores = scores or {'estimator': 'none', 'images': {}}
    report = {'schema_version': 1, 'title': title, 'created_at': datetime.now(timezone.utc).isoformat(),
              'estimator': scores.get('estimator', 'unknown'), 'estimator_version': scores.get('version'),
              'estimator_device': scores.get('device'), 'au_delta_scale': au_delta_scale,
              'intensity_range': [0, 5], 'metadata': metadata or {}, 'cases': []}
    assets = {}
    for case in cases:
        for key in ('source', 'result'):
            path = case[key]
            if path not in assets:
                # Keep distinct input paths distinct even when their pixels match:
                # FER sample counts must remain stable after moving/replotting a run.
                digest = hashlib.sha256(path.encode('utf-8') + b'\0' + Path(path).read_bytes()).hexdigest()
                asset = f'images/{digest}.png'
                with Image.open(path) as image:
                    ImageOps.exif_transpose(image).convert('RGB').save(output_dir / asset)
                assets[path] = asset
            case[f'{key}_asset'] = assets[path]
        source, result = _score(scores, case['source']), _score(scores, case['result'])
        report['cases'].append({**case, 'source_score': source, 'result_score': result,
                                **_evaluate(case, source, result, au_delta_scale)})
    report['scored_cases'] = sum(c['status'] == 'scored' for c in report['cases'])
    report['grids'] = _draw_sheets(report, output_dir)
    report['cell_grids'] = [f'cell_grids/{name}' for name in render_cell_sheets(
        report['cases'], output_dir / 'cell_grids',
        stage='libreface_scored' if report['scored_cases'] else 'unscored')]
    if evidence:
        from .evidence import write_evidence
        report['evidence'] = write_evidence(report, output_dir, formats=figure_formats)
    portable_cases = [{**{key: case[key] for key in CASE_METADATA_FIELDS if key in case},
                       'source': case['source_asset'], 'result': case['result_asset']} for case in report['cases']]
    (output_dir / 'manifest.json').write_text(json.dumps({'cases': portable_cases, 'metadata': report['metadata']},
                                                       indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    fields = ['case', 'sample_id', 'dataset_id', 'dataset_name', 'dataset_split', 'source_emotion',
              'source_valence', 'source_arousal', 'label', 'source', 'result', 'seed', 'inference_steps',
              'generation_seconds', 'cell_id', 'cell_A', 'target_au', 'control_aus',
              'selection_status', 'acceptance_status', 'edit_type', 'edit_au', 'edit_level', 'status', 'au',
              'requested_delta', 'source_intensity', 'result_intensity', 'measured_delta',
              'expected_delta', 'absolute_error', 'target_out_of_range', 'edited_au_mae', 'unchanged_au_drift']
    with (output_dir / 'scores.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, case in enumerate(report['cases']):
            for row in case['au_rows']:
                entry = {key: case.get(key) for key in fields if key in case}
                entry.update(row, case=i + 1)
                # Prevent spreadsheet programs interpreting labels or paths as formulas.
                for key in ('sample_id', 'dataset_id', 'dataset_name', 'dataset_split', 'source_emotion', 'label', 'source', 'result'):
                    if isinstance(entry.get(key), str) and entry[key].startswith(('=', '+', '-', '@')):
                        entry[key] = "'" + entry[key]
                writer.writerow(entry)
    _write_html(report, output_dir)
    return report


def report_from_args(cases, output_dir, args, metadata=None):
    validate_scale(args.au_delta_scale)
    cases = validate_cases(cases)
    scores = None
    if args.au_backend == 'libreface':
        paths = [case[key] for case in cases for key in ('source', 'result')]
        scores = score_with_libreface(paths, python=args.au_python, device=args.au_device,
                                     weights_dir=args.au_weights_dir)
    report = write_verification_report(cases, output_dir, scores=scores, au_delta_scale=args.au_delta_scale,
                                      title=args.report_title, metadata=metadata, evidence=args.evidence,
                                      figure_formats=args.figure_formats)
    print(f"Report: {Path(output_dir).resolve() / 'report.html'}")
    print(f"AU scores: {report['scored_cases']}/{len(cases)} cases scored.")
    if report['scored_cases'] != len(cases):
        print('Some AU scores are unavailable. See scoring errors in report.html/results.json.')
    return report
