"""Per-source visual evidence sheets for audited A -> A+x cell experiments."""

import hashlib
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont, ImageOps


COLORS = {'ink': '#17243b', 'muted': '#64748b', 'line': '#cbd5e1',
          'panel': '#f8fafc', 'target': '#c2544d', 'kept': '#4f8550',
          'pending': '#94a3b8', 'pass': '#3f7f43', 'fail': '#b83f36'}
ROI = {'AU5': (0.12, 0.16, 0.88, 0.50), 'AU25': (0.18, 0.50, 0.82, 0.86)}


def _font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _image(path, size):
    with Image.open(path) as image:
        return ImageOps.pad(ImageOps.exif_transpose(image).convert('RGB'), size,
                            method=Image.Resampling.LANCZOS, color='#e5e7eb')


def _crop(path, target, size):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        box = ROI.get(target, (0.15, 0.25, 0.85, 0.75))
        crop = image.crop((int(box[0] * image.width), int(box[1] * image.height),
                           int(box[2] * image.width), int(box[3] * image.height)))
        return ImageOps.fit(crop, size, method=Image.Resampling.LANCZOS)


def _score(case, role, au):
    score = case.get(f'{role}_score') or {}
    if score.get('status') != 'ok':
        return None
    return score.get('intensities', {}).get(au)


def _badge(draw, x, y, number, state):
    color = COLORS['pass'] if state is True else COLORS['fail'] if state is False else COLORS['pending']
    draw.rectangle((x, y, x + 28, y + 25), fill=color)
    draw.text((x + 9, y + 2), str(number), font=_font(15, True), fill='white')


def _bar(draw, x, y, width, label, value, baseline, is_target):
    draw.text((x, y + 5), label, font=_font(14), fill=COLORS['ink'])
    bx = x + 55
    draw.rectangle((bx, y, bx + width, y + 27), outline=COLORS['line'], width=2, fill='white')
    if value is None:
        draw.text((bx + 7, y + 4), 'LF N/A', font=_font(13), fill=COLORS['muted'])
        return
    fill = int(width * max(0, min(5, value)) / 5)
    draw.rectangle((bx, y, bx + fill, y + 27), fill=COLORS['target'] if is_target else COLORS['kept'])
    if baseline is not None:
        marker = bx + int(width * max(0, min(5, baseline)) / 5)
        draw.line((marker, y - 2, marker, y + 29), fill='black', width=3)
    delta = '' if baseline is None else f'  Δ{value - baseline:+.2f}'
    draw.text((bx + width + 6, y + 4), f'{value:.2f}{delta}', font=_font(13, True), fill=COLORS['ink'])


def _case_level(case):
    target = case.get('target_au') or case.get('dataset_metadata', {}).get('target_au')
    return float(case.get('requested_aus', {}).get(target, 0.0))


def cell_sheet_key(cases):
    first = cases[0]
    source = first.get('input_source') or first['source']
    token = hashlib.sha256(f"{first.get('cell_id')}\0{source}".encode()).hexdigest()[:12]
    source_name = Path(source).stem.replace('_aligned', '')
    safe_cell = re.sub(r'[^A-Za-z0-9_.-]+', '_', first.get('cell_id') or 'cell')
    return f'{safe_cell}__{source_name}__{token}.png'


def render_cell_sheet(cases, output_path, stage='progress'):
    if not cases:
        raise ValueError('A cell sheet needs at least one case.')
    ordered = sorted(cases, key=_case_level)
    first = ordered[0]
    meta = first.get('dataset_metadata', {})
    target = first.get('target_au') or meta.get('target_au')
    A = first.get('cell_A') or meta.get('cell_A') or []
    cell_id = first.get('cell_id') or meta.get('cell_id') or 'cell'
    source_path = first.get('input_source') or first['source']
    source_id = Path(source_path).stem.replace('_aligned', '')
    columns = [('Original', source_path, None)]
    for case in ordered:
        level = _case_level(case)
        columns.append(('Control +0' if level == 0 else f'MagicFace {target} {level:+g}', case['result'], case))

    col_w, gap, margin = 275, 22, 45
    canvas_w = margin * 2 + len(columns) * col_w + (len(columns) - 1) * gap
    canvas_h = 990
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
    draw = ImageDraw.Draw(canvas)
    title = f'{source_id}  |  Anger A={"+".join(name[2:] for name in A)} → +{target[2:]}  |  MagicFace'
    draw.text((canvas_w // 2, 12), title, anchor='ma', font=_font(24, True), fill=COLORS['ink'])
    threshold = first.get('target_threshold_og') or meta.get('target_threshold_og')
    subtitle = (f'{cell_id} | stage={stage} | seed={first.get("seed")} | steps={first.get("inference_steps")} | '
                f'OG threshold {target}={threshold if threshold is not None else "N/A"} (reference only; not applied to LF)')
    draw.text((canvas_w // 2, 46), subtitle, anchor='ma', font=_font(14), fill=COLORS['muted'])

    baseline_case = next((case for case in ordered if _case_level(case) == 0), ordered[0])
    row_aus = [target] + [name for name in A if name != target]
    for index, (heading, path, case) in enumerate(columns):
        x = margin + index * (col_w + gap)
        draw.text((x + col_w // 2, 72), heading, anchor='ma', font=_font(16, True), fill=COLORS['ink'])
        canvas.paste(_image(path, (col_w, 250)), (x, 92))
        if case is None:
            draw.text((x, 355), f'label: Anger\nsource: {source_id}', font=_font(14), fill=COLORS['ink'])
        else:
            gates = case.get('official_gates', {})
            for gate in range(1, 5):
                _badge(draw, x + gate * 34 - 34, 352, gate, gates.get(f'gate{gate}'))
            gate_text = ('GATES: N/E' if not gates else
                         ' / '.join(f'{key}={value}' for key, value in gates.items()))
            draw.text((x + 148, 355), gate_text, font=_font(11, True), fill=COLORS['muted'])
        canvas.paste(_crop(path, target, (col_w, 130)), (x, 390))
        draw.rectangle((x, 390, x + col_w, 520), outline=COLORS['line'], width=2)
        if case is not None:
            baseline_scores = {au: _score(baseline_case, 'result', au) for au in row_aus}
            for row_index, au in enumerate(row_aus):
                _bar(draw, x, 555 + row_index * 50, 135, au,
                     _score(case, 'result', au), baseline_scores[au], au == target)
            level = _case_level(case)
            status = case.get('acceptance_status') or meta.get('acceptance_status') or 'not_evaluated'
            short_status = 'UNGATED' if str(status).startswith('not_evaluated') else str(status)
            draw.text((x, 730), f'request={level:+g} | {short_status}', font=_font(12, True),
                      fill=COLORS['fail'] if short_status == 'UNGATED' else COLORS['muted'])
        else:
            for row_index, au in enumerate(row_aus):
                _bar(draw, x, 555 + row_index * 50, 135, au,
                     _score(first, 'source', au), None, au == target)

    legend_y = 820
    draw.line((margin, legend_y, canvas_w - margin, legend_y), fill=COLORS['line'], width=2)
    draw.text((margin, legend_y + 14),
              'Bars: LibreFace intensity 0–5. Black marker: generated +0 baseline. '
              'Gray gate badges: OpenGraphAU / AUCANet / dlib not evaluated.',
              font=_font(14), fill=COLORS['ink'])
    draw.text((margin, legend_y + 44),
              f'Rarity evidence: n(A)={meta.get("n_A")}, n_exact(A+x)={meta.get("n_Ax_exact")}, '
              f'n_superset(A+x)={meta.get("n_Ax_superset")}, other emotions={meta.get("n_Ax_other_emotions")}, '
              f'lift={meta.get("x_lift")}, probe AUROC={meta.get("x_probe_auroc")}.',
              font=_font(14), fill=COLORS['ink'])
    draw.text((margin, legend_y + 74),
              'Selection prequalified by shared rules. Generated images are not T1/T2 until official gates (1)(2)(3″)(4) pass.',
              font=_font(14, True), fill=COLORS['fail'])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def render_cell_sheets(cases, output_dir, stage='progress'):
    groups = {}
    for case in cases:
        if not case.get('cell_id'):
            continue
        key = (case['cell_id'], case.get('input_source') or case['source'])
        groups.setdefault(key, []).append(case)
    files = []
    for group in groups.values():
        name = cell_sheet_key(group)
        render_cell_sheet(group, Path(output_dir) / name, stage=stage)
        files.append(name)
    return sorted(files)
