"""Apply audited rare-cell source allowlists before any image generation."""

import copy
import hashlib
import json
from pathlib import Path

from .au import AU_NAMES, edit_metadata


def source_id(path):
    stem = Path(path).stem
    return stem[:-len('_aligned')] if stem.endswith('_aligned') else stem


def _parse_A(value):
    parts = [part.strip().upper() for part in str(value).split('+') if part.strip()]
    result = [part if part.startswith('AU') else f'AU{part}' for part in parts]
    if not result or any(name not in AU_NAMES for name in result):
        raise ValueError(f'Invalid cell A configuration: {value!r}')
    return result


def apply_cell_selection(dataset, selection_path, requests):
    path = Path(selection_path).expanduser().resolve()
    document = json.loads(path.read_text(encoding='utf-8'))
    cells = document.get('cells')
    if not isinstance(cells, list) or not cells:
        raise ValueError('Cell selection must contain a non-empty cells list.')
    if 'anger' not in str(document.get('scope', '')).lower():
        raise ValueError('Current experiment is restricted to Anger cells.')
    baselines = [request for request in requests if edit_metadata(request)['edit_type'] == 'zero_baseline']
    if len(baselines) != 1:
        raise ValueError('Rare-cell experiment requires exactly one zero-edit baseline.')

    by_source = {}
    for item in dataset['items']:
        key = source_id(item['source'])
        if key in by_source:
            raise ValueError(f'Duplicate dataset source ID before cell selection: {key}')
        by_source[key] = item

    selected, missing = [], []
    seen_pairs = set()
    for cell in cells:
        cell_id = cell.get('cell_id')
        emotion = str(cell.get('emotion', '')).lower()
        target = str(cell.get('x', '')).upper()
        sources = cell.get('sources_aligned_S1')
        if not cell_id or emotion != 'anger' or target not in AU_NAMES or not isinstance(sources, list):
            raise ValueError(f'Invalid Anger cell entry: {cell_id!r}')
        if cell.get('status') != 'to_generate' or not all(
                cell.get(key) is True for key in ('R1_R7_pass', 'R8_codebook', 'R10_drawable')):
            raise ValueError(f'Cell is not approved for generation: {cell_id}')
        A = _parse_A(cell.get('A'))
        cell_requests = [copy.deepcopy(baselines[0])]
        cell_requests += [copy.deepcopy(request) for request in requests
                          if request.get(target, 0) != 0
                          and sum(value != 0 for value in request.values()) == 1]
        levels = sorted(request[target] for request in cell_requests if request.get(target, 0) != 0)
        if levels != [1, 2, 3, 4]:
            raise ValueError(f'{cell_id} needs {target} levels +1,+2,+3,+4; got {levels}')
        for source in sources:
            key = (cell_id, source)
            if key in seen_pairs:
                raise ValueError(f'Duplicate cell/source pair: {cell_id}/{source}')
            seen_pairs.add(key)
            base = by_source.get(source)
            if base is None:
                missing.append({'cell_id': cell_id, 'source_id': source})
                continue
            if base.get('fer', {}).get('source_true') != 'anger':
                raise ValueError(f'{source} is not labelled anger in the selected dataset.')
            item = copy.deepcopy(base)
            item['id'] = f'{cell_id}:{source}'
            item['generation_requests'] = cell_requests
            item['metadata'].update({
                'cell_id': cell_id, 'cell_A': A, 'target_au': target,
                'control_aus': cell.get('controls_y', []),
                'selection_status': 'prequalified_R1_R8_R10',
                'acceptance_status': 'not_evaluated_requires_gates_1_2_3_4',
                'n_A': cell.get('n_A'), 'n_Ax_exact': cell.get('n_Ax_exact'),
                'n_Ax_superset': cell.get('n_Ax_superset'),
                'n_Ax_other_emotions': cell.get('n_Ax_other_emotions'),
                'x_lift': cell.get('x_lift'), 'x_emfacs': cell.get('x_emfacs'),
                'x_probe_auroc': cell.get('x_probe_auroc'),
            })
            selected.append(item)
    if missing:
        preview = ', '.join(f"{row['cell_id']}/{row['source_id']}" for row in missing[:5])
        raise ValueError(f'{len(missing)} audited cell sources are missing from the dataset: {preview}')
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(dataset.get('metadata', {}))
    metadata['cell_selection'] = {
        'file': str(path), 'sha256': checksum, 'version': document.get('version'),
        'scope': document.get('scope'), 'cell_count': len(cells),
        'cell_source_pairs': len(selected),
        'unique_sources': len({source_id(item['source']) for item in selected}),
        'pre_selection_images': len(dataset['items']),
    }
    return {**dataset, 'items': selected, 'metadata': metadata,
            'kind': f"{dataset['kind']}_rare_cells"}
