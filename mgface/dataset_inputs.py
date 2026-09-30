"""Discover a dataset from one folder/manifest path, without loading models."""

import csv
import hashlib
import json
from pathlib import Path


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def item_key(item):
    return hashlib.sha256(item['id'].encode('utf-8')).hexdigest()[:16]


def _resolve(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Each dataset item needs a non-empty source path.')
    return str((base / Path(value).expanduser()).resolve())


def discover_dataset(path, exclude_dir=None):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f'Dataset does not exist: {path}')
    metadata = {}
    if path.is_dir():
        manifests = [path / name for name in ('dataset.json', 'dataset.csv') if (path / name).is_file()]
        if len(manifests) > 1:
            raise ValueError('Both dataset.json and dataset.csv exist. Pass the desired manifest path explicitly.')
        if manifests:
            return discover_dataset(manifests[0], exclude_dir=exclude_dir)
        image_root = path / 'images' if (path / 'images').is_dir() else path
        backgrounds = path / 'backgrounds'
        excluded = Path(exclude_dir).resolve() if exclude_dir else None
        files = sorted(p for p in image_root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                       and not p.stem.lower().endswith('_bg')
                       and not any(part.startswith('.') or part in ('backgrounds', 'prepared', 'generated', 'runs')
                                   for part in p.relative_to(image_root).parts[:-1])
                       and not (excluded and p.resolve().is_relative_to(excluded)))
        rows = []
        background_index = {}
        for source in files:
            relative = source.relative_to(image_root)
            candidates = []
            roots = [(source.parent, source.stem + '_bg')]
            if backgrounds.is_dir():
                roots += [(backgrounds / relative.parent, source.stem),
                          (backgrounds / relative.parent, source.stem + '_bg')]
            for parent, stem in roots:
                if parent.is_dir():
                    if parent not in background_index:
                        index = {}
                        for p in parent.iterdir():
                            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                                index.setdefault(p.stem, []).append(p.resolve())
                        background_index[parent] = index
                    candidates += background_index[parent].get(stem, [])
            candidates = list(dict.fromkeys(candidates))
            row = {'id': str(relative), 'source': str(source),
                   'background': str(candidates[0]) if len(candidates) == 1 else None}
            if len(candidates) > 1:
                row['discovery_error'] = 'Ambiguous background files; choose one explicitly in dataset.json.'
            rows.append(row)
        base = path
        kind = 'folder'
    elif path.suffix.lower() == '.json':
        document = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(document, dict) or not isinstance(document.get('items'), list):
            raise ValueError('Dataset JSON must contain an items list (not a report cases list).')
        rows, metadata, base, kind = document['items'], document.get('metadata', {}), path.parent, 'json'
    elif path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        base, kind = path.parent, 'csv'
    else:
        raise ValueError('Dataset path must be a folder, dataset.json, or dataset.csv.')
    if not rows:
        raise ValueError(f'No dataset images found: {path}')
    if not isinstance(metadata, dict):
        raise ValueError('Dataset metadata must be an object.')
    items, ids = [], set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'Dataset item {i + 1} must be an object.')
        source = _resolve(row.get('source'), base)
        identifier = row.get('id') or f'{i + 1:06d}:{row["source"]}'
        if not isinstance(identifier, str) or identifier in ids:
            raise ValueError(f'Dataset IDs must be unique strings: {identifier!r}')
        ids.add(identifier)
        if not isinstance(row.get('fer', {}), dict) or not isinstance(row.get('metadata', {}), dict):
            raise ValueError('Item fer and metadata fields must be objects.')
        fer = dict(row.get('fer', {}))
        for key in ('source_true', 'source_pred'):
            if row.get(key):
                fer[key] = row[key]
        if set(fer) - {'source_true', 'source_pred'}:
            raise ValueError('Input dataset FER annotations may contain only source_true/source_pred. '
                             'Result labels belong to generated images and must be added afterwards.')
        if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in fer.values()):
            raise ValueError('FER labels must be non-empty strings or null.')
        extra = {key: value for key, value in row.items() if key not in
                 ('id', 'source', 'background', 'fer', 'metadata', 'source_true', 'source_pred', 'discovery_error')}
        item = {'id': identifier, 'source': source,
                'background': _resolve(row['background'], base) if row.get('background') else None,
                'fer': fer, 'metadata': {**extra, **row.get('metadata', {})}}
        if row.get('discovery_error'):
            item['discovery_error'] = row['discovery_error']
        items.append(item)
    return {'path': str(path), 'kind': kind, 'metadata': metadata, 'items': items}


def inspect_item(item):
    """Check files before any expensive download; return whether preprocessing is needed."""
    from PIL import Image

    if item.get('discovery_error'):
        raise ValueError(item['discovery_error'])
    sizes = {}
    for key in ('source', 'background'):
        if item.get(key):
            with Image.open(item[key]) as image:
                image.load()
                sizes[key] = image.size
    if item.get('background'):
        if sizes['source'] != (512, 512) or sizes['background'] != (512, 512):
            raise ValueError('Prepared source and background must both be 512x512. '
                             'For raw images, omit background and use automatic preprocessing.')
        return False
    return True
