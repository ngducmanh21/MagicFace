"""Readers for native RAF-DB Basic and AffectNet annotation formats.

Numeric label mappings are format-specific. Custom/reindexed releases should
use an explicit generic manifest rather than guessing a mapping from folders.
"""

import csv
import math
from pathlib import Path


RAF_LABELS = {1: 'surprise', 2: 'fear', 3: 'disgust', 4: 'happy',
              5: 'sad', 6: 'anger', 7: 'neutral'}
AFFECTNET_LABELS = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise',
                    4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt'}
RAF_FILES = ('list_patition_label.txt', 'list_partition_label.txt')
SPLITS = {'train': 'train', 'training': 'train', 'train_set': 'train', 'training_set': 'train',
          'val': 'val', 'valid': 'val', 'validation': 'val', 'val_set': 'val', 'validation_set': 'val',
          'test': 'test', 'testing': 'test', 'test_set': 'test'}
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def canonical_split(value):
    return SPLITS.get(str(value).strip().lower(), str(value).strip().lower())


def _integer(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not an expression ID.')
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f'Expected integer expression ID, got {value!r}')
    return int(number)


def _dimension(value):
    """AffectNet's missing/sentinel/out-of-range VA values are not measurements."""
    try:
        number = float(value)
        return number if math.isfinite(number) and -1 <= number <= 1 else None
    except (ValueError, TypeError):
        return None


def _csv_is_affectnet(path):
    try:
        with path.open(newline='', encoding='utf-8-sig') as stream:
            fields = next(csv.reader(stream), [])
        return {'subDirectory_filePath', 'expression'}.issubset(fields)
    except (OSError, UnicodeError, csv.Error):
        return False


def _raf_annotations(root):
    return [base / 'EmoLabel' / name for base in (root, root / 'basic') for name in RAF_FILES
            if (base / 'EmoLabel' / name).is_file()]


def _affect_csvs(root):
    return [base / folder / name for base in (root, root / 'Manually_Annotated')
            for folder in ('', 'file_lists', 'Manually_Annotated_file_lists')
            for name in ('training.csv', 'validation.csv') if (base / folder / name).is_file()
            and _csv_is_affectnet(base / folder / name)]


def _npy_roots(root):
    candidates = [root] + [root / name for name in ('train_set', 'val_set', 'validation_set', 'test_set')]
    return [p for p in candidates if (p / 'annotations').is_dir()
            and next((p / 'annotations').glob('*_exp.npy'), None) is not None]


def _class_folder_roots(root):
    candidates = [root] + [root / name for name in ('train', 'val', 'test', 'training', 'validation')]
    return [p for p in candidates if p.is_dir() and canonical_split(p.name) in ('train', 'val', 'test')
            and any(child.is_dir() and child.name.isdecimal() for child in p.iterdir())]


def detect_dataset_type(path):
    if path.is_file():
        if path.name in RAF_FILES:
            return 'rafdb'
        if path.suffix.lower() == '.csv' and _csv_is_affectnet(path):
            return 'affectnet'
        return 'generic'
    # User-written manifests take precedence over automatic native detection.
    if any((path / name).is_file() for name in ('dataset.json', 'dataset.csv')):
        return 'generic'
    raf = bool(_raf_annotations(path))
    affect = bool(_affect_csvs(path) or _npy_roots(path))
    if raf and affect:
        raise ValueError('Both RAF-DB and AffectNet annotations found. Pass a specific dataset root.')
    return 'rafdb' if raf else 'affectnet' if affect else 'generic'


def _item(name, source, label, split, label_id, annotation_file, **extra):
    return {'id': f'{name}:{split}:{extra.pop("sample_key")}', 'source': str(source.resolve()),
            'background': None, 'fer': {'source_true': label},
            'metadata': {'dataset_name': name, 'split': split, 'source_emotion': label,
                         'original_label_id': label_id, 'annotation_file': str(annotation_file), **extra}}


def _finish(path, kind, items, excluded, metadata, split):
    if split != 'all':
        items = [item for item in items if item['metadata']['split'] == split]
    if not items:
        raise ValueError(f'No supported {kind} samples for split={split!r}. '
                         'Check the annotation format, split and class selection.')
    ids, sources = set(), set()
    for item in items:
        if item['id'] in ids or item['source'] in sources:
            raise ValueError(f'Duplicate image in {kind} annotations: {item["source"]}')
        ids.add(item['id'])
        sources.add(item['source'])
    return {'path': str(path), 'kind': kind, 'items': items, 'excluded_annotations': excluded,
            'metadata': {**metadata, 'requested_split': split, 'excluded_annotation_count': len(excluded)}}


def read_rafdb(path, split='all', annotations=None, image_root=None, image_version='auto'):
    path = Path(path).resolve()
    files = [Path(annotations).resolve()] if annotations else ([path] if path.is_file() else _raf_annotations(path))
    if len(files) != 1:
        raise ValueError('RAF-DB needs one EmoLabel/list_patition_label.txt. '
                         'Pass the basic dataset root or --annotations explicitly.')
    annotation_file = files[0]
    root = annotation_file.parent.parent if annotation_file.parent.name.lower() == 'emolabel' else (
        path if path.is_dir() else path.parent)
    version = image_version
    if version == 'auto':
        version = 'aligned' if (root / 'Image/aligned').is_dir() else 'original'
    images = Path(image_root).resolve() if image_root else root / 'Image' / version
    if not images.is_dir():
        raise ValueError(f'RAF-DB image directory missing: {images}. Use --image-root / --raf-images.')
    items = []
    for line_number, line in enumerate(annotation_file.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            filename, value = line.rsplit(maxsplit=1)
            label_id = _integer(value)
            label = RAF_LABELS[label_id]
        except (ValueError, KeyError) as exc:
            raise ValueError(f'{annotation_file}:{line_number}: expected RAF-DB Basic labels 1..7; '
                             'custom/zero-based labels need a generic manifest.') from exc
        sample_split = filename.split('_', 1)[0].lower()
        if sample_split not in ('train', 'test'):
            raise ValueError(f'Unknown RAF-DB split in filename: {filename}')
        if split != 'all' and sample_split != split:
            continue
        relative = Path(filename)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unexpected RAF-DB filename: {filename}')
        if version == 'aligned' and not relative.stem.endswith('_aligned'):
            relative = relative.with_name(relative.stem + '_aligned' + relative.suffix)
        items.append(_item('rafdb', images / relative, label, sample_split, label_id, annotation_file,
                           sample_key=filename, image_version=version, annotation_line=line_number))
    return _finish(path, 'rafdb', items, [], {'label_mapping': RAF_LABELS, 'image_root': str(images),
                   'image_version': version, 'annotation_files': [str(annotation_file)]}, split)


def _affectnet_image(relative, base, image_root):
    relative = Path(relative.replace('\\', '/'))
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('AffectNet image paths must be relative to the image root.')
    roots = [Path(image_root).resolve()] if image_root else [base / 'Manually_Annotated_Images', base / 'images', base]
    candidates = list(dict.fromkeys((root / relative).resolve() for root in roots))
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) > 1:
        raise ValueError(f'Ambiguous AffectNet image {relative}; specify --image-root.')
    return existing[0] if existing else candidates[0]


def _affect_label(value, classes):
    label_id = _integer(value)
    if label_id in (8, 9, 10):
        return label_id, None, 'non_target_expression'
    if label_id not in AFFECTNET_LABELS:
        raise ValueError(f'Unknown AffectNet expression ID: {label_id}')
    if label_id == 7 and classes == 7:
        return label_id, None, 'contempt_excluded_for_7_classes'
    return label_id, AFFECTNET_LABELS[label_id], None


def _npy_scalar(path):
    import numpy as np
    value = np.load(path, allow_pickle=False)
    if value.size != 1:
        raise ValueError(f'Expected scalar annotation: {path}')
    return value.item()


def read_affectnet(path, split='all', annotations=None, image_root=None, classes=8):
    path = Path(path).resolve()
    csvs = [Path(annotations).resolve()] if annotations else ([path] if path.is_file() else _affect_csvs(path))
    npy_roots = [] if csvs else _npy_roots(path)
    folder_roots = [] if csvs or npy_roots else _class_folder_roots(path)
    if classes not in (7, 8):
        raise ValueError('AffectNet supports 7 or 8 target expression classes.')
    if not csvs and not npy_roots and not folder_roots:
        raise ValueError('AffectNet needs native training.csv/validation.csv or split/images + '
                         'split/annotations/*_exp.npy, or explicit AffectNet split/0..7 class folders.')
    items, excluded, annotation_files = [], [], []
    for csv_path in csvs:
        sample_split = canonical_split(csv_path.stem)
        if sample_split not in ('train', 'val', 'test'):
            raise ValueError('Native AffectNet CSV filename must identify its split: training.csv, validation.csv or test.csv.')
        if split != 'all' and sample_split != split:
            continue
        annotation_files.append(str(csv_path))
        if 'Automatically_Annotated' in csv_path.parts:
            raise ValueError('Automatically annotated predictions are not source ground truth. '
                             'Use a generic manifest with source_pred for those images.')
        base = csv_path.parent.parent if csv_path.parent.name in ('file_lists', 'Manually_Annotated_file_lists') else csv_path.parent
        with csv_path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if not {'subDirectory_filePath', 'expression'}.issubset(reader.fieldnames or []):
                raise ValueError('Expected native AffectNet CSV columns: subDirectory_filePath, expression.')
            for line_number, row in enumerate(reader, 2):
                key = row['subDirectory_filePath']
                try:
                    if not key:
                        raise ValueError('Empty AffectNet image path.')
                    label_id, label, reason = _affect_label(row['expression'], classes)
                    if reason:
                        excluded.append({'annotation': str(csv_path), 'sample': key, 'split': sample_split, 'reason': reason})
                        continue
                    source = _affectnet_image(key, base, image_root)
                    items.append(_item('affectnet', source, label, sample_split, label_id, csv_path,
                                       sample_key=key, annotation_line=line_number,
                                       valence=_dimension(row.get('valence')), arousal=_dimension(row.get('arousal')),
                                       original_bbox={k: row[k] for k in ('face_x', 'face_y', 'face_width', 'face_height') if k in row}))
                except (ValueError, TypeError) as exc:
                    excluded.append({'annotation': str(csv_path), 'sample': key, 'split': sample_split,
                                     'reason': f'invalid_annotation: {exc}'})
    for root in npy_roots:
        sample_split = canonical_split(root.name)
        if sample_split not in ('train', 'val', 'test'):
            raise ValueError('AffectNet NPY root must be named train_set, val_set/validation_set or test_set.')
        if split != 'all' and sample_split != split:
            continue
        annotations_dir = root / 'annotations'
        annotation_files.append(str(annotations_dir))
        images = Path(image_root).resolve() if image_root else root / 'images'
        for label_path in sorted(annotations_dir.glob('*_exp.npy')):
            key = label_path.name[:-len('_exp.npy')]
            try:
                label_id, label, reason = _affect_label(_npy_scalar(label_path), classes)
                if reason:
                    excluded.append({'annotation': str(label_path), 'sample': key, 'split': sample_split, 'reason': reason})
                    continue
                candidates = [images / (key + suffix) for suffix in IMAGE_EXTENSIONS if (images / (key + suffix)).is_file()]
                if len(candidates) > 1:
                    raise ValueError('Multiple image extensions match this annotation.')
                source = candidates[0] if candidates else images / (key + '.jpg')
                dimensions = {}
                for dimension, suffix in (('valence', 'val'), ('arousal', 'aro')):
                    optional = annotations_dir / f'{key}_{suffix}.npy'
                    try:
                        dimensions[dimension] = _dimension(_npy_scalar(optional)) if optional.is_file() else None
                    except (OSError, ValueError, TypeError):
                        dimensions[dimension] = None
                items.append(_item('affectnet', source, label, sample_split, label_id, label_path,
                                   sample_key=key, **dimensions))
            except (OSError, ValueError, TypeError) as exc:
                excluded.append({'annotation': str(label_path), 'sample': key, 'split': sample_split,
                                 'reason': f'invalid_annotation: {exc}'})
    # Only reached with explicitly selected AffectNet type: numeric folders alone
    # do not identify a dataset or its label ordering during auto-detection.
    for root in folder_roots:
        sample_split = canonical_split(root.name)
        if split != 'all' and sample_split != split:
            continue
        annotation_files.append(str(root))
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdecimal()):
            label_id, label, reason = _affect_label(class_dir.name, classes)
            for source in sorted(p for p in class_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                                 and not p.stem.lower().endswith('_bg')):
                key = str(source.relative_to(root))
                if reason:
                    excluded.append({'annotation': str(class_dir), 'sample': key,
                                     'split': sample_split, 'reason': reason})
                    continue
                items.append(_item('affectnet', source, label, sample_split, label_id, class_dir,
                                   sample_key=key, annotation_origin='class_folder', valence=None, arousal=None))
    return _finish(path, 'affectnet', items, excluded, {'label_mapping': AFFECTNET_LABELS,
                   'expression_classes': classes, 'annotation_files': annotation_files,
                   'layout': 'csv' if csvs else 'npy' if npy_roots else 'class_folders'}, split)
