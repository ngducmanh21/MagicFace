"""Run pretrained MagicFace once per dataset and export aggregate measurements."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

from inference import generate_edits, load_pipeline, model_metadata
from mgface.au import edit_metadata, parse_au_request
from mgface.cell_selection import apply_cell_selection
from mgface.dataset_inputs import discover_dataset, inspect_item, item_key
from mgface.verification import add_report_arguments, report_from_args, validate_scale


ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def plan_dataset(args):
    dataset = discover_dataset(args.dataset, exclude_dir=args.output_dir, dataset_type=args.dataset_type,
                               split=args.split, annotations=args.annotations, image_root=args.image_root,
                               raf_images=args.raf_images, affectnet_classes=args.affectnet_classes)
    if args.cell_selection:
        dataset = apply_cell_selection(dataset, args.cell_selection, args.au_requests)
    selected = dataset['items'][:args.limit] if args.limit else dataset['items']
    if args.min_images and len(selected) < args.min_images:
        raise ValueError(f'Experiment requires at least {args.min_images} selected images, found {len(selected)}.')
    ready, raw, errors = [], [], []
    for item in selected:
        try:
            needs_preprocessing = inspect_item(item)
            if needs_preprocessing and args.prepared_only:
                raise ValueError('Missing background/pose. Remove --prepared_only to preprocess raw images.')
            (raw if needs_preprocessing else ready).append(item)
        except (OSError, ValueError) as exc:
            errors.append({'dataset_id': item['id'], 'source': item['source'],
                           'stage': 'input', 'error': str(exc)})
    return dataset, selected, ready, raw, errors


def prepare_raw_images(items, args, output):
    if not items:
        return {}, {}
    request, response = output / 'preprocess_request.json', output / 'preprocess_results.json'
    write_json(request, [{**item, 'key': item_key(item)} for item in items])
    assets = args.preprocess_assets
    provenance = {'repo_id': 'mengtingwei/magicface', 'requested_revision': args.preprocess_revision}
    try:
        if assets:
            assets = Path(assets).expanduser().resolve()
            provenance = {'local_assets': str(assets)}
        else:
            from huggingface_hub import snapshot_download
            assets = Path(snapshot_download(
                repo_id='mengtingwei/magicface', revision=args.preprocess_revision,
                allow_patterns=['79999_iter.pth', 'checkpoints/**', 'third_party/**', 'third_party_files/**'],
            ))
            provenance['resolved_revision'] = assets.name
        for relative in ('79999_iter.pth', 'third_party/model_resnet_d3dfr.py',
                         'third_party/d3dfr/bfm.py', 'checkpoints/third_party/d3dfr_res50_nofc.pth',
                         'checkpoints/third_party/BFM_model_front.mat'):
            if not (assets / relative).is_file():
                raise FileNotFoundError(f'Missing preprocessing asset: {assets / relative}')
        subprocess.run([args.preprocess_python, str(ROOT / 'mgface/preprocess_worker.py'),
                        '--request', str(request), '--response', str(response),
                        '--output', str(output / 'prepared'), '--assets', str(assets)], check=True)
        return json.loads(response.read_text(encoding='utf-8')), provenance
    except (OSError, ValueError, ImportError, subprocess.SubprocessError) as exc:
        message = f'{type(exc).__name__}: {exc}. For raw images install requirements-preprocess.txt.'
        return {item['id']: {'status': 'error', 'error': message} for item in items}, provenance


def _csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            row = dict(row)
            for key, value in row.items():
                if isinstance(value, dict):
                    row[key] = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, str) and value.startswith(('=', '+', '-', '@')):
                    row[key] = "'" + value
            writer.writerow(row)


def execute_dataset(args, loader=load_pipeline, generator=generate_edits,
                    preparer=prepare_raw_images, reporter=report_from_args):
    dataset, selected, ready, raw, errors = plan_dataset(args)
    request_counts = [len(item.get('generation_requests', args.au_requests)) for item in selected]
    counts = {'dataset_images': len(dataset['items']), 'selected_images': len(selected),
              'prepared_inputs': len(ready), 'raw_inputs': len(raw), 'invalid_inputs': len(errors),
              'edits_per_image': request_counts[0] if len(set(request_counts)) == 1 else None,
              'planned_edits': sum(request_counts),
              'unique_selected_sources': len({item['source'] for item in selected}),
              'minimum_images': args.min_images}
    counts.update(dataset_kind=dataset['kind'], excluded_annotations=len(dataset.get('excluded_annotations', [])),
                  selected_class_counts=dict(Counter(item['fer'].get('source_true') or 'unlabeled' for item in selected)),
                  selected_split_counts=dict(Counter(item['metadata'].get('split') or 'unspecified' for item in selected)))
    print(json.dumps(counts, indent=2), flush=True)
    if args.inspect:
        for error in errors[:20]:
            print(f"{error['dataset_id']}: {error['error']}")
        return {'inspection': counts, 'errors': errors}

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory is not empty: {output}. Use a new run directory.')
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    cases = []
    done_ids = set()
    previous_error_count = -1
    metadata = {**model_metadata(args), 'dataset_path': dataset['path'], 'dataset_kind': dataset['kind'],
                'dataset_metadata': dataset['metadata'], 'created_at': datetime.now(timezone.utc).isoformat(),
                'seed': args.seed, 'inference_steps': args.inference_steps,
                'requested_variations': args.au_requests, 'preprocessing': {}}
    designs = [edit_metadata(request) for request in args.au_requests]
    metadata['experiment_design'] = {
        'zero_baselines': sum(item['edit_type'] == 'zero_baseline' for item in designs),
        'single_au_conditions': sum(item['edit_type'] == 'single_au' for item in designs),
        'combination_conditions': sum(item['edit_type'] == 'combination' for item in designs),
        'active_aus': sorted({au for item in designs for au in item['active_aus']}),
        'seed_fixed_per_condition': args.seed,
        'inference_steps_fixed': args.inference_steps,
        'prompt_fixed': metadata['prompt'],
    }
    write_json(output / 'dataset_inputs.json', {**dataset, 'items': selected})
    if args.cell_selection:
        write_json(output / 'cell_selection.json',
                   json.loads(Path(args.cell_selection).read_text(encoding='utf-8')))
    _csv(output / 'excluded_annotations.csv', dataset.get('excluded_annotations', []),
         ['annotation', 'sample', 'split', 'reason'])

    def save_progress(status):
        nonlocal previous_error_count
        summary = {**counts, 'status': status, 'generated_edits': len(cases),
                   'not_generated_edits': counts['planned_edits'] - len(cases),
                   'images_with_output': len(done_ids), 'failed_events': len(errors),
                   'elapsed_seconds': perf_counter() - started, 'metadata': metadata}
        write_json(output / 'dataset_summary.json', summary)
        # Avoid rewriting a large manifest after every edit; samples.jsonl is
        # appended synchronously for every successful output.
        if status != 'generating' or len(cases) % 50 == 0:
            write_json(output / 'generation_manifest.json', {'cases': cases, 'metadata': metadata})
        if len(errors) != previous_error_count:
            write_json(output / 'failures.json', errors)
            _csv(output / 'failures.csv', errors, ['dataset_id', 'source', 'stage', 'error'])
            previous_error_count = len(errors)
        return summary

    save_progress('preparing')
    prepared, provenance = preparer(raw, args, output)
    metadata['preprocessing'] = provenance
    for item in raw:
        result = prepared.get(item['id'], {'status': 'error', 'error': 'No preprocessing result returned.'})
        if result['status'] == 'ok':
            candidate = {**item, 'input_source': item['source'],
                         'source': result['source'], 'background': result['background']}
            try:
                if inspect_item(candidate):
                    raise ValueError('Preprocessing returned no background.')
                ready.append(candidate)
            except (OSError, ValueError) as exc:
                errors.append({'dataset_id': item['id'], 'source': item['source'], 'stage': 'preprocess', 'error': str(exc)})
        else:
            errors.append({'dataset_id': item['id'], 'source': item['source'],
                           'stage': 'preprocess', 'error': result['error']})
    order = {item['id']: i for i, item in enumerate(selected)}
    ready.sort(key=lambda item: order[item['id']])
    if not ready:
        summary = save_progress('no_valid_inputs')
        _csv(output / 'failures.csv', errors, ['dataset_id', 'source', 'stage', 'error'])
        return summary

    pipeline = embeddings = None
    try:
        save_progress('loading_model')
        pipeline, embeddings = loader(args)  # One model load for the entire dataset.
        for i, item in enumerate(ready):
            print(f"Image {i + 1}/{len(ready)}: {item['id']}", flush=True)
            job = SimpleNamespace(**vars(args))
            job.img_path, job.bg_path = item['source'], item['background']
            job.saved_path = str(output / 'generated' / item_key(item))
            job.au_requests = item.get('generation_requests', args.au_requests)

            def on_case(case, variant_index):
                for key, value in edit_metadata(case['requested_aus']).items():
                    case.setdefault(key, value)
                case.update(dataset_id=item['id'], sample_id=f'{item_key(item)}_{variant_index + 1:03d}',
                            input_source=item.get('input_source', item['source']),
                            dataset_metadata=item['metadata'], fer=item['fer'],
                            dataset_name=item['metadata'].get('dataset_name', dataset['kind']),
                            dataset_split=item['metadata'].get('split', 'unspecified'),
                            source_emotion=item['fer'].get('source_true'),
                            source_valence=item['metadata'].get('valence'),
                            source_arousal=item['metadata'].get('arousal'))
                case.update(cell_id=item['metadata'].get('cell_id'),
                            cell_A=item['metadata'].get('cell_A'),
                            target_au=item['metadata'].get('target_au'),
                            control_aus=item['metadata'].get('control_aus'),
                            selection_status=item['metadata'].get('selection_status'),
                            acceptance_status=item['metadata'].get('acceptance_status'))
                case['label'] = f"{item['id']} / {case['label']}"
                cases.append(case)
                done_ids.add(item['id'])
                with (output / 'samples.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + '\n')
                save_progress('generating')

            try:
                generator(job, pipeline, embeddings, on_case=on_case)
            except Exception as exc:
                errors.append({'dataset_id': item['id'], 'source': item['source'],
                               'stage': 'generation', 'error': f'{type(exc).__name__}: {exc}'})
                save_progress('generating')
    except BaseException as exc:
        errors.append({'dataset_id': '', 'source': '', 'stage': 'model_or_run',
                       'error': f'{type(exc).__name__}: {exc}'})
        save_progress('interrupted_or_failed')
        raise
    finally:
        del pipeline, embeddings
        gc.collect()
        # Free the diffusion models before scoring, including when --au_device is CUDA.
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
            sys.modules['torch'].cuda.empty_cache()

    _csv(output / 'samples.csv', cases, ['sample_id', 'dataset_id', 'input_source', 'source', 'background',
                                        'result', 'requested_aus', 'seed', 'inference_steps', 'generation_seconds',
                                        'edit_type', 'edit_au', 'edit_level', 'dataset_name', 'dataset_split',
                                        'source_emotion', 'source_valence', 'source_arousal', 'cell_id',
                                        'cell_A', 'target_au', 'control_aus', 'selection_status', 'acceptance_status'])
    _csv(output / 'failures.csv', errors, ['dataset_id', 'source', 'stage', 'error'])
    metadata['dataset_counts'] = {**counts, 'images_with_output': len(done_ids),
                                  'generated_edits': len(cases), 'failed_events': len(errors)}
    save_progress('scoring_and_reporting')
    if cases:
        try:
            report = reporter(cases, output, args, metadata=metadata)
            counts['au_scored_pairs'] = report['scored_cases']
            counts['au_unscored_pairs'] = len(cases) - report['scored_cases']
        except BaseException:
            save_progress('report_failed')
            raise
    scoring_errors = args.au_backend != 'none' and counts.get('au_unscored_pairs', 0) > 0
    summary = save_progress('complete_with_errors' if errors or scoring_errors else 'complete')
    print(f"Dataset summary: {output / 'dataset_summary.json'}", flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dataset_type', choices=('auto', 'generic', 'rafdb', 'affectnet'), default='auto')
    parser.add_argument('--split', choices=('all', 'train', 'val', 'test'), default='all')
    parser.add_argument('--annotations', help='Native annotation file when it is outside the usual directory.')
    parser.add_argument('--image_root', help='Directory relative to which annotation image paths are resolved.')
    parser.add_argument('--raf_images', choices=('auto', 'aligned', 'original'), default='auto')
    parser.add_argument('--affectnet_classes', type=int, choices=(7, 8), default=8)
    parser.add_argument('--cell_selection', help='Audited rare-cell allowlist; filters sources before --limit.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--au_test', default='AU4+AU1')
    parser.add_argument('--AU_variation', action='append')
    parser.add_argument('--seed', type=int, default=424)
    parser.add_argument('--inference_steps', type=int, default=50)
    parser.add_argument('--limit', type=int, default=0, help='Maximum input images; zero means all.')
    parser.add_argument('--min_images', type=int, default=0, help='Fail if fewer selected images are available.')
    parser.add_argument('--inspect', action='store_true', help='Inspect inputs without downloads, GPU or output writes.')
    parser.add_argument('--prepared_only', action='store_true')
    parser.add_argument('--preprocess_python', default=sys.executable)
    parser.add_argument('--preprocess_assets', default=None)
    parser.add_argument('--preprocess_revision', default='main')
    parser.add_argument('--pretrained_model_name_or_path', default='sd-legacy/stable-diffusion-v1-5')
    parser.add_argument('--ID_unet_path', default='mengtingwei/magicface')
    parser.add_argument('--denoising_unet_path', default='mengtingwei/magicface')
    parser.add_argument('--revision', default=None)
    parser.add_argument('--variant', default=None)
    add_report_arguments(parser)
    args = parser.parse_args(argv)
    try:
        args.au_requests = [parse_au_request(args.au_test, value)
                            for value in (args.AU_variation or ['0+0', '2+1', '4+2'])]
        validate_scale(args.au_delta_scale)
        if args.limit < 0 or args.min_images < 0 or args.inference_steps <= 0:
            raise ValueError('limit/min_images must be non-negative and inference_steps must be positive.')
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    if not args.inspect:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit('NVIDIA CUDA is required for dataset inference. Use --inspect to check inputs first.')
    summary = execute_dataset(args)
    if not args.inspect and not summary.get('generated_edits'):
        raise SystemExit('No images were generated. Inspect failures.csv / dataset_summary.json.')


if __name__ == '__main__':
    main()
