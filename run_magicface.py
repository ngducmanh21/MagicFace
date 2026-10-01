#!/usr/bin/env python3
"""Short commands for MagicFace inference and AU/FER reports. See docs/QUICKSTART_VI.md."""

import argparse
from decimal import Decimal
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from mgface.au import parse_au_request


ROOT = Path(__file__).resolve().parent
CONFIG_KEYS = {'image', 'background', 'output_dir', 'aus', 'variations', 'seed',
               'inference_steps', 'title', 'au_python', 'au_backend', 'au_device',
               'au_delta_scale', 'figure_formats', 'base_model', 'id_model', 'denoising_model',
               'limit', 'min_images', 'dataset_type', 'split', 'raf_images',
               'affectnet_classes', 'single_au_only', 'require_zero_baseline', 'cell_selection',
               'cell_selection_sha256', 'shared_source_sha256', 'expected_cell_count',
               'expected_cell_source_pairs', 'expected_unique_sources'}


def resolve_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Duong dan phai la chuoi khong rong.')
    return (base / Path(value).expanduser()).resolve()


def python_path(value, base):
    if value is None:
        return sys.executable
    if not isinstance(value, str) or not value.strip():
        raise ValueError('au_python phai la duong dan Python, hoac null.')
    if '/' not in value and '\\' not in value and not value.startswith('~'):
        return value
    # Do not call Path.resolve(): a virtualenv's bin/python is normally a
    # symlink, and resolving it discards that environment's site-packages.
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return os.path.abspath(os.fspath(path))


def report_options(backend, python, device, title, formats, scale):
    if backend not in ('libreface', 'none'):
        raise ValueError('au_backend phai la libreface hoac none.')
    if not isinstance(formats, list) or not formats or set(formats) - {'png', 'svg', 'pdf'}:
        raise ValueError('figure_formats phai la danh sach png, svg, pdf.')
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float))
                              or not math.isfinite(scale) or scale == 0):
        raise ValueError('au_delta_scale phai la so huu han khac 0, hoac null.')
    if not isinstance(title, str) or not isinstance(device, str):
        raise ValueError('title va au_device phai la chuoi.')
    options = ['--evidence', '--au_backend', backend, '--au_python', python,
               '--au_device', device, '--report_title', title, '--figure_formats', *formats]
    if scale is not None:
        options += ['--au_delta_scale', str(scale)]
    return options


def check_python_imports(python, modules):
    """Fail before model/asset downloads when a delegated Python lacks dependencies."""
    statement = '; '.join(f'import {module}' for module in modules)
    result = subprocess.run([python, '-c', statement], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        detail = detail[-1] if detail else f'exit code {result.returncode}'
        raise RuntimeError(f'{python}: {detail}')


def build_command(args):
    """Return argv and output directory. Building/dry-running never loads models."""
    if args.command == 'preview':
        output = Path(args.output).expanduser().resolve()
        return [sys.executable, str(ROOT / 'verify_results.py'),
                '--manifest', str(ROOT / 'examples/verification_preview.json'),
                '--output_dir', str(output), '--au_backend', 'none', '--evidence',
                '--figure_formats', 'png', '--report_title',
                'Layout preview - unchanged images, no measurements'], output

    if args.command == 'report':
        path = Path(args.results or args.manifest).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'Khong tim thay file: {path}')
        output = Path(args.output).expanduser().resolve()
        options = report_options('none' if args.no_au else 'libreface',
                                 python_path(args.au_python or os.environ.get('MAGICFACE_AU_PYTHON'), Path.cwd()), args.au_device,
                                 args.title, args.formats, args.au_delta_scale)
        return [sys.executable, str(ROOT / 'verify_results.py'),
                '--results_json' if args.results else '--manifest', str(path),
                '--output_dir', str(output), *options], output

    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or set(config) - CONFIG_KEYS:
        raise ValueError(f'Config chi ho tro cac key: {", ".join(sorted(CONFIG_KEYS))}')
    base = config_path.parent
    is_dataset = args.command == 'dataset'
    if is_dataset:
        dataset = Path(args.path).expanduser().resolve()
        if not dataset.exists():
            raise ValueError(f'Khong tim thay dataset: {dataset}')
        default_output = ROOT / 'runs' / (dataset.stem + '_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    else:
        image, background = (resolve_path(config[key], base) for key in ('image', 'background'))
        for path in (image, background):
            if not path.is_file():
                raise ValueError(f'Khong tim thay anh: {path}')
        default_output = None
    output = (Path(args.output).expanduser().resolve() if args.output
              else resolve_path(config['output_dir'], base) if 'output_dir' in config else default_output)
    if output is None:
        raise ValueError('Can output_dir trong config hoac --output.')
    aus, variations = config['aus'], config['variations']
    if not isinstance(aus, list) or not aus or not all(isinstance(name, str) for name in aus):
        raise ValueError('aus phai la danh sach ten AU, vi du ["AU4", "AU1"].')
    if not isinstance(variations, list) or not variations:
        raise ValueError('variations phai la danh sach khong rong.')
    names = '+'.join(aus)
    changes = []
    requests = []
    for values in variations:
        if not isinstance(values, list) or len(values) != len(aus):
            raise ValueError('Moi dong variations phai co dung mot gia tri cho moi AU.')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ValueError('Gia tri AU phai la so.')
        # Fixed-point formatting keeps '+' in scientific exponents out of the AU delimiter.
        serialized = '+'.join(format(Decimal(str(v)), 'f') for v in values)
        requests.append(parse_au_request(names, serialized))
        changes.append(f'--AU_variation={serialized}')
    if len({tuple(request[name] for name in aus) for request in requests}) != len(requests):
        raise ValueError('variations khong duoc trung nhau.')
    if config.get('require_zero_baseline', False):
        baselines = sum(all(value == 0 for value in request.values()) for request in requests)
        if baselines != 1:
            raise ValueError('Config yeu cau dung mot zero-edit baseline.')
    if config.get('single_au_only', False):
        invalid = [request for request in requests if sum(value != 0 for value in request.values()) > 1]
        if invalid:
            raise ValueError(f'single_au_only cam AU combinations: {invalid[0]}')
    seed, steps = config.get('seed', 424), config.get('inference_steps', 50)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed phai la so nguyen trong [0, 2**63).')
    if type(steps) is not int or steps <= 0:
        raise ValueError('inference_steps phai la so nguyen duong.')
    backend = 'none' if args.no_au else config.get('au_backend', 'libreface')
    au_python = (python_path(args.au_python, Path.cwd()) if args.au_python
                 else python_path(config['au_python'], base) if config.get('au_python')
                 else python_path(os.environ.get('MAGICFACE_AU_PYTHON'), Path.cwd()))
    options = report_options(backend, au_python, config.get('au_device', 'cpu'),
                             config.get('title', 'MagicFace / AU review'),
                             config.get('figure_formats', ['png', 'svg', 'pdf']),
                             config.get('au_delta_scale'))
    if is_dataset:
        explicit_limit = args.limit is not None
        limit = args.limit if explicit_limit else config.get('limit', 0)
        min_images = (args.min_images if args.min_images is not None else
                      0 if explicit_limit else config.get('min_images', 0))
        dataset_type = args.dataset_type or config.get('dataset_type', 'auto')
        split = args.split or config.get('split', 'all')
        raf_images = args.raf_images or config.get('raf_images', 'auto')
        affectnet_classes = (args.affectnet_classes if args.affectnet_classes is not None
                             else config.get('affectnet_classes', 8))
        if limit < 0 or min_images < 0:
            raise ValueError('--limit va --min-images phai >= 0.')
        command = [sys.executable, str(ROOT / 'inference_dataset.py'), '--dataset', str(dataset),
                   '--output_dir', str(output), '--limit', str(limit), '--min_images', str(min_images),
                   '--preprocess_python', python_path(args.preprocess_python, Path.cwd()),
                   '--dataset_type', dataset_type, '--split', split,
                   '--raf_images', raf_images, '--affectnet_classes', str(affectnet_classes)]
        cell_selection = args.cell_selection or config.get('cell_selection')
        if cell_selection:
            selection_path = Path(cell_selection).expanduser()
            if not selection_path.is_absolute():
                selection_path = base / selection_path
            command += ['--cell_selection', os.path.abspath(os.fspath(selection_path))]
            for key, flag in (
                    ('cell_selection_sha256', '--cell_selection_sha256'),
                    ('shared_source_sha256', '--shared_source_sha256'),
                    ('expected_cell_count', '--expected_cell_count'),
                    ('expected_cell_source_pairs', '--expected_cell_source_pairs'),
                    ('expected_unique_sources', '--expected_unique_sources')):
                if config.get(key) is not None:
                    command += [flag, str(config[key])]
        for value, flag in ((args.annotations, '--annotations'), (args.image_root, '--image_root')):
            if value:
                command += [flag, str(Path(value).expanduser().resolve())]
        if args.inspect:
            command.append('--inspect')
        if args.prepared_only:
            command.append('--prepared_only')
        if args.preprocess_assets:
            command += ['--preprocess_assets', str(Path(args.preprocess_assets).expanduser().resolve())]
    else:
        command = [sys.executable, str(ROOT / 'inference.py'), '--img_path', str(image),
                   '--bg_path', str(background), '--saved_path', str(output / 'generated'),
                   '--verification_dir', str(output)]
    command += ['--au_test', names, *changes, '--seed', str(seed), '--inference_steps', str(steps), *options]
    for key, flag in (('base_model', '--pretrained_model_name_or_path'),
                      ('id_model', '--ID_unet_path'), ('denoising_model', '--denoising_unet_path')):
        if key in config:
            value = config[key]
            if not isinstance(value, str) or not value:
                raise ValueError(f'{key} phai la Hugging Face model ID hoac duong dan.')
            if value.startswith(('.', '~', '/')):
                value = str(resolve_path(value, base))
            command.extend((flag, value))
    return command, output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preview = commands.add_parser('preview', help='Xem layout report; khong can GPU, weights hay LibreFace.')
    preview.add_argument('--output', default=str(ROOT / 'runs/tutorial_preview'))
    run = commands.add_parser('run', help='Sinh anh + report tu config JSON; can NVIDIA CUDA.')
    run.add_argument('--config', default=str(ROOT / 'configs/inference_demo.json'))
    run.add_argument('--output', help='Ghi de output_dir trong config.')
    dataset = commands.add_parser('dataset', help='Mot path = mot dataset; chay pretrained weights va report chung.')
    dataset.add_argument('path', help='Thu muc anh, dataset.json hoac dataset.csv.')
    dataset.add_argument('--config', default=str(ROOT / 'configs/dataset_demo.json'))
    dataset.add_argument('--output', help='Thu muc output moi; mac dinh tao run theo ten dataset + timestamp.')
    dataset.add_argument('--limit', type=int, default=None, help='Ghi de so anh trong config; 0 = tat ca.')
    dataset.add_argument('--min-images', type=int, default=None, help='Ghi de so anh toi thieu; explicit --limit tu dong cho phep smoke test nho.')
    dataset.add_argument('--dataset-type', choices=('auto', 'generic', 'rafdb', 'affectnet'), default=None)
    dataset.add_argument('--split', choices=('all', 'train', 'val', 'test'), default=None, help='Loc split truoc khi ap dung --limit.')
    dataset.add_argument('--annotations', help='File nhan goc neu khong nam dung vi tri mac dinh.')
    dataset.add_argument('--image-root', help='Thu muc anh neu khac cau truc mac dinh.')
    dataset.add_argument('--raf-images', choices=('auto', 'aligned', 'original'), default=None)
    dataset.add_argument('--affectnet-classes', type=int, choices=(7, 8), default=None)
    dataset.add_argument('--cell-selection', help='Ghi de file audited rare-cell allowlist trong config.')
    dataset.add_argument('--inspect', action='store_true', help='Kiem tra dataset, khong can GPU hay tai weights.')
    dataset.add_argument('--prepared-only', action='store_true', help='Chi nhan anh da co background/pose.')
    dataset.add_argument('--preprocess-python', help='Python da cai requirements-preprocess.txt; mac dinh Python hien tai.')
    dataset.add_argument('--preprocess-assets', help='Thu muc assets preprocessing co san; mac dinh tai tu Hugging Face.')
    report = commands.add_parser('report', help='Xuat report tu ket qua/manifest da co.')
    source = report.add_mutually_exclusive_group(required=True)
    source.add_argument('--results', help='results.json da co scores; khong chay lai model.')
    source.add_argument('--manifest', help='Manifest anh; cham AU neu khong dung --no-au.')
    report.add_argument('--output', required=True)
    report.add_argument('--au-device', default='cpu')
    report.add_argument('--au-delta-scale', type=float)
    report.add_argument('--title', default='MagicFace / AU and FER evidence')
    report.add_argument('--formats', nargs='+', choices=('png', 'svg', 'pdf'), default=['png', 'svg', 'pdf'])
    for sub in (run, report, dataset):
        sub.add_argument('--au-python', help='Python executable cua moi truong LibreFace rieng.')
        sub.add_argument('--no-au', action='store_true', help='Khong cham AU moi; scores cu van duoc giu voi --results.')
    for sub in (preview, run, report, dataset):
        sub.add_argument('--dry-run', action='store_true', help='In lenh se chay; khong tai model hay tao output.')
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        command, output = build_command(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(shlex.join(command), flush=True)
    inspection = args.command == 'dataset' and args.inspect
    if inspection:
        print('Inspection only: no output directory or report will be created.', flush=True)
    else:
        print(f'Report: {output / "report.html"}', flush=True)
    if args.dry_run:
        return 0
    if args.command == 'dataset' and not inspection:
        # Validate annotations, split, paths and actual images before checking
        # CUDA or optional AU-scoring dependencies.
        try:
            from inference_dataset import parse_args as parse_dataset_args, plan_dataset
            dataset_args = parse_dataset_args(command[2:])
            dataset, selected, ready, raw, errors = plan_dataset(dataset_args)
            print(f"Dataset preflight: type={dataset['kind']}, selected={len(selected)}, "
                  f"prepared={len(ready)}, preprocess={len(raw)}, invalid={len(errors)}", flush=True)
            for error in errors[:5]:
                print(f"  invalid {error['dataset_id']}: {error['error']}", file=sys.stderr)
            if raw:
                try:
                    modules = ['torch', 'torchvision', 'cv2', 'scipy']
                    if any(not (item.get('metadata', {}).get('dataset_name') == 'rafdb'
                               and item.get('metadata', {}).get('image_version') == 'aligned')
                           for item in raw):
                        modules += ['onnxruntime', 'insightface']
                    check_python_imports(dataset_args.preprocess_python, modules)
                except (OSError, RuntimeError) as exc:
                    print(f'Moi truong preprocessing chua san sang: {exc}\n'
                          f'Python dang dung: {dataset_args.preprocess_python}\n'
                          'Cai requirements-preprocess.txt trong moi truong MagicFace rieng, '
                          'roi chay CLI bang Python do.', file=sys.stderr)
                    return 2
        except (ImportError, OSError, ValueError, SystemExit) as exc:
            print(f'Dataset khong hop le: {exc}\n'
                  'RAF-DB: truyen root basic/, hoac truyen --annotations va --image-root.\n'
                  'Neu thieu package, hay chay CLI trong moi truong MagicFace da cai requirements.', file=sys.stderr)
            return 2
    if args.command in ('run', 'dataset') and not inspection:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('NVIDIA CUDA chua san sang. Kiem tra nvidia-smi va ban PyTorch CUDA.')
        except (ImportError, RuntimeError, OSError) as exc:
            print(f'Khong the chay inference: {exc}\nXem docs/QUICKSTART_VI.md; preview khong can GPU.', file=sys.stderr)
            return 2
    # Avoid downloading diffusion weights only to discover that scoring cannot import.
    needs_au = not inspection and (args.command in ('run', 'dataset') or (args.command == 'report' and args.manifest))
    if needs_au and command[command.index('--au_backend') + 1] == 'libreface':
        au_python = command[command.index('--au_python') + 1]
        try:
            subprocess.run([au_python, '-c', 'import libreface'], cwd=ROOT, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f'Khong import duoc LibreFace voi {au_python}: {exc}\n'
                  'Generation chua bat dau. Chon mot trong hai cach:\n'
                  '  1) Them --no-au de sinh anh/report truoc (AU se la N/A).\n'
                  '  2) Tao moi truong AU rieng, roi truyen --au-python /path/to/python.\n'
                  '     uv venv --python 3.9 --seed .venv-au\n'
                  '     .venv-au/bin/python -m pip install -r requirements-au.txt', file=sys.stderr)
            return 2
    try:
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
