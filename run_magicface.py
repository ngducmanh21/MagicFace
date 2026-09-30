#!/usr/bin/env python3
"""Short commands for MagicFace inference and AU/FER reports. See docs/QUICKSTART_VI.md."""

import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

from mgface.au import parse_au_request


ROOT = Path(__file__).resolve().parent
CONFIG_KEYS = {'image', 'background', 'output_dir', 'aus', 'variations', 'seed',
               'inference_steps', 'title', 'au_python', 'au_backend', 'au_device',
               'au_delta_scale', 'figure_formats', 'base_model', 'id_model', 'denoising_model'}


def resolve_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Duong dan phai la chuoi khong rong.')
    return (base / Path(value).expanduser()).resolve()


def python_path(value, base):
    if value is None:
        return sys.executable
    if not isinstance(value, str) or not value.strip():
        raise ValueError('au_python phai la duong dan Python, hoac null.')
    return str(resolve_path(value, base)) if '/' in value or '\\' in value or value.startswith('~') else value


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
                                 python_path(args.au_python, Path.cwd()), args.au_device,
                                 args.title, args.formats, args.au_delta_scale)
        return [sys.executable, str(ROOT / 'verify_results.py'),
                '--results_json' if args.results else '--manifest', str(path),
                '--output_dir', str(output), *options], output

    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or set(config) - CONFIG_KEYS:
        raise ValueError(f'Config chi ho tro cac key: {", ".join(sorted(CONFIG_KEYS))}')
    base = config_path.parent
    image, background = (resolve_path(config[key], base) for key in ('image', 'background'))
    for path in (image, background):
        if not path.is_file():
            raise ValueError(f'Khong tim thay anh: {path}')
    output = (Path(args.output).expanduser().resolve() if args.output
              else resolve_path(config['output_dir'], base))
    aus, variations = config['aus'], config['variations']
    if not isinstance(aus, list) or not aus or not all(isinstance(name, str) for name in aus):
        raise ValueError('aus phai la danh sach ten AU, vi du ["AU4", "AU1"].')
    if not isinstance(variations, list) or not variations:
        raise ValueError('variations phai la danh sach khong rong.')
    names = '+'.join(aus)
    changes = []
    for values in variations:
        if not isinstance(values, list) or len(values) != len(aus):
            raise ValueError('Moi dong variations phai co dung mot gia tri cho moi AU.')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ValueError('Gia tri AU phai la so.')
        # Fixed-point formatting keeps '+' in scientific exponents out of the AU delimiter.
        serialized = '+'.join(format(Decimal(str(v)), 'f') for v in values)
        parse_au_request(names, serialized)
        changes.append(f'--AU_variation={serialized}')
    seed, steps = config.get('seed', 424), config.get('inference_steps', 50)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed phai la so nguyen trong [0, 2**63).')
    if type(steps) is not int or steps <= 0:
        raise ValueError('inference_steps phai la so nguyen duong.')
    backend = 'none' if args.no_au else config.get('au_backend', 'libreface')
    au_python = (python_path(args.au_python, Path.cwd()) if args.au_python
                 else python_path(config.get('au_python'), base))
    options = report_options(backend, au_python, config.get('au_device', 'cpu'),
                             config.get('title', 'MagicFace / AU review'),
                             config.get('figure_formats', ['png', 'svg', 'pdf']),
                             config.get('au_delta_scale'))
    command = [sys.executable, str(ROOT / 'inference.py'), '--img_path', str(image),
               '--bg_path', str(background), '--saved_path', str(output / 'generated'),
               '--verification_dir', str(output), '--au_test', names, *changes,
               '--seed', str(seed), '--inference_steps', str(steps), *options]
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
    report = commands.add_parser('report', help='Xuat report tu ket qua/manifest da co.')
    source = report.add_mutually_exclusive_group(required=True)
    source.add_argument('--results', help='results.json da co scores; khong chay lai model.')
    source.add_argument('--manifest', help='Manifest anh; cham AU neu khong dung --no-au.')
    report.add_argument('--output', required=True)
    report.add_argument('--au-device', default='cpu')
    report.add_argument('--au-delta-scale', type=float)
    report.add_argument('--title', default='MagicFace / AU and FER evidence')
    report.add_argument('--formats', nargs='+', choices=('png', 'svg', 'pdf'), default=['png', 'svg', 'pdf'])
    for sub in (run, report):
        sub.add_argument('--au-python', help='Python executable cua moi truong LibreFace rieng.')
        sub.add_argument('--no-au', action='store_true', help='Khong cham AU moi; scores cu van duoc giu voi --results.')
    for sub in (preview, run, report):
        sub.add_argument('--dry-run', action='store_true', help='In lenh se chay; khong tai model hay tao output.')
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        command, output = build_command(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(shlex.join(command), flush=True)
    print(f'Report: {output / "report.html"}', flush=True)
    if args.dry_run:
        return 0
    if args.command == 'run':
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('NVIDIA CUDA chua san sang. Kiem tra nvidia-smi va ban PyTorch CUDA.')
        except (ImportError, RuntimeError, OSError) as exc:
            print(f'Khong the chay inference: {exc}\nXem docs/QUICKSTART_VI.md; preview khong can GPU.', file=sys.stderr)
            return 2
    # Avoid downloading diffusion weights only to discover that scoring cannot import.
    needs_au = args.command == 'run' or (args.command == 'report' and args.manifest)
    if needs_au and command[command.index('--au_backend') + 1] == 'libreface':
        au_python = command[command.index('--au_python') + 1]
        try:
            subprocess.run([au_python, '-c', 'import libreface'], cwd=ROOT, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f'Khong import duoc LibreFace voi {au_python}: {exc}\n'
                  'Cai requirements-au.txt hoac dung --no-au de chay anh truoc.', file=sys.stderr)
            return 2
    try:
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
