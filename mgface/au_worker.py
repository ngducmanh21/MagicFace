"""Run AU estimation in a separate Python environment from diffusion inference."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import tempfile

# This file is executed by path, including from an external Python environment.
from au import normalize_intensities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--response', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--weights_dir', required=True)
    args = parser.parse_args()
    paths = json.loads(Path(args.request).read_text())
    results = {}
    try:
        import libreface
        version = importlib.metadata.version('libreface')
    except Exception as exc:
        message = f'LibreFace unavailable: {exc}. Install requirements-au.txt in the AU Python environment.'
        results = {path: {'status': 'error', 'error': message} for path in paths}
        version = None
    else:
        for index, path in enumerate(paths):
            print(f'AU scoring {index + 1}/{len(paths)}: {Path(path).name}', flush=True)
            try:
                with tempfile.TemporaryDirectory(prefix='magicface-au-') as temp:
                    aligned, _, _ = libreface.get_aligned_image(path, temp_dir=temp)
                    # Use the joint model's intensity head, with the same AU ordering
                    # as MagicFace. Binary detections from the other head are ignored.
                    _, raw = libreface.get_au_intensities_and_detect_aus(
                        aligned, device=args.device, weights_download_dir=args.weights_dir,
                    )
                results[path] = {'status': 'ok', 'intensities': normalize_intensities(raw)}
            except Exception as exc:
                # Missing faces / weights are visible in the report, not zeros.
                results[path] = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
    Path(args.response).write_text(json.dumps({
        'estimator': 'LibreFace / joint AU intensity head',
        'version': version, 'device': args.device, 'images': results,
    }, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
