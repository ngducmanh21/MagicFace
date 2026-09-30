"""Prepare raw dataset images in a separate process, releasing GPU memory afterwards."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def aligned_input(item):
    """RAF aligned crops already encode face location; detecting/cropping again is harmful."""
    metadata = item.get('metadata') or {}
    return metadata.get('dataset_name') == 'rafdb' and metadata.get('image_version') == 'aligned'


def prepare_aligned_source(input_path, output_path):
    from PIL import Image, ImageOps

    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        if image.width != image.height:
            raise ValueError(f'RAF aligned input must be square, got {image.size}')
        image.resize((512, 512), Image.Resampling.LANCZOS).save(output_path)


def prepare(items, output, assets):
    from PIL import Image, ImageOps

    # The original preprocessing modules expect these paths relative to cwd.
    os.chdir(assets)
    sys.path.insert(0, str(ROOT / 'utils'))
    sys.path.insert(0, str(assets))
    import retrieve_bg
    preprocess = None
    if any(not aligned_input(item) for item in items):
        import preprocess as preprocess_module
        preprocess = preprocess_module

    results = {}
    for index, item in enumerate(items):
        print(f'Preparing {index + 1}/{len(items)}: {item["id"]}', flush=True)
        directory = output / item['key']
        directory.mkdir(parents=True, exist_ok=True)
        source, background = directory / 'source.png', directory / 'background.png'
        try:
            if aligned_input(item):
                prepare_aligned_source(item['source'], source)
            else:
                with tempfile.TemporaryDirectory(prefix='magicface-input-') as temp:
                    normalized = Path(temp) / 'input.png'
                    with Image.open(item['source']) as image:
                        ImageOps.exif_transpose(image).convert('RGB').save(normalized)
                    preprocess.crop_one_image(SimpleNamespace(img_path=str(normalized), save_path=str(source)))
            retrieve_bg.make_bg_for_one_image(SimpleNamespace(
                img_path=str(source), save_path=str(background), assume_aligned=aligned_input(item)))
            for path in (source, background):
                with Image.open(path) as image:
                    image.load()
                    if image.size != (512, 512):
                        raise ValueError(f'Unexpected prepared image size: {image.size}')
            results[item['id']] = {'status': 'ok', 'source': str(source), 'background': str(background)}
        except Exception as exc:
            results[item['id']] = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
            print(f'Failed {item["id"]}: {results[item["id"]]["error"]}', file=sys.stderr, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--response', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--assets', required=True)
    args = parser.parse_args()
    response = Path(args.response).resolve()
    items = json.loads(Path(args.request).read_text(encoding='utf-8'))
    try:
        results = prepare(items, Path(args.output).resolve(), Path(args.assets).resolve())
    except Exception as exc:
        results = {item['id']: {'status': 'error', 'error': f'Preprocessing setup failed: {type(exc).__name__}: {exc}'}
                   for item in items}
    response.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
