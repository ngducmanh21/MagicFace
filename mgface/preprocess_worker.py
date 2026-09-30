"""Prepare raw dataset images in a separate process, releasing GPU memory afterwards."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def prepare(items, output, assets):
    from PIL import Image, ImageOps

    # The original preprocessing modules expect these paths relative to cwd.
    os.chdir(assets)
    sys.path.insert(0, str(ROOT / 'utils'))
    sys.path.insert(0, str(assets))
    import preprocess
    import retrieve_bg

    results = {}
    for index, item in enumerate(items):
        print(f'Preparing {index + 1}/{len(items)}: {item["id"]}', flush=True)
        directory = output / item['key']
        directory.mkdir(parents=True, exist_ok=True)
        source, background = directory / 'source.png', directory / 'background.png'
        try:
            with tempfile.TemporaryDirectory(prefix='magicface-input-') as temp:
                normalized = Path(temp) / 'input.png'
                with Image.open(item['source']) as image:
                    ImageOps.exif_transpose(image).convert('RGB').save(normalized)
                preprocess.crop_one_image(SimpleNamespace(img_path=str(normalized), save_path=str(source)))
            retrieve_bg.make_bg_for_one_image(SimpleNamespace(img_path=str(source), save_path=str(background)))
            for path in (source, background):
                with Image.open(path) as image:
                    image.load()
                    if image.size != (512, 512):
                        raise ValueError(f'Unexpected prepared image size: {image.size}')
            results[item['id']] = {'status': 'ok', 'source': str(source), 'background': str(background)}
        except Exception as exc:
            results[item['id']] = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
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
