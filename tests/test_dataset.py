"""Dataset orchestration tests use synthetic images and fake generation, never model scores."""

import ast
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from inference_dataset import execute_dataset, parse_args, plan_dataset
from mgface.dataset_inputs import discover_dataset, inspect_item, item_key
from mgface.preprocess_worker import aligned_input, prepare_aligned_source
from mgface.verification import report_from_args
from run_magicface import ROOT, build_command, parse_args as parse_cli
from verify_results import load_saved_results


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'dataset'
        for name, color in (('class_a', 'red'), ('class_b', 'blue')):
            directory = self.dataset / name
            directory.mkdir(parents=True)
            Image.new('RGB', (512, 512), color).save(directory / 'same.png')
            Image.new('RGB', (512, 512), 'black').save(directory / 'same_bg.png')
        self.output = self.root / 'output'

    def args(self, *extra):
        return parse_args(['--dataset', str(self.dataset), '--output_dir', str(self.output),
                           '--au_backend', 'none', '--figure_formats', 'png', *extra])

    @staticmethod
    def fake_generate(args, pipeline, embeddings, on_case):
        directory = Path(args.saved_path)
        directory.mkdir(parents=True)
        for i, request in enumerate(args.au_requests):
            target = directory / f'fake_{i}.png'
            with Image.open(args.img_path) as image:
                image.save(target)
            on_case({'source': args.img_path, 'result': str(target), 'background': args.bg_path,
                     'requested_aus': request, 'label': 'SYNTHETIC GENERATION FIXTURE', 'seed': args.seed,
                     'inference_steps': args.inference_steps, 'generation_seconds': 0.01}, i)

    def test_folder_pairs_skip_backgrounds_and_keep_duplicate_basenames(self):
        dataset = discover_dataset(self.dataset)
        self.assertEqual(len(dataset['items']), 2)
        self.assertEqual(len({item_key(item) for item in dataset['items']}), 2)
        self.assertTrue(all(not inspect_item(item) for item in dataset['items']))

    def test_images_and_backgrounds_trees(self):
        source = self.root / 'split/images/person/x.jpg'
        background = self.root / 'split/backgrounds/person/x_bg.png'
        source.parent.mkdir(parents=True)
        background.parent.mkdir(parents=True)
        Image.new('RGB', (512, 512)).save(source)
        Image.new('RGB', (512, 512)).save(background)
        data = discover_dataset(self.root / 'split')
        self.assertEqual(len(data['items']), 1)
        self.assertEqual(data['items'][0]['background'], str(background))

    def test_raw_images_require_preprocessing_and_ambiguous_background_is_error(self):
        raw = self.dataset / 'raw.jpg'
        Image.new('RGB', (300, 200)).save(raw)
        dataset, selected, ready, raw_items, errors = plan_dataset(self.args())
        self.assertEqual((len(ready), len(raw_items), len(errors)), (2, 1, 0))
        _, _, _, _, errors = plan_dataset(self.args('--prepared_only'))
        self.assertEqual(len(errors), 1)
        Image.new('RGB', (512, 512)).save(self.dataset / 'class_a/same_bg.jpg')
        _, _, _, _, errors = plan_dataset(self.args())
        self.assertIn('Ambiguous', errors[0]['error'])

    def test_csv_metadata_and_manifest_relative_paths(self):
        manifest = self.dataset / 'dataset.csv'
        manifest.write_text('id,source,background,source_true,source_pred,split\na,class_a/same.png,class_a/same_bg.png,happy,happy,test\n')
        data = discover_dataset(self.dataset)
        self.assertEqual(data['kind'], 'csv')
        self.assertEqual(data['items'][0]['fer']['source_true'], 'happy')
        self.assertEqual(data['items'][0]['metadata']['split'], 'test')
        self.assertEqual(data['items'][0]['source'], str(self.dataset / 'class_a/same.png'))

    def test_duplicate_ids_and_result_labels_are_rejected(self):
        path = self.dataset / 'dataset.json'
        row = {'id': 'duplicate', 'source': 'class_a/same.png'}
        for items in ([row, row], [{**row, 'fer': {'result_true': 'happy'}}]):
            path.write_text(json.dumps({'items': items}))
            with self.assertRaises(ValueError):
                discover_dataset(path)

    def test_one_model_load_for_all_images_and_metadata_survives_replot(self):
        loader = Mock(return_value=(object(), object()))
        with contextlib.redirect_stdout(io.StringIO()):
            summary = execute_dataset(self.args(), loader=loader, generator=self.fake_generate)
        loader.assert_called_once()
        self.assertEqual(summary['generated_edits'], 6)
        self.assertEqual(summary['images_with_output'], 2)
        self.assertEqual(summary['status'], 'complete')
        with (self.output / 'samples.csv').open() as stream:
            samples = list(csv.DictReader(stream))
        self.assertEqual(len(samples), 6)
        self.assertEqual(len({row['result'] for row in samples}), 6)
        self.assertTrue(all(row['edit_type'] in ('zero_baseline', 'combination') for row in samples))
        self.assertEqual(len((self.output / 'samples.jsonl').read_text().splitlines()), 6)
        _, cases, _ = load_saved_results(self.output / 'results.json')
        self.assertEqual(len({case['dataset_id'] for case in cases}), 2)
        self.assertEqual(cases[0]['generation_seconds'], 0.01)
        self.assertIn('Dataset coverage', (self.output / 'report.html').read_text())

    def test_partial_generation_and_invalid_input_preserve_successes(self):
        (self.dataset / 'corrupt.jpg').write_bytes(b'not an image')
        calls = 0

        def generator(args, pipeline, embeddings, on_case):
            nonlocal calls
            calls += 1
            if calls == 2:
                args.au_requests = args.au_requests[:1]
            self.fake_generate(args, pipeline, embeddings, on_case)
            if calls == 2:
                raise RuntimeError('synthetic failure after first edit')

        with contextlib.redirect_stdout(io.StringIO()):
            summary = execute_dataset(self.args(), loader=Mock(return_value=(None, None)), generator=generator)
        self.assertEqual(summary['generated_edits'], 4)
        self.assertEqual(summary['planned_edits'], 9)
        self.assertEqual(summary['not_generated_edits'], 5)
        self.assertEqual(summary['failed_events'], 2)
        self.assertEqual(summary['status'], 'complete_with_errors')
        manifest = json.loads((self.output / 'generation_manifest.json').read_text())
        self.assertEqual(len(manifest['cases']), 4)
        failures = json.loads((self.output / 'failures.json').read_text())
        self.assertEqual({failure['stage'] for failure in failures}, {'input', 'generation'})

    def test_inspection_does_not_download_models_or_write_output(self):
        loader, preparer, reporter = Mock(), Mock(), Mock()
        with contextlib.redirect_stdout(io.StringIO()):
            result = execute_dataset(self.args('--inspect', '--limit', '1'), loader=loader,
                                     preparer=preparer, reporter=reporter)
        self.assertEqual(result['inspection']['selected_images'], 1)
        for mock in (loader, preparer, reporter):
            mock.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_raw_preprocessing_keeps_original_source_provenance(self):
        raw = self.dataset / 'raw.jpg'
        Image.new('RGB', (100, 150), 'white').save(raw)
        def preparer(items, args, output):
            self.assertEqual(len(items), 1)
            return {items[0]['id']: {'status': 'ok', 'source': str(self.dataset / 'class_a/same.png'),
                                    'background': str(self.dataset / 'class_a/same_bg.png')}}, {'test': True}
        with contextlib.redirect_stdout(io.StringIO()):
            execute_dataset(self.args(), loader=Mock(return_value=(None, None)), generator=self.fake_generate,
                            preparer=preparer)
        result = json.loads((self.output / 'results.json').read_text())
        raw_cases = [case for case in result['cases'] if case['dataset_id'] == 'raw.jpg']
        self.assertEqual(len(raw_cases), 3)
        self.assertEqual(raw_cases[0]['input_source'], str(raw))

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        marker = self.output / 'existing.txt'
        marker.write_text('keep')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            execute_dataset(self.args(), loader=Mock())
        self.assertEqual(marker.read_text(), 'keep')

    def test_all_preprocessing_failures_do_not_load_diffusion(self):
        raw_dir = self.root / 'raw_only'
        raw_dir.mkdir()
        Image.new('RGB', (100, 200)).save(raw_dir / 'x.png')
        loader, reporter = Mock(), Mock()
        def preparer(items, args, output):
            return {item['id']: {'status': 'error', 'error': 'Synthetic no-face fixture'} for item in items}, {}
        with contextlib.redirect_stdout(io.StringIO()):
            summary = execute_dataset(self.args('--dataset', str(raw_dir)), loader=loader,
                                      reporter=reporter, preparer=preparer)
        self.assertEqual(summary['status'], 'no_valid_inputs')
        self.assertEqual(summary['generated_edits'], 0)
        self.assertEqual(summary['failed_events'], 1)
        self.assertTrue((self.output / 'failures.csv').is_file())
        loader.assert_not_called()
        reporter.assert_not_called()

    def test_au_environment_variable_does_not_change_preprocess_python(self):
        _, args = parse_cli(['dataset', str(self.dataset)])
        with patch.dict(os.environ, {'MAGICFACE_AU_PYTHON': '/custom/au/python'}):
            command, _ = build_command(args)
        self.assertEqual(command[command.index('--au_python') + 1], '/custom/au/python')
        self.assertEqual(command[command.index('--preprocess_python') + 1], sys.executable)

    def test_dataset_cli_inspects_without_au_environment_or_gpu(self):
        _, args = parse_cli(['dataset', str(self.dataset), '--inspect', '--limit', '1',
                             '--output', str(self.output), '--au-python', '/missing/python'])
        command, output = build_command(args)
        self.assertIn('--dataset', command)
        self.assertEqual(output, self.output)
        result = subprocess.run([sys.executable, str(ROOT / 'run_magicface.py'), 'dataset', str(self.dataset),
                                 '--inspect', '--limit', '1', '--output', str(self.output),
                                 '--au-python', '/missing/python'],
                                cwd='/', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"prepared_inputs": 1', result.stdout)
        self.assertFalse(self.output.exists())

    def test_portrait_crop_uses_bbox_height(self):
        module = ast.parse((ROOT / 'utils/preprocess.py').read_text())
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'get_bbox')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'get_bbox', 'exec'), namespace)
        box = namespace['get_bbox'](np.array([0, 0, 10, 40]), 0.75)
        np.testing.assert_allclose(box[0], [-25, -10])
        np.testing.assert_allclose(box[2], [35, 50])

    def test_raf_aligned_input_is_resized_without_face_crop(self):
        source, output = self.root / 'aligned_224.jpg', self.root / 'source.png'
        Image.new('RGB', (224, 224), 'purple').save(source)
        item = {'source': str(source), 'metadata': {'dataset_name': 'rafdb', 'image_version': 'aligned'}}
        self.assertTrue(aligned_input(item))
        prepare_aligned_source(source, output)
        with Image.open(output) as image:
            self.assertEqual(image.size, (512, 512))
            self.assertEqual(image.mode, 'RGB')

    def test_non_aligned_and_non_square_inputs_are_not_silently_distorted(self):
        generic = {'metadata': {'dataset_name': 'affectnet', 'image_version': 'aligned'}}
        self.assertFalse(aligned_input(generic))
        source = self.root / 'not_square.jpg'
        Image.new('RGB', (224, 200)).save(source)
        with self.assertRaisesRegex(ValueError, 'must be square'):
            prepare_aligned_source(source, self.root / 'source.png')

    def test_aligned_landmark_template_scales_from_256_to_512(self):
        module = ast.parse((ROOT / 'utils/retrieve_bg.py').read_text())
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'aligned_face_landmarks5')
        template = np.array([[10, 20], [30, 40]], dtype=np.float32)
        namespace = {'np': np, 'datasets_faceswap': SimpleNamespace(mean_face_lm5p_256=template)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'aligned_face_landmarks5', 'exec'), namespace)
        np.testing.assert_allclose(namespace['aligned_face_landmarks5']((512, 512)), template * 2)
        with self.assertRaisesRegex(ValueError, 'must be square'):
            namespace['aligned_face_landmarks5']((512, 480))


if __name__ == '__main__':
    unittest.main()
