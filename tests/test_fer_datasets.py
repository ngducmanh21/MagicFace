"""Native dataset fixtures contain synthetic images and annotation files only."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from inference_dataset import execute_dataset, parse_args, plan_dataset
from mgface.dataset_inputs import discover_dataset
from mgface.fer_datasets import AFFECTNET_LABELS, RAF_LABELS
from run_magicface import build_command, parse_args as cli_args

ROOT = Path(__file__).resolve().parents[1]


class NativeDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def image(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (100, 100), 'gray').save(path)

    def raf(self):
        root = self.root / 'RAF-DB'
        labels = root / 'EmoLabel/list_patition_label.txt'
        labels.parent.mkdir(parents=True)
        lines = []
        for label in RAF_LABELS:
            name = f'train_{label:05d}.jpg'
            lines.append(f'{name} {label}')
            self.image(root / 'Image/original' / name)
            self.image(root / 'Image/aligned' / name.replace('.jpg', '_aligned.jpg'))
        lines.append('test_0001.jpg 4')
        self.image(root / 'Image/aligned/test_0001_aligned.jpg')
        self.image(root / 'Image/original/test_0001.jpg')
        labels.write_text('\n'.join(lines))
        return root, labels

    def affect_csv(self):
        root = self.root / 'AffectNet'
        base = root / 'Manually_Annotated'
        folder = base / 'file_lists'
        folder.mkdir(parents=True)
        for split, labels in (('training', list(range(11))), ('validation', [4])):
            with (folder / f'{split}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['subDirectory_filePath', 'expression', 'valence', 'arousal'])
                writer.writeheader()
                for label in labels:
                    name = f'1/{split}_{label}.jpg'
                    self.image(base / 'Manually_Annotated_Images' / name)
                    writer.writerow({'subDirectory_filePath': name, 'expression': label, 'valence': 0.25, 'arousal': -2})
        return root

    def affect_npy(self):
        root = self.root / 'AffectNetNPY'
        for name, label in (('train_set', 1), ('val_set', 5)):
            split = root / name
            (split / 'annotations').mkdir(parents=True)
            self.image(split / 'images/000001.jpg')
            np.save(split / 'annotations/000001_exp.npy', np.array(label))
            np.save(split / 'annotations/000001_val.npy', np.array(-0.25))
            np.save(split / 'annotations/000001_aro.npy', np.array(0.75))
        return root

    def test_raf_native_labels_and_aligned_paths(self):
        root, labels = self.raf()
        dataset = discover_dataset(root)
        self.assertEqual(dataset['kind'], 'rafdb')
        self.assertEqual(len(dataset['items']), 8)
        for item in dataset['items'][:7]:
            index = item['metadata']['original_label_id']
            self.assertEqual(item['fer'], {'source_true': RAF_LABELS[index]})
            self.assertTrue(item['source'].endswith('_aligned.jpg'))
            self.assertEqual(item['metadata']['split'], 'train')
        original = discover_dataset(root, split='test', raf_images='original')
        self.assertEqual(original['items'][0]['source'], str(root / 'Image/original/test_0001.jpg'))
        self.assertEqual(original['items'][0]['fer']['source_true'], 'happy')

    def test_raf_aligned_224_path_finds_labels_in_ancestor(self):
        root, _ = self.raf()
        aligned_224 = root / 'Image/aligned_224'
        aligned_224.mkdir()
        for label in RAF_LABELS:
            self.image(aligned_224 / f'train_{label:05d}_aligned.jpg')
        self.image(aligned_224 / 'test_0001_aligned.jpg')
        dataset = discover_dataset(aligned_224, dataset_type='rafdb', split='test')
        self.assertEqual(dataset['kind'], 'rafdb')
        self.assertEqual(dataset['metadata']['image_root'], str(aligned_224))
        self.assertEqual(dataset['metadata']['image_version'], 'aligned')
        self.assertEqual(dataset['items'][0]['source'], str(aligned_224 / 'test_0001_aligned.jpg'))

    def test_raf_aligned_224_accepts_unsuffixed_resized_files(self):
        root, _ = self.raf()
        aligned_224 = root / 'Image/aligned_224'
        aligned_224.mkdir()
        for label in RAF_LABELS:
            self.image(aligned_224 / f'train_{label:05d}.jpg')
        self.image(aligned_224 / 'test_0001.jpg')
        dataset = discover_dataset(aligned_224, dataset_type='rafdb', split='test')
        self.assertEqual(dataset['items'][0]['source'], str(aligned_224 / 'test_0001.jpg'))
        self.assertEqual(dataset['items'][0]['metadata']['image_naming'], 'plain')

    def test_raf_prefers_aligned_suffix_when_both_conventions_exist(self):
        root, _ = self.raf()
        aligned = root / 'Image/aligned'
        # Official suffixed image already exists; add a resized/renamed alternative.
        self.image(aligned / 'test_0001.jpg')
        dataset = discover_dataset(root, dataset_type='rafdb', split='test', raf_images='aligned')
        self.assertEqual(dataset['items'][0]['source'], str(aligned / 'test_0001_aligned.jpg'))
        self.assertEqual(dataset['items'][0]['metadata']['image_naming'], 'aligned_suffix')

    def test_raf_aligned_224_is_auto_detected_from_ancestor_labels(self):
        root, _ = self.raf()
        aligned_224 = root / 'Image/aligned_224'
        aligned_224.mkdir()
        for label in RAF_LABELS:
            self.image(aligned_224 / f'train_{label:05d}_aligned.jpg')
        self.image(aligned_224 / 'test_0001_aligned.jpg')
        dataset = discover_dataset(aligned_224, split='test')
        self.assertEqual(dataset['kind'], 'rafdb')

    def test_raf_zero_based_labels_fail_instead_of_being_remapped(self):
        root, labels = self.raf()
        labels.write_text('train_00001.jpg 0')
        with self.assertRaisesRegex(ValueError, 'labels 1..7'):
            discover_dataset(root)

    def test_split_filter_is_applied_before_limit(self):
        root, _ = self.raf()
        args = parse_args(['--dataset', str(root), '--output_dir', str(self.root / 'out'),
                           '--split', 'test', '--limit', '1', '--inspect'])
        dataset, selected, ready, raw, errors = plan_dataset(args)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['metadata']['split'], 'test')
        self.assertEqual((len(raw), len(ready), len(errors)), (1, 0, 0))
        with self.assertRaises(ValueError):
            discover_dataset(root, split='val')

    def test_affect_csv_nested_manual_release_and_label_mapping(self):
        root = self.affect_csv()
        dataset = discover_dataset(root, split='train')
        self.assertEqual(dataset['kind'], 'affectnet')
        self.assertEqual(len(dataset['items']), 8)
        self.assertEqual(len(dataset['excluded_annotations']), 3)
        for item in dataset['items']:
            index = item['metadata']['original_label_id']
            self.assertEqual(item['fer'], {'source_true': AFFECTNET_LABELS[index]})
            self.assertTrue(Path(item['source']).is_file())
            self.assertEqual(item['metadata']['valence'], 0.25)
            self.assertIsNone(item['metadata']['arousal'])
        seven = discover_dataset(root, split='train', affectnet_classes=7)
        self.assertEqual(len(seven['items']), 7)
        self.assertEqual(len(seven['excluded_annotations']), 4)
        self.assertFalse(any(i['fer']['source_true'] == 'contempt' for i in seven['items']))
        validation = discover_dataset(root, split='val')
        self.assertEqual(len(validation['items']), 1)
        self.assertEqual(validation['items'][0]['fer']['source_true'], 'fear')

    def test_affect_npy_scalars_splits_and_valence_arousal(self):
        root = self.affect_npy()
        dataset = discover_dataset(root, split='val')
        self.assertEqual(len(dataset['items']), 1)
        item = dataset['items'][0]
        self.assertEqual(item['fer'], {'source_true': 'disgust'})
        self.assertEqual(item['metadata']['valence'], -0.25)
        self.assertEqual(item['metadata']['arousal'], 0.75)
        self.assertEqual(item['metadata']['split'], 'val')
        self.assertEqual(dataset['metadata']['layout'], 'npy')

    def test_npy_objects_and_fractional_class_ids_are_excluded(self):
        root = self.affect_npy()
        annotations = root / 'val_set/annotations'
        np.save(annotations / 'bad_exp.npy', np.array({'not': 'numeric'}, dtype=object))
        np.save(annotations / 'fraction_exp.npy', np.array(1.5))
        dataset = discover_dataset(root, split='val')
        self.assertEqual(len(dataset['items']), 1)
        self.assertEqual(len(dataset['excluded_annotations']), 2)

    def test_numeric_class_folders_require_explicit_affectnet_mapping(self):
        root = self.root / 'class_folders'
        self.image(root / 'train/0/a.jpg')
        self.image(root / 'val/6/b.jpg')
        generic = discover_dataset(root)
        self.assertEqual(generic['kind'], 'folder')
        self.assertTrue(all(not item['fer'] for item in generic['items']))
        affect = discover_dataset(root, dataset_type='affectnet', split='val')
        self.assertEqual(len(affect['items']), 1)
        self.assertEqual(affect['items'][0]['fer'], {'source_true': 'anger'})
        self.assertEqual(affect['metadata']['layout'], 'class_folders')

    def test_duplicate_annotations_are_not_counted_twice(self):
        root, labels = self.raf()
        labels.write_text(labels.read_text() + '\ntrain_00001.jpg 1')
        with self.assertRaisesRegex(ValueError, 'Duplicate image'):
            discover_dataset(root)

    def test_explicit_annotation_and_image_root_overrides(self):
        root, labels = self.raf()
        _, args = cli_args(['dataset', str(root), '--config', str(ROOT / 'configs/inference_demo.json'),
                            '--dataset-type', 'rafdb', '--split', 'test',
                            '--annotations', str(labels), '--image-root', str(root / 'Image/original'),
                            '--raf-images', 'original', '--limit', '1', '--inspect'])
        command, _ = build_command(args)
        self.assertIn('--annotations', command)
        parsed = parse_args(command[2:])
        with contextlib.redirect_stdout(io.StringIO()):
            summary = execute_dataset(parsed, loader=Mock())
        self.assertEqual(summary['inspection']['selected_class_counts'], {'happy': 1})
        self.assertEqual(summary['inspection']['selected_split_counts'], {'test': 1})

    def test_source_labels_and_dimensions_are_kept_in_exports(self):
        from verify_results import load_saved_results
        root = self.affect_csv()
        output = self.root / 'export'
        args = parse_args(['--dataset', str(root), '--split', 'val', '--output_dir', str(output),
                           '--au_backend', 'none', '--AU_variation', '1+0', '--evidence', '--figure_formats', 'png'])

        def prepare(items, args, output):
            source, background = output / 'source.png', output / 'background.png'
            Image.new('RGB', (512, 512), 'gray').save(source)
            Image.new('RGB', (512, 512), 'black').save(background)
            return {items[0]['id']: {'status': 'ok', 'source': str(source), 'background': str(background)}}, {'fixture': True}

        def generate(args, pipeline, embeddings, on_case):
            result = Path(args.saved_path) / 'synthetic.png'
            result.parent.mkdir(parents=True)
            Image.new('RGB', (512, 512), 'white').save(result)
            on_case({'source': args.img_path, 'result': str(result), 'background': args.bg_path,
                     'label': 'SYNTHETIC TEST FIXTURE', 'requested_aus': args.au_requests[0],
                     'seed': args.seed, 'inference_steps': args.inference_steps, 'generation_seconds': 0.01}, 0)

        with contextlib.redirect_stdout(io.StringIO()):
            execute_dataset(args, loader=Mock(return_value=(None, None)), preparer=prepare, generator=generate)
        with (output / 'samples.csv').open() as stream:
            sample = next(csv.DictReader(stream))
        self.assertEqual(sample['dataset_name'], 'affectnet')
        self.assertEqual(sample['dataset_split'], 'val')
        self.assertEqual(sample['source_emotion'], 'fear')
        self.assertEqual(sample['source_valence'], '0.25')
        self.assertEqual(sample['source_arousal'], '')
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['source_label_groups'][0]['unique_source_images'], 1)
        self.assertEqual(summary['fer']['source']['status'], 'unavailable')
        self.assertEqual(summary['fer']['result']['status'], 'unavailable')
        self.assertTrue((output / 'figures/fig_source_emotion_counts.png').is_file())
        self.assertTrue((output / 'tables/au_by_source_emotion.csv').is_file())
        _, cases, _ = load_saved_results(output / 'results.json')
        self.assertEqual(cases[0]['source_emotion'], 'fear')
        self.assertEqual(cases[0]['source_valence'], 0.25)
        self.assertNotIn('result_true', cases[0]['fer'])


if __name__ == '__main__':
    unittest.main()
