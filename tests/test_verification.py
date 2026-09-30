"""Verification tests use synthetic images and explicit synthetic score fixtures."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from inference import parse_args
from mgface.au import AU_NAMES, normalize_intensities, parse_au_request
from mgface.verification import score_with_libreface, write_verification_report


ROOT = Path(__file__).resolve().parents[1]


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.result = self.root / 'source.png', self.root / 'result.png'
        Image.new('RGB', (64, 48), 'gray').save(self.source)
        Image.new('RGB', (48, 64), 'white').save(self.result)
        self.case = {'source': str(self.source), 'result': str(self.result),
                     'requested_aus': {'AU4': 2, 'AU1': 0}, 'label': 'Synthetic test', 'seed': 424}
        before = dict.fromkeys(AU_NAMES, 1.0)
        after = {**before, 'AU4': 2.0, 'AU6': 1.3}
        self.scores = {'estimator': 'synthetic test fixture', 'version': 'test', 'images': {
            str(self.source): {'status': 'ok', 'intensities': before},
            str(self.result): {'status': 'ok', 'intensities': after},
        }}

    def report(self, **kwargs):
        return write_verification_report([self.case], self.root / 'report', **kwargs)

    def test_request_order_fractional_and_negative(self):
        self.assertEqual(parse_au_request('AU4+AU1', '-2+0.5'), {'AU4': -2, 'AU1': 0.5})

    def test_invalid_requests_rejected(self):
        for names, values in [('AU4+AU1', '2'), ('AU4+AU4', '1+2'),
                              ('AU99', '1'), ('AU1', 'nan'), ('AU1', 'inf'), ('', '')]:
            with self.subTest(names=names, values=values), self.assertRaises(ValueError):
                parse_au_request(names, values)

    def test_only_complete_finite_intensity_scores_are_accepted(self):
        raw = {f'au_{name[2:]}_intensity': 0.5 for name in AU_NAMES}
        self.assertEqual(normalize_intensities(raw)['AU4'], 0.5)
        with self.assertRaises(KeyError):
            normalize_intensities({f'au_{name[2:]}': 1 for name in AU_NAMES})
        for invalid in (float('nan'), -0.1, 5.1):
            with self.assertRaises(ValueError):
                normalize_intensities({**raw, 'au_4_intensity': invalid})

    def test_inference_accepts_multiple_variations_and_validates_before_models(self):
        args = ['--img_path', str(self.source), '--bg_path', str(self.result),
                '--au_test', 'AU4+AU1', '--AU_variation=2+1', '--AU_variation=-2+-1']
        parsed = parse_args(args)
        self.assertEqual(parsed.au_requests, [{'AU4': 2, 'AU1': 1}, {'AU4': -2, 'AU1': -1}])
        self.assertEqual(parsed.seed, 424)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(args + ['--au_delta_scale', 'nan'])

    def test_no_scale_does_not_fabricate_error_metric(self):
        report = self.report(scores=self.scores)
        case = report['cases'][0]
        au4 = next(row for row in case['au_rows'] if row['au'] == 'AU4')
        self.assertEqual(au4['measured_delta'], 1.0)
        self.assertIsNone(au4['absolute_error'])
        self.assertIsNone(case['edited_au_mae'])
        self.assertAlmostEqual(case['unchanged_au_drift'], 0.3 / 11)
        with (self.root / 'report/scores.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[0]['expected_delta'], '')
        self.assertEqual(report['estimator_version'], 'test')

    def test_calibrated_error_and_out_of_range_target(self):
        report = self.report(scores=self.scores, au_delta_scale=0.5)
        self.assertEqual(report['cases'][0]['edited_au_mae'], 0)
        report = self.report(scores=self.scores, au_delta_scale=3)
        row = next(r for r in report['cases'][0]['au_rows'] if r['au'] == 'AU4')
        self.assertEqual(row['expected_delta'], 6)
        self.assertEqual(row['absolute_error'], 5)
        self.assertTrue(row['target_out_of_range'])
        for invalid in (0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.report(au_delta_scale=invalid)

    def test_failed_estimation_is_not_a_zero_measurement(self):
        self.scores['images'][str(self.result)] = {'status': 'error', 'error': 'No face landmarks'}
        self.case['label'] = '<script>alert(1)</script>'
        report = self.report(scores=self.scores, au_delta_scale=1)
        case = report['cases'][0]
        self.assertEqual(report['scored_cases'], 0)
        self.assertIsNone(case['unchanged_au_drift'])
        self.assertIsNone(case['au_rows'][0]['result_intensity'])
        self.assertEqual(case['au_rows'][0]['source_intensity'], 1)
        document = (self.root / 'report/report.html').read_text()
        self.assertIn('No face landmarks', document)
        self.assertNotIn('<script>alert(1)</script>', document)
        self.assertIn('&lt;script&gt;', document)

    def test_invalid_score_bundle_is_visible_failure(self):
        self.scores['images'][str(self.source)]['intensities'].pop('AU4')
        report = self.report(scores=self.scores)
        self.assertEqual(report['scored_cases'], 0)
        self.assertIn('Invalid AU scores', report['cases'][0]['source_score']['error'])

    def test_pagination_assets_and_csv_labels(self):
        self.case['label'] = '=synthetic test label'
        report = write_verification_report([self.case] * 5, self.root / 'report')
        self.assertEqual(len(report['grids']), 2)
        self.assertEqual(len(list((self.root / 'report/images').glob('*.png'))), 2)
        for name in report['grids']:
            with Image.open(self.root / 'report' / name) as image:
                self.assertEqual(image.width, 1440)
                image.verify()
        with (self.root / 'report/scores.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 60)
        self.assertTrue(rows[0]['label'].startswith("'="))

    def test_cli_resolves_images_relative_to_manifest(self):
        case = {**self.case, 'source': 'source.png', 'result': 'result.png'}
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps({'cases': [case]}))
        run = subprocess.run([sys.executable, str(ROOT / 'verify_results.py'),
                              '--manifest', str(manifest), '--output_dir', str(self.root / 'cli'),
                              '--au_backend', 'none'], cwd='/', capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads((self.root / 'cli/results.json').read_text())
        self.assertEqual(report['cases'][0]['source'], str(self.source))
        self.assertEqual(report['scored_cases'], 0)

    def test_worker_protocol_with_explicit_synthetic_estimator(self):
        # No external estimator is downloaded. This fixture verifies isolation,
        # alignment-before-scoring, intensity extraction and per-image failures.
        fake = self.root / 'fake'
        fake.mkdir()
        (fake / 'libreface.py').write_text('''
def get_aligned_image(path, temp_dir):
    if path.endswith('result.png'):
        raise RuntimeError('Synthetic no-face fixture')
    return path + '.aligned', None, None

def get_au_intensities_and_detect_aus(path, device, weights_download_dir):
    assert path.endswith('.aligned')
    aus = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)
    return {}, {f'au_{au}_intensity': 1.25 for au in aus}
''')
        info = fake / 'libreface-0.0.dist-info'
        info.mkdir()
        (info / 'METADATA').write_text('Name: libreface\nVersion: 0.0\n')
        with patch.dict(os.environ, {'PYTHONPATH': str(fake)}):
            scores = score_with_libreface([self.source, self.result, self.source])
        self.assertEqual(len(scores['images']), 2)
        self.assertEqual(scores['images'][str(self.source)]['intensities']['AU4'], 1.25)
        self.assertEqual(scores['images'][str(self.result)]['status'], 'error')
        self.assertEqual(scores['version'], '0.0')

    def test_missing_au_python_produces_inspectable_failure(self):
        scores = score_with_libreface([self.source], python=str(self.root / 'missing-python'))
        self.assertEqual(scores['images'][str(self.source)]['status'], 'error')


if __name__ == '__main__':
    unittest.main()
