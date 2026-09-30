"""Synthetic fixtures test evidence arithmetic and exports, not model accuracy."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from mgface.au import AU_NAMES
from mgface.evidence import classification_metrics, summarize_evidence, validate_fer_annotations
from mgface.verification import write_verification_report
from verify_results import load_saved_results


ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.png'
        Image.new('RGB', (24, 24), 'gray').save(self.source)
        before = dict.fromkeys(AU_NAMES, 0.5)
        self.scores = {'estimator': 'SYNTHETIC TEST FIXTURE', 'version': 'test', 'images': {
            str(self.source): {'status': 'ok', 'intensities': before}}}
        self.cases = []
        for i, color in enumerate(('red', 'blue', 'green')):
            path = self.root / f'result_{i}.png'
            Image.new('RGB', (24, 24), color).save(path)
            self.scores['images'][str(path)] = ({'status': 'ok', 'intensities': {**before, 'AU4': 1.5 + i}}
                                               if i < 2 else {'status': 'error', 'error': 'Synthetic missing-face fixture'})
            self.cases.append({'source': str(self.source), 'result': str(path),
                               'requested_aus': {'AU4': i + 1}, 'label': f'Synthetic edit {i}',
                               'fer': {'source_true': 'class_a', 'source_pred': 'class_a',
                                       'result_true': 'class_b' if i < 2 else None,
                                       'result_pred': ('class_b' if i == 0 else 'class_a')}})

    def report(self, **kwargs):
        return write_verification_report(self.cases, self.root / 'report', scores=self.scores,
                                          title='SYNTHETIC TEST DATA - not model evaluation', **kwargs)

    def test_confusion_orientation_and_macro_convention(self):
        result = classification_metrics([('a', 'a'), ('a', 'b'), ('b', 'b')], ['a', 'b', 'c'])
        self.assertEqual(result['confusion_counts'], [[1, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.assertEqual(result['confusion_row_normalized'][0], [0.5, 0.5, 0])
        self.assertEqual(result['confusion_row_normalized'][2], [None, None, None])
        self.assertAlmostEqual(result['accuracy'], 2 / 3)
        self.assertAlmostEqual(result['macro_precision'], 0.75)
        self.assertAlmostEqual(result['macro_recall'], 0.75)
        self.assertAlmostEqual(result['macro_f1'], 2 / 3)

    def test_fer_deduplicates_source_and_excludes_missing_labels(self):
        summary = summarize_evidence(self.report())
        self.assertEqual(summary['fer']['source']['n_evaluated'], 1)
        self.assertEqual(summary['fer']['result']['n_evaluated'], 2)
        self.assertEqual(summary['fer']['result']['n_missing_labels'], 1)
        self.assertEqual(summary['fer']['result']['accuracy'], 0.5)
        self.assertEqual(summary['fer']['source']['accuracy'], 1)
        self.assertEqual(summary['n_scored_pairs'], 2)
        self.assertEqual(summary['n_unscored_pairs'], 1)

    def test_missing_scores_do_not_become_mean_zero_or_error_zero(self):
        summary = summarize_evidence(self.report())
        au = next(row for row in summary['au'] if row['au'] == 'AU4')
        self.assertEqual(au['n_scored_pairs'], 2)
        self.assertEqual(au['source_mean'], 0.5)
        self.assertEqual(au['result_mean'], 2)
        self.assertEqual(au['change_mean'], 1.5)
        self.assertIsNone(au['edited_mae'])
        failed = summary['au_control_response'][2]
        self.assertEqual(failed['n_scored'], 0)
        self.assertIsNone(failed['change_mean'])
        self.assertIsNone(summary['au_control_response'][0]['change_std'])

    def test_conflicting_annotations_rejected_even_across_source_result_roles(self):
        cases = [dict(case) for case in self.cases]
        cases[1]['fer'] = {**cases[1]['fer'], 'source_true': 'different_label'}
        with self.assertRaisesRegex(ValueError, 'Conflicting FER'):
            validate_fer_annotations(cases)
        cases = [self.cases[0], {**self.cases[1], 'result': str(self.source),
                               'fer': {'result_true': 'class_b'}}]
        with self.assertRaisesRegex(ValueError, 'Conflicting FER'):
            validate_fer_annotations(cases)

    def test_requested_emotion_is_never_assumed_to_be_ground_truth(self):
        for case in self.cases:
            case.pop('fer')
            case['target_emotion'] = 'happy'
        summary = summarize_evidence(self.report())
        for role in ('source', 'result'):
            self.assertEqual(summary['fer'][role]['status'], 'unavailable')
            self.assertIsNone(summary['fer'][role]['accuracy'])

    def test_exports_have_consistent_metrics_links_and_checksums(self):
        report = self.report(evidence=True, figure_formats=('png', 'svg', 'pdf'), au_delta_scale=1)
        output = self.root / 'report'
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['fer']['result']['accuracy'], 0.5)
        with (output / 'tables/fer_result_per_class.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[1]['class'], 'class_b')
        self.assertEqual(rows[1]['support'], '2')
        document = (output / 'report.html').read_text()
        for figure in report['evidence']['figures']:
            for path in figure['files']:
                self.assertTrue((output / path).is_file())
                self.assertIn(path, document)
                if path.endswith('.pdf'):
                    self.assertTrue((output / path).read_bytes().startswith(b'%PDF'))
        index = json.loads((output / 'artifact_index.json').read_text())
        for artifact in index:
            self.assertEqual(artifact['sha256'], hashlib.sha256((output / artifact['path']).read_bytes()).hexdigest())
        self.assertEqual(len(report['evidence']['figures']), 5)

    def test_results_can_be_moved_and_replotted_without_estimator(self):
        self.report()
        relocated = self.root / 'moved'
        shutil.move(str(self.root / 'report'), relocated)
        previous, cases, scores = load_saved_results(relocated / 'results.json')
        self.assertEqual(previous['scored_cases'], 2)
        self.assertEqual(cases[0]['fer']['result_true'], 'class_b')
        self.assertTrue(Path(cases[0]['source']).is_file())
        run = subprocess.run([sys.executable, str(ROOT / 'verify_results.py'),
                              '--results_json', str(relocated / 'results.json'),
                              '--output_dir', str(self.root / 'replot'), '--evidence', '--figure_formats', 'png',
                              '--au_python', '/does/not/exist'], cwd='/', capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        summary = json.loads((self.root / 'replot/summary.json').read_text())
        self.assertEqual(summary['n_scored_pairs'], 2)
        self.assertEqual(summary['fer']['result']['accuracy'], 0.5)

    def test_invalid_fer_fields_and_figure_formats_are_rejected(self):
        for fer in ({'true': 'a'}, {'source_true': ''}, {'result_pred': 1}, []):
            with self.subTest(fer=fer), self.assertRaises(ValueError):
                validate_fer_annotations([{**self.cases[0], 'fer': fer}])
        with self.assertRaises(ValueError):
            self.report(evidence=True, figure_formats=('jpg',))

    def test_replot_preserves_sample_counts_for_identical_image_copies(self):
        self.cases = self.cases[:2]
        shutil.copyfile(self.cases[0]['result'], self.cases[1]['result'])
        self.cases[1]['fer'] = dict(self.cases[0]['fer'])
        self.scores['images'][self.cases[1]['result']] = self.scores['images'][self.cases[0]['result']]
        original = self.report()
        previous, cases, scores = load_saved_results(self.root / 'report/results.json')
        replotted = write_verification_report(cases, self.root / 'replot', scores=scores)
        self.assertEqual(summarize_evidence(original)['fer']['result']['n_evaluated'], 2)
        self.assertEqual(summarize_evidence(replotted)['fer']['result']['n_evaluated'], 2)


if __name__ == '__main__':
    unittest.main()
