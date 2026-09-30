"""CLI tests exercise config conversion and real CPU-only report commands."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from inference import parse_args as parse_inference_args
from run_magicface import ROOT, build_command, main, parse_args


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='magicface CLI ')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config = json.loads((ROOT / 'configs/inference_demo.json').read_text())
        self.config.update(image=str(ROOT / 'test_images/00381.png'),
                           background=str(ROOT / 'test_images/00381_bg.png'), output_dir='output with spaces')
        self.path = self.directory / 'experiment.json'
        self.path.write_text(json.dumps(self.config))

    def test_config_paths_and_negative_fractional_values_reach_inference(self):
        self.config['variations'] = [[-2, 0.0000001], [2.25, -1.5]]
        self.path.write_text(json.dumps(self.config))
        _, args = parse_args(['run', '--config', str(self.path), '--no-au'])
        command, output = build_command(args)
        parsed = parse_inference_args(command[2:])
        self.assertEqual(parsed.au_requests, [{'AU4': -2, 'AU1': 1e-7}, {'AU4': 2.25, 'AU1': -1.5}])
        self.assertEqual(output, self.directory / 'output with spaces')
        self.assertEqual(parsed.verification_dir, str(output))
        self.assertEqual(parsed.au_backend, 'none')
        self.assertTrue(parsed.evidence)

    def test_bad_config_is_rejected_without_side_effects(self):
        for change in ({'aus': ['AU4', 'AU4']}, {'variations': [[1]]}, {'seed': True},
                       {'figure_formats': ['gif']}, {'typo': 1}, {'variations': [[float('nan'), 0]]}):
            self.path.write_text(json.dumps({**self.config, **change}))
            _, args = parse_args(['run', '--config', str(self.path)])
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_command(args)
        self.assertFalse((self.directory / 'output with spaces').exists())

    def test_dry_run_never_executes_subprocess_or_creates_output(self):
        with patch('run_magicface.subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()):
            code = main(['run', '--config', str(self.path), '--dry-run'])
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertFalse((self.directory / 'output with spaces').exists())

    def test_no_cuda_returns_actionable_error_before_download(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        stderr = io.StringIO()
        with patch.dict(sys.modules, {'torch': fake_torch}), patch('run_magicface.subprocess.run') as run, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = main(['run', '--config', str(self.path), '--no-au'])
        self.assertEqual(code, 2)
        self.assertIn('CUDA chua san sang', stderr.getvalue())
        run.assert_not_called()

    def test_missing_au_environment_stops_before_inference(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with patch.dict(sys.modules, {'torch': fake_torch}), \
                patch('run_magicface.subprocess.run', side_effect=FileNotFoundError('missing AU Python')) as run, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(['run', '--config', str(self.path), '--au-python', '/missing/python'])
        self.assertEqual(code, 2)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], ['/missing/python', '-c', 'import libreface'])

    def test_missing_au_message_offers_generation_without_scores(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        stderr = io.StringIO()
        with patch.dict(sys.modules, {'torch': fake_torch}), \
                patch('run_magicface.subprocess.run', side_effect=FileNotFoundError('missing AU Python')), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            code = main(['run', '--config', str(self.path), '--au-python', '/missing/python'])
        self.assertEqual(code, 2)
        self.assertIn('--no-au', stderr.getvalue())
        self.assertIn('Generation chua bat dau', stderr.getvalue())

    def test_preview_and_cached_report_work_outside_repo_without_gpu(self):
        output = self.directory / 'preview'
        command = [sys.executable, str(ROOT / 'run_magicface.py')]
        preview = subprocess.run(command + ['preview', '--output', str(output)], cwd='/', capture_output=True, text=True)
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertTrue((output / 'report.html').is_file())
        cached = subprocess.run(command + ['report', '--results', str(output / 'results.json'),
                                '--output', str(self.directory / 'cached'), '--formats', 'png',
                                '--au-python', '/missing/python'], cwd='/', capture_output=True, text=True)
        self.assertEqual(cached.returncode, 0, cached.stderr)
        summary = json.loads((self.directory / 'cached/summary.json').read_text())
        self.assertEqual(summary['n_cases'], 3)
        self.assertEqual(summary['n_scored_pairs'], 0)

    def test_child_failure_is_returned_to_caller(self):
        with patch('run_magicface.subprocess.run', return_value=SimpleNamespace(returncode=7)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['preview', '--output', str(self.directory / 'out')]), 7)


if __name__ == '__main__':
    unittest.main()
