"""Visual sheet tests use synthetic images and explicit synthetic scores."""

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from mgface.au import AU_NAMES, edit_metadata
from mgface.cell_sheet import render_cell_sheet, render_cell_sheets


class CellSheetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'train_00023.jpg'
        Image.new('RGB', (224, 224), '#d7c5a8').save(self.source)
        intensities = {name: 0.25 for name in AU_NAMES}
        self.cases = []
        for level in range(5):
            result = self.root / f'edit_{level}.png'
            Image.new('RGB', (512, 512), (120 + level * 20, 80, 70)).save(result)
            requested = {'AU5': float(level), 'AU25': 0.0}
            self.cases.append({
                'source': str(self.source), 'input_source': str(self.source), 'result': str(result),
                'requested_aus': requested, 'cell_id': 'anger_4_25_au5',
                'cell_A': ['AU4', 'AU25'], 'target_au': 'AU5', 'source_emotion': 'anger',
                'seed': 424, 'inference_steps': 50, 'target_threshold_og': 0.5616,
                'acceptance_status': 'not_evaluated_requires_gates_1_2_3_4',
                'source_score': {'status': 'ok', 'intensities': intensities},
                'result_score': {'status': 'ok', 'intensities': {**intensities, 'AU5': 0.3 + level}},
                'dataset_metadata': {'cell_A': ['AU4', 'AU25'], 'target_au': 'AU5',
                                     'n_A': 34, 'n_Ax_exact': 3, 'n_Ax_superset': 23,
                                     'n_Ax_other_emotions': 12, 'x_lift': 0.967,
                                     'x_probe_auroc': 0.9311,
                                     'acceptance_status': 'not_evaluated_requires_gates_1_2_3_4'},
                **edit_metadata(requested),
            })

    def test_progress_and_scored_sheets_render_with_expected_layout(self):
        progress = render_cell_sheet(
            [{key: value for key, value in case.items() if not key.endswith('_score')}
             for case in self.cases], self.root / 'progress.png', stage='generation_pending_scores')
        scored = render_cell_sheet(self.cases, self.root / 'scored.png', stage='libreface_scored')
        for path in (progress, scored):
            with Image.open(path) as image:
                self.assertEqual(image.size, (1850, 990))
                image.verify()
        self.assertNotEqual(progress.read_bytes(), scored.read_bytes())

    def test_group_renderer_outputs_one_sheet_per_cell_source(self):
        files = render_cell_sheets(self.cases + self.cases, self.root / 'groups', stage='scored')
        self.assertEqual(len(files), 1)
        self.assertTrue((self.root / 'groups' / files[0]).is_file())


if __name__ == '__main__':
    unittest.main()
