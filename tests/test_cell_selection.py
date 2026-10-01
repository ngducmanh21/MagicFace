"""Rare-cell selection tests use the audited Anger config with synthetic paths."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from mgface.au import edit_metadata
from mgface.cell_selection import apply_cell_selection


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / 'configs/anger_cells_magicface.json'


class CellSelectionTests(unittest.TestCase):
    def dataset(self):
        document = json.loads(SELECTION.read_text())
        sources = sorted({source for cell in document['cells'] for source in cell['sources_aligned_S1']})
        return {'kind': 'rafdb', 'metadata': {}, 'excluded_annotations': [],
                'items': [{'id': f'rafdb:train:{source}.jpg', 'source': f'/data/{source}.jpg',
                           'background': None, 'fer': {'source_true': 'anger'},
                           'metadata': {'dataset_name': 'rafdb', 'split': 'train'}} for source in sources]}

    @staticmethod
    def requests():
        return ([{'AU5': 0.0, 'AU25': 0.0}] +
                [{'AU5': float(level), 'AU25': 0.0} for level in range(1, 5)] +
                [{'AU5': 0.0, 'AU25': float(level)} for level in range(1, 5)])

    def test_audited_anger_selection_has_only_rule_approved_pairs(self):
        result = apply_cell_selection(self.dataset(), SELECTION, self.requests())
        self.assertEqual(len(result['items']), 38)
        self.assertEqual(result['metadata']['cell_selection']['unique_sources'], 32)
        self.assertEqual(result['metadata']['cell_selection']['cell_count'], 5)
        self.assertEqual(result['metadata']['cell_selection']['pre_selection_images'], 32)
        self.assertEqual({item['fer']['source_true'] for item in result['items']}, {'anger'})
        self.assertEqual({item['metadata']['target_au'] for item in result['items']}, {'AU5', 'AU25'})
        for item in result['items']:
            requests = item['generation_requests']
            self.assertEqual(len(requests), 5)
            self.assertEqual(sum(edit_metadata(request)['edit_type'] == 'zero_baseline'
                                 for request in requests), 1)
            self.assertTrue(all(sum(value != 0 for value in request.values()) <= 1 for request in requests))
            target = item['metadata']['target_au']
            self.assertEqual(sorted(request[target] for request in requests if request[target]), [1, 2, 3, 4])
            self.assertEqual(item['metadata']['acceptance_status'],
                             'not_evaluated_requires_gates_1_2_3_4')

    def test_missing_or_non_anger_sources_are_rejected(self):
        dataset = self.dataset()
        dataset['items'].pop()
        with self.assertRaisesRegex(ValueError, 'missing from the dataset'):
            apply_cell_selection(dataset, SELECTION, self.requests())
        dataset = self.dataset()
        dataset['items'][0]['fer']['source_true'] = 'sad'
        with self.assertRaisesRegex(ValueError, 'not labelled anger'):
            apply_cell_selection(dataset, SELECTION, self.requests())

    def test_unapproved_cell_status_or_gate_is_rejected(self):
        document = json.loads(SELECTION.read_text())
        for key, value in (('status', 'deferred'), ('R10_drawable', False)):
            changed = copy.deepcopy(document)
            changed['cells'][0][key] = value
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'cells.json'
                path.write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'not approved'):
                    apply_cell_selection(self.dataset(), path, self.requests())

    def test_pinned_checksums_and_scope_counts_reject_drift(self):
        expected = 'd5db8048354ea53a2edd7c51caee0befdffc52115ee77a3c23d2b00cbca052da'
        source = '7ce3342c42d391d9a534ccfda6877eccb246c14822417f3ce82ec8c8495fc741'
        result = apply_cell_selection(
            self.dataset(), SELECTION, self.requests(), expected, source,
            {'cell_count': 5, 'cell_source_pairs': 38, 'unique_sources': 32})
        audit = result['metadata']['cell_selection']
        self.assertFalse(audit['random_fallback'])
        self.assertEqual(audit['allowed_emotions'], ['anger'])
        self.assertEqual(audit['allowed_targets'], ['AU25', 'AU5'])
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            apply_cell_selection(self.dataset(), SELECTION, self.requests(), '0' * 64)
        with self.assertRaisesRegex(ValueError, 'scope count mismatch'):
            apply_cell_selection(self.dataset(), SELECTION, self.requests(), expected, source,
                                 {'cell_count': 6})


if __name__ == '__main__':
    unittest.main()
