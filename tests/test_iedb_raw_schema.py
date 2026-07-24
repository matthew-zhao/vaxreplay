from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from vaxreplay.iedb.raw_schema import IedbEpisodeSpec, IedbSnapshotManifest


def _fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'iedb_fictional_history'


class IedbRawSchemaTest(unittest.TestCase):
    def test_rejects_snapshot_path_traversal(self) -> None:
        value = json.loads((_fixture_root() / 'snapshot_decision' / 'snapshot.json').read_text(encoding='utf-8'))
        value['tables'][0]['relative_path'] = '../outside.json'

        with self.assertRaises(ValidationError):
            IedbSnapshotManifest.model_validate_json(json.dumps(value))

    def test_rejects_unordered_or_incomplete_query_receipt(self) -> None:
        value = json.loads((_fixture_root() / 'snapshot_decision' / 'snapshot.json').read_text(encoding='utf-8'))
        value['tables'][0]['source_url'] = 'https://query-api.iedb.org/tcell_search'

        with self.assertRaises(ValidationError):
            IedbSnapshotManifest.model_validate_json(json.dumps(value))

    def test_snapshot_table_url_must_use_the_exact_origin_and_endpoint(self) -> None:
        cases = (
            (
                'https://query-api.iedb.org.evil.example/tcell_search?order=tcell_id',
                'exact source_base_url origin',
            ),
            (
                'https://query-api.iedb.org/bcell_search?order=tcell_id',
                'exactly match its declared endpoint',
            ),
        )
        for source_url, message in cases:
            value = json.loads((_fixture_root() / 'snapshot_decision' / 'snapshot.json').read_text(encoding='utf-8'))
            value['tables'][0]['source_url'] = source_url
            value['tables'][0]['request_sha256'] = hashlib.sha256(source_url.encode()).hexdigest()

            with self.subTest(source_url=source_url), self.assertRaisesRegex(ValidationError, message):
                IedbSnapshotManifest.model_validate_json(json.dumps(value))

    def test_rejects_non_integral_horizon(self) -> None:
        value = json.loads((_fixture_root() / 'spec.json').read_text(encoding='utf-8'))
        value['outcome_as_of'] = '2020-06-29T00:00:01Z'

        with self.assertRaises(ValidationError):
            IedbEpisodeSpec.model_validate_json(json.dumps(value))

    def test_rejects_label_endpoint_outside_evidence(self) -> None:
        value = json.loads((_fixture_root() / 'spec.json').read_text(encoding='utf-8'))
        value['label_endpoint'] = 'mhc_search'

        with self.assertRaises(ValidationError):
            IedbEpisodeSpec.model_validate_json(json.dumps(value))

    def test_rejects_an_unimplemented_ranking_rubric(self) -> None:
        value = json.loads((_fixture_root() / 'spec.json').read_text(encoding='utf-8'))
        value['ranking_rubric_version'] = 'unimplemented-rubric-v99'

        with self.assertRaisesRegex(ValidationError, 'iedb-qualitative-binary-v1'):
            IedbEpisodeSpec.model_validate_json(json.dumps(value))

    def test_real_snapshot_requires_iedb_license_and_rights_review(self) -> None:
        value = json.loads((_fixture_root() / 'snapshot_decision' / 'snapshot.json').read_text(encoding='utf-8'))
        value['synthetic'] = False

        with self.assertRaises(ValidationError):
            IedbSnapshotManifest.model_validate_json(json.dumps(value))


if __name__ == '__main__':
    unittest.main()
