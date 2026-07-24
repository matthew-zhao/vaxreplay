from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from vaxreplay.baselines import oracle_submission
from vaxreplay.bundle import (
    BundleIntegrityError,
    EpisodeBundle,
    canonical_json_bytes,
    jsonl_text,
    ranking_labels_commitment,
)
from vaxreplay.case_schema import RANKING_REWARD_VERSION, LabelCommitmentScheme
from vaxreplay.environment import VaxReplayEnvironment
from vaxreplay.iedb.adapter import (
    IedbAdapterError,
    audit_episode,
    build_episode,
    export_public_episode,
    load_snapshot,
    load_snapshot_history,
    normalize_assay,
)
from vaxreplay.iedb.raw_schema import (
    IedbEndpoint,
    IedbEpisodeSpec,
    IedbPrivateAudit,
    IedbSnapshotManifest,
    QualitativePolarity,
)
from vaxreplay.prompt import build_episode_prompt
from vaxreplay.scoring import make_submission_evaluator


def _fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'iedb_fictional_history'


def _spec() -> IedbEpisodeSpec:
    return IedbEpisodeSpec.model_validate_json((_fixture_root() / 'spec.json').read_bytes())


def _snapshot_roots() -> list[Path]:
    return [
        _fixture_root() / 'snapshot_decision',
        _fixture_root() / 'snapshot_outcome',
    ]


class IedbAdapterTest(unittest.TestCase):
    def test_builds_a_bound_time_frozen_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / 'episode'
            bundle = build_episode(spec=_spec(), snapshot_roots=_snapshot_roots(), output_root=output)

            self.assertEqual(bundle.manifest.candidate_ids, ['cand-alpha', 'cand-beta'])
            self.assertEqual(len(bundle.visible_evidence), 4)
            self.assertEqual(len(bundle.evidence), 4)
            self.assertIsNotNone(bundle.manifest.source_provenance)
            self.assertEqual(
                bundle.manifest.label_commitment_scheme,
                LabelCommitmentScheme.HMAC_SHA256,
            )
            self.assertEqual(bundle.manifest.reward_version, RANKING_REWARD_VERSION)

            prompt = build_episode_prompt(bundle)
            self.assertIn('Fictional early cellular study', prompt)
            self.assertNotIn('Fictional held-out cellular validation', prompt)
            self.assertNotIn('Positive-High', prompt)
            self.assertNotIn('IEDB_EPITOPE:900001', prompt)
            self.assertTrue(all('?' not in evidence.provenance_url for evidence in bundle.evidence))
            self.assertTrue(all('SHA-256' not in evidence.derivation for evidence in bundle.evidence))

            labels = bundle.private_labels
            assert labels is not None
            outcomes = {outcome.candidate_id: outcome.outcome for outcome in labels.outcomes}
            self.assertEqual(outcomes, {'cand-alpha': 1, 'cand-beta': 0})
            assert bundle.ranking_labels is not None
            self.assertEqual(
                {label.candidate_id: label.relevance_grade for label in bundle.ranking_labels},
                {'cand-alpha': 1, 'cand-beta': 0},
            )
            assessments = {
                (assessment.candidate_id, assessment.dimension): assessment.conclusion.value
                for assessment in labels.assessments_gold
            }
            self.assertEqual(assessments[('cand-alpha', 'prior_t_cell_response')], 'favorable')
            self.assertEqual(assessments[('cand-alpha', 'prior_b_cell_response')], 'concern')
            self.assertEqual(assessments[('cand-beta', 'prior_t_cell_response')], 'concern')
            self.assertEqual(assessments[('cand-beta', 'prior_b_cell_response')], 'favorable')

            evaluator = make_submission_evaluator(bundle)
            oracle = oracle_submission(bundle)
            score = evaluator.score(oracle)
            self.assertEqual(score.reward, 1.0)
            self.assertEqual(score.ndcg_at_k, 1.0)
            self.assertEqual(score.pairwise_concordance, 1.0)
            self.assertEqual(score.top_k_utility, 1.0)
            self.assertEqual(score.ranking_reward, 1.0)
            step = VaxReplayEnvironment(bundle, evaluator).step(oracle.model_dump_json())
            self.assertEqual(step.reporting_reward, 1.0)
            self.assertEqual(step.metrics['ranking_reward'], 1.0)
            self.assertEqual(audit_episode(output)['candidate_count'], 2)

    def test_first_seen_snapshot_controls_availability_not_publication_year(self) -> None:
        spec = _spec()
        history = load_snapshot_history(
            _snapshot_roots(),
            required_endpoints=spec.evidence_endpoints,
            outcome_as_of=spec.outcome_as_of,
        )
        outcome_state = history.states['fictional-iedb-20200629'][IedbEndpoint.TCELL]
        late_assay = outcome_state['IEDB_ASSAY:700101']

        self.assertEqual(late_assay.assay.reference_dates, ['2009'])
        self.assertGreater(late_assay.first_seen_at, spec.decision_at)

    def test_post_cutoff_correction_is_not_misclassified_as_a_new_assay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / 'episode'
            build_episode(spec=_spec(), snapshot_roots=_snapshot_roots(), output_root=output)
            audit = IedbPrivateAudit.model_validate_json((output / 'private' / 'iedb_audit.json').read_bytes())

            source_ids = {source.assay_iri for outcome in audit.outcomes for source in outcome.sources}
            self.assertEqual(source_ids, {'IEDB_ASSAY:700101', 'IEDB_ASSAY:700102'})
            self.assertNotIn('IEDB_ASSAY:700001', source_ids)

    def test_public_export_excludes_private_labels_keys_and_future_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode = root / 'episode'
            public = root / 'public'
            build_episode(spec=_spec(), snapshot_roots=_snapshot_roots(), output_root=episode)
            metadata_canary = 'PRIVATE ORGANIZER METADATA CANARY'
            (episode / 'DATASET_CARD.md').write_text(metadata_canary, encoding='utf-8')
            (episode / 'ADAPTER_REPORT.json').write_text(metadata_canary, encoding='utf-8')

            public_bundle = export_public_episode(episode, public)

            self.assertFalse((public / 'private').exists())
            self.assertFalse((public / 'ranking_labels.jsonl').exists())
            self.assertEqual(len(public_bundle.evidence), len(public_bundle.visible_evidence))
            exported_text = '\n'.join(path.read_text(encoding='utf-8') for path in public.iterdir() if path.is_file())
            self.assertNotIn('Fictional held-out cellular validation', exported_text)
            self.assertNotIn('IEDB_REFERENCE:899999', exported_text)
            self.assertNotIn('IEDB_ASSAY:700101', exported_text)
            self.assertNotIn('IEDB_EPITOPE:900001', exported_text)
            self.assertNotIn('positive_candidate_count', exported_text)
            self.assertNotIn('negative_candidate_count', exported_text)
            self.assertNotIn(metadata_canary, exported_text)

    def test_public_export_rejects_symlinked_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode = root / 'episode'
            build_episode(spec=_spec(), snapshot_roots=_snapshot_roots(), output_root=episode)
            dataset_card = episode / 'DATASET_CARD.md'
            dataset_card.unlink()
            dataset_card.symlink_to(root / 'private-organizer-notes.txt')

            with self.assertRaisesRegex(IedbAdapterError, 'symbolic link'):
                export_public_episode(episode, root / 'public')

    def test_public_export_blocks_real_iedb_rows_until_linkage_risk_is_solved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode = root / 'episode'
            bundle = build_episode(spec=_spec(), snapshot_roots=_snapshot_roots(), output_root=episode)
            real_manifest = bundle.manifest.model_copy(update={'synthetic': False})
            (episode / 'manifest.json').write_bytes(canonical_json_bytes(real_manifest) + b'\n')

            with self.assertRaisesRegex(IedbAdapterError, 'linkage-risk evaluation'):
                export_public_episode(episode, root / 'public')

    def test_public_export_canonicalizes_duplicate_key_covert_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode = root / 'episode'
            public = root / 'public'
            build_episode(spec=_spec(), snapshot_roots=_snapshot_roots(), output_root=episode)
            canary = 'PRIVATE FUTURE LABEL CANARY'
            for relative_path, duplicate_field in (
                ('manifest.json', 'episode_id'),
                ('candidates.jsonl', 'candidate_id'),
                ('evidence.jsonl', 'body'),
            ):
                path = episode / relative_path
                original = path.read_text(encoding='utf-8')
                path.write_text(
                    original.replace('{', f'{{"{duplicate_field}":"{canary}",', 1),
                    encoding='utf-8',
                )

            exported = export_public_episode(episode, public)
            exported_text = '\n'.join(path.read_text(encoding='utf-8') for path in public.iterdir() if path.is_file())

            self.assertEqual(exported.manifest.episode_id, 'iedb-fictional-cohort-001')
            self.assertNotIn(canary, exported_text)

    def test_public_evidence_omits_secret_query_filters_and_row_lookup_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            secret_structure = 'IEDB_EPITOPE:PRIVATE_MAPPING_CANARY'
            snapshots = []
            for snapshot_name in ('snapshot_decision', 'snapshot_outcome'):
                copied = root / snapshot_name
                shutil.copytree(_fixture_root() / snapshot_name, copied)
                manifest_path = copied / 'snapshot.json'
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                for table in manifest['tables']:
                    source_url = f'{table["source_url"]}&structure_iri=eq.{secret_structure}'
                    table['source_url'] = source_url
                    table['request_sha256'] = hashlib.sha256(source_url.encode()).hexdigest()
                manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
                snapshots.append(copied)

            private = root / 'private'
            public = root / 'public'
            bundle = build_episode(spec=_spec(), snapshot_roots=snapshots, output_root=private)
            export_public_episode(private, public)
            exported_text = '\n'.join(path.read_text(encoding='utf-8') for path in public.iterdir() if path.is_file())

            self.assertNotIn(secret_structure, exported_text)
            self.assertTrue(all('?' not in record.provenance_url for record in bundle.evidence))
            self.assertTrue(all('normalized row SHA-256' not in record.derivation for record in bundle.evidence))

    def test_evidence_ids_are_keyed_against_offline_row_enumeration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = build_episode(
                spec=_spec(),
                snapshot_roots=_snapshot_roots(),
                output_root=root / 'first',
                label_commitment_key=b'a' * 32,
            )
            second = build_episode(
                spec=_spec(),
                snapshot_roots=_snapshot_roots(),
                output_root=root / 'second',
                label_commitment_key=b'b' * 32,
            )

            self.assertEqual(
                [record.body for record in first.evidence],
                [record.body for record in second.evidence],
            )
            self.assertNotEqual(
                {record.evidence_id for record in first.evidence},
                {record.evidence_id for record in second.evidence},
            )

    def test_rejects_tampered_raw_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory) / 'snapshot'
            shutil.copytree(_fixture_root() / 'snapshot_decision', copied)
            table_path = copied / 'tcell_search.json'
            table_path.write_text(
                table_path.read_text(encoding='utf-8').replace('Positive', 'Negative', 1),
                encoding='utf-8',
            )

            with self.assertRaisesRegex(IedbAdapterError, 'file hash mismatch'):
                load_snapshot(copied)

    def test_rejects_snapshot_file_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            copied = root / 'snapshot'
            shutil.copytree(_fixture_root() / 'snapshot_decision', copied)
            linked_table = copied / 'tcell_search.json'
            linked_table.unlink()
            linked_table.symlink_to((_fixture_root() / 'snapshot_decision' / 'tcell_search.json').resolve())

            with self.assertRaisesRegex(IedbAdapterError, 'symbolic links'):
                load_snapshot(copied)

    def test_loads_jsonl_snapshot_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory) / 'snapshot'
            shutil.copytree(_fixture_root() / 'snapshot_decision', copied)
            json_path = copied / 'tcell_search.json'
            rows = json.loads(json_path.read_text(encoding='utf-8'))
            jsonl_path = copied / 'tcell_search.jsonl'
            jsonl_path.write_text(
                ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows),
                encoding='utf-8',
            )
            json_path.unlink()
            manifest_path = copied / 'snapshot.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            table = manifest['tables'][0]
            table['relative_path'] = 'tcell_search.jsonl'
            table['format'] = 'jsonl'
            table['sha256'] = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
            table['byte_count'] = jsonl_path.stat().st_size
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

            snapshot = load_snapshot(copied)

            self.assertEqual(len(snapshot.rows_by_endpoint[IedbEndpoint.TCELL]), 2)

    def test_rejects_snapshot_when_api_metrics_change_during_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory) / 'snapshot'
            shutil.copytree(_fixture_root() / 'snapshot_decision', copied)
            after_path = copied / 'api_metrics_after.json'
            after = json.loads(after_path.read_text(encoding='utf-8'))
            after[0]['record_count'] = 3
            after_path.write_text(json.dumps(after), encoding='utf-8')
            manifest_path = copied / 'snapshot.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest['api_metrics_after_sha256'] = hashlib.sha256(after_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

            with self.assertRaisesRegex(IedbAdapterError, 'changed during capture'):
                load_snapshot(copied)

    def test_rejects_tampered_private_hmac_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / 'episode'
            build_episode(spec=_spec(), snapshot_roots=_snapshot_roots(), output_root=output)
            (output / 'private' / 'label_commitment_key.hex').write_text('00' * 32 + '\n', encoding='ascii')

            with self.assertRaisesRegex(BundleIntegrityError, 'key does not match'):
                EpisodeBundle.load(output, include_private=True)

    def test_private_episode_spec_is_retained_and_bound_for_reaudit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / 'episode'
            build_episode(spec=_spec(), snapshot_roots=_snapshot_roots(), output_root=output)
            spec_path = output / 'private' / 'iedb_episode_spec.json'
            retained = IedbEpisodeSpec.model_validate_json(spec_path.read_bytes())
            self.assertEqual(retained, _spec())

            tampered = retained.model_copy(update={'episode_id': 'tampered-episode'})
            spec_path.write_bytes(canonical_json_bytes(tampered))
            with self.assertRaisesRegex(IedbAdapterError, 'spec does not match'):
                audit_episode(output)

    def test_rejects_an_unimplemented_ranking_rubric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            revised_spec = _spec().model_copy(update={'ranking_rubric_version': 'fictional-revised-v2'})
            with self.assertRaisesRegex(IedbAdapterError, 'unsupported IEDB ranking rubric'):
                build_episode(
                    spec=revised_spec,
                    snapshot_roots=_snapshot_roots(),
                    output_root=Path(temporary_directory) / 'revised',
                )

    def test_audit_rejects_committed_grades_that_violate_the_binary_rubric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / 'episode'
            bundle = build_episode(spec=_spec(), snapshot_roots=_snapshot_roots(), output_root=output)
            assert bundle.private_labels is not None
            assert bundle.ranking_labels is not None
            assert bundle.label_commitment_key is not None
            swapped_grades = tuple(
                label.model_copy(update={'relevance_grade': 1 - label.relevance_grade})
                for label in bundle.ranking_labels
                if label.relevance_grade is not None
            )
            commitment = ranking_labels_commitment(
                bundle.private_labels,
                swapped_grades,
                bundle.manifest.label_commitment_scheme,
                key=bundle.label_commitment_key,
            )
            tampered_manifest = bundle.manifest.model_copy(update={'labels_sha256': commitment})
            (output / 'manifest.json').write_bytes(canonical_json_bytes(tampered_manifest) + b'\n')
            (output / 'private' / 'ranking_labels.jsonl').write_text(
                jsonl_text(swapped_grades),
                encoding='utf-8',
            )

            with self.assertRaisesRegex(IedbAdapterError, 'binary outcome rubric'):
                audit_episode(output)

    def test_requires_exact_boundary_snapshots(self) -> None:
        shifted = _spec().model_copy(update={'decision_at': _spec().decision_at.replace(day=2)})
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(IedbAdapterError, 'exactly equal'):
                build_episode(
                    spec=shifted,
                    snapshot_roots=_snapshot_roots(),
                    output_root=Path(temporary_directory) / 'episode',
                )

    def test_rejects_query_drift_between_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory) / 'outcome'
            shutil.copytree(_fixture_root() / 'snapshot_outcome', copied)
            manifest_path = copied / 'snapshot.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            url = 'https://query-api.iedb.org/tcell_search?order=tcell_id&reference_id=eq.1'
            manifest['tables'][0]['source_url'] = url
            manifest['tables'][0]['request_sha256'] = hashlib.sha256(url.encode()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

            with self.assertRaisesRegex(IedbAdapterError, 'exact same canonical query'):
                load_snapshot_history(
                    [_snapshot_roots()[0], copied],
                    required_endpoints=_spec().evidence_endpoints,
                    outcome_as_of=_spec().outcome_as_of,
                )

    def test_snapshot_schema_rejects_torn_table_builds(self) -> None:
        value = json.loads((_fixture_root() / 'snapshot_decision' / 'snapshot.json').read_text(encoding='utf-8'))
        value['tables'][0]['source_build_at'] = '2020-01-02T00:00:00Z'

        with self.assertRaises(ValidationError):
            IedbSnapshotManifest.model_validate_json(json.dumps(value))

    def test_normalizes_positive_variants_without_using_quantitative_thresholds(self) -> None:
        assay = normalize_assay(
            IedbEndpoint.TCELL,
            {
                'tcell_iri': 'IEDB_ASSAY:1',
                'structure_iri': 'IEDB_EPITOPE:1',
                'reference_iri': 'IEDB_REFERENCE:1',
                'qualitative_measure': 'Positive-Low',
                'assay_names': '<strong>ELISPOT</strong>',
                'quantitative_measure': 0,
            },
        )

        self.assertEqual(assay.polarity, QualitativePolarity.POSITIVE)
        self.assertEqual(assay.assay_names, ['ELISPOT'])

        unknown = normalize_assay(
            IedbEndpoint.TCELL,
            {
                'tcell_iri': 'IEDB_ASSAY:2',
                'structure_iri': 'IEDB_EPITOPE:2',
                'reference_iri': 'IEDB_REFERENCE:2',
                'qualitative_measure': 'Positive-looking',
            },
        )
        self.assertEqual(unknown.polarity, QualitativePolarity.UNKNOWN)

    def test_normalizes_mhc_search_identifier_shape(self) -> None:
        assay = normalize_assay(
            IedbEndpoint.MHC,
            {
                'elution_iri': 'IEDB_ASSAY:3',
                'structure_iri': 'IEDB_EPITOPE:3',
                'reference_iri': 'IEDB_REFERENCE:3',
                'qualitative_measure': 'Negative',
            },
        )

        self.assertEqual(assay.assay_iri, 'IEDB_ASSAY:3')
        self.assertEqual(assay.polarity, QualitativePolarity.NEGATIVE)


if __name__ == '__main__':
    unittest.main()
