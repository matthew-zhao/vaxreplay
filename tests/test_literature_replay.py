from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from vaxreplay.bundle import canonical_json_bytes
from vaxreplay.case_inventory import CaseUniverseDisposition, CaseUniverseEntry
from vaxreplay.case_schema import EvidenceStance
from vaxreplay.literature.adapter import (
    LiteratureAdmissionError,
    LiteratureEpisodeSpec,
    build_literature_episode,
    make_release_source_material_verifier,
    verify_decision_package,
    verify_outcome_package,
)
from vaxreplay.literature.schema import (
    ArchivedLiteratureDocument,
    ArchiveProof,
    CandidatePanelManifest,
    DecisionSeal,
    ExtractionKind,
    ExtractionRunManifest,
    LiteratureByteSpan,
    LiteratureClaim,
    LiteratureCorpusManifest,
    LiteratureDecisionPackage,
    LiteratureOutcomeJoinAudit,
    LiteratureOutcomePackage,
    OutcomeJoinRecord,
    OutcomeJoinStatus,
    PanelCandidate,
    literature_decision_content_sha256,
    literature_decision_package_sha256,
    literature_model_sha256,
    literature_outcome_package_sha256,
    validate_outcome_package_against_decision,
)
from vaxreplay.temporal_schema import DecisionTimeConfig, TemporalReceiptAuthority

UTC = timezone.utc
_KEY = bytes(range(32))


def _fixture_root() -> Path:
    return Path(__file__).parent / 'fixtures' / 'literature_fictional_replay'


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _span(body: bytes, quote: str) -> LiteratureByteSpan:
    encoded = quote.encode('utf-8')
    start = body.index(encoded)
    return LiteratureByteSpan(start=start, end=start + len(encoded), quote=quote)


def _packages() -> tuple[LiteratureDecisionPackage, LiteratureOutcomePackage]:
    decision_root = _fixture_root() / 'decision'
    panel_bytes = (decision_root / 'documents' / 'panel.txt').read_bytes()
    evidence_bytes = (decision_root / 'documents' / 'evidence.txt').read_bytes()
    panel_proof_bytes = (decision_root / 'proofs' / 'panel-proof.json').read_bytes()
    evidence_proof_bytes = (decision_root / 'proofs' / 'evidence-proof.json').read_bytes()
    seal_bytes = (decision_root / 'proofs' / 'decision-seal.json').read_bytes()
    extraction_receipt_bytes = (decision_root / 'proofs' / 'extraction-run.json').read_bytes()
    discovery_protocol_bytes = (decision_root / 'protocols' / 'discovery.json').read_bytes()
    decision_at = datetime(2020, 1, 1, tzinfo=UTC)

    documents = (
        ArchivedLiteratureDocument(
            document_id='evidence-paper',
            version_id='evidence-paper-v1',
            canonical_id='PMID:FICT0002',
            raw_path='documents/evidence.txt',
            raw_sha256=_sha(evidence_bytes),
            raw_bytes=len(evidence_bytes),
            media_type='text/plain',
            text_path='documents/evidence.txt',
            text_sha256=_sha(evidence_bytes),
            text_bytes=len(evidence_bytes),
            text_derivation='identity-utf8-v1',
            resolved_available_at=datetime(2019, 12, 14, tzinfo=UTC),
            selected_proof_id='evidence-proof',
            archive_proofs=(
                ArchiveProof(
                    proof_id='evidence-proof',
                    document_id='evidence-paper',
                    version_id='evidence-paper-v1',
                    artifact_sha256=_sha(evidence_bytes),
                    artifact_bytes=len(evidence_bytes),
                    witnessed_at=datetime(2019, 12, 14, tzinfo=UTC),
                    authority_type=TemporalReceiptAuthority.INDEPENDENT_ARCHIVE,
                    authority_id='fixture-independent-archive',
                    proof_path='proofs/evidence-proof.json',
                    proof_sha256=_sha(evidence_proof_bytes),
                    proof_bytes=len(evidence_proof_bytes),
                    verification_uri='https://fixture-archive.invalid/evidence-paper-v1',
                    fixture_only=True,
                ),
            ),
            source_url='https://publications.invalid/evidence-paper-v1',
            license_id='CC-BY-4.0',
            license_url='https://creativecommons.org/licenses/by/4.0/',
            citation='Fictional evidence paper v1',
        ),
        ArchivedLiteratureDocument(
            document_id='panel-paper',
            version_id='panel-paper-v1',
            canonical_id='PMID:FICT0001',
            raw_path='documents/panel.txt',
            raw_sha256=_sha(panel_bytes),
            raw_bytes=len(panel_bytes),
            media_type='text/plain',
            text_path='documents/panel.txt',
            text_sha256=_sha(panel_bytes),
            text_bytes=len(panel_bytes),
            text_derivation='identity-utf8-v1',
            resolved_available_at=datetime(2019, 10, 2, tzinfo=UTC),
            selected_proof_id='panel-proof',
            archive_proofs=(
                ArchiveProof(
                    proof_id='panel-proof',
                    document_id='panel-paper',
                    version_id='panel-paper-v1',
                    artifact_sha256=_sha(panel_bytes),
                    artifact_bytes=len(panel_bytes),
                    witnessed_at=datetime(2019, 10, 2, tzinfo=UTC),
                    authority_type=TemporalReceiptAuthority.INDEPENDENT_ARCHIVE,
                    authority_id='fixture-independent-archive',
                    proof_path='proofs/panel-proof.json',
                    proof_sha256=_sha(panel_proof_bytes),
                    proof_bytes=len(panel_proof_bytes),
                    verification_uri='https://fixture-archive.invalid/panel-paper-v1',
                    fixture_only=True,
                ),
            ),
            source_url='https://publications.invalid/panel-paper-v1',
            license_id='CC-BY-4.0',
            license_url='https://creativecommons.org/licenses/by/4.0/',
            citation='Fictional panel paper v1',
        ),
    )
    corpus = LiteratureCorpusManifest(
        corpus_id='fictional-corpus-v1',
        episode_id='fictional-literature-replay-001',
        decision_at=decision_at,
        discovery_protocol_path='protocols/discovery.json',
        discovery_protocol_sha256=_sha(discovery_protocol_bytes),
        discovery_protocol_bytes=len(discovery_protocol_bytes),
        fixture_only=True,
        documents=documents,
    )
    panel_quotes = {
        'target-alpha': 'row-001 | target-alpha | included under the preregistered complete-table rule.',
        'target-beta': 'row-002 | target-beta | included under the preregistered complete-table rule.',
        'target-gamma': 'row-003 | target-gamma | included under the preregistered complete-table rule.',
        'target-delta': 'row-004 | target-delta | included under the preregistered complete-table rule.',
    }
    candidate_rows = {
        'target-alpha': 'row-001',
        'target-beta': 'row-002',
        'target-delta': 'row-004',
        'target-gamma': 'row-003',
    }
    candidate_spec_sha = _sha(b'include every matching row; source-row order; outcomes forbidden')
    panel = CandidatePanelManifest(
        panel_id='fictional-complete-panel-v1',
        episode_id=corpus.episode_id,
        candidate_set_definition_sha256=candidate_spec_sha,
        matching_source_row_ids=('row-001', 'row-002', 'row-003', 'row-004'),
        included_source_row_ids=('row-001', 'row-002', 'row-003', 'row-004'),
        candidates=tuple(
            PanelCandidate(
                candidate_id=candidate_id,
                source_document_id='panel-paper',
                source_version_id='panel-paper-v1',
                source_row_id=candidate_rows[candidate_id],
                source_span=_span(panel_bytes, panel_quotes[candidate_id]),
            )
            for candidate_id in sorted(candidate_rows)
        ),
    )
    claim_quotes = {
        'target-alpha': 'Target alpha produced a strong cellular response in the fictional pre-cutoff assay.',
        'target-beta': 'Target beta showed weak stability in the fictional pre-cutoff assay.',
        'target-gamma': 'Target gamma produced a moderate cellular response in the fictional pre-cutoff assay.',
        'target-delta': 'Target delta showed inconsistent stability in the fictional pre-cutoff assay.',
    }
    claim_stances = {
        'target-alpha': EvidenceStance.SUPPORT,
        'target-beta': EvidenceStance.CONCERN,
        'target-delta': EvidenceStance.CONCERN,
        'target-gamma': EvidenceStance.SUPPORT,
    }
    extraction = ExtractionRunManifest(
        extraction_id='fixture-deterministic-extractor-v1',
        corpus_sha256=literature_model_sha256(corpus),
        panel_sha256=literature_model_sha256(panel),
        extractor_kind=ExtractionKind.DETERMINISTIC,
        extractor_id='fixture-exact-span-extractor-v1',
        extractor_code_sha256=_sha(b'fixture extractor code v1'),
        prompt_sha256=_sha(b'no LLM prompt; strict deterministic extractor'),
        config_sha256=_sha(b'fixture extraction config v1'),
        runner_receipt_path='proofs/extraction-run.json',
        runner_receipt_sha256=_sha(extraction_receipt_bytes),
        runner_receipt_bytes=len(extraction_receipt_bytes),
        claims=tuple(
            LiteratureClaim(
                claim_id=f'claim-{candidate_id}',
                document_id='evidence-paper',
                document_version_id='evidence-paper-v1',
                text_sha256=_sha(evidence_bytes),
                candidate_id=candidate_id,
                dimension='prior_response',
                stance=claim_stances[candidate_id],
                span=_span(evidence_bytes, claim_quotes[candidate_id]),
            )
            for candidate_id in sorted(claim_quotes)
        ),
    )
    evidence_spec_sha = _sha(b'exact admitted text views and exact byte-span claims')
    outcome_spec_sha = _sha(b'binary later fictional validation and four-level utility grade')
    evaluation_config = LiteratureEpisodeSpec(
        episode_id=corpus.episode_id,
        lineage_group_id='fictional-literature-lineage-alpha',
        synthetic=True,
        portfolio_size=2,
        target_id='later_validation',
        horizon_days=180,
        required_dimensions=('prior_response',),
        adjudication_version='fictional-literature-binary-v1',
    )
    decision_content_sha = literature_decision_content_sha256(
        episode_id=corpus.episode_id,
        decision_at=decision_at,
        corpus=corpus,
        panel=panel,
        extraction=extraction,
        evaluation_config=evaluation_config,
        candidate_set_definition_sha256=candidate_spec_sha,
        evidence_acquisition_spec_sha256=evidence_spec_sha,
        outcome_adjudication_spec_sha256=outcome_spec_sha,
    )
    decision = LiteratureDecisionPackage(
        episode_id=corpus.episode_id,
        decision_at=decision_at,
        corpus=corpus,
        panel=panel,
        extraction=extraction,
        evaluation_config=evaluation_config,
        candidate_set_definition_sha256=candidate_spec_sha,
        evidence_acquisition_spec_sha256=evidence_spec_sha,
        outcome_adjudication_spec_sha256=outcome_spec_sha,
        decision_content_sha256=decision_content_sha,
        seal=DecisionSeal(
            seal_id='fixture-decision-seal-v1',
            decision_content_sha256=decision_content_sha,
            witnessed_at=datetime(2020, 1, 2, tzinfo=UTC),
            authority_type=TemporalReceiptAuthority.PUBLIC_TRANSPARENCY_LOG,
            authority_id='fixture-transparency-log',
            proof_path='proofs/decision-seal.json',
            proof_sha256=_sha(seal_bytes),
            proof_bytes=len(seal_bytes),
            verification_uri='https://fixture-log.invalid/decision-seal-v1',
            fixture_only=True,
        ),
    )

    raw_outcome = (_fixture_root() / 'outcome' / 'outcomes.json').read_bytes()
    outcome_proof_bytes = (_fixture_root() / 'outcome' / 'outcome-proof.json').read_bytes()
    values = {
        'target-alpha': (1, 1.0, 4),
        'target-beta': (0, 0.0, 0),
        'target-delta': (0, 0.2, 1),
        'target-gamma': (1, 0.8, 3),
    }
    join_audit = LiteratureOutcomeJoinAudit(
        episode_id=corpus.episode_id,
        decision_content_sha256=decision_content_sha,
        label_join_started_at=datetime(2020, 7, 6, 0, 0, 1, tzinfo=UTC),
        panel_candidate_ids=tuple(sorted(values)),
        records=tuple(
            OutcomeJoinRecord(
                candidate_id=candidate_id,
                status=OutcomeJoinStatus.OBSERVED,
                outcome=values[candidate_id][0],
                candidate_utility=values[candidate_id][1],
                relevance_grade=values[candidate_id][2],
                source_record_ids=(f'outcome-{candidate_id}',),
            )
            for candidate_id in sorted(values)
        ),
        unmatched_outcome_record_ids=('outcome-outcome-only-decoy',),
    )
    outcome = LiteratureOutcomePackage(
        episode_id=corpus.episode_id,
        decision_package_sha256=literature_decision_package_sha256(decision),
        raw_outcome_path='outcomes.json',
        raw_outcome_sha256=_sha(raw_outcome),
        raw_outcome_bytes=len(raw_outcome),
        source_id='fictional-later-outcomes',
        version_id='fictional-later-outcomes-v1',
        availability_proof=ArchiveProof(
            proof_id='fictional-outcome-proof',
            document_id='fictional-later-outcomes',
            version_id='fictional-later-outcomes-v1',
            artifact_sha256=_sha(raw_outcome),
            artifact_bytes=len(raw_outcome),
            witnessed_at=datetime(2020, 7, 6, tzinfo=UTC),
            authority_type=TemporalReceiptAuthority.INDEPENDENT_ARCHIVE,
            authority_id='fixture-independent-archive',
            proof_path='outcome-proof.json',
            proof_sha256=_sha(outcome_proof_bytes),
            proof_bytes=len(outcome_proof_bytes),
            verification_uri='https://fixture-archive.invalid/fictional-later-outcomes-v1',
            fixture_only=True,
        ),
        outcome_available_at=datetime(2020, 7, 6, tzinfo=UTC),
        target_id='later_validation',
        horizon_days=180,
        source_url='https://outcomes.invalid/fictional-later-validation',
        license_id='CC0-1.0',
        license_url='https://creativecommons.org/publicdomain/zero/1.0/',
        citation='Fictional later validation outcomes',
        join_audit=join_audit,
    )
    return decision, outcome


def _archive_verifier(document: ArchivedLiteratureDocument, proof: ArchiveProof, payload: bytes) -> bool:
    parsed = json.loads(payload)
    return (
        document.document_id == proof.document_id
        and proof.authority_id == 'fixture-independent-archive'
        and parsed.get('fixture_only') is True
    )


def _seal_verifier(seal: DecisionSeal, payload: bytes) -> bool:
    parsed = json.loads(payload)
    return seal.authority_id == 'fixture-transparency-log' and parsed.get('fixture_only') is True


def _outcome_verifier(package: LiteratureOutcomePackage, payload: bytes) -> bool:
    parsed = json.loads(payload)
    return package.license_id == 'CC0-1.0' and parsed.get('outcome_canary') == 'OUTCOME-CANARY-7X9Q'


def _outcome_archive_verifier(
    package: LiteratureOutcomePackage,
    proof: ArchiveProof,
    payload: bytes,
) -> bool:
    parsed = json.loads(payload)
    return (
        proof.artifact_sha256 == package.raw_outcome_sha256
        and proof.authority_id == 'fixture-independent-archive'
        and parsed.get('fixture_only') is True
    )


def _panel_verifier(panel: CandidatePanelManifest, files: Mapping[str, bytes]) -> bool:
    panel_text = files['documents/panel.txt'].decode('utf-8')
    enumerated_rows = tuple(
        sorted(line.split(' | ', 1)[0] for line in panel_text.splitlines() if line.startswith('row-'))
    )
    return panel.complete and panel.matching_source_row_ids == enumerated_rows


def _extraction_verifier(extraction: ExtractionRunManifest, payload: bytes) -> bool:
    parsed = json.loads(payload)
    return (
        extraction.label_blind
        and not extraction.network_allowed
        and not extraction.outcome_namespace_mounted
        and parsed.get('fixture_only') is True
        and parsed.get('label_blind') is True
        and parsed.get('outcome_namespace_mounted') is False
    )


def _corpus_verifier(
    corpus: LiteratureCorpusManifest,
    protocol_bytes: bytes,
    files: Mapping[str, bytes],
) -> bool:
    parsed = json.loads(protocol_bytes)
    return (
        corpus.inventory_complete
        and parsed.get('outcome_fields_permitted') is False
        and tuple(document.document_id for document in corpus.documents) == ('evidence-paper', 'panel-paper')
        and 'documents/evidence.txt' in files
        and 'documents/panel.txt' in files
    )


def _text_derivation_verifier(document: ArchivedLiteratureDocument, raw: bytes, text: bytes) -> bool:
    return document.text_derivation == 'identity-utf8-v1' and raw == text


class LiteratureReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        shutil.copytree(_fixture_root() / 'decision', self.root / 'decision')
        shutil.copytree(_fixture_root() / 'outcome', self.root / 'outcome')
        self.decision_package, self.outcome_package = _packages()

    def _verify_decision(self):
        return verify_decision_package(
            self.root / 'decision',
            self.decision_package,
            archive_proof_verifier=_archive_verifier,
            decision_seal_verifier=_seal_verifier,
            panel_completeness_verifier=_panel_verifier,
            extraction_run_verifier=_extraction_verifier,
            corpus_completeness_verifier=_corpus_verifier,
            text_derivation_verifier=_text_derivation_verifier,
            allow_fixture=True,
        )

    def _verify_outcome(self, decision):
        return verify_outcome_package(
            self.root / 'outcome',
            self.outcome_package,
            decision,
            outcome_source_verifier=_outcome_verifier,
            outcome_archive_proof_verifier=_outcome_archive_verifier,
        )

    def test_two_stage_fixture_builds_a_valid_v1_episode_without_canary_leakage(self) -> None:
        decision = self._verify_decision()
        outcome = self._verify_outcome(decision)
        built = build_literature_episode(
            decision=decision,
            outcome=outcome,
            output_root=self.root / 'episode',
            label_commitment_key=_KEY,
            archive_proof_verifier=_archive_verifier,
            decision_seal_verifier=_seal_verifier,
            panel_completeness_verifier=_panel_verifier,
            extraction_run_verifier=_extraction_verifier,
            corpus_completeness_verifier=_corpus_verifier,
            text_derivation_verifier=_text_derivation_verifier,
            outcome_source_verifier=_outcome_verifier,
            outcome_archive_proof_verifier=_outcome_archive_verifier,
            allow_fixture=True,
        )

        self.assertEqual(built.bundle.manifest.candidate_ids, sorted(built.bundle.manifest.candidate_ids))
        self.assertNotIn('outcome-only-decoy', built.bundle.manifest.candidate_ids)
        self.assertEqual(
            built.source_audit.outcome_package.join_audit.unmatched_outcome_record_ids, ('outcome-outcome-only-decoy',)
        )
        public_bytes = canonical_json_bytes(built.bundle.public_view())
        self.assertNotIn(b'OUTCOME-CANARY-7X9Q', public_bytes)
        self.assertNotIn(b'outcome-only-decoy', public_bytes)
        self.assertNotIn(self.outcome_package.raw_outcome_sha256.encode(), public_bytes)
        self.assertNotIn(self.outcome_package.source_url.encode(), public_bytes)
        self.assertNotIn(literature_outcome_package_sha256(self.outcome_package).encode(), public_bytes)

        release_verifier = make_release_source_material_verifier(
            archive_proof_verifier=_archive_verifier,
            decision_seal_verifier=_seal_verifier,
            panel_completeness_verifier=_panel_verifier,
            extraction_run_verifier=_extraction_verifier,
            corpus_completeness_verifier=_corpus_verifier,
            text_derivation_verifier=_text_derivation_verifier,
            outcome_source_verifier=_outcome_verifier,
            outcome_archive_proof_verifier=_outcome_archive_verifier,
            allow_fixture=True,
        )
        material = SimpleNamespace(
            label_derivation_audit=built.source_audit_bytes,
            protocol_artifacts={
                'candidate_set_definition': (b'include every matching row; source-row order; outcomes forbidden'),
                'evidence_acquisition_spec': b'exact admitted text views and exact byte-span claims',
                'outcome_adjudication_spec': (b'binary later fictional validation and four-level utility grade'),
            },
            raw_outcome_source=built.source_material_files['outcome/outcomes.json'],
            admission=SimpleNamespace(
                decision_snapshot=SimpleNamespace(config=DecisionTimeConfig.from_manifest(built.bundle.manifest))
            ),
        )
        self.assertTrue(
            release_verifier(
                built.bundle.manifest.episode_id,
                built.bundle,
                material,
                built.source_audit_bytes,
                built.source_material_files,
                CaseUniverseEntry(
                    case_id='fictional-literature-case',
                    lineage_group_id=built.bundle.manifest.lineage_group_id,
                    disposition=CaseUniverseDisposition.PREELIGIBLE,
                    decision_package_sha256=literature_decision_package_sha256(built.source_audit.decision_package),
                ),
            )
        )
        self.assertFalse(
            release_verifier(
                built.bundle.manifest.episode_id,
                built.bundle,
                SimpleNamespace(**{**vars(material), 'raw_outcome_source': b'unrelated outcome bytes'}),
                built.source_audit_bytes,
                built.source_material_files,
                CaseUniverseEntry(
                    case_id='fictional-literature-case',
                    lineage_group_id=built.bundle.manifest.lineage_group_id,
                    disposition=CaseUniverseDisposition.PREELIGIBLE,
                    decision_package_sha256=literature_decision_package_sha256(built.source_audit.decision_package),
                ),
            )
        )
        self.assertFalse(
            release_verifier(
                built.bundle.manifest.episode_id,
                built.bundle,
                material,
                built.source_audit_bytes,
                built.source_material_files,
                CaseUniverseEntry(
                    case_id='unrelated-case',
                    lineage_group_id=built.bundle.manifest.lineage_group_id,
                    disposition=CaseUniverseDisposition.PREELIGIBLE,
                    decision_package_sha256='f' * 64,
                ),
            )
        )

    def test_production_admission_rejects_fixture_archive_authorities(self) -> None:
        with self.assertRaisesRegex(LiteratureAdmissionError, 'fixture-only'):
            verify_decision_package(
                self.root / 'decision',
                self.decision_package,
                archive_proof_verifier=_archive_verifier,
                decision_seal_verifier=_seal_verifier,
                panel_completeness_verifier=_panel_verifier,
                extraction_run_verifier=_extraction_verifier,
                corpus_completeness_verifier=_corpus_verifier,
                text_derivation_verifier=_text_derivation_verifier,
            )

    def test_tampered_document_and_extra_outcome_file_fail_closed(self) -> None:
        panel = self.root / 'decision' / 'documents' / 'panel.txt'
        panel.write_bytes(panel.read_bytes() + b'tamper')
        with self.subTest(case='document'), self.assertRaisesRegex(LiteratureAdmissionError, 'raw document'):
            self._verify_decision()

        panel.write_bytes((_fixture_root() / 'decision' / 'documents' / 'panel.txt').read_bytes())
        decision = self._verify_decision()
        (self.root / 'outcome' / 'extra.json').write_text('{}', encoding='utf-8')
        with (
            self.subTest(case='outcome-inventory'),
            self.assertRaisesRegex(
                LiteratureAdmissionError,
                'allowlist',
            ),
        ):
            self._verify_outcome(decision)

    def test_exact_spans_and_seal_before_join_are_enforced(self) -> None:
        claims = list(self.decision_package.extraction.claims)
        claims[0] = claims[0].model_copy(
            update={'span': claims[0].span.model_copy(update={'quote': claims[0].span.quote + 'x'})}
        )
        extraction = self.decision_package.extraction.model_copy(update={'claims': tuple(claims)})
        with self.subTest(case='derived-hash'), self.assertRaisesRegex(ValidationError, 'decision content hash'):
            LiteratureDecisionPackage.model_validate({**self.decision_package.model_dump(), 'extraction': extraction})

        changed_config = self.decision_package.evaluation_config.model_copy(update={'portfolio_size': 1})
        with self.subTest(case='reward-config'), self.assertRaisesRegex(ValidationError, 'decision content hash'):
            LiteratureDecisionPackage.model_validate(
                {**self.decision_package.model_dump(), 'evaluation_config': changed_config}
            )

        early_join = self.outcome_package.join_audit.model_copy(
            update={'label_join_started_at': self.decision_package.seal.witnessed_at}
        )
        early_outcome = self.outcome_package.model_copy(update={'join_audit': early_join})
        with self.subTest(case='seal-order'), self.assertRaisesRegex(ValueError, 'only after'):
            validate_outcome_package_against_decision(early_outcome, self.decision_package)

        earlier_proof = self.outcome_package.availability_proof.model_copy(
            update={'witnessed_at': self.outcome_package.outcome_available_at - timedelta(days=1)}
        )
        with (
            self.subTest(case='outcome-witness'),
            self.assertRaisesRegex(ValidationError, 'must equal its independently verified'),
        ):
            LiteratureOutcomePackage.model_validate(
                {**self.outcome_package.model_dump(), 'availability_proof': earlier_proof}
            )

    def test_verified_source_bytes_cannot_be_replaced_before_build(self) -> None:
        decision = self._verify_decision()
        outcome = self._verify_outcome(decision)
        forged_files = dict(decision.source_files)
        forged_files['decision/documents/evidence.txt'] += b' OUTCOME-CANARY-7X9Q'
        forged_decision = replace(decision, source_files=forged_files)

        with self.assertRaisesRegex(LiteratureAdmissionError, 'declared hash and byte count'):
            build_literature_episode(
                decision=forged_decision,
                outcome=outcome,
                output_root=self.root / 'forged-episode',
                label_commitment_key=_KEY,
                archive_proof_verifier=_archive_verifier,
                decision_seal_verifier=_seal_verifier,
                panel_completeness_verifier=_panel_verifier,
                extraction_run_verifier=_extraction_verifier,
                corpus_completeness_verifier=_corpus_verifier,
                text_derivation_verifier=_text_derivation_verifier,
                outcome_source_verifier=_outcome_verifier,
                outcome_archive_proof_verifier=_outcome_archive_verifier,
                allow_fixture=True,
            )

    def test_future_outcomes_cannot_change_or_shrink_the_frozen_panel(self) -> None:
        missing_records = self.outcome_package.join_audit.records[:-1]
        with self.assertRaisesRegex(ValidationError, 'retain every frozen panel'):
            LiteratureOutcomePackage.model_validate(
                {
                    **self.outcome_package.model_dump(),
                    'join_audit': {
                        **self.outcome_package.join_audit.model_dump(),
                        'records': missing_records,
                    },
                }
            )

        decision = self._verify_decision()
        censored = self.outcome_package.join_audit.records[-1].model_copy(
            update={
                'status': OutcomeJoinStatus.MISSING,
                'outcome': None,
                'candidate_utility': None,
                'relevance_grade': None,
                'censor_reason': 'no later assay',
                'source_record_ids': (),
            }
        )
        audit = self.outcome_package.join_audit.model_copy(
            update={'records': (*self.outcome_package.join_audit.records[:-1], censored)}
        )
        incomplete_package = self.outcome_package.model_copy(update={'join_audit': audit})
        verified_outcome = replace(
            self._verify_outcome(decision),
            package=incomplete_package,
        )
        with self.assertRaisesRegex(LiteratureAdmissionError, 'observed labels for every'):
            build_literature_episode(
                decision=decision,
                outcome=verified_outcome,
                output_root=self.root / 'incomplete-episode',
                label_commitment_key=_KEY,
                archive_proof_verifier=_archive_verifier,
                decision_seal_verifier=_seal_verifier,
                panel_completeness_verifier=_panel_verifier,
                extraction_run_verifier=_extraction_verifier,
                corpus_completeness_verifier=_corpus_verifier,
                text_derivation_verifier=_text_derivation_verifier,
                outcome_source_verifier=_outcome_verifier,
                outcome_archive_proof_verifier=_outcome_archive_verifier,
                allow_fixture=True,
            )


if __name__ == '__main__':
    unittest.main()
