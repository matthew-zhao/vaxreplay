"""Two-stage adapter from sealed literature replay material to a VaxReplay episode."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from vaxreplay.bundle import (
    EpisodeBundle,
    body_sha256,
    canonical_json_bytes,
    jsonl_text,
    ranking_labels_commitment,
    records_sha256,
)
from vaxreplay.case_inventory import CaseUniverseEntry
from vaxreplay.case_schema import (
    ANTIGEN_TARGET_PRIORITIZATION_TASK,
    RANKING_REWARD_VERSION,
    AssessmentConclusion,
    CandidateRecord,
    EpisodeManifest,
    EvidenceRecord,
    EvidenceStance,
    ForecastTarget,
    GoldAssessmentRecord,
    GoldEvidenceRecord,
    LabelCommitmentScheme,
    OutcomeRecord,
    PrivateLabels,
    SourceType,
    StrictModel,
)
from vaxreplay.literature.schema import (
    ArchivedLiteratureDocument,
    ArchiveProof,
    CandidatePanelManifest,
    DecisionSeal,
    ExtractionRunManifest,
    LiteratureByteSpan,
    LiteratureCorpusManifest,
    LiteratureDecisionPackage,
    LiteratureEvaluationConfig,
    LiteratureOutcomePackage,
    LiteratureSourceAudit,
    LiteratureSourceFileBinding,
    OutcomeJoinStatus,
    literature_decision_package_sha256,
    validate_outcome_package_against_decision,
)
from vaxreplay.ranking_schema import RankingLabelV1
from vaxreplay.temporal_schema import DecisionTimeConfig

LITERATURE_ADAPTER_ID = 'vaxreplay.literature.v0.1'
_MAX_SOURCE_FILE_BYTES = 256 * 1024 * 1024
_MAX_SOURCE_FILES = 100_000

type ArchiveProofVerifier = Callable[[ArchivedLiteratureDocument, ArchiveProof, bytes], bool]
type DecisionSealVerifier = Callable[[DecisionSeal, bytes], bool]
type PanelCompletenessVerifier = Callable[[CandidatePanelManifest, Mapping[str, bytes]], bool]
type ExtractionRunVerifier = Callable[[ExtractionRunManifest, bytes], bool]
type CorpusCompletenessVerifier = Callable[[LiteratureCorpusManifest, bytes, Mapping[str, bytes]], bool]
type TextDerivationVerifier = Callable[[ArchivedLiteratureDocument, bytes, bytes], bool]
type OutcomeSourceVerifier = Callable[[LiteratureOutcomePackage, bytes], bool]
type OutcomeArchiveProofVerifier = Callable[[LiteratureOutcomePackage, ArchiveProof, bytes], bool]


class LiteratureAdmissionError(ValueError):
    """Raised when archived source, decision seal, or outcome material fails closed."""


# Compatibility alias for the first development API.  The object now lives in the sealed
# decision package, so callers cannot choose reward-affecting fields after loading outcomes.
LiteratureEpisodeSpec = LiteratureEvaluationConfig


@dataclass(frozen=True)
class VerifiedDecisionPackage:
    package: LiteratureDecisionPackage
    source_files: dict[str, bytes]


@dataclass(frozen=True)
class VerifiedOutcomePackage:
    package: LiteratureOutcomePackage
    source_files: dict[str, bytes]


@dataclass(frozen=True)
class BuiltLiteratureEpisode:
    bundle: EpisodeBundle
    source_audit: LiteratureSourceAudit
    source_audit_bytes: bytes
    source_material_files: dict[str, bytes]


@dataclass(frozen=True)
class _DerivedLiteratureEpisode:
    candidates: tuple[CandidateRecord, ...]
    evidence: tuple[EvidenceRecord, ...]
    private_labels: PrivateLabels
    ranking_labels: tuple[RankingLabelV1, ...]


def verify_decision_package(
    source_root: Path,
    package: LiteratureDecisionPackage,
    *,
    archive_proof_verifier: ArchiveProofVerifier,
    decision_seal_verifier: DecisionSealVerifier,
    panel_completeness_verifier: PanelCompletenessVerifier,
    extraction_run_verifier: ExtractionRunVerifier,
    corpus_completeness_verifier: CorpusCompletenessVerifier,
    text_derivation_verifier: TextDerivationVerifier,
    allow_fixture: bool = False,
) -> VerifiedDecisionPackage:
    """Verify the source-only decision namespace without accepting an outcome input."""

    root = _resolve_directory(source_root, 'literature decision source')
    package = _canonical_reparse(package, LiteratureDecisionPackage, 'decision package')
    if package.corpus.fixture_only and not allow_fixture:
        raise LiteratureAdmissionError('fixture-only literature cannot enter production admission')

    expected_paths = {
        package.corpus.discovery_protocol_path,
        package.seal.proof_path,
        package.extraction.runner_receipt_path,
        *(document.raw_path for document in package.corpus.documents),
        *(document.text_path for document in package.corpus.documents),
        *(proof.proof_path for document in package.corpus.documents for proof in document.archive_proofs),
    }
    _require_exact_file_inventory(root, expected_paths)
    source_files = {path: _read_regular_file(root, path) for path in sorted(expected_paths)}
    discovery_protocol = source_files[package.corpus.discovery_protocol_path]
    _require_binding(
        discovery_protocol,
        package.corpus.discovery_protocol_sha256,
        package.corpus.discovery_protocol_bytes,
        'corpus discovery protocol',
    )
    try:
        corpus_verified = corpus_completeness_verifier(package.corpus, discovery_protocol, source_files)
    except Exception as error:
        raise LiteratureAdmissionError(f'corpus completeness verifier failed: {error}') from error
    if not corpus_verified:
        raise LiteratureAdmissionError('corpus completeness verifier rejected the source inventory')
    for document in package.corpus.documents:
        raw = source_files[document.raw_path]
        text_bytes = source_files[document.text_path]
        _require_binding(raw, document.raw_sha256, document.raw_bytes, f'raw document {document.document_id}')
        _require_binding(
            text_bytes,
            document.text_sha256,
            document.text_bytes,
            f'text view {document.document_id}',
        )
        try:
            derived = text_derivation_verifier(document, raw, text_bytes)
        except Exception as error:
            raise LiteratureAdmissionError(
                f'text derivation verifier failed for {document.document_id}: {error}'
            ) from error
        if not derived:
            raise LiteratureAdmissionError(f'text derivation verifier rejected {document.document_id}')
        try:
            text_bytes.decode('utf-8')
        except UnicodeDecodeError as error:
            raise LiteratureAdmissionError(f'text view for {document.document_id} is not UTF-8') from error
        for proof in document.archive_proofs:
            proof_bytes = source_files[proof.proof_path]
            _require_binding(proof_bytes, proof.proof_sha256, proof.proof_bytes, f'archive proof {proof.proof_id}')
            if proof.fixture_only and not allow_fixture:
                raise LiteratureAdmissionError('fixture archive authorities are rejected in production')
            try:
                verified = archive_proof_verifier(document, proof, proof_bytes)
            except Exception as error:
                raise LiteratureAdmissionError(f'archive verifier failed for {proof.proof_id}: {error}') from error
            if not verified:
                raise LiteratureAdmissionError(f'archive verifier rejected {proof.proof_id}')

    seal_bytes = source_files[package.seal.proof_path]
    _require_binding(seal_bytes, package.seal.proof_sha256, package.seal.proof_bytes, 'decision seal proof')
    try:
        seal_verified = decision_seal_verifier(package.seal, seal_bytes)
    except Exception as error:
        raise LiteratureAdmissionError(f'decision seal verifier failed: {error}') from error
    if not seal_verified:
        raise LiteratureAdmissionError('decision seal verifier rejected the package')

    extraction_receipt = source_files[package.extraction.runner_receipt_path]
    _require_binding(
        extraction_receipt,
        package.extraction.runner_receipt_sha256,
        package.extraction.runner_receipt_bytes,
        'extraction run receipt',
    )
    try:
        extraction_verified = extraction_run_verifier(package.extraction, extraction_receipt)
    except Exception as error:
        raise LiteratureAdmissionError(f'extraction run verifier failed: {error}') from error
    if not extraction_verified:
        raise LiteratureAdmissionError('extraction run verifier rejected the label-blind extraction')

    documents = {document.document_id: document for document in package.corpus.documents}
    for candidate in package.panel.candidates:
        document = documents[candidate.source_document_id]
        _verify_span(
            source_files[document.text_path],
            candidate.source_span,
            f'panel candidate {candidate.candidate_id}',
        )
    for exclusion in package.panel.exclusions:
        document = documents[exclusion.source_document_id]
        _verify_span(
            source_files[document.text_path], exclusion.source_span, f'panel exclusion {exclusion.source_row_id}'
        )
    for claim in package.extraction.claims:
        document = documents[claim.document_id]
        _verify_span(source_files[document.text_path], claim.span, f'extracted claim {claim.claim_id}')
    try:
        panel_verified = panel_completeness_verifier(package.panel, source_files)
    except Exception as error:
        raise LiteratureAdmissionError(f'panel completeness verifier failed: {error}') from error
    if not panel_verified:
        raise LiteratureAdmissionError('panel completeness verifier rejected the frozen candidate panel')

    return VerifiedDecisionPackage(
        package=package,
        source_files={f'decision/{path}': payload for path, payload in source_files.items()},
    )


def verify_outcome_package(
    source_root: Path,
    package: LiteratureOutcomePackage,
    decision: VerifiedDecisionPackage,
    *,
    outcome_source_verifier: OutcomeSourceVerifier,
    outcome_archive_proof_verifier: OutcomeArchiveProofVerifier,
) -> VerifiedOutcomePackage:
    """Join access begins here, after the independently sealed decision package is verified."""

    root = _resolve_directory(source_root, 'literature outcome source')
    package = _canonical_reparse(package, LiteratureOutcomePackage, 'outcome package')
    try:
        validate_outcome_package_against_decision(package, decision.package)
    except ValueError as error:
        raise LiteratureAdmissionError(str(error)) from error
    _require_exact_file_inventory(root, {package.raw_outcome_path, package.availability_proof.proof_path})
    raw = _read_regular_file(root, package.raw_outcome_path)
    proof_bytes = _read_regular_file(root, package.availability_proof.proof_path)
    _require_binding(raw, package.raw_outcome_sha256, package.raw_outcome_bytes, 'raw outcome source')
    _require_binding(
        proof_bytes,
        package.availability_proof.proof_sha256,
        package.availability_proof.proof_bytes,
        'outcome availability proof',
    )
    try:
        proof_verified = outcome_archive_proof_verifier(package, package.availability_proof, proof_bytes)
    except Exception as error:
        raise LiteratureAdmissionError(f'outcome archive verifier failed: {error}') from error
    if not proof_verified:
        raise LiteratureAdmissionError('outcome archive verifier rejected the availability proof')
    try:
        verified = outcome_source_verifier(package, raw)
    except Exception as error:
        raise LiteratureAdmissionError(f'outcome source verifier failed: {error}') from error
    if not verified:
        raise LiteratureAdmissionError('outcome source verifier rejected the package')
    return VerifiedOutcomePackage(
        package=package,
        source_files={
            f'outcome/{package.raw_outcome_path}': raw,
            f'outcome/{package.availability_proof.proof_path}': proof_bytes,
        },
    )


def build_literature_episode(
    *,
    decision: VerifiedDecisionPackage,
    outcome: VerifiedOutcomePackage,
    output_root: Path,
    label_commitment_key: bytes,
    archive_proof_verifier: ArchiveProofVerifier,
    decision_seal_verifier: DecisionSealVerifier,
    panel_completeness_verifier: PanelCompletenessVerifier,
    extraction_run_verifier: ExtractionRunVerifier,
    corpus_completeness_verifier: CorpusCompletenessVerifier,
    text_derivation_verifier: TextDerivationVerifier,
    outcome_source_verifier: OutcomeSourceVerifier,
    outcome_archive_proof_verifier: OutcomeArchiveProofVerifier,
    allow_fixture: bool = False,
) -> BuiltLiteratureEpisode:
    """Normalize a sealed decision plus later outcome join into a private V1 episode.

    Reward-affecting configuration is read only from the pre-outcome decision package.  There is
    intentionally no post-outcome ``spec`` argument.
    """

    package = decision.package
    outcome_package = outcome.package
    spec = package.evaluation_config
    validate_outcome_package_against_decision(outcome_package, package)
    text_by_document_id = _revalidate_bound_build_inputs(decision, outcome)
    if (spec.target_id, spec.horizon_days) != (outcome_package.target_id, outcome_package.horizon_days):
        raise LiteratureAdmissionError('sealed evaluation target does not match the outcome package')
    if len(label_commitment_key) < 32:
        raise LiteratureAdmissionError('literature test episodes require a label HMAC key of at least 32 bytes')

    derived = _derive_literature_episode(package, outcome_package, text_by_document_id)
    candidate_ids = tuple(candidate.candidate_id for candidate in derived.candidates)
    commitment = ranking_labels_commitment(
        derived.private_labels,
        derived.ranking_labels,
        LabelCommitmentScheme.HMAC_SHA256,
        key=label_commitment_key,
    )
    manifest = EpisodeManifest(
        episode_id=spec.episode_id,
        lineage_group_id=spec.lineage_group_id,
        synthetic=spec.synthetic,
        task_type=ANTIGEN_TARGET_PRIORITIZATION_TASK,
        split=spec.split,
        decision_at=package.decision_at,
        portfolio_size=spec.portfolio_size,
        candidate_ids=list(candidate_ids),
        forecast_targets=[ForecastTarget(target_id=spec.target_id, horizon_days=spec.horizon_days)],
        required_dimensions=list(spec.required_dimensions),
        evidence_sha256=records_sha256(derived.evidence),
        candidates_sha256=records_sha256(derived.candidates),
        labels_sha256=commitment,
        label_commitment_scheme=LabelCommitmentScheme.HMAC_SHA256,
        label_commitment_key_id=_sha256(label_commitment_key),
        adjudication_version=spec.adjudication_version,
        source_provenance=None,
        reward_version=RANKING_REWARD_VERSION,
    )
    source_files = {**decision.source_files, **outcome.source_files}
    source_audit = LiteratureSourceAudit(
        episode_id=spec.episode_id,
        episode_manifest_sha256=_sha256(canonical_json_bytes(manifest)),
        fixture_only=package.corpus.fixture_only,
        decision_package=package,
        outcome_package=outcome_package,
        files=tuple(
            LiteratureSourceFileBinding(path=path, sha256=_sha256(payload), byte_count=len(payload))
            for path, payload in sorted(source_files.items())
        ),
    )
    source_audit_bytes = canonical_json_bytes(source_audit)
    provisional_bundle = EpisodeBundle(
        root=output_root.expanduser().resolve(strict=False),
        manifest=manifest,
        candidates=derived.candidates,
        evidence=derived.evidence,
        private_labels=derived.private_labels,
        label_commitment_key=label_commitment_key,
        ranking_labels=derived.ranking_labels,
    )
    provisional_bundle.validate_integrity()
    if not _verify_literature_source_audit(
        episode_id=spec.episode_id,
        bundle=provisional_bundle,
        audit_bytes=source_audit_bytes,
        source_files=source_files,
        archive_proof_verifier=archive_proof_verifier,
        decision_seal_verifier=decision_seal_verifier,
        panel_completeness_verifier=panel_completeness_verifier,
        extraction_run_verifier=extraction_run_verifier,
        corpus_completeness_verifier=corpus_completeness_verifier,
        text_derivation_verifier=text_derivation_verifier,
        outcome_source_verifier=outcome_source_verifier,
        outcome_archive_proof_verifier=outcome_archive_proof_verifier,
        allow_fixture=allow_fixture,
    ):
        raise LiteratureAdmissionError('trusted source verifiers rejected episode construction inputs')
    _write_episode(
        output_root,
        manifest=manifest,
        candidates=derived.candidates,
        evidence=derived.evidence,
        private_labels=derived.private_labels,
        ranking_labels=derived.ranking_labels,
        label_key=label_commitment_key,
        spec=spec,
        source_audit=source_audit,
    )
    bundle = EpisodeBundle.load(output_root.expanduser().resolve(), include_private=True)
    return BuiltLiteratureEpisode(
        bundle=bundle,
        source_audit=source_audit,
        source_audit_bytes=source_audit_bytes,
        source_material_files=source_files,
    )


def make_release_source_material_verifier(
    *,
    archive_proof_verifier: ArchiveProofVerifier,
    decision_seal_verifier: DecisionSealVerifier,
    panel_completeness_verifier: PanelCompletenessVerifier,
    extraction_run_verifier: ExtractionRunVerifier,
    corpus_completeness_verifier: CorpusCompletenessVerifier,
    text_derivation_verifier: TextDerivationVerifier,
    outcome_source_verifier: OutcomeSourceVerifier,
    outcome_archive_proof_verifier: OutcomeArchiveProofVerifier,
    allow_fixture: bool = False,
) -> Callable[[str, EpisodeBundle, object, bytes, Mapping[str, bytes], CaseUniverseEntry], bool]:
    """Create the required Tier-B release callback from concrete trust-policy verifiers."""

    def verify(
        episode_id: str,
        bundle: EpisodeBundle,
        temporal_material: object,
        audit_bytes: bytes,
        source_files: Mapping[str, bytes],
        case_universe_entry: CaseUniverseEntry,
    ) -> bool:
        return _verify_literature_source_audit(
            episode_id=episode_id,
            bundle=bundle,
            temporal_material=temporal_material,
            case_universe_entry=case_universe_entry,
            audit_bytes=audit_bytes,
            source_files=source_files,
            archive_proof_verifier=archive_proof_verifier,
            decision_seal_verifier=decision_seal_verifier,
            panel_completeness_verifier=panel_completeness_verifier,
            extraction_run_verifier=extraction_run_verifier,
            corpus_completeness_verifier=corpus_completeness_verifier,
            text_derivation_verifier=text_derivation_verifier,
            outcome_source_verifier=outcome_source_verifier,
            outcome_archive_proof_verifier=outcome_archive_proof_verifier,
            allow_fixture=allow_fixture,
        )

    return verify


def _derive_literature_episode(
    package: LiteratureDecisionPackage,
    outcome_package: LiteratureOutcomePackage,
    text_by_document_id: Mapping[str, str],
) -> _DerivedLiteratureEpisode:
    """Purely derive every public and private record from the sealed replay packages."""

    spec = package.evaluation_config
    if (spec.target_id, spec.horizon_days) != (outcome_package.target_id, outcome_package.horizon_days):
        raise LiteratureAdmissionError('sealed evaluation target does not match the outcome package')
    candidate_ids = tuple(candidate.candidate_id for candidate in package.panel.candidates)
    if set(text_by_document_id) != {document.document_id for document in package.corpus.documents}:
        raise LiteratureAdmissionError('derived text inventory does not match the sealed corpus')

    candidates = tuple(CandidateRecord(episode_id=spec.episode_id, candidate_id=value) for value in candidate_ids)
    claims_by_document: dict[str, list] = {document.document_id: [] for document in package.corpus.documents}
    for claim in package.extraction.claims:
        claims_by_document[claim.document_id].append(claim)
    evidence_id_by_document: dict[str, str] = {}
    evidence: list[EvidenceRecord] = []
    for index, document in enumerate(package.corpus.documents):
        evidence_id = f'lit-evidence-{index:04d}'
        evidence_id_by_document[document.document_id] = evidence_id
        body = text_by_document_id[document.document_id]
        evidence.append(
            EvidenceRecord(
                episode_id=spec.episode_id,
                evidence_id=evidence_id,
                source_type=SourceType.JOURNAL_ABSTRACT,
                collected_at=None,
                available_at=document.resolved_available_at,
                title=document.citation,
                body=body,
                body_sha256=body_sha256(body),
                related_candidate_ids=sorted(
                    {claim.candidate_id for claim in claims_by_document[document.document_id]}
                ),
                provenance_url=document.source_url,
                license_id=document.license_id,
                derivation=f'exact admitted text view via {document.text_derivation}',
            )
        )

    gold_evidence = tuple(
        GoldEvidenceRecord(
            episode_id=spec.episode_id,
            candidate_id=claim.candidate_id,
            dimension=claim.dimension,
            evidence_id=evidence_id_by_document[claim.document_id],
            stance=claim.stance,
            quote=claim.span.quote,
        )
        for claim in package.extraction.claims
    )
    stances: dict[tuple[str, str], set[EvidenceStance]] = {}
    for claim in package.extraction.claims:
        stances.setdefault((claim.candidate_id, claim.dimension), set()).add(claim.stance)
    assessments = tuple(
        GoldAssessmentRecord(
            episode_id=spec.episode_id,
            candidate_id=candidate_id,
            dimension=dimension,
            conclusion=_assessment_conclusion(stances.get((candidate_id, dimension), set())),
        )
        for candidate_id in candidate_ids
        for dimension in spec.required_dimensions
    )

    joins = {record.candidate_id: record for record in outcome_package.join_audit.records}
    incomplete = sorted(
        candidate_id for candidate_id, record in joins.items() if record.status != OutcomeJoinStatus.OBSERVED
    )
    if incomplete:
        raise LiteratureAdmissionError(
            f'V1 episode construction requires observed labels for every frozen candidate: {incomplete}'
        )
    outcomes: list[OutcomeRecord] = []
    ranking_labels: list[RankingLabelV1] = []
    for candidate_id in candidate_ids:
        joined = joins[candidate_id]
        assert joined.outcome is not None
        assert joined.candidate_utility is not None
        assert joined.relevance_grade is not None
        outcomes.append(
            OutcomeRecord(
                episode_id=spec.episode_id,
                candidate_id=candidate_id,
                target_id=spec.target_id,
                horizon_days=spec.horizon_days,
                outcome=joined.outcome,
                candidate_utility=joined.candidate_utility,
                revealed_at=outcome_package.outcome_available_at,
            )
        )
        ranking_labels.append(
            RankingLabelV1(
                episode_id=spec.episode_id,
                candidate_id=candidate_id,
                relevance_grade=joined.relevance_grade,
            )
        )
    return _DerivedLiteratureEpisode(
        candidates=candidates,
        evidence=tuple(evidence),
        private_labels=PrivateLabels(
            outcomes=outcomes,
            assessments_gold=list(assessments),
            evidence_gold=list(gold_evidence),
        ),
        ranking_labels=tuple(ranking_labels),
    )


def _verify_literature_source_audit(
    *,
    episode_id: str,
    bundle: EpisodeBundle,
    audit_bytes: bytes,
    source_files: Mapping[str, bytes],
    archive_proof_verifier: ArchiveProofVerifier,
    decision_seal_verifier: DecisionSealVerifier,
    panel_completeness_verifier: PanelCompletenessVerifier,
    extraction_run_verifier: ExtractionRunVerifier,
    corpus_completeness_verifier: CorpusCompletenessVerifier,
    text_derivation_verifier: TextDerivationVerifier,
    outcome_source_verifier: OutcomeSourceVerifier,
    outcome_archive_proof_verifier: OutcomeArchiveProofVerifier,
    allow_fixture: bool,
    temporal_material: object | None = None,
    case_universe_entry: CaseUniverseEntry | None = None,
) -> bool:
    """Reverify exact sources and prove that they deterministically produce this bundle."""

    try:
        audit = LiteratureSourceAudit.model_validate_json(audit_bytes)
        if canonical_json_bytes(audit) != audit_bytes or audit.episode_id != episode_id:
            return False
        if audit.episode_manifest_sha256 != bundle.manifest_sha256:
            return False
        if audit.fixture_only and not allow_fixture:
            return False
        expected = {binding.path: binding for binding in audit.files}
        if set(source_files) != set(expected):
            return False
        for path, binding in expected.items():
            payload = source_files[path]
            if len(payload) != binding.byte_count or _sha256(payload) != binding.sha256:
                return False

        corpus = audit.decision_package.corpus
        discovery_protocol = source_files[f'decision/{corpus.discovery_protocol_path}']
        _require_binding(
            discovery_protocol,
            corpus.discovery_protocol_sha256,
            corpus.discovery_protocol_bytes,
            'corpus discovery protocol',
        )
        decision_files = {
            path.removeprefix('decision/'): payload
            for path, payload in source_files.items()
            if path.startswith('decision/')
        }
        if not corpus_completeness_verifier(corpus, discovery_protocol, decision_files):
            return False
        texts: dict[str, str] = {}
        for document in corpus.documents:
            raw = source_files[f'decision/{document.raw_path}']
            text = source_files[f'decision/{document.text_path}']
            _require_binding(raw, document.raw_sha256, document.raw_bytes, document.document_id)
            _require_binding(text, document.text_sha256, document.text_bytes, f'{document.document_id} text')
            if not text_derivation_verifier(document, raw, text):
                return False
            texts[document.document_id] = text.decode('utf-8')
            for proof in document.archive_proofs:
                proof_bytes = source_files[f'decision/{proof.proof_path}']
                _require_binding(proof_bytes, proof.proof_sha256, proof.proof_bytes, proof.proof_id)
                if proof.fixture_only and not allow_fixture:
                    return False
                if not archive_proof_verifier(document, proof, proof_bytes):
                    return False

        seal = audit.decision_package.seal
        seal_bytes = source_files[f'decision/{seal.proof_path}']
        _require_binding(seal_bytes, seal.proof_sha256, seal.proof_bytes, 'decision seal')
        if not decision_seal_verifier(seal, seal_bytes):
            return False
        extraction = audit.decision_package.extraction
        extraction_receipt = source_files[f'decision/{extraction.runner_receipt_path}']
        _require_binding(
            extraction_receipt,
            extraction.runner_receipt_sha256,
            extraction.runner_receipt_bytes,
            'extraction receipt',
        )
        if not extraction_run_verifier(extraction, extraction_receipt):
            return False
        documents = {document.document_id: document for document in corpus.documents}
        for candidate in audit.decision_package.panel.candidates:
            document = documents[candidate.source_document_id]
            _verify_span(
                source_files[f'decision/{document.text_path}'],
                candidate.source_span,
                candidate.candidate_id,
            )
        for exclusion in audit.decision_package.panel.exclusions:
            document = documents[exclusion.source_document_id]
            _verify_span(
                source_files[f'decision/{document.text_path}'],
                exclusion.source_span,
                exclusion.source_row_id,
            )
        for claim in extraction.claims:
            document = documents[claim.document_id]
            _verify_span(source_files[f'decision/{document.text_path}'], claim.span, claim.claim_id)
        if not panel_completeness_verifier(audit.decision_package.panel, decision_files):
            return False

        raw_outcome = source_files[f'outcome/{audit.outcome_package.raw_outcome_path}']
        outcome_proof = audit.outcome_package.availability_proof
        outcome_proof_bytes = source_files[f'outcome/{outcome_proof.proof_path}']
        _require_binding(
            raw_outcome,
            audit.outcome_package.raw_outcome_sha256,
            audit.outcome_package.raw_outcome_bytes,
            'raw outcome source',
        )
        if not outcome_source_verifier(audit.outcome_package, raw_outcome):
            return False
        _require_binding(
            outcome_proof_bytes,
            outcome_proof.proof_sha256,
            outcome_proof.proof_bytes,
            'outcome availability proof',
        )
        if outcome_proof.fixture_only and not allow_fixture:
            return False
        if not outcome_archive_proof_verifier(
            audit.outcome_package,
            outcome_proof,
            outcome_proof_bytes,
        ):
            return False
        validate_outcome_package_against_decision(audit.outcome_package, audit.decision_package)

        derived = _derive_literature_episode(audit.decision_package, audit.outcome_package, texts)
        spec = audit.decision_package.evaluation_config
        expected_config = {
            'episode_id': spec.episode_id,
            'lineage_group_id': spec.lineage_group_id,
            'synthetic': spec.synthetic,
            'task_type': ANTIGEN_TARGET_PRIORITIZATION_TASK,
            'split': spec.split,
            'decision_at': audit.decision_package.decision_at,
            'portfolio_size': spec.portfolio_size,
            'candidate_ids': list(candidate.candidate_id for candidate in derived.candidates),
            'forecast_targets': [ForecastTarget(target_id=spec.target_id, horizon_days=spec.horizon_days)],
            'required_dimensions': list(spec.required_dimensions),
            'adjudication_version': spec.adjudication_version,
            'reward_version': RANKING_REWARD_VERSION,
        }
        if any(getattr(bundle.manifest, field) != value for field, value in expected_config.items()):
            return False
        if bundle.manifest.source_provenance is not None:
            return False
        if (
            bundle.candidates != derived.candidates
            or bundle.evidence != derived.evidence
            or bundle.private_labels != derived.private_labels
            or bundle.ranking_labels != derived.ranking_labels
        ):
            return False
        bundle.validate_integrity()

        if temporal_material is not None:
            if (
                case_universe_entry is None
                or case_universe_entry.lineage_group_id != bundle.manifest.lineage_group_id
                or case_universe_entry.decision_package_sha256
                != literature_decision_package_sha256(audit.decision_package)
            ):
                return False
            label_audit = getattr(temporal_material, 'label_derivation_audit', None)
            protocol_artifacts = getattr(temporal_material, 'protocol_artifacts', None)
            temporal_raw_outcome = getattr(temporal_material, 'raw_outcome_source', None)
            admission = getattr(temporal_material, 'admission', None)
            if label_audit != audit_bytes or temporal_raw_outcome != raw_outcome:
                return False
            if not isinstance(protocol_artifacts, Mapping):
                return False
            expected_protocol_hashes = {
                'candidate_set_definition': audit.decision_package.candidate_set_definition_sha256,
                'evidence_acquisition_spec': audit.decision_package.evidence_acquisition_spec_sha256,
                'outcome_adjudication_spec': audit.decision_package.outcome_adjudication_spec_sha256,
            }
            if set(protocol_artifacts) != set(expected_protocol_hashes):
                return False
            if any(
                not isinstance(protocol_artifacts[name], bytes) or _sha256(protocol_artifacts[name]) != expected_hash
                for name, expected_hash in expected_protocol_hashes.items()
            ):
                return False
            if admission is None or admission.decision_snapshot.config != DecisionTimeConfig.from_manifest(
                bundle.manifest
            ):
                return False
    except Exception:
        return False
    return True


def _assessment_conclusion(stances: set[EvidenceStance]) -> AssessmentConclusion:
    if not stances:
        return AssessmentConclusion.INSUFFICIENT
    if stances == {EvidenceStance.SUPPORT}:
        return AssessmentConclusion.FAVORABLE
    if stances == {EvidenceStance.CONCERN}:
        return AssessmentConclusion.CONCERN
    return AssessmentConclusion.MIXED


def _revalidate_bound_build_inputs(
    decision: VerifiedDecisionPackage,
    outcome: VerifiedOutcomePackage,
) -> dict[str, str]:
    """Re-derive public text from bound files; a dataclass instance is not an authority token."""

    package = _canonical_reparse(decision.package, LiteratureDecisionPackage, 'verified decision package')
    outcome_package = _canonical_reparse(outcome.package, LiteratureOutcomePackage, 'verified outcome package')
    validate_outcome_package_against_decision(outcome_package, package)
    expected_decision_paths = {
        f'decision/{package.corpus.discovery_protocol_path}',
        f'decision/{package.seal.proof_path}',
        f'decision/{package.extraction.runner_receipt_path}',
        *(f'decision/{document.raw_path}' for document in package.corpus.documents),
        *(f'decision/{document.text_path}' for document in package.corpus.documents),
        *(f'decision/{proof.proof_path}' for document in package.corpus.documents for proof in document.archive_proofs),
    }
    if set(decision.source_files) != expected_decision_paths:
        raise LiteratureAdmissionError('verified decision source inventory changed before episode construction')
    expected_outcome_path = f'outcome/{outcome_package.raw_outcome_path}'
    expected_outcome_proof_path = f'outcome/{outcome_package.availability_proof.proof_path}'
    if set(outcome.source_files) != {expected_outcome_path, expected_outcome_proof_path}:
        raise LiteratureAdmissionError('verified outcome source inventory changed before episode construction')

    texts: dict[str, str] = {}
    discovery_protocol = decision.source_files[f'decision/{package.corpus.discovery_protocol_path}']
    _require_binding(
        discovery_protocol,
        package.corpus.discovery_protocol_sha256,
        package.corpus.discovery_protocol_bytes,
        'corpus discovery protocol',
    )
    documents = {document.document_id: document for document in package.corpus.documents}
    for document in package.corpus.documents:
        raw = decision.source_files[f'decision/{document.raw_path}']
        text_bytes = decision.source_files[f'decision/{document.text_path}']
        _require_binding(raw, document.raw_sha256, document.raw_bytes, document.document_id)
        _require_binding(text_bytes, document.text_sha256, document.text_bytes, f'{document.document_id} text')
        try:
            texts[document.document_id] = text_bytes.decode('utf-8')
        except UnicodeDecodeError as error:
            raise LiteratureAdmissionError(f'text view for {document.document_id} is not UTF-8') from error
    for candidate in package.panel.candidates:
        document = documents[candidate.source_document_id]
        _verify_span(
            decision.source_files[f'decision/{document.text_path}'],
            candidate.source_span,
            candidate.candidate_id,
        )
    for claim in package.extraction.claims:
        document = documents[claim.document_id]
        _verify_span(decision.source_files[f'decision/{document.text_path}'], claim.span, claim.claim_id)
    raw_outcome = outcome.source_files[expected_outcome_path]
    _require_binding(
        raw_outcome,
        outcome_package.raw_outcome_sha256,
        outcome_package.raw_outcome_bytes,
        'raw outcome source',
    )
    proof_bytes = outcome.source_files[expected_outcome_proof_path]
    _require_binding(
        proof_bytes,
        outcome_package.availability_proof.proof_sha256,
        outcome_package.availability_proof.proof_bytes,
        'outcome availability proof',
    )
    return texts


def _verify_span(text: bytes, span: LiteratureByteSpan, label: str) -> None:
    quote = span.quote.encode('utf-8')
    if span.end > len(text) or text[span.start : span.end] != quote:
        raise LiteratureAdmissionError(f'{label} does not match its exact admitted byte span')


def _canonical_reparse[ModelT: StrictModel](value: ModelT, model: type[ModelT], label: str) -> ModelT:
    try:
        reparsed = model.model_validate_json(canonical_json_bytes(value))
    except ValueError as error:
        raise LiteratureAdmissionError(f'invalid {label}: {error}') from error
    return reparsed


def _require_binding(payload: bytes, sha256: str, byte_count: int, label: str) -> None:
    if len(payload) != byte_count or _sha256(payload) != sha256:
        raise LiteratureAdmissionError(f'{label} does not match its declared hash and byte count')


def _resolve_directory(path: Path, label: str) -> Path:
    supplied = path.expanduser()
    if supplied.is_symlink():
        raise LiteratureAdmissionError(f'{label} cannot be a symlink')
    resolved = supplied.resolve()
    if not resolved.is_dir():
        raise LiteratureAdmissionError(f'{label} does not exist: {resolved}')
    return resolved


def _require_exact_file_inventory(root: Path, expected: set[str]) -> None:
    actual: set[str] = set()
    for index, path in enumerate(root.rglob('*'), start=1):
        if index > _MAX_SOURCE_FILES:
            raise LiteratureAdmissionError('literature source inventory exceeds the file-count limit')
        if path.is_symlink():
            raise LiteratureAdmissionError(f'literature source cannot contain symlinks: {path}')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise LiteratureAdmissionError(f'literature source contains a non-regular entry: {path}')
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise LiteratureAdmissionError(f'literature source file allowlist mismatch; missing={missing}, extra={extra}')


def _read_regular_file(root: Path, relative_path: str) -> bytes:
    path = root / relative_path
    if path.is_symlink() or not path.is_file():
        raise LiteratureAdmissionError(f'literature source path is not a regular file: {relative_path}')
    payload = path.read_bytes()
    if len(payload) > _MAX_SOURCE_FILE_BYTES:
        raise LiteratureAdmissionError(f'literature source file exceeds the size limit: {relative_path}')
    return payload


def _write_episode(
    output_root: Path,
    *,
    manifest: EpisodeManifest,
    candidates: tuple[CandidateRecord, ...],
    evidence: tuple[EvidenceRecord, ...],
    private_labels: PrivateLabels,
    ranking_labels: tuple[RankingLabelV1, ...],
    label_key: bytes,
    spec: LiteratureEvaluationConfig,
    source_audit: LiteratureSourceAudit,
) -> None:
    target = output_root.expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise LiteratureAdmissionError(f'episode output already exists: {target}')
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    try:
        private = staging / 'private'
        private.mkdir()
        (staging / 'manifest.json').write_bytes(canonical_json_bytes(manifest))
        (staging / 'candidates.jsonl').write_text(jsonl_text(candidates), encoding='utf-8')
        (staging / 'evidence.jsonl').write_text(jsonl_text(evidence), encoding='utf-8')
        (private / 'outcomes.jsonl').write_text(jsonl_text(tuple(private_labels.outcomes)), encoding='utf-8')
        (private / 'assessments_gold.jsonl').write_text(
            jsonl_text(tuple(private_labels.assessments_gold)),
            encoding='utf-8',
        )
        (private / 'evidence_gold.jsonl').write_text(
            jsonl_text(tuple(private_labels.evidence_gold)),
            encoding='utf-8',
        )
        (private / 'ranking_labels.jsonl').write_text(jsonl_text(ranking_labels), encoding='utf-8')
        (private / 'label_commitment_key.hex').write_text(label_key.hex() + '\n', encoding='ascii')
        (private / 'literature_evaluation_config.json').write_bytes(canonical_json_bytes(spec))
        (private / 'literature_source_audit.json').write_bytes(canonical_json_bytes(source_audit))
        for path in staging.rglob('*'):
            path.chmod(0o700 if path.is_dir() else 0o600)
        staging.chmod(0o700)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
