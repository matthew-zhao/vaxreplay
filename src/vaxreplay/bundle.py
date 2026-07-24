"""Loading, hashing, and integrity validation for VaxReplay episode bundles."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from vaxreplay.case_schema import (
    RANKING_REWARD_VERSION,
    AssessmentConclusion,
    CandidateRecord,
    EpisodeManifest,
    EvidenceRecord,
    GoldAssessmentRecord,
    GoldEvidenceRecord,
    LabelCommitmentScheme,
    OutcomeRecord,
    PrivateLabels,
)
from vaxreplay.ranking_schema import RankingLabelV1


class BundleIntegrityError(ValueError):
    """Raised when an episode bundle fails a deterministic integrity check."""


def body_sha256(body: str) -> str:
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def canonical_json_bytes(value: BaseModel | Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode='json')
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True,
    ).encode('utf-8')


def records_sha256(records: Iterable[BaseModel]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_json_bytes(record))
        digest.update(b'\n')
    return digest.hexdigest()


def manifest_sha256(manifest: EpisodeManifest) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def labels_sha256(labels: PrivateLabels) -> str:
    return labels_commitment(labels, LabelCommitmentScheme.SHA256)


def labels_commitment(
    labels: PrivateLabels,
    scheme: LabelCommitmentScheme,
    *,
    key: bytes | None = None,
) -> str:
    payload = canonical_json_bytes(labels)
    if scheme == LabelCommitmentScheme.SHA256:
        if key is not None:
            raise ValueError('SHA-256 label commitments do not accept a key')
        return hashlib.sha256(payload).hexdigest()
    if key is None or len(key) < 32:
        raise ValueError('HMAC-SHA256 label commitments require a key of at least 32 bytes')
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def ranking_labels_commitment(
    labels: PrivateLabels,
    ranking_labels: Sequence[RankingLabelV1],
    scheme: LabelCommitmentScheme,
    *,
    key: bytes | None = None,
) -> str:
    payload = canonical_json_bytes(
        {
            'private_labels': labels.model_dump(mode='json'),
            'ranking_labels': [label.model_dump(mode='json') for label in ranking_labels],
        }
    )
    if scheme == LabelCommitmentScheme.SHA256:
        if key is not None:
            raise ValueError('SHA-256 ranking-label commitments do not accept a key')
        return hashlib.sha256(payload).hexdigest()
    if key is None or len(key) < 32:
        raise ValueError('HMAC-SHA256 ranking-label commitments require a key of at least 32 bytes')
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _load_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    try:
        with path.open(encoding='utf-8') as source:
            for line in source:
                if line.strip():
                    records.append(model.model_validate_json(line))
    except OSError as error:
        raise BundleIntegrityError(f'cannot read {path}: {error}') from error
    except ValueError as error:
        raise BundleIntegrityError(f'invalid record in {path}: {error}') from error
    if not records:
        raise BundleIntegrityError(f'{path} must contain at least one record')
    return tuple(records)


@dataclass(frozen=True)
class EpisodeBundle:
    root: Path
    manifest: EpisodeManifest
    candidates: tuple[CandidateRecord, ...]
    evidence: tuple[EvidenceRecord, ...]
    private_labels: PrivateLabels | None = None
    label_commitment_key: bytes | None = None
    ranking_labels: tuple[RankingLabelV1, ...] | None = None

    @classmethod
    def load(cls, root: Path, *, include_private: bool = False) -> EpisodeBundle:
        try:
            manifest = EpisodeManifest.model_validate_json((root / 'manifest.json').read_text(encoding='utf-8'))
        except OSError as error:
            raise BundleIntegrityError(f'cannot read {root / "manifest.json"}: {error}') from error
        except ValueError as error:
            raise BundleIntegrityError(f'invalid manifest: {error}') from error

        candidates = _load_jsonl(root / 'candidates.jsonl', CandidateRecord)
        evidence = _load_jsonl(root / 'evidence.jsonl', EvidenceRecord)
        private_labels = None
        label_commitment_key = None
        ranking_labels = None
        if include_private:
            private_labels = PrivateLabels(
                outcomes=list(_load_jsonl(root / 'private' / 'outcomes.jsonl', OutcomeRecord)),
                assessments_gold=list(_load_jsonl(root / 'private' / 'assessments_gold.jsonl', GoldAssessmentRecord)),
                evidence_gold=list(_load_jsonl(root / 'private' / 'evidence_gold.jsonl', GoldEvidenceRecord)),
            )
            if manifest.label_commitment_scheme == LabelCommitmentScheme.HMAC_SHA256:
                label_commitment_key = _load_label_commitment_key(root, manifest)
            if manifest.reward_version == RANKING_REWARD_VERSION:
                ranking_labels = _load_jsonl(root / 'private' / 'ranking_labels.jsonl', RankingLabelV1)
        bundle = cls(
            root=root,
            manifest=manifest,
            candidates=candidates,
            evidence=evidence,
            private_labels=private_labels,
            label_commitment_key=label_commitment_key,
            ranking_labels=ranking_labels,
        )
        bundle.validate_integrity()
        return bundle

    @property
    def manifest_sha256(self) -> str:
        return manifest_sha256(self.manifest)

    @property
    def visible_evidence(self) -> tuple[EvidenceRecord, ...]:
        return tuple(record for record in self.evidence if record.available_at <= self.manifest.decision_at)

    @property
    def evidence_by_id(self) -> dict[str, EvidenceRecord]:
        return {record.evidence_id: record for record in self.evidence}

    def public_view(self) -> dict[str, Any]:
        return {
            'manifest': self.manifest.model_dump(mode='json'),
            'manifest_sha256': self.manifest_sha256,
            'candidates': [record.model_dump(mode='json') for record in self.candidates],
            'evidence': [record.model_dump(mode='json') for record in self.visible_evidence],
        }

    def validate_integrity(self) -> None:
        if records_sha256(self.candidates) != self.manifest.candidates_sha256:
            raise BundleIntegrityError('candidate snapshot hash does not match the manifest')
        if records_sha256(self.evidence) != self.manifest.evidence_sha256:
            raise BundleIntegrityError('evidence snapshot hash does not match the manifest')

        candidate_ids = [record.candidate_id for record in self.candidates if record.eligible]
        if candidate_ids != self.manifest.candidate_ids:
            raise BundleIntegrityError('eligible candidate order must exactly match manifest candidate_ids')
        if len({record.candidate_id for record in self.candidates}) != len(self.candidates):
            raise BundleIntegrityError('candidate IDs must be unique')
        if len({record.evidence_id for record in self.evidence}) != len(self.evidence):
            raise BundleIntegrityError('evidence IDs must be unique')

        candidate_id_set = set(self.manifest.candidate_ids)
        for candidate in self.candidates:
            self._require_episode_id(candidate.episode_id, f'candidate {candidate.candidate_id}')
        for record in self.evidence:
            self._require_episode_id(record.episode_id, f'evidence {record.evidence_id}')
            if body_sha256(record.body) != record.body_sha256:
                raise BundleIntegrityError(f'evidence body hash mismatch for {record.evidence_id}')
            unknown_candidates = set(record.related_candidate_ids) - candidate_id_set
            if unknown_candidates:
                raise BundleIntegrityError(
                    f'evidence {record.evidence_id} references unknown candidates {sorted(unknown_candidates)}'
                )

        if self.private_labels is not None:
            self._validate_private_labels(self.private_labels)
            if self.manifest.reward_version == RANKING_REWARD_VERSION:
                self._validate_ranking_labels()

    def _validate_private_labels(self, labels: PrivateLabels) -> None:
        try:
            if self.manifest.reward_version == RANKING_REWARD_VERSION:
                if self.ranking_labels is None:
                    raise ValueError('V1 episodes require private ranking labels')
                commitment = ranking_labels_commitment(
                    labels,
                    self.ranking_labels,
                    self.manifest.label_commitment_scheme,
                    key=self.label_commitment_key,
                )
            else:
                commitment = labels_commitment(
                    labels,
                    self.manifest.label_commitment_scheme,
                    key=self.label_commitment_key,
                )
        except ValueError as error:
            raise BundleIntegrityError(str(error)) from error
        if commitment != self.manifest.labels_sha256:
            raise BundleIntegrityError('private label hash does not match the manifest commitment')
        candidate_ids = set(self.manifest.candidate_ids)
        expected_targets = {
            (candidate_id, target.target_id, target.horizon_days)
            for candidate_id in candidate_ids
            for target in self.manifest.forecast_targets
        }
        outcome_keys = {(outcome.candidate_id, outcome.target_id, outcome.horizon_days) for outcome in labels.outcomes}
        if outcome_keys != expected_targets or len(outcome_keys) != len(labels.outcomes):
            raise BundleIntegrityError('private outcomes must cover every candidate and forecast target exactly once')

        utility_by_candidate: dict[str, float] = {}
        if all(outcome.outcome is None for outcome in labels.outcomes):
            raise BundleIntegrityError('private outcomes require at least one non-censored forecast label')
        for outcome in labels.outcomes:
            self._require_episode_id(outcome.episode_id, f'outcome {outcome.candidate_id}')
            if outcome.revealed_at <= self.manifest.decision_at:
                raise BundleIntegrityError('private outcomes must be revealed after the decision cutoff')
            previous_utility = utility_by_candidate.setdefault(outcome.candidate_id, outcome.candidate_utility)
            if previous_utility != outcome.candidate_utility:
                raise BundleIntegrityError('candidate utility must be consistent across forecast targets')

        visible_evidence = {record.evidence_id: record for record in self.visible_evidence}
        required_dimensions = set(self.manifest.required_dimensions)
        expected_assessments = {
            (candidate_id, dimension) for candidate_id in candidate_ids for dimension in required_dimensions
        }
        assessment_keys = [(gold.candidate_id, gold.dimension) for gold in labels.assessments_gold]
        if set(assessment_keys) != expected_assessments or len(assessment_keys) != len(set(assessment_keys)):
            raise BundleIntegrityError(
                'gold assessments must cover every candidate and required dimension exactly once'
            )
        for assessment in labels.assessments_gold:
            self._require_episode_id(
                assessment.episode_id,
                f'gold assessment {assessment.candidate_id}/{assessment.dimension}',
            )

        gold_dimensions = {(gold.candidate_id, gold.dimension) for gold in labels.evidence_gold}
        gold_keys = [
            (gold.candidate_id, gold.dimension, gold.evidence_id, gold.stance, gold.quote)
            for gold in labels.evidence_gold
        ]
        if len(gold_keys) != len(set(gold_keys)):
            raise BundleIntegrityError('gold evidence records must be unique')
        non_insufficient_assessments = {
            (gold.candidate_id, gold.dimension)
            for gold in labels.assessments_gold
            if gold.conclusion != AssessmentConclusion.INSUFFICIENT
        }
        if not non_insufficient_assessments.issubset(gold_dimensions):
            raise BundleIntegrityError('non-insufficient gold assessments require at least one gold evidence span')
        for gold in labels.evidence_gold:
            self._require_episode_id(gold.episode_id, f'gold evidence {gold.evidence_id}')
            if gold.candidate_id not in candidate_ids:
                raise BundleIntegrityError(f'gold evidence references unknown candidate {gold.candidate_id}')
            if gold.dimension not in required_dimensions:
                raise BundleIntegrityError(f'gold evidence references unknown dimension {gold.dimension}')
            evidence = visible_evidence.get(gold.evidence_id)
            if evidence is None:
                raise BundleIntegrityError(f'gold evidence {gold.evidence_id} is not visible at the cutoff')
            if gold.quote not in evidence.body:
                raise BundleIntegrityError(f'gold quote is not present in evidence {gold.evidence_id}')

    def _validate_ranking_labels(self) -> None:
        ranking_labels = self.ranking_labels
        if ranking_labels is None:
            raise BundleIntegrityError('V1 episodes require private ranking labels')
        candidate_ids = [label.candidate_id for label in ranking_labels]
        if candidate_ids != self.manifest.candidate_ids:
            raise BundleIntegrityError('V1 ranking labels must cover candidates exactly once in manifest order')
        for label in ranking_labels:
            self._require_episode_id(label.episode_id, f'ranking label {label.candidate_id}')
        if any(label.relevance_grade is None for label in ranking_labels):
            raise BundleIntegrityError('official V1 episodes cannot contain censored ranking labels')
        grades = [label.relevance_grade for label in ranking_labels]
        assert all(grade is not None for grade in grades)
        observed_grades = [grade for grade in grades if grade is not None]
        if len(set(observed_grades)) < 2:
            raise BundleIntegrityError('V1 ranking labels require at least two distinct grades')
        if self.manifest.portfolio_size >= len(candidate_ids):
            raise BundleIntegrityError('V1 portfolio_size must be smaller than the candidate count')
        k = self.manifest.portfolio_size
        best = sum(sorted(observed_grades, reverse=True)[:k])
        worst = sum(sorted(observed_grades)[:k])
        if best <= 0 or best == worst:
            raise BundleIntegrityError('V1 ranking labels have a degenerate top-k utility range')

    def _require_episode_id(self, episode_id: str, label: str) -> None:
        if episode_id != self.manifest.episode_id:
            raise BundleIntegrityError(f'{label} has episode_id {episode_id}, expected {self.manifest.episode_id}')


def resolve_episode_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise BundleIntegrityError(f'episode root does not exist: {root}')
    return root


def _load_label_commitment_key(root: Path, manifest: EpisodeManifest) -> bytes:
    path = root / 'private' / 'label_commitment_key.hex'
    try:
        key = bytes.fromhex(path.read_text(encoding='ascii').strip())
    except OSError as error:
        raise BundleIntegrityError(f'cannot read {path}: {error}') from error
    except ValueError as error:
        raise BundleIntegrityError(f'invalid hexadecimal label commitment key in {path}') from error
    if len(key) < 32:
        raise BundleIntegrityError('label commitment key must contain at least 32 bytes')
    key_id = hashlib.sha256(key).hexdigest()
    if key_id != manifest.label_commitment_key_id:
        raise BundleIntegrityError('label commitment key does not match label_commitment_key_id')
    return key


def jsonl_text(records: Sequence[BaseModel]) -> str:
    return ''.join(canonical_json_bytes(record).decode('utf-8') + '\n' for record in records)
